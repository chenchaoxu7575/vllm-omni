# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Realtime-VLA style Triton decoder fast path for pi0.5.

This module ports the Pi05 action-expert decoder kernel pipeline from
realtime-vla-flash, but keeps vLLM-Omni's preprocessing, prefix forward, and
checkpoint loading.  The fast path consumes prefix KV produced by the regular
PaliGemma prefix pass and runtime-packs weights from the current PyTorch
modules into the layout expected by the Triton kernels.

The decoder intentionally keeps realtime-vla's split attention structure:
QK matmul, prefix/suffix softmax, then AV matmul.  A fused streaming attention
prototype is left below for isolated experiments, but it is not used by the hot
path because it makes the A100 batch=1 graph replay slower than the split GEMMs.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
    from triton.language.extra import libdevice
except ImportError:  # pragma: no cover - optional CUDA fast path
    triton = None
    tl = None
    libdevice = None


def is_available() -> bool:
    return triton is not None and tl is not None and torch.cuda.is_available()


def _validate_prefix_suffix_softmax_window(
    prefix_len: int,
    suffix_len: int,
    *,
    block_size: int = 1024,
) -> None:
    total_len = int(prefix_len) + int(suffix_len)
    if total_len > int(block_size):
        raise ValueError(
            "realtime Triton split attention requires "
            f"prefix_len + suffix_len <= {block_size}; got "
            f"{int(prefix_len)} + {int(suffix_len)} = {total_len}. "
            "Use the safe/default path or add a larger softmax block."
        )


if triton is not None and tl is not None:

    @triton.jit
    def _matmul_small_bias(
        inp_ptr,
        weight_ptr,
        out_ptr,
        bias_ptr,
        seq_len: tl.constexpr,
        features: tl.constexpr,
        hidden: tl.constexpr,
        block_n: tl.constexpr,
        block_m: tl.constexpr,
        block_k: tl.constexpr,
    ):
        pid = tl.program_id(0)
        psize = tl.num_programs(0)
        grid_i = tl.cdiv(seq_len, block_n)
        grid_j = tl.cdiv(hidden, block_m)
        for p in range(pid, grid_i * grid_j, psize):
            i = (p // grid_j) * block_n
            j = (p % grid_j) * block_m
            offs_i = i + tl.arange(0, block_n)
            offs_j = j + tl.arange(0, block_m)
            bias = tl.load(bias_ptr + offs_j, mask=offs_j < hidden, other=0.0).to(tl.float32)
            acc = tl.zeros((block_n, block_m), dtype=tl.float32) + bias[None, :]
            for k in range(0, features, block_k):
                offs_k = k + tl.arange(0, block_k)
                x = tl.load(
                    inp_ptr + offs_i[:, None] * features + offs_k[None, :],
                    mask=(offs_i[:, None] < seq_len) & (offs_k[None, :] < features),
                    other=0.0,
                )
                w = tl.load(
                    weight_ptr + offs_k[:, None] * hidden + offs_j[None, :],
                    mask=(offs_k[:, None] < features) & (offs_j[None, :] < hidden),
                    other=0.0,
                )
                acc = tl.dot(x, w, acc)
            tl.store(
                out_ptr + offs_i[:, None] * hidden + offs_j[None, :],
                acc.to(tl.bfloat16),
                mask=(offs_i[:, None] < seq_len) & (offs_j[None, :] < hidden),
            )

    @triton.jit
    def _matmul_small_res_gate(
        inp_ptr,
        weight_ptr,
        out_ptr,
        res_ptr,
        gate_ptr,
        seq_len: tl.constexpr,
        features: tl.constexpr,
        hidden: tl.constexpr,
        block_n: tl.constexpr,
        block_m: tl.constexpr,
        block_k: tl.constexpr,
    ):
        pid = tl.program_id(0)
        psize = tl.num_programs(0)
        grid_i = tl.cdiv(seq_len, block_n)
        grid_j = tl.cdiv(hidden, block_m)
        for p in range(pid, grid_i * grid_j, psize):
            i = (p // grid_j) * block_n
            j = (p % grid_j) * block_m
            offs_i = i + tl.arange(0, block_n)
            offs_j = j + tl.arange(0, block_m)
            acc = tl.load(
                res_ptr + offs_i[:, None] * hidden + offs_j[None, :],
                mask=(offs_i[:, None] < seq_len) & (offs_j[None, :] < hidden),
                other=0.0,
            ).to(tl.float32)
            matmul_acc = tl.zeros((block_n, block_m), dtype=tl.float32)
            for k in range(0, features, block_k):
                offs_k = k + tl.arange(0, block_k)
                x = tl.load(
                    inp_ptr + offs_i[:, None] * features + offs_k[None, :],
                    mask=(offs_i[:, None] < seq_len) & (offs_k[None, :] < features),
                    other=0.0,
                )
                w = tl.load(
                    weight_ptr + offs_k[:, None] * hidden + offs_j[None, :],
                    mask=(offs_k[:, None] < features) & (offs_j[None, :] < hidden),
                    other=0.0,
                )
                matmul_acc = tl.dot(x, w, matmul_acc)
            gate = tl.load(
                gate_ptr + offs_i[:, None] * hidden + offs_j[None, :],
                mask=(offs_i[:, None] < seq_len) & (offs_j[None, :] < hidden),
                other=0.0,
            ).to(tl.float32)
            acc += matmul_acc * gate
            tl.store(
                out_ptr + offs_i[:, None] * hidden + offs_j[None, :],
                acc.to(tl.bfloat16),
                mask=(offs_i[:, None] < seq_len) & (offs_j[None, :] < hidden),
            )

    @triton.jit
    def _matmul_small_res_gate_oproj(
        inp_ptr,
        weight_ptr,
        out_ptr,
        res_ptr,
        gate_ptr,
        seq_len: tl.constexpr,
        features: tl.constexpr,
        hidden: tl.constexpr,
        block_n: tl.constexpr,
        block_m: tl.constexpr,
        block_k: tl.constexpr,
    ):
        pid = tl.program_id(0)
        psize = tl.num_programs(0)
        grid_i = tl.cdiv(seq_len, block_n)
        grid_j = tl.cdiv(hidden, block_m)
        for p in range(pid, grid_i * grid_j, psize):
            i = (p // grid_j) * block_n
            j = (p % grid_j) * block_m
            offs_i = i + tl.arange(0, block_n)
            offs_j = j + tl.arange(0, block_m)
            acc = tl.load(
                res_ptr + offs_i[:, None] * hidden + offs_j[None, :],
                mask=(offs_i[:, None] < seq_len) & (offs_j[None, :] < hidden),
                other=0.0,
            ).to(tl.float32)
            matmul_acc = tl.zeros((block_n, block_m), dtype=tl.float32)
            for k in range(0, features, block_k):
                offs_k = k + tl.arange(0, block_k)
                x = tl.load(
                    inp_ptr + offs_i[:, None] * features + offs_k[None, :],
                    mask=(offs_i[:, None] < seq_len) & (offs_k[None, :] < features),
                    other=0.0,
                )
                w = tl.load(
                    weight_ptr + offs_k[:, None] * hidden + offs_j[None, :],
                    mask=(offs_k[:, None] < features) & (offs_j[None, :] < hidden),
                    other=0.0,
                )
                matmul_acc = tl.dot(x, w, matmul_acc)
            gate = tl.load(
                gate_ptr + offs_i[:, None] * hidden + offs_j[None, :],
                mask=(offs_i[:, None] < seq_len) & (offs_j[None, :] < hidden),
                other=0.0,
            ).to(tl.float32)
            acc += matmul_acc * gate
            tl.store(
                out_ptr + offs_i[:, None] * hidden + offs_j[None, :],
                acc.to(tl.bfloat16),
                mask=(offs_i[:, None] < seq_len) & (offs_j[None, :] < hidden),
            )

    @triton.jit
    def _matmul_small_res_gate_ffn_down(
        inp_ptr,
        weight_ptr,
        out_ptr,
        res_ptr,
        gate_ptr,
        seq_len: tl.constexpr,
        features: tl.constexpr,
        hidden: tl.constexpr,
        block_n: tl.constexpr,
        block_m: tl.constexpr,
        block_k: tl.constexpr,
    ):
        pid = tl.program_id(0)
        psize = tl.num_programs(0)
        grid_i = tl.cdiv(seq_len, block_n)
        grid_j = tl.cdiv(hidden, block_m)
        for p in range(pid, grid_i * grid_j, psize):
            i = (p // grid_j) * block_n
            j = (p % grid_j) * block_m
            offs_i = i + tl.arange(0, block_n)
            offs_j = j + tl.arange(0, block_m)
            acc = tl.load(
                res_ptr + offs_i[:, None] * hidden + offs_j[None, :],
                mask=(offs_i[:, None] < seq_len) & (offs_j[None, :] < hidden),
                other=0.0,
            ).to(tl.float32)
            matmul_acc = tl.zeros((block_n, block_m), dtype=tl.float32)
            for k in range(0, features, block_k):
                offs_k = k + tl.arange(0, block_k)
                x = tl.load(
                    inp_ptr + offs_i[:, None] * features + offs_k[None, :],
                    mask=(offs_i[:, None] < seq_len) & (offs_k[None, :] < features),
                    other=0.0,
                )
                w = tl.load(
                    weight_ptr + offs_k[:, None] * hidden + offs_j[None, :],
                    mask=(offs_k[:, None] < features) & (offs_j[None, :] < hidden),
                    other=0.0,
                )
                matmul_acc = tl.dot(x, w, matmul_acc)
            gate = tl.load(
                gate_ptr + offs_i[:, None] * hidden + offs_j[None, :],
                mask=(offs_i[:, None] < seq_len) & (offs_j[None, :] < hidden),
                other=0.0,
            ).to(tl.float32)
            acc += matmul_acc * gate
            tl.store(
                out_ptr + offs_i[:, None] * hidden + offs_j[None, :],
                acc.to(tl.bfloat16),
                mask=(offs_i[:, None] < seq_len) & (offs_j[None, :] < hidden),
            )

    @triton.jit
    def _matmul_small_bias_res(
        inp_ptr,
        weight_ptr,
        out_ptr,
        bias_ptr,
        res_ptr,
        seq_len: tl.constexpr,
        features: tl.constexpr,
        hidden: tl.constexpr,
        block_n: tl.constexpr,
        block_m: tl.constexpr,
        block_k: tl.constexpr,
    ):
        pid = tl.program_id(0)
        psize = tl.num_programs(0)
        grid_i = tl.cdiv(seq_len, block_n)
        grid_j = tl.cdiv(hidden, block_m)
        for p in range(pid, grid_i * grid_j, psize):
            i = (p // grid_j) * block_n
            j = (p % grid_j) * block_m
            offs_i = i + tl.arange(0, block_n)
            offs_j = j + tl.arange(0, block_m)
            acc = tl.load(
                res_ptr + offs_i[:, None] * hidden + offs_j[None, :],
                mask=(offs_i[:, None] < seq_len) & (offs_j[None, :] < hidden),
                other=0.0,
            ).to(tl.float32)
            acc += tl.load(
                bias_ptr + offs_j[None, :],
                mask=offs_j[None, :] < hidden,
                other=0.0,
            ).to(tl.float32)
            for k in range(0, features, block_k):
                offs_k = k + tl.arange(0, block_k)
                x = tl.load(
                    inp_ptr + offs_i[:, None] * features + offs_k[None, :],
                    mask=(offs_i[:, None] < seq_len) & (offs_k[None, :] < features),
                    other=0.0,
                )
                w = tl.load(
                    weight_ptr + offs_k[:, None] * hidden + offs_j[None, :],
                    mask=(offs_k[:, None] < features) & (offs_j[None, :] < hidden),
                    other=0.0,
                )
                acc = tl.dot(x, w, acc)
            tl.store(
                out_ptr + offs_i[:, None] * hidden + offs_j[None, :],
                acc.to(tl.bfloat16),
                mask=(offs_i[:, None] < seq_len) & (offs_j[None, :] < hidden),
            )

    @triton.jit
    def _matmul_small_res(
        inp_ptr,
        weight_ptr,
        out_ptr,
        res_ptr,
        seq_len: tl.constexpr,
        features: tl.constexpr,
        hidden: tl.constexpr,
        block_n: tl.constexpr,
        block_m: tl.constexpr,
        block_k: tl.constexpr,
    ):
        pid = tl.program_id(0)
        psize = tl.num_programs(0)
        grid_i = tl.cdiv(seq_len, block_n)
        grid_j = tl.cdiv(hidden, block_m)
        for p in range(pid, grid_i * grid_j, psize):
            i = (p // grid_j) * block_n
            j = (p % grid_j) * block_m
            offs_i = i + tl.arange(0, block_n)
            offs_j = j + tl.arange(0, block_m)
            acc = tl.load(
                res_ptr + offs_i[:, None] * hidden + offs_j[None, :],
                mask=(offs_i[:, None] < seq_len) & (offs_j[None, :] < hidden),
                other=0.0,
            ).to(tl.float32)
            for k in range(0, features, block_k):
                offs_k = k + tl.arange(0, block_k)
                x = tl.load(
                    inp_ptr + offs_i[:, None] * features + offs_k[None, :],
                    mask=(offs_i[:, None] < seq_len) & (offs_k[None, :] < features),
                    other=0.0,
                )
                w = tl.load(
                    weight_ptr + offs_k[:, None] * hidden + offs_j[None, :],
                    mask=(offs_k[:, None] < features) & (offs_j[None, :] < hidden),
                    other=0.0,
                )
                acc = tl.dot(x, w, acc)
            tl.store(
                out_ptr + offs_i[:, None] * hidden + offs_j[None, :],
                acc.to(tl.bfloat16),
                mask=(offs_i[:, None] < seq_len) & (offs_j[None, :] < hidden),
            )

    @triton.jit
    def _rms_norm_kernel(
        x_ptr,
        weight_ptr,
        out_ptr,
        seq_len: tl.constexpr,
        features: tl.constexpr,
        block_size: tl.constexpr,
    ):
        pid = tl.program_id(0)
        psize = tl.num_programs(0)
        for i in range(pid, seq_len, psize):
            row_x_offset = i * features
            sum_sq = tl.zeros((block_size,), dtype=tl.float32)
            for j in range(0, features, block_size):
                cols = j + tl.arange(0, block_size)
                mask = cols < features
                x_val = tl.load(x_ptr + row_x_offset + cols, mask=mask, other=0.0).to(tl.float32)
                sum_sq += x_val * x_val
            rms_factor = tl.rsqrt(tl.sum(sum_sq) / features + 1e-6)
            for j in range(0, features, block_size):
                cols = j + tl.arange(0, block_size)
                mask = cols < features
                x_val = tl.load(x_ptr + row_x_offset + cols, mask=mask, other=0.0).to(tl.float32)
                weight = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)
                out = x_val * rms_factor * (1.0 + weight)
                tl.store(out_ptr + row_x_offset + cols, out.to(tl.bfloat16), mask=mask)

    @triton.jit
    def _adarms_norm_kernel(
        x_ptr,
        style_ptr,
        normed_x_ptr,
        gate_ptr,
        seq_len: tl.constexpr,
        features: tl.constexpr,
        block_size: tl.constexpr,
        rows_per_sample: tl.constexpr,
    ):
        pid = tl.program_id(0)
        psize = tl.num_programs(0)
        for i in range(pid, seq_len, psize):
            row_x_offset = i * features
            # The modulation is per sample, not per row, so rows belonging to
            # sample b read style block b. With seq_len == rows_per_sample this
            # is always block 0, i.e. the single-sample behaviour.
            style_base = style_ptr + (i // rows_per_sample) * 3 * features
            sum_sq = tl.zeros((block_size,), dtype=tl.float32)
            for j in range(0, features, block_size):
                cols = j + tl.arange(0, block_size)
                mask = cols < features
                x_val = tl.load(x_ptr + row_x_offset + cols, mask=mask, other=0.0).to(tl.float32)
                sum_sq += x_val * x_val
            rms_factor = tl.rsqrt(tl.sum(sum_sq) / features + 1e-6)
            for j in range(0, features, block_size):
                cols = j + tl.arange(0, block_size)
                mask = cols < features
                x_val = tl.load(x_ptr + row_x_offset + cols, mask=mask, other=0.0).to(tl.float32)
                scale = tl.load(style_base + cols, mask=mask, other=0.0).to(tl.float32)
                shift = tl.load(style_base + features + cols, mask=mask, other=0.0).to(tl.float32)
                gate = tl.load(style_base + 2 * features + cols, mask=mask, other=0.0).to(tl.float32)
                out = x_val * rms_factor * (1.0 + scale) + shift
                tl.store(normed_x_ptr + row_x_offset + cols, out.to(tl.bfloat16), mask=mask)
                tl.store(gate_ptr + row_x_offset + cols, gate.to(tl.bfloat16), mask=mask)

    @triton.jit
    def _build_gemma_rope_from_positions(
        position_ids_ptr,
        inv_freq_ptr,
        rope_cos_ptr,
        rope_sin_ptr,
        seq_len: tl.constexpr,
        head_dim: tl.constexpr,
        attention_scaling: tl.constexpr,
        block_m: tl.constexpr,
        block_half: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        offs_i = pid * block_m + tl.arange(0, block_m)[:, None]
        offs_d = tl.arange(0, block_half)[None, :]
        half: tl.constexpr = head_dim // 2
        pos = tl.load(position_ids_ptr + offs_i, mask=offs_i < seq_len, other=0).to(tl.float32)
        inv = tl.load(inv_freq_ptr + offs_d, mask=offs_d < half, other=0.0).to(tl.float32)
        phase = pos * inv
        cos = tl.cos(phase) * attention_scaling
        sin = tl.sin(phase) * attention_scaling
        tl.store(
            rope_cos_ptr + offs_i * head_dim + offs_d,
            cos.to(tl.bfloat16),
            mask=(offs_i < seq_len) & (offs_d < half),
        )
        tl.store(
            rope_cos_ptr + offs_i * head_dim + offs_d + half,
            cos.to(tl.bfloat16),
            mask=(offs_i < seq_len) & (offs_d < half),
        )
        tl.store(
            rope_sin_ptr + offs_i * head_dim + offs_d,
            sin.to(tl.bfloat16),
            mask=(offs_i < seq_len) & (offs_d < half),
        )
        tl.store(
            rope_sin_ptr + offs_i * head_dim + offs_d + half,
            sin.to(tl.bfloat16),
            mask=(offs_i < seq_len) & (offs_d < half),
        )

    @triton.jit
    def _matmul_qkv(
        inp_ptr,
        seq_len: tl.constexpr,
        features: tl.constexpr,
        head_dim: tl.constexpr,
        num_heads: tl.constexpr,
        weight_qkv_ptr,
        q_ptr,
        k_ptr,
        v_ptr,
        block_m: tl.constexpr,
        block_n: tl.constexpr,
        block_k: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        psize = tl.num_programs(axis=0)
        grid_m = tl.cdiv(seq_len, block_m)
        grid_n = tl.cdiv((num_heads + 2) * head_dim, block_n)
        while pid < grid_m * grid_n:
            pid_m = pid // grid_n
            pid_n = pid % grid_n
            start_i = pid_m * block_m
            start_j = pid_n * block_n
            offs_i = start_i + tl.arange(0, block_m)[:, None]
            offs_j = start_j + tl.arange(0, block_n)[None, :]
            acc = tl.zeros((block_m, block_n), dtype=tl.float32)
            for k in range(0, features, block_k):
                offs_k = k + tl.arange(0, block_k)
                x = tl.load(
                    inp_ptr + offs_i * features + offs_k[None, :],
                    mask=(offs_i < seq_len) & (offs_k[None, :] < features),
                    other=0.0,
                )
                w = tl.load(
                    weight_qkv_ptr + offs_k[:, None] * ((num_heads + 2) * head_dim) + offs_j,
                    mask=(offs_k[:, None] < features) & (offs_j < (num_heads + 2) * head_dim),
                    other=0.0,
                )
                acc = tl.dot(x, w, acc)
            acc = acc.to(tl.bfloat16)
            if start_j < num_heads * head_dim:
                out_ptr = q_ptr
                out_stride = num_heads * head_dim
            elif start_j < (num_heads + 1) * head_dim:
                out_ptr = k_ptr
                out_stride = head_dim
            else:
                out_ptr = v_ptr
                out_stride = head_dim
            tl.store(
                out_ptr + offs_i * out_stride + offs_j % out_stride,
                acc,
                mask=(offs_i < seq_len) & (offs_j < (num_heads + 2) * head_dim),
            )
            pid += psize

    @triton.jit
    def _matmul_gemma_rope_qkv(
        inp_ptr,
        seq_len: tl.constexpr,
        features: tl.constexpr,
        head_dim: tl.constexpr,
        num_heads: tl.constexpr,
        weight_qkv_ptr,
        rope_cos_ptr,
        rope_sin_ptr,
        q_ptr,
        k_ptr,
        v_ptr,
        block_m: tl.constexpr,
        block_half: tl.constexpr,
        block_k: tl.constexpr,
    ):
        pid_m = tl.program_id(axis=0)
        pid_h = tl.program_id(axis=1)
        start_i = pid_m * block_m
        offs_i = start_i + tl.arange(0, block_m)[:, None]
        offs_d = tl.arange(0, block_half)[None, :]
        half: tl.constexpr = head_dim // 2
        offs_lo = offs_d
        offs_hi = offs_d + half
        base_j = pid_h * head_dim

        acc_lo = tl.zeros((block_m, block_half), dtype=tl.float32)
        acc_hi = tl.zeros((block_m, block_half), dtype=tl.float32)
        for k in range(0, features, block_k):
            offs_k = k + tl.arange(0, block_k)
            x = tl.load(
                inp_ptr + offs_i * features + offs_k[None, :],
                mask=(offs_i < seq_len) & (offs_k[None, :] < features),
                other=0.0,
            )
            w_lo = tl.load(
                weight_qkv_ptr + offs_k[:, None] * ((num_heads + 2) * head_dim) + base_j + offs_lo,
                mask=(offs_k[:, None] < features) & (offs_lo < head_dim),
                other=0.0,
            )
            w_hi = tl.load(
                weight_qkv_ptr + offs_k[:, None] * ((num_heads + 2) * head_dim) + base_j + offs_hi,
                mask=(offs_k[:, None] < features) & (offs_hi < head_dim),
                other=0.0,
            )
            acc_lo = tl.dot(x, w_lo, acc_lo)
            acc_hi = tl.dot(x, w_hi, acc_hi)

        if pid_h < num_heads + 1:
            cos = tl.load(
                rope_cos_ptr + offs_i * head_dim + offs_lo,
                mask=(offs_i < seq_len) & (offs_lo < half),
                other=1.0,
            ).to(tl.float32)
            sin = tl.load(
                rope_sin_ptr + offs_i * head_dim + offs_lo,
                mask=(offs_i < seq_len) & (offs_lo < half),
                other=0.0,
            ).to(tl.float32)
            out_lo = acc_lo * cos - acc_hi * sin
            out_hi = acc_hi * cos + acc_lo * sin
        else:
            out_lo = acc_lo
            out_hi = acc_hi

        if pid_h < num_heads:
            q_row = offs_i * num_heads + pid_h
            tl.store(
                q_ptr + q_row * head_dim + offs_lo,
                out_lo.to(tl.bfloat16),
                mask=(offs_i < seq_len) & (offs_lo < half),
            )
            tl.store(
                q_ptr + q_row * head_dim + offs_hi,
                out_hi.to(tl.bfloat16),
                mask=(offs_i < seq_len) & (offs_lo < half),
            )
        elif pid_h == num_heads:
            tl.store(
                k_ptr + offs_i * head_dim + offs_lo,
                out_lo.to(tl.bfloat16),
                mask=(offs_i < seq_len) & (offs_lo < half),
            )
            tl.store(
                k_ptr + offs_i * head_dim + offs_hi,
                out_hi.to(tl.bfloat16),
                mask=(offs_i < seq_len) & (offs_lo < half),
            )
        else:
            tl.store(
                v_ptr + offs_i * head_dim + offs_lo,
                out_lo.to(tl.bfloat16),
                mask=(offs_i < seq_len) & (offs_lo < half),
            )
            tl.store(
                v_ptr + offs_i * head_dim + offs_hi,
                out_hi.to(tl.bfloat16),
                mask=(offs_i < seq_len) & (offs_lo < half),
            )

    @triton.jit
    def _apply_gemma_rope(
        x_ptr,
        out_ptr,
        rope_cos_ptr,
        rope_sin_ptr,
        rows: tl.constexpr,
        head_dim: tl.constexpr,
        heads_per_token: tl.constexpr,
        block_m: tl.constexpr,
        block_half: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        offs_r = pid * block_m + tl.arange(0, block_m)[:, None]
        half: tl.constexpr = head_dim // 2
        offs_d = tl.arange(0, block_half)[None, :]
        offs_d_hi = offs_d + half
        token = offs_r // heads_per_token
        x_lo = tl.load(
            x_ptr + offs_r * head_dim + offs_d,
            mask=(offs_r < rows) & (offs_d < half),
            other=0.0,
        ).to(tl.float32)
        x_hi = tl.load(
            x_ptr + offs_r * head_dim + offs_d_hi,
            mask=(offs_r < rows) & (offs_d < half),
            other=0.0,
        ).to(tl.float32)
        cos_lo = tl.load(
            rope_cos_ptr + token * head_dim + offs_d,
            mask=(offs_r < rows) & (offs_d < half),
            other=1.0,
        ).to(tl.float32)
        sin_lo = tl.load(
            rope_sin_ptr + token * head_dim + offs_d,
            mask=(offs_r < rows) & (offs_d < half),
            other=0.0,
        ).to(tl.float32)
        cos_hi = tl.load(
            rope_cos_ptr + token * head_dim + offs_d_hi,
            mask=(offs_r < rows) & (offs_d < half),
            other=1.0,
        ).to(tl.float32)
        sin_hi = tl.load(
            rope_sin_ptr + token * head_dim + offs_d_hi,
            mask=(offs_r < rows) & (offs_d < half),
            other=0.0,
        ).to(tl.float32)
        out_lo = x_lo * cos_lo - x_hi * sin_lo
        out_hi = x_hi * cos_hi + x_lo * sin_hi
        tl.store(
            out_ptr + offs_r * head_dim + offs_d,
            out_lo.to(tl.bfloat16),
            mask=(offs_r < rows) & (offs_d < half),
        )
        tl.store(
            out_ptr + offs_r * head_dim + offs_d_hi,
            out_hi.to(tl.bfloat16),
            mask=(offs_r < rows) & (offs_d < half),
        )

    @triton.jit
    def _matmul_rope_qkv(
        inp_ptr,
        seq_len: tl.constexpr,
        features: tl.constexpr,
        head_dim: tl.constexpr,
        num_heads: tl.constexpr,
        weight_qkv_ptr,
        rope_weights_ptr,
        q_ptr,
        k_ptr,
        v_ptr,
        block_m: tl.constexpr,
        block_n: tl.constexpr,
        block_k: tl.constexpr,
        kv_rows_per_sample: tl.constexpr,
        kv_sample_stride: tl.constexpr,
        kv_row_offset: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        psize = tl.num_programs(axis=0)
        grid_m = tl.cdiv(seq_len, block_m)
        grid_n = tl.cdiv((num_heads + 2) * head_dim, block_n)
        while pid < grid_m * grid_n:
            pid_m = pid // grid_n
            pid_n = pid % grid_n
            start_i = pid_m * block_m
            start_j = pid_n * block_n
            offs_i = start_i + tl.arange(0, block_m)[:, None]
            offs_j = start_j + tl.arange(0, block_n)[None, :]
            acc = tl.zeros((block_m, block_n), dtype=tl.float32)
            for k in range(0, features, block_k):
                offs_k = k + tl.arange(0, block_k)
                x = tl.load(
                    inp_ptr + offs_i * features + offs_k[None, :],
                    mask=(offs_i < seq_len) & (offs_k[None, :] < features),
                    other=0.0,
                )
                w = tl.load(
                    weight_qkv_ptr + offs_k[:, None] * ((num_heads + 2) * head_dim) + offs_j,
                    mask=(offs_k[:, None] < features) & (offs_j < (num_heads + 2) * head_dim),
                    other=0.0,
                )
                acc = tl.dot(x, w, acc)
            qk_cols: tl.constexpr = (num_heads + 1) * head_dim
            if start_j < qk_cols:
                x0, x1 = tl.split(acc.reshape(block_m, block_n // 2, 2))
                x_cossin = tl.load(
                    rope_weights_ptr + offs_i * head_dim + offs_j % head_dim,
                    mask=(offs_i < seq_len) & (offs_j < qk_cols),
                    other=0.0,
                )
                x_cos, x_sin = tl.split(x_cossin.reshape(block_m, block_n // 2, 2))
                rot0 = x0 * x_cos - x1 * x_sin
                rot1 = x1 * x_cos + x0 * x_sin
                acc = tl.interleave(rot0, rot1)
            acc = acc.to(tl.bfloat16)
            # Q rows stay folded (sample b owns a contiguous block), but K/V go
            # into the decoder's cache, where each sample's prefix and suffix sit
            # together. Suffix row t of sample b therefore lands at
            # b*kv_sample_stride + kv_row_offset + t.
            kv_row = (
                (offs_i // kv_rows_per_sample) * kv_sample_stride
                + kv_row_offset
                + (offs_i % kv_rows_per_sample)
            )
            if start_j < num_heads * head_dim:
                out_ptr = q_ptr
                out_stride = num_heads * head_dim
                out_row = offs_i
            elif start_j < (num_heads + 1) * head_dim:
                out_ptr = k_ptr
                out_stride = head_dim
                out_row = kv_row
            else:
                out_ptr = v_ptr
                out_stride = head_dim
                out_row = kv_row
            tl.store(
                out_ptr + out_row * out_stride + offs_j % out_stride,
                acc,
                mask=(offs_i < seq_len) & (offs_j < (num_heads + 2) * head_dim),
            )
            pid += psize

    @triton.jit
    def _matmul_abt_scale(
        q_ptr,
        k_ptr,
        out_ptr,
        rows_q: tl.constexpr,
        rows_k: tl.constexpr,
        cols: tl.constexpr,
        scale_factor: tl.constexpr,
        block_m: tl.constexpr,
        block_n: tl.constexpr,
        block_k: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        psize = tl.num_programs(axis=0)
        # axis 1 is the batch index; a 1-D launch leaves it 0, so single-sample
        # callers keep their exact previous behaviour.
        pid_b = tl.program_id(axis=1)
        q_base = q_ptr + pid_b * rows_q * cols
        k_base = k_ptr + pid_b * rows_k * cols
        out_base = out_ptr + pid_b * rows_q * rows_k
        grid_m = tl.cdiv(rows_q, block_m)
        grid_n = tl.cdiv(rows_k, block_n)
        while pid < grid_m * grid_n:
            pid_m = pid // grid_n
            pid_n = pid % grid_n
            offs_i = pid_m * block_m + tl.arange(0, block_m)
            offs_j = pid_n * block_n + tl.arange(0, block_n)
            acc = tl.zeros((block_m, block_n), dtype=tl.float32)
            for k in range(0, cols, block_k):
                offs_k = k + tl.arange(0, block_k)
                q = tl.load(q_base + offs_i[:, None] * cols + offs_k[None, :], mask=offs_i[:, None] < rows_q, other=0)
                key = tl.load(k_base + offs_j[:, None] * cols + offs_k[None, :], mask=offs_j[:, None] < rows_k, other=0)
                acc = tl.dot(q, tl.trans(key), acc)
            acc = acc * scale_factor
            tl.store(
                out_base + offs_i[:, None] * rows_k + offs_j[None, :],
                acc,
                mask=(offs_i[:, None] < rows_q) & (offs_j[None, :] < rows_k),
            )
            pid += psize

    @triton.jit
    def _attention_prefix_suffix_fused(
        q_ptr,
        k_ptr,
        v_ptr,
        out_ptr,
        prefix_mask_ptr,
        rows_q: tl.constexpr,
        keys_prefix: tl.constexpr,
        keys_suffix: tl.constexpr,
        head_dim: tl.constexpr,
        scale_factor: tl.constexpr,
        block_m: tl.constexpr,
        block_n: tl.constexpr,
        block_d: tl.constexpr,
    ):
        pid_m = tl.program_id(axis=0)
        offs_m = pid_m * block_m + tl.arange(0, block_m)
        offs_n = tl.arange(0, block_n)
        offs_d = tl.arange(0, block_d)
        total_keys: tl.constexpr = keys_prefix + keys_suffix

        q = tl.load(
            q_ptr + offs_m[:, None] * head_dim + offs_d[None, :],
            mask=(offs_m[:, None] < rows_q) & (offs_d[None, :] < head_dim),
            other=0.0,
        )
        m_i = tl.full((block_m,), -float("inf"), tl.float32)
        l_i = tl.zeros((block_m,), tl.float32)
        acc = tl.zeros((block_m, block_d), tl.float32)

        for start_n in tl.range(0, total_keys, block_n):
            cur_n = start_n + offs_n
            is_prefix = cur_n < keys_prefix
            prefix_ok = tl.load(prefix_mask_ptr + cur_n, mask=cur_n < keys_prefix, other=0).to(tl.int1)
            valid_key = (is_prefix & prefix_ok) | (~is_prefix & (cur_n < total_keys))
            k = tl.load(
                k_ptr + cur_n[None, :] * head_dim + offs_d[:, None],
                mask=(cur_n[None, :] < total_keys) & (offs_d[:, None] < head_dim),
                other=0.0,
            )
            qk = tl.dot(q, k) * scale_factor
            qk = tl.where((offs_m[:, None] < rows_q) & valid_key[None, :], qk, -float("inf"))

            m_new = tl.maximum(m_i, tl.max(qk, axis=1))
            alpha = tl.exp(m_i - m_new)
            p = tl.exp(qk - m_new[:, None])
            l_new = l_i * alpha + tl.sum(p, axis=1)

            v = tl.load(
                v_ptr + cur_n[:, None] * head_dim + offs_d[None, :],
                mask=valid_key[:, None] & (offs_d[None, :] < head_dim),
                other=0.0,
            )
            acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
            m_i = m_new
            l_i = l_new

        acc = acc / l_i[:, None]
        tl.store(
            out_ptr + offs_m[:, None] * head_dim + offs_d[None, :],
            acc.to(tl.bfloat16),
            mask=(offs_m[:, None] < rows_q) & (offs_d[None, :] < head_dim),
        )

    @triton.jit
    def _softmax_prefix_suffix(
        inp_ptr,
        queries: tl.constexpr,
        keys_prefix: tl.constexpr,
        keys_suffix: tl.constexpr,
        valid_prefix_len_ptr,
        out_ptr,
        block_m: tl.constexpr,
        block_size: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        psize = tl.num_programs(axis=0)
        big_neg = -2.3819763e38
        total_keys: tl.constexpr = keys_prefix + keys_suffix
        valid_prefix_len = tl.load(valid_prefix_len_ptr).to(tl.int32)
        valid_prefix_len = tl.maximum(0, tl.minimum(valid_prefix_len, keys_prefix))
        for i in range(pid * block_m, queries, psize * block_m):
            offs_i = i + tl.arange(0, block_m)[:, None]
            offs_j = tl.arange(0, block_size)[None, :]
            in_bounds = (offs_i < queries) & (offs_j < total_keys)
            is_prefix = offs_j < keys_prefix
            prefix_ok = is_prefix & (offs_j < valid_prefix_len)
            suffix_ok = ~is_prefix
            mask = in_bounds & (prefix_ok | suffix_ok)
            vals = tl.load(inp_ptr + offs_i * total_keys + offs_j, mask=mask, other=big_neg)
            vals = tl.exp(vals - tl.max(vals, axis=1, keep_dims=True))
            vals = vals / tl.sum(vals, axis=1, keep_dims=True, dtype=tl.float32)
            tl.store(out_ptr + offs_i * total_keys + offs_j, vals.to(tl.bfloat16), mask=in_bounds)

    @triton.jit
    def _softmax_masklen(
        inp_ptr,
        queries: tl.constexpr,
        keys: tl.constexpr,
        valid_keys_len_ptr,
        out_ptr,
        block_m: tl.constexpr,
        block_size: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        psize = tl.num_programs(axis=0)
        big_neg = -2.3819763e38
        valid_keys_len = tl.load(valid_keys_len_ptr).to(tl.int32)
        valid_keys_len = tl.maximum(0, tl.minimum(valid_keys_len, keys))
        for i in range(pid * block_m, queries, psize * block_m):
            offs_i = i + tl.arange(0, block_m)[:, None]
            offs_j = tl.arange(0, block_size)[None, :]
            mask = (offs_i < queries) & (offs_j < valid_keys_len)
            vals = tl.load(inp_ptr + offs_i * keys + offs_j, mask=mask, other=big_neg)
            vals = tl.exp(vals - tl.max(vals, axis=1, keep_dims=True))
            vals = vals / tl.sum(vals, axis=1, keep_dims=True, dtype=tl.float32)
            tl.store(out_ptr + offs_i * keys + offs_j, vals.to(tl.bfloat16), mask=(offs_i < queries) & (offs_j < keys))

    @triton.jit
    def _softmax_mask_vector(
        inp_ptr,
        queries: tl.constexpr,
        keys: tl.constexpr,
        key_mask_ptr,
        out_ptr,
        block_m: tl.constexpr,
        block_size: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        psize = tl.num_programs(axis=0)
        # axis 1 is the batch index; a 1-D launch leaves it 0.
        pid_b = tl.program_id(axis=1)
        inp_base = inp_ptr + pid_b * queries * keys
        out_base = out_ptr + pid_b * queries * keys
        mask_base = key_mask_ptr + pid_b * keys
        big_neg = -2.3819763e38
        for i in range(pid * block_m, queries, psize * block_m):
            offs_i = i + tl.arange(0, block_m)[:, None]
            offs_j = tl.arange(0, block_size)[None, :]
            key_ok = tl.load(mask_base + offs_j, mask=offs_j < keys, other=0).to(tl.int1)
            mask = (offs_i < queries) & (offs_j < keys) & key_ok
            vals = tl.load(inp_base + offs_i * keys + offs_j, mask=mask, other=big_neg)
            vals = tl.exp(vals - tl.max(vals, axis=1, keep_dims=True))
            vals = vals / tl.sum(vals, axis=1, keep_dims=True, dtype=tl.float32)
            tl.store(out_base + offs_i * keys + offs_j, vals.to(tl.bfloat16), mask=(offs_i < queries) & (offs_j < keys))

    @triton.jit
    def _softmax_prefix_suffix_mask_vector(
        inp_ptr,
        queries: tl.constexpr,
        keys_prefix: tl.constexpr,
        keys_suffix: tl.constexpr,
        prefix_mask_ptr,
        out_ptr,
        block_m: tl.constexpr,
        block_size: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        psize = tl.num_programs(axis=0)
        # axis 1 is the batch index; a 1-D launch leaves it 0.
        pid_b = tl.program_id(axis=1)
        big_neg = -2.3819763e38
        total_keys: tl.constexpr = keys_prefix + keys_suffix
        inp_base = inp_ptr + pid_b * queries * total_keys
        out_base = out_ptr + pid_b * queries * total_keys
        mask_base = prefix_mask_ptr + pid_b * keys_prefix
        for i in range(pid * block_m, queries, psize * block_m):
            offs_i = i + tl.arange(0, block_m)[:, None]
            offs_j = tl.arange(0, block_size)[None, :]
            is_prefix = offs_j < keys_prefix
            prefix_ok = tl.load(mask_base + offs_j, mask=offs_j < keys_prefix, other=0).to(tl.int1)
            suffix_ok = ~is_prefix
            mask = (offs_i < queries) & (offs_j < total_keys) & ((is_prefix & prefix_ok) | suffix_ok)
            vals = tl.load(inp_base + offs_i * total_keys + offs_j, mask=mask, other=big_neg)
            vals = tl.exp(vals - tl.max(vals, axis=1, keep_dims=True))
            vals = vals / tl.sum(vals, axis=1, keep_dims=True, dtype=tl.float32)
            tl.store(
                out_base + offs_i * total_keys + offs_j,
                vals.to(tl.bfloat16),
                mask=(offs_i < queries) & (offs_j < total_keys),
            )

    @triton.jit
    def _matmul_small(
        inp_ptr,
        weight_ptr,
        out_ptr,
        seq_len: tl.constexpr,
        features: tl.constexpr,
        hidden: tl.constexpr,
        block_n: tl.constexpr,
        block_m: tl.constexpr,
        block_k: tl.constexpr,
    ):
        pid = tl.program_id(0)
        psize = tl.num_programs(0)
        grid_i = tl.cdiv(seq_len, block_n)
        grid_j = tl.cdiv(hidden, block_m)
        for p in range(pid, grid_i * grid_j, psize):
            i = (p // grid_j) * block_n
            j = (p % grid_j) * block_m
            offs_i = i + tl.arange(0, block_n)
            offs_j = j + tl.arange(0, block_m)
            acc = tl.zeros((block_n, block_m), dtype=tl.float32)
            for k in range(0, features, block_k):
                offs_k = k + tl.arange(0, block_k)
                x = tl.load(
                    inp_ptr + offs_i[:, None] * features + offs_k[None, :],
                    mask=(offs_i[:, None] < seq_len) & (offs_k[None, :] < features),
                    other=0.0,
                )
                w = tl.load(
                    weight_ptr + offs_k[:, None] * hidden + offs_j[None, :],
                    mask=(offs_k[:, None] < features) & (offs_j[None, :] < hidden),
                    other=0.0,
                )
                acc = tl.dot(x, w, acc)
            tl.store(
                out_ptr + offs_i[:, None] * hidden + offs_j[None, :],
                acc.to(tl.bfloat16),
                mask=(offs_i[:, None] < seq_len) & (offs_j[None, :] < hidden),
            )

    @triton.jit
    def _matmul_small_bmm(
        inp_ptr,
        weight_ptr,
        out_ptr,
        seq_len: tl.constexpr,
        features: tl.constexpr,
        hidden: tl.constexpr,
        block_n: tl.constexpr,
        block_m: tl.constexpr,
        block_k: tl.constexpr,
    ):
        """``_matmul_small`` for the attention-times-V role.

        Identical maths, but the right-hand operand is V -- per sample, not a
        shared weight -- so it is offset by the batch index too.  A 1-D launch
        leaves ``pid_b`` at 0 and reproduces ``_matmul_small`` exactly.
        """
        pid = tl.program_id(0)
        psize = tl.num_programs(0)
        pid_b = tl.program_id(1)
        inp_base = inp_ptr + pid_b * seq_len * features
        w_base = weight_ptr + pid_b * features * hidden
        out_base = out_ptr + pid_b * seq_len * hidden
        grid_i = tl.cdiv(seq_len, block_n)
        grid_j = tl.cdiv(hidden, block_m)
        for p in range(pid, grid_i * grid_j, psize):
            i = (p // grid_j) * block_n
            j = (p % grid_j) * block_m
            offs_i = i + tl.arange(0, block_n)
            offs_j = j + tl.arange(0, block_m)
            acc = tl.zeros((block_n, block_m), dtype=tl.float32)
            for k in range(0, features, block_k):
                offs_k = k + tl.arange(0, block_k)
                x = tl.load(
                    inp_base + offs_i[:, None] * features + offs_k[None, :],
                    mask=(offs_i[:, None] < seq_len) & (offs_k[None, :] < features),
                    other=0.0,
                )
                w = tl.load(
                    w_base + offs_k[:, None] * hidden + offs_j[None, :],
                    mask=(offs_k[:, None] < features) & (offs_j[None, :] < hidden),
                    other=0.0,
                )
                acc = tl.dot(x, w, acc)
            tl.store(
                out_base + offs_i[:, None] * hidden + offs_j[None, :],
                acc.to(tl.bfloat16),
                mask=(offs_i[:, None] < seq_len) & (offs_j[None, :] < hidden),
            )

    @triton.jit
    def _matmul_small_gate(
        inp_ptr,
        weight1_ptr,
        weight2_ptr,
        out_ptr,
        seq_len: tl.constexpr,
        features: tl.constexpr,
        hidden: tl.constexpr,
        block_n: tl.constexpr,
        block_m: tl.constexpr,
        block_k: tl.constexpr,
    ):
        pid1 = tl.program_id(axis=0)
        psize1 = tl.num_programs(axis=0)
        pid2 = tl.program_id(axis=1)
        psize2 = tl.num_programs(axis=1)
        for i in range(pid1 * block_n, seq_len, psize1 * block_n):
            for j in range(pid2 * block_m, hidden, psize2 * block_m):
                offs_i = i + tl.arange(0, block_n)
                offs_j = j + tl.arange(0, block_m)
                acc = tl.zeros((block_n, block_m), dtype=tl.float32)
                acc2 = tl.zeros((block_n, block_m), dtype=tl.float32)
                for k in range(0, features, block_k):
                    offs_k = k + tl.arange(0, block_k)
                    x = tl.load(
                        inp_ptr + offs_i[:, None] * features + offs_k[None, :],
                        mask=(offs_i[:, None] < seq_len) & (offs_k[None, :] < features),
                        other=0.0,
                    )
                    w1 = tl.load(
                        weight1_ptr + offs_k[:, None] * hidden + offs_j[None, :],
                        mask=(offs_k[:, None] < features) & (offs_j[None, :] < hidden),
                        other=0.0,
                    )
                    w2 = tl.load(
                        weight2_ptr + offs_k[:, None] * hidden + offs_j[None, :],
                        mask=(offs_k[:, None] < features) & (offs_j[None, :] < hidden),
                        other=0.0,
                    )
                    acc = tl.dot(x, w1, acc)
                    acc2 = tl.dot(x, w2, acc2)
                acc = acc * tl.sigmoid(1.5957691216057308 * acc * (1 + 0.044715 * acc * acc))
                tl.store(
                    out_ptr + offs_i[:, None] * hidden + offs_j[None, :],
                    (acc * acc2).to(tl.bfloat16),
                    mask=(offs_i[:, None] < seq_len) & (offs_j[None, :] < hidden),
                )

    @triton.jit
    def _matmul_small_gate_gelu_tanh(
        inp_ptr,
        weight1_ptr,
        weight2_ptr,
        out_ptr,
        seq_len: tl.constexpr,
        features: tl.constexpr,
        hidden: tl.constexpr,
        block_n: tl.constexpr,
        block_m: tl.constexpr,
        block_k: tl.constexpr,
    ):
        pid1 = tl.program_id(axis=0)
        psize1 = tl.num_programs(axis=0)
        pid2 = tl.program_id(axis=1)
        psize2 = tl.num_programs(axis=1)
        for i in range(pid1 * block_n, seq_len, psize1 * block_n):
            for j in range(pid2 * block_m, hidden, psize2 * block_m):
                offs_i = i + tl.arange(0, block_n)
                offs_j = j + tl.arange(0, block_m)
                gate_acc = tl.zeros((block_n, block_m), dtype=tl.float32)
                up_acc = tl.zeros((block_n, block_m), dtype=tl.float32)
                for k in range(0, features, block_k):
                    offs_k = k + tl.arange(0, block_k)
                    x = tl.load(
                        inp_ptr + offs_i[:, None] * features + offs_k[None, :],
                        mask=(offs_i[:, None] < seq_len) & (offs_k[None, :] < features),
                        other=0.0,
                    )
                    w1 = tl.load(
                        weight1_ptr + offs_k[:, None] * hidden + offs_j[None, :],
                        mask=(offs_k[:, None] < features) & (offs_j[None, :] < hidden),
                        other=0.0,
                    )
                    w2 = tl.load(
                        weight2_ptr + offs_k[:, None] * hidden + offs_j[None, :],
                        mask=(offs_k[:, None] < features) & (offs_j[None, :] < hidden),
                        other=0.0,
                    )
                    gate_acc = tl.dot(x, w1, gate_acc)
                    up_acc = tl.dot(x, w2, up_acc)
                gelu = (
                    0.5
                    * gate_acc
                    * (
                        1.0
                        + libdevice.tanh(0.7978845608028654 * (gate_acc + 0.044715 * gate_acc * gate_acc * gate_acc))
                    )
                )
                tl.store(
                    out_ptr + offs_i[:, None] * hidden + offs_j[None, :],
                    (gelu * up_acc).to(tl.bfloat16),
                    mask=(offs_i[:, None] < seq_len) & (offs_j[None, :] < hidden),
                )


if triton is None or tl is None:
    _attention_prefix_suffix_fused = None


def _as_bf16_contig(t: torch.Tensor) -> torch.Tensor:
    return t.detach().to(device="cuda", dtype=torch.bfloat16).contiguous()


def _linear_weight_t(linear: nn.Linear) -> torch.Tensor:
    return _as_bf16_contig(linear.weight.t())


def _linear_bias(linear: nn.Linear, out_features: int) -> torch.Tensor:
    if linear.bias is None:
        return torch.zeros(out_features, device="cuda", dtype=torch.bfloat16)
    return _as_bf16_contig(linear.bias)


def _prefix_mlp_torch_impl(
    x_normed: torch.Tensor,
    residual: torch.Tensor,
    gate_w: torch.Tensor,
    up_w: torch.Tensor,
    down_w: torch.Tensor,
) -> torch.Tensor:
    gate = F.gelu(torch.matmul(x_normed, gate_w), approximate="tanh")
    hidden = gate * torch.matmul(x_normed, up_w)
    return torch.matmul(hidden, down_w) + residual


def _prefix_mlp_torch_packed_gate_up_impl(
    x_normed: torch.Tensor,
    residual: torch.Tensor,
    gate_up_w: torch.Tensor,
    down_w: torch.Tensor,
) -> torch.Tensor:
    gate_up = torch.matmul(x_normed, gate_up_w)
    gate, up = torch.chunk(gate_up, 2, dim=-1)
    hidden = F.gelu(gate, approximate="tanh") * up
    return torch.matmul(hidden, down_w) + residual


@dataclass
class _PrefixEncoderBuffers:
    x: torch.Tensor
    x_normed: torch.Tensor
    q_raw: torch.Tensor
    q: torch.Tensor
    k_raw: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    logits: torch.Tensor
    attn: torch.Tensor
    hidden: torch.Tensor
    valid_prefix_len: torch.Tensor
    prefix_mask: torch.Tensor
    rope_cos: torch.Tensor
    rope_sin: torch.Tensor


@dataclass
class _DecoderBuffers:
    noise: torch.Tensor
    x: torch.Tensor
    x_normed: torch.Tensor
    gate: torch.Tensor
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    logits: torch.Tensor
    attn: torch.Tensor
    hidden: torch.Tensor
    valid_prefix_len: torch.Tensor
    prefix_mask: torch.Tensor
    rope: torch.Tensor


class Pi05RealtimePrefixEncoder:
    """Fixed-shape realtime prefix encoder for the Pi05 Triton decoder.

    This is intentionally narrower than the regular PaliGemma forward path:
    batch=1, one KV head, and ``prefix_len <= 1024``.  It consumes already-built
    ``prefix_embs`` plus the real prefix mask/position ids from vLLM-Omni
    preprocessing and returns the same per-layer K/V shape expected by
    ``Pi05RealtimeTritonDecoder``.
    """

    def __init__(self, model: Any):
        if not is_available():
            raise RuntimeError("Pi05RealtimePrefixEncoder requires CUDA and Triton")
        self.model = model
        language_model = model.paligemma_with_expert.paligemma.model.language_model
        self.layers = language_model.layers
        self.num_layers = len(self.layers)
        self.hidden_size = int(language_model.config.hidden_size)
        first_attn = self.layers[0].self_attn
        self.head_dim = int(first_attn.head_dim)
        self.num_heads = int(first_attn.q_proj.out_features // self.head_dim)
        self.num_kv_heads = int(first_attn.k_proj.out_features // self.head_dim)
        self.rope_inv_freq = language_model.rotary_emb.inv_freq.detach().to(device="cuda").contiguous()
        self.rope_attention_scaling = float(getattr(language_model.rotary_emb, "attention_scaling", 1.0))
        if self.hidden_size != 2048 or self.head_dim != 256 or self.num_heads != 8 or self.num_kv_heads != 1:
            raise ValueError(
                "realtime prefix encoder currently specializes pi0.5 base: "
                f"H={self.hidden_size}, heads={self.num_heads}, kv_heads={self.num_kv_heads}, D={self.head_dim}"
            )
        self.use_torch_prefix_mlp = os.environ.get("PI05_REALTIME_PREFIX_TORCH_MLP", "1").lower() not in {
            "0",
            "false",
            "no",
        }
        self.compile_torch_prefix_mlp = os.environ.get("PI05_REALTIME_PREFIX_COMPILE_MLP", "1").lower() not in {
            "0",
            "false",
            "no",
        }
        self.prefix_mlp_compile_mode = os.environ.get(
            "PI05_REALTIME_PREFIX_COMPILE_MODE",
            "max-autotune-no-cudagraphs",
        ).strip()
        if self.prefix_mlp_compile_mode.lower() in {"", "default", "none"}:
            self.prefix_mlp_compile_mode = ""
        self.prefix_mlp_compile_fullgraph = os.environ.get(
            "PI05_REALTIME_PREFIX_COMPILE_FULLGRAPH",
            "1",
        ).lower() not in {
            "0",
            "false",
            "no",
        }
        self.use_packed_prefix_gate_up = os.environ.get(
            "PI05_REALTIME_PREFIX_PACKED_GATE_UP",
            "1",
        ).lower() not in {
            "0",
            "false",
            "no",
        }
        self._prefix_mlp_fn = None
        self._prefix_mlp_packed_fn = None
        self.weights = self._pack_weights()
        self.buffers_by_shape: dict[tuple[int, torch.dtype], _PrefixEncoderBuffers] = {}

    def _pack_weights(self) -> dict[str, torch.Tensor]:
        weights: dict[str, torch.Tensor] = {
            "attn_qkv_w": torch.empty(
                self.num_layers,
                self.hidden_size,
                (self.num_heads + 2) * self.head_dim,
                device="cuda",
                dtype=torch.bfloat16,
            ),
            "attn_o_w": torch.empty(
                self.num_layers,
                self.hidden_size,
                self.hidden_size,
                device="cuda",
                dtype=torch.bfloat16,
            ),
            "ffn_gate_w": torch.empty(
                self.num_layers,
                self.hidden_size,
                16384,
                device="cuda",
                dtype=torch.bfloat16,
            ),
            "ffn_up_w": torch.empty(
                self.num_layers,
                self.hidden_size,
                16384,
                device="cuda",
                dtype=torch.bfloat16,
            ),
            "ffn_gate_up_w": torch.empty(
                self.num_layers,
                self.hidden_size,
                32768,
                device="cuda",
                dtype=torch.bfloat16,
            ),
            "ffn_down_w": torch.empty(
                self.num_layers,
                16384,
                self.hidden_size,
                device="cuda",
                dtype=torch.bfloat16,
            ),
            "input_norm_w": torch.empty(
                self.num_layers,
                self.hidden_size,
                device="cuda",
                dtype=torch.bfloat16,
            ),
            "post_norm_w": torch.empty(
                self.num_layers,
                self.hidden_size,
                device="cuda",
                dtype=torch.bfloat16,
            ),
        }
        for idx, layer in enumerate(self.layers):
            attn = layer.self_attn
            weights["attn_qkv_w"][idx].copy_(
                torch.cat(
                    [
                        attn.q_proj.weight.detach().t(),
                        attn.k_proj.weight.detach().t(),
                        attn.v_proj.weight.detach().t(),
                    ],
                    dim=1,
                ).to(device="cuda", dtype=torch.bfloat16)
            )
            weights["attn_o_w"][idx].copy_(_linear_weight_t(attn.o_proj))
            weights["ffn_gate_w"][idx].copy_(_linear_weight_t(layer.mlp.gate_proj))
            weights["ffn_up_w"][idx].copy_(_linear_weight_t(layer.mlp.up_proj))
            weights["ffn_gate_up_w"][idx].copy_(
                torch.cat(
                    [
                        _linear_weight_t(layer.mlp.gate_proj),
                        _linear_weight_t(layer.mlp.up_proj),
                    ],
                    dim=1,
                )
            )
            weights["ffn_down_w"][idx].copy_(_linear_weight_t(layer.mlp.down_proj))
            weights["input_norm_w"][idx].copy_(_as_bf16_contig(layer.input_layernorm.weight))
            weights["post_norm_w"][idx].copy_(_as_bf16_contig(layer.post_attention_layernorm.weight))
        return weights

    def _build_rope(self, rows: int) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.empty(rows, self.head_dim, device="cuda", dtype=torch.bfloat16),
            torch.empty(rows, self.head_dim, device="cuda", dtype=torch.bfloat16),
        )

    def _get_buffers(self, batch: int, prefix_len: int, dtype: torch.dtype) -> _PrefixEncoderBuffers:
        key = (int(batch), int(prefix_len), dtype)
        cached = self.buffers_by_shape.get(key)
        if cached is not None:
            return cached
        _validate_prefix_suffix_softmax_window(prefix_len, 0)
        # Batch is folded into the token dimension: row ``b * prefix_len + t``.
        # Every weight-shared kernel (norms, o-proj, MLP, QKV+RoPE) then works
        # unchanged on ``rows`` rows; only the three attention kernels, whose
        # K/V operand is per-sample, need an explicit batch index.
        rows = batch * prefix_len
        rows_q = rows * self.num_heads
        rope_cos, rope_sin = self._build_rope(rows)
        buffers = _PrefixEncoderBuffers(
            x=torch.empty(rows, self.hidden_size, device="cuda", dtype=torch.bfloat16),
            x_normed=torch.empty(rows, self.hidden_size, device="cuda", dtype=torch.bfloat16),
            q_raw=torch.empty(rows_q, self.head_dim, device="cuda", dtype=torch.bfloat16),
            q=torch.empty(rows_q, self.head_dim, device="cuda", dtype=torch.bfloat16),
            k_raw=torch.empty(rows, self.head_dim, device="cuda", dtype=torch.bfloat16),
            k=torch.empty(self.num_layers, rows, self.head_dim, device="cuda", dtype=torch.bfloat16),
            v=torch.empty(self.num_layers, rows, self.head_dim, device="cuda", dtype=torch.bfloat16),
            logits=torch.empty(rows_q, prefix_len, device="cuda", dtype=torch.float32),
            attn=torch.empty(rows_q, prefix_len, device="cuda", dtype=torch.bfloat16),
            hidden=torch.empty(rows, 16384, device="cuda", dtype=torch.bfloat16),
            valid_prefix_len=torch.empty(batch, device="cuda", dtype=torch.int32),
            prefix_mask=torch.empty(rows, device="cuda", dtype=torch.int32),
            rope_cos=rope_cos,
            rope_sin=rope_sin,
        )
        self.buffers_by_shape[key] = buffers
        return buffers

    @staticmethod
    def _valid_prefix_len(prefix_pad_masks: torch.Tensor) -> torch.Tensor:
        """Per-sample count of valid prefix tokens, shape ``[B]``."""
        return prefix_pad_masks.reshape(prefix_pad_masks.shape[0], -1).sum(dim=1).to(torch.int32)

    def _get_prefix_mlp_fn(self):
        if self._prefix_mlp_fn is not None:
            return self._prefix_mlp_fn
        fn = _prefix_mlp_torch_impl
        if self.compile_torch_prefix_mlp and hasattr(torch, "compile"):
            compile_kwargs: dict[str, object] = {"fullgraph": self.prefix_mlp_compile_fullgraph}
            if self.prefix_mlp_compile_mode:
                compile_kwargs["mode"] = self.prefix_mlp_compile_mode
            fn = torch.compile(_prefix_mlp_torch_impl, **compile_kwargs)
        self._prefix_mlp_fn = fn
        return fn

    def _get_prefix_mlp_packed_fn(self):
        if self._prefix_mlp_packed_fn is not None:
            return self._prefix_mlp_packed_fn
        fn = _prefix_mlp_torch_packed_gate_up_impl
        if self.compile_torch_prefix_mlp and hasattr(torch, "compile"):
            compile_kwargs: dict[str, object] = {"fullgraph": self.prefix_mlp_compile_fullgraph}
            if self.prefix_mlp_compile_mode:
                compile_kwargs["mode"] = self.prefix_mlp_compile_mode
            fn = torch.compile(_prefix_mlp_torch_packed_gate_up_impl, **compile_kwargs)
        self._prefix_mlp_packed_fn = fn
        return fn

    def _run_prefix_mlp(
        self,
        buffers: _PrefixEncoderBuffers,
        layer_idx: int,
        prefix_len: int,
    ) -> None:
        if self.use_torch_prefix_mlp:
            if self.use_packed_prefix_gate_up:
                out = self._get_prefix_mlp_packed_fn()(
                    buffers.x_normed,
                    buffers.x,
                    self.weights["ffn_gate_up_w"][layer_idx],
                    self.weights["ffn_down_w"][layer_idx],
                )
            else:
                out = self._get_prefix_mlp_fn()(
                    buffers.x_normed,
                    buffers.x,
                    self.weights["ffn_gate_w"][layer_idx],
                    self.weights["ffn_up_w"][layer_idx],
                    self.weights["ffn_down_w"][layer_idx],
                )
            buffers.x.copy_(out)
            return

        _matmul_small_gate_gelu_tanh[((prefix_len + 63) // 64, (16384 + 127) // 128)](
            buffers.x_normed,
            self.weights["ffn_gate_w"][layer_idx],
            self.weights["ffn_up_w"][layer_idx],
            buffers.hidden,
            prefix_len,
            self.hidden_size,
            16384,
            block_n=64,
            block_m=128,
            block_k=64,
        )
        block_n = 64 if prefix_len < 512 else 128
        _matmul_small_res[((prefix_len + block_n - 1) // block_n) * (self.hidden_size // 64),](
            buffers.hidden,
            self.weights["ffn_down_w"][layer_idx],
            buffers.x,
            buffers.x,
            prefix_len,
            16384,
            self.hidden_size,
            block_n=block_n,
            block_m=64,
            block_k=64,
        )

    def __call__(
        self,
        *,
        prefix_embs: torch.Tensor,
        prefix_pad_masks: torch.Tensor,
        prefix_position_ids: torch.Tensor | None = None,
        valid_prefix_len: int | None = None,
        output_k: torch.Tensor | None = None,
        output_v: torch.Tensor | None = None,
        return_prefix_kv: bool = True,
    ) -> list[tuple[torch.Tensor, torch.Tensor]] | None:
        if prefix_embs.dim() != 3 or prefix_embs.shape[-1] != self.hidden_size:
            raise ValueError(f"unexpected prefix_embs shape for realtime prefix encoder: {tuple(prefix_embs.shape)}")
        batch = int(prefix_embs.shape[0])
        prefix_len = int(prefix_embs.shape[1])
        if prefix_len > 1024:
            raise ValueError(f"realtime prefix encoder requires prefix_len <= 1024, got {prefix_len}")
        if valid_prefix_len is None:
            valid_prefix_len = self._valid_prefix_len(prefix_pad_masks)
        buffers = self._get_buffers(batch, prefix_len, prefix_embs.dtype)
        rows = batch * prefix_len
        # B=1 keeps the direct-write path (the prefix encoder writes K/V straight
        # into the decoder's buffers).  For B>1 the decoder lays K/V out per
        # sample as prefix+suffix, which is not the contiguous [B*prefix_len]
        # block this loop produces, so write locally and let the caller copy.
        direct_write = batch == 1
        target_k = buffers.k if (output_k is None or not direct_write) else output_k
        target_v = buffers.v if (output_v is None or not direct_write) else output_v
        if target_k.shape[0] != self.num_layers or target_k.shape[1] < rows or target_k.shape[2] != self.head_dim:
            raise ValueError(f"unexpected realtime prefix output_k shape: {tuple(target_k.shape)}")
        if target_v.shape[0] != self.num_layers or target_v.shape[1] < rows or target_v.shape[2] != self.head_dim:
            raise ValueError(f"unexpected realtime prefix output_v shape: {tuple(target_v.shape)}")
        buffers.x.copy_(prefix_embs.reshape(rows, self.hidden_size).to(torch.bfloat16))
        buffers.valid_prefix_len.copy_(
            valid_prefix_len.to(torch.int32)
            if isinstance(valid_prefix_len, torch.Tensor)
            else torch.full((batch,), int(valid_prefix_len), dtype=torch.int32, device=buffers.valid_prefix_len.device)
        )
        buffers.prefix_mask.copy_(prefix_pad_masks.reshape(rows).to(torch.int32))
        if prefix_position_ids is None:
            prefix_position_ids = torch.cumsum(prefix_pad_masks.to(torch.int64), dim=1) - 1
        # Positions are indexed by the same folded row id, so per-sample RoPE
        # falls out of the flattening without touching the kernel.
        _build_gemma_rope_from_positions[((rows + 15) // 16,)](
            prefix_position_ids.reshape(rows),
            self.rope_inv_freq,
            buffers.rope_cos,
            buffers.rope_sin,
            rows,
            self.head_dim,
            self.rope_attention_scaling,
            block_m=16,
            block_half=128,
        )
        rows_q_per = prefix_len * self.num_heads
        rows_q = rows * self.num_heads
        scale = 1.0 / math.sqrt(float(self.head_dim))

        for layer_idx in range(self.num_layers):
            _rms_norm_kernel[(rows,)](
                buffers.x,
                self.weights["input_norm_w"][layer_idx],
                buffers.x_normed,
                rows,
                self.hidden_size,
                block_size=1024,
            )
            _matmul_gemma_rope_qkv[((rows + 31) // 32, self.num_heads + 2)](
                buffers.x_normed,
                rows,
                self.hidden_size,
                self.head_dim,
                self.num_heads,
                self.weights["attn_qkv_w"][layer_idx],
                buffers.rope_cos,
                buffers.rope_sin,
                buffers.q,
                target_k[layer_idx, :rows],
                target_v[layer_idx, :rows],
                block_m=32,
                block_half=128,
                block_k=64,
            )
            if layer_idx == self.num_layers - 1:
                continue
            # Attention is the only place folding breaks down: sample b's queries
            # must see only sample b's K/V, so these three carry a batch index.
            _matmul_abt_scale[(((rows_q_per + 31) // 32) * ((prefix_len + 31) // 32), batch)](
                buffers.q,
                target_k[layer_idx, :rows],
                buffers.logits,
                rows_q_per,
                prefix_len,
                self.head_dim,
                scale,
                block_m=32,
                block_n=32,
                block_k=64,
            )
            _softmax_mask_vector[((rows_q_per + 3) // 4, batch)](
                buffers.logits,
                rows_q_per,
                prefix_len,
                buffers.prefix_mask,
                buffers.attn,
                block_m=4,
                block_size=1024,
            )
            _matmul_small_bmm[(((rows_q_per + 31) // 32) * (self.head_dim // 32), batch)](
                buffers.attn,
                target_v[layer_idx, :rows],
                buffers.q_raw,
                rows_q_per,
                prefix_len,
                self.head_dim,
                block_n=32,
                block_m=32,
                block_k=64,
            )
            _matmul_small_res[((rows + 63) // 64) * (self.hidden_size // 64),](
                buffers.q_raw,
                self.weights["attn_o_w"][layer_idx],
                buffers.x,
                buffers.x,
                rows,
                self.hidden_size,
                self.hidden_size,
                block_n=64,
                block_m=64,
                block_k=64,
            )
            _rms_norm_kernel[(rows,)](
                buffers.x,
                self.weights["post_norm_w"][layer_idx],
                buffers.x_normed,
                rows,
                self.hidden_size,
                block_size=1024,
            )
            self._run_prefix_mlp(buffers, layer_idx, rows)

        if not return_prefix_kv:
            return None
        return [
            (
                target_k[layer_idx, :rows].view(batch, 1, prefix_len, self.head_dim),
                target_v[layer_idx, :rows].view(batch, 1, prefix_len, self.head_dim),
            )
            for layer_idx in range(self.num_layers)
        ]


class Pi05RealtimeTritonDecoder:
    """Runtime-packed decoder that mirrors realtime-vla Pi05's action expert."""

    def __init__(self, model: Any):
        if not is_available():
            raise RuntimeError("Pi05RealtimeTritonDecoder requires CUDA and Triton")
        self.model = model
        self.device = model.action_in_proj.weight.device
        self.chunk_size = int(model.action_horizon)
        self.action_dim = int(model.action_dim)
        self.num_layers = len(model.paligemma_with_expert.gemma_expert.model.layers)
        self.hidden_size = int(model.expert_width)
        if self.chunk_size > 64 or self.action_dim != 32 or self.hidden_size != 1024 or self.num_layers != 18:
            raise ValueError("realtime Triton decoder currently specializes pi0.5 base: chunk=50, dim=32, H=1024, L=18")
        self.decoder_oproj_block_n = int(os.environ.get("PI05_REALTIME_DECODER_OPROJ_BLOCK_N", "32"))
        self.decoder_oproj_block_m = int(os.environ.get("PI05_REALTIME_DECODER_OPROJ_BLOCK_M", "32"))
        self.decoder_oproj_block_k = int(os.environ.get("PI05_REALTIME_DECODER_OPROJ_BLOCK_K", "256"))
        if 1024 % self.decoder_oproj_block_m != 0:
            raise ValueError(f"PI05_REALTIME_DECODER_OPROJ_BLOCK_M must divide 1024; got {self.decoder_oproj_block_m}.")
        if 2048 % self.decoder_oproj_block_k != 0:
            raise ValueError(f"PI05_REALTIME_DECODER_OPROJ_BLOCK_K must divide 2048; got {self.decoder_oproj_block_k}.")
        self.decoder_qkv_block_m = int(os.environ.get("PI05_REALTIME_DECODER_QKV_BLOCK_M", "32"))
        self.decoder_qkv_block_n = int(os.environ.get("PI05_REALTIME_DECODER_QKV_BLOCK_N", "64"))
        self.decoder_qkv_block_k = int(os.environ.get("PI05_REALTIME_DECODER_QKV_BLOCK_K", "128"))
        if self.decoder_qkv_block_n % 2 != 0 or 256 % self.decoder_qkv_block_n != 0:
            raise ValueError(
                f"PI05_REALTIME_DECODER_QKV_BLOCK_N must be an even divisor of 256; got {self.decoder_qkv_block_n}."
            )
        if 1024 % self.decoder_qkv_block_k != 0:
            raise ValueError(f"PI05_REALTIME_DECODER_QKV_BLOCK_K must divide 1024; got {self.decoder_qkv_block_k}.")
        self.decoder_fused_attention = os.environ.get(
            "PI05_REALTIME_DECODER_FUSED_ATTN",
            "0",
        ).lower() not in {
            "0",
            "false",
            "no",
        }
        self.decoder_fused_attention_block_m = int(os.environ.get("PI05_REALTIME_DECODER_FUSED_ATTN_BLOCK_M", "16"))
        self.decoder_fused_attention_block_n = int(os.environ.get("PI05_REALTIME_DECODER_FUSED_ATTN_BLOCK_N", "64"))
        self.decoder_ffn_gate_block_n = int(os.environ.get("PI05_REALTIME_DECODER_FFN_GATE_BLOCK_N", "64"))
        self.decoder_ffn_gate_block_m = int(os.environ.get("PI05_REALTIME_DECODER_FFN_GATE_BLOCK_M", "64"))
        self.decoder_ffn_gate_block_k = int(os.environ.get("PI05_REALTIME_DECODER_FFN_GATE_BLOCK_K", "64"))
        if 1024 % self.decoder_ffn_gate_block_k != 0:
            raise ValueError(
                f"PI05_REALTIME_DECODER_FFN_GATE_BLOCK_K must divide 1024; got {self.decoder_ffn_gate_block_k}."
            )
        self.decoder_ffn_down_block_n = int(os.environ.get("PI05_REALTIME_DECODER_FFN_DOWN_BLOCK_N", "16"))
        self.decoder_ffn_down_block_m = int(os.environ.get("PI05_REALTIME_DECODER_FFN_DOWN_BLOCK_M", "64"))
        self.decoder_ffn_down_block_k = int(os.environ.get("PI05_REALTIME_DECODER_FFN_DOWN_BLOCK_K", "256"))
        if 1024 % self.decoder_ffn_down_block_m != 0:
            raise ValueError(
                f"PI05_REALTIME_DECODER_FFN_DOWN_BLOCK_M must divide 1024; got {self.decoder_ffn_down_block_m}."
            )
        if 4096 % self.decoder_ffn_down_block_k != 0:
            raise ValueError(
                f"PI05_REALTIME_DECODER_FFN_DOWN_BLOCK_K must divide 4096; got {self.decoder_ffn_down_block_k}."
            )
        self.weights = self._pack_weights()
        self.final_weights_by_steps: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        self.buffers_by_prefix: dict[tuple[int, int, torch.dtype], _DecoderBuffers] = {}

    def _pack_weights(self) -> dict[str, torch.Tensor]:
        expert_layers = self.model.paligemma_with_expert.gemma_expert.model.layers
        weights: dict[str, torch.Tensor] = {
            "action_in_w": _linear_weight_t(self.model.action_in_proj),
            "action_in_b": _linear_bias(self.model.action_in_proj, self.hidden_size),
            "action_out_w": _linear_weight_t(self.model.action_out_proj),
            "action_out_b": _linear_bias(self.model.action_out_proj, self.action_dim),
            "attn_qkv_w": torch.empty(
                self.num_layers,
                self.hidden_size,
                2560,
                device="cuda",
                dtype=torch.bfloat16,
            ),
            "attn_o_w": torch.empty(
                self.num_layers,
                self.hidden_size * 2,
                self.hidden_size,
                device="cuda",
                dtype=torch.bfloat16,
            ),
            "ffn_gate_w": torch.empty(
                self.num_layers,
                self.hidden_size,
                4096,
                device="cuda",
                dtype=torch.bfloat16,
            ),
            "ffn_up_w": torch.empty(
                self.num_layers,
                self.hidden_size,
                4096,
                device="cuda",
                dtype=torch.bfloat16,
            ),
            "ffn_down_w": torch.empty(
                self.num_layers,
                4096,
                self.hidden_size,
                device="cuda",
                dtype=torch.bfloat16,
            ),
        }
        for idx, layer in enumerate(expert_layers):
            attn = layer.self_attn
            weights["attn_qkv_w"][idx].copy_(
                torch.cat(
                    [
                        attn.q_proj.weight.detach().t(),
                        attn.k_proj.weight.detach().t(),
                        attn.v_proj.weight.detach().t(),
                    ],
                    dim=1,
                ).to(device="cuda", dtype=torch.bfloat16)
            )
            weights["attn_o_w"][idx].copy_(_linear_weight_t(attn.o_proj))
            weights["ffn_gate_w"][idx].copy_(_linear_weight_t(layer.mlp.gate_proj))
            weights["ffn_up_w"][idx].copy_(_linear_weight_t(layer.mlp.up_proj))
            weights["ffn_down_w"][idx].copy_(_linear_weight_t(layer.mlp.down_proj))
        return weights

    def _get_buffers(self, batch: int, prefix_len: int, dtype: torch.dtype) -> _DecoderBuffers:
        # Batch is folded into the token dimension, as in the prefix encoder.
        # K/V keep prefix and suffix adjacent per sample (row b*total_len + t)
        # because the attention kernels walk them as one contiguous key range.
        #
        # The RoPE table is deliberately NOT part of the key: it depends on the
        # per-sample valid prefix length, which would otherwise multiply the
        # number of buffer sets the way it does for the CUDA graph cache. It is
        # filled during setup instead, alongside the mask.
        key = (int(batch), int(prefix_len), dtype)
        cached = self.buffers_by_prefix.get(key)
        if cached is not None:
            return cached
        total_len = prefix_len + self.chunk_size
        rows = batch * self.chunk_size
        buffers = _DecoderBuffers(
            noise=torch.empty(rows, self.action_dim, device="cuda", dtype=torch.bfloat16),
            x=torch.empty(rows, self.hidden_size, device="cuda", dtype=torch.bfloat16),
            x_normed=torch.empty(rows, self.hidden_size, device="cuda", dtype=torch.bfloat16),
            gate=torch.empty(rows, self.hidden_size, device="cuda", dtype=torch.bfloat16),
            q=torch.empty(rows * 8, 256, device="cuda", dtype=torch.bfloat16),
            k=torch.empty(self.num_layers, batch * total_len, 256, device="cuda", dtype=torch.bfloat16),
            v=torch.empty(self.num_layers, batch * total_len, 256, device="cuda", dtype=torch.bfloat16),
            logits=torch.empty(rows * 8, total_len, device="cuda", dtype=torch.float32),
            attn=torch.empty(rows * 8, total_len, device="cuda", dtype=torch.bfloat16),
            hidden=torch.empty(rows, 4096, device="cuda", dtype=torch.bfloat16),
            valid_prefix_len=torch.empty(batch, device="cuda", dtype=torch.int32),
            prefix_mask=torch.empty(batch * prefix_len, device="cuda", dtype=torch.int32),
            rope=torch.empty(rows, 256, device="cuda", dtype=torch.bfloat16),
        )
        self.buffers_by_prefix[key] = buffers
        return buffers

    def _fill_rope(self, rope: torch.Tensor, valid_prefix_lens: torch.Tensor) -> None:
        """Write the suffix rotary table for each sample into ``rope``.

        Suffix positions continue from the sample's own *valid* prefix length,
        not the padded one, so a batch whose samples have different valid
        lengths needs one table per sample.
        """
        batch = int(valid_prefix_lens.numel())
        starts = valid_prefix_lens.to(device="cuda", dtype=torch.float32)[:, None]
        offsets = torch.arange(self.chunk_size, device="cuda", dtype=torch.float32)[None, :]
        pos = (starts + offsets).reshape(batch * self.chunk_size)
        inv_freq = 1.0 / (10000 ** (torch.arange(0, 256, 2, dtype=torch.float32, device="cuda") / 256))
        phase = pos[:, None] * inv_freq[None, :]
        table = torch.cat([torch.cos(phase)[:, :, None], torch.sin(phase)[:, :, None]], dim=2)
        rope.copy_(table.reshape(batch * self.chunk_size, 256).to(torch.bfloat16))

    @staticmethod
    def _as_valid_lens(valid_prefix_len, batch: int) -> torch.Tensor:
        """Normalise a scalar-or-tensor valid length to an ``[B]`` int32 tensor."""
        if isinstance(valid_prefix_len, torch.Tensor):
            return valid_prefix_len.reshape(-1).to(device="cuda", dtype=torch.int32)
        return torch.full((batch,), int(valid_prefix_len), device="cuda", dtype=torch.int32)

    def _copy_prefix_kv(
        self,
        prefix_kv: list[tuple[torch.Tensor, torch.Tensor]],
        prefix_pad_masks: torch.Tensor | None,
        buffers: _DecoderBuffers,
        valid_prefix_len,
    ) -> int:
        batch = int(prefix_kv[0][0].shape[0])
        prefix_len = int(prefix_kv[0][0].shape[2])
        total_len = prefix_len + self.chunk_size
        for idx, (k_prefix, v_prefix) in enumerate(prefix_kv):
            # K/V are laid out [num_layers, B*total_len, D]; sample b owns rows
            # [b*total_len, b*total_len + total_len), prefix first then suffix.
            k_dst = buffers.k[idx].view(batch, total_len, 256)
            v_dst = buffers.v[idx].view(batch, total_len, 256)
            k_dst[:, :prefix_len].copy_(k_prefix[:, 0].to(torch.bfloat16))
            v_dst[:, :prefix_len].copy_(v_prefix[:, 0].to(torch.bfloat16))
        self._copy_prefix_metadata(prefix_pad_masks, buffers, valid_prefix_len, batch=batch)
        return prefix_len

    def _copy_prefix_metadata(
        self,
        prefix_pad_masks: torch.Tensor | None,
        buffers: _DecoderBuffers,
        valid_prefix_len,
        batch: int | None = None,
    ) -> None:
        if batch is None:
            batch = int(buffers.valid_prefix_len.numel())
        valid_lens = self._as_valid_lens(valid_prefix_len, batch)
        buffers.valid_prefix_len.copy_(valid_lens)
        if prefix_pad_masks is None:
            prefix_len = buffers.prefix_mask.numel() // batch
            mask = buffers.prefix_mask.view(batch, prefix_len)
            mask.zero_()
            for b in range(batch):
                mask[b, : int(valid_lens[b])] = 1
        else:
            buffers.prefix_mask.copy_(prefix_pad_masks.reshape(-1).to(torch.int32))
        # Must happen here, in setup, not inside the captured region: the table
        # depends on the per-sample valid lengths, which change between calls.
        self._fill_rope(buffers.rope, valid_lens)

    def prepare_prefix_buffers(
        self,
        *,
        prefix_len: int,
        valid_prefix_len,
        dtype: torch.dtype,
        prefix_pad_masks: torch.Tensor | None,
        batch: int = 1,
    ) -> _DecoderBuffers:
        _validate_prefix_suffix_softmax_window(prefix_len, self.chunk_size)
        buffers = self._get_buffers(batch, prefix_len, dtype)
        self._copy_prefix_metadata(prefix_pad_masks, buffers, valid_prefix_len, batch=batch)
        return buffers

    def _get_final_weights(self, num_steps: int) -> tuple[torch.Tensor, torch.Tensor]:
        cached = self.final_weights_by_steps.get(num_steps)
        if cached is not None:
            return cached
        dt = -1.0 / num_steps
        final_w = (self.weights["action_out_w"] * dt).contiguous()
        final_b = (self.weights["action_out_b"] * dt).contiguous()
        self.final_weights_by_steps[num_steps] = (final_w, final_b)
        return final_w, final_b

    def __call__(
        self,
        *,
        prefix_kv: list[tuple[torch.Tensor, torch.Tensor]] | None,
        prefix_pad_masks: torch.Tensor | None = None,
        valid_prefix_len: int,
        x_t: torch.Tensor,
        adarms_modulations: list[Any],
        num_steps: int,
        decoder_buffers: _DecoderBuffers | None = None,
        prefix_len: int | None = None,
    ) -> torch.Tensor:
        if x_t.dim() != 3 or x_t.shape[1] != self.chunk_size or x_t.shape[2] != self.action_dim:
            raise ValueError(f"unexpected action tensor shape for realtime Triton decoder: {tuple(x_t.shape)}")
        batch = int(x_t.shape[0])
        if num_steps != len(adarms_modulations):
            raise ValueError("realtime Triton decoder requires precomputed AdaRMS for each denoise step")
        if prefix_len is None:
            if prefix_kv is None:
                raise ValueError("prefix_len is required when prefix_kv is None")
            prefix_len = int(prefix_kv[0][0].shape[2])
        _validate_prefix_suffix_softmax_window(prefix_len, self.chunk_size)
        if decoder_buffers is None:
            if prefix_kv is None:
                raise ValueError("prefix_kv is required when decoder_buffers is None")
            buffers = self._get_buffers(batch, prefix_len, x_t.dtype)
            self._copy_prefix_kv(prefix_kv, prefix_pad_masks, buffers, valid_prefix_len)
        else:
            buffers = decoder_buffers
            self._copy_prefix_metadata(prefix_pad_masks, buffers, valid_prefix_len, batch=batch)
        buffers.noise.copy_(x_t.reshape(batch * self.chunk_size, self.action_dim).to(torch.bfloat16))
        final_w, final_b = self._get_final_weights(num_steps)
        for step in range(num_steps):
            self._step(
                buffers, adarms_modulations[step], prefix_len, valid_prefix_len, final_w, final_b, batch=batch
            )
        return buffers.noise.view(batch, self.chunk_size, self.action_dim).to(dtype=x_t.dtype)

    def _step(
        self,
        buffers: _DecoderBuffers,
        step_mods: Any,
        prefix_len: int,
        valid_prefix_len: int,
        final_w: torch.Tensor,
        final_b: torch.Tensor,
        batch: int = 1,
    ) -> None:
        seq_len = self.chunk_size
        # Rows carry the batch: row b*chunk_size + t. Weight-shared kernels are
        # row-parallel and need nothing beyond the larger row count.
        rows = batch * seq_len
        _matmul_small_bias[((rows + 31) // 32) * (1024 // 32),](
            buffers.noise,
            self.weights["action_in_w"],
            buffers.x,
            self.weights["action_in_b"],
            rows,
            32,
            1024,
            block_n=32,
            block_m=32,
            block_k=32,
        )
        total_keys = prefix_len + seq_len
        for layer_idx in range(self.num_layers):
            input_style, post_style = step_mods.layer_modulations[layer_idx]
            _adarms_norm_kernel[(rows,)](
                buffers.x,
                input_style.reshape(-1),
                buffers.x_normed,
                buffers.gate,
                rows,
                1024,
                block_size=512,
                rows_per_sample=seq_len,
            )
            _matmul_rope_qkv[(128,)](
                buffers.x_normed,
                rows,
                1024,
                256,
                8,
                self.weights["attn_qkv_w"][layer_idx],
                buffers.rope,
                buffers.q,
                buffers.k[layer_idx],
                buffers.v[layer_idx],
                block_m=self.decoder_qkv_block_m,
                block_n=self.decoder_qkv_block_n,
                block_k=self.decoder_qkv_block_k,
                kv_rows_per_sample=seq_len,
                kv_sample_stride=total_keys,
                kv_row_offset=prefix_len,
            )
            rows_q = seq_len * 8
            if self.decoder_fused_attention:
                if batch != 1:
                    raise ValueError(
                        "PI05_REALTIME_DECODER_FUSED_ATTN is batch=1 only; it is off by "
                        "default and measured slower than the split path"
                    )
                fused_attn_grid = (
                    (rows_q + self.decoder_fused_attention_block_m - 1) // self.decoder_fused_attention_block_m,
                )
                _attention_prefix_suffix_fused[fused_attn_grid](
                    buffers.q,
                    buffers.k[layer_idx, :total_keys],
                    buffers.v[layer_idx, :total_keys],
                    buffers.q,
                    buffers.prefix_mask,
                    rows_q,
                    prefix_len,
                    seq_len,
                    256,
                    1.0 / math.sqrt(256.0),
                    block_m=self.decoder_fused_attention_block_m,
                    block_n=self.decoder_fused_attention_block_n,
                    block_d=256,
                )
            else:
                # Sample b's queries must see only sample b's K/V, so these three
                # take the batch on grid axis 1 and index their per-sample
                # operands from it.
                _matmul_abt_scale[(((rows_q + 31) // 32) * ((total_keys + 31) // 32), batch)](
                    buffers.q,
                    buffers.k[layer_idx],
                    buffers.logits,
                    rows_q,
                    total_keys,
                    256,
                    1.0 / math.sqrt(256.0),
                    block_m=32,
                    block_n=32,
                    block_k=64,
                )
                _softmax_prefix_suffix_mask_vector[((rows_q + 3) // 4, batch)](
                    buffers.logits,
                    rows_q,
                    prefix_len,
                    seq_len,
                    buffers.prefix_mask,
                    buffers.attn,
                    block_m=4,
                    block_size=1024,
                )
                _matmul_small_bmm[(((rows_q + 31) // 32) * (256 // 32), batch)](
                    buffers.attn,
                    buffers.v[layer_idx],
                    buffers.q,
                    rows_q,
                    total_keys,
                    256,
                    block_n=32,
                    block_m=32,
                    block_k=64,
                )
            _matmul_small_res_gate_oproj[(128,)](
                buffers.q,
                self.weights["attn_o_w"][layer_idx],
                buffers.x,
                buffers.x,
                buffers.gate,
                rows,
                2048,
                1024,
                block_n=self.decoder_oproj_block_n,
                block_m=self.decoder_oproj_block_m,
                block_k=self.decoder_oproj_block_k,
            )
            _adarms_norm_kernel[(rows,)](
                buffers.x,
                post_style.reshape(-1),
                buffers.x_normed,
                buffers.gate,
                rows,
                1024,
                block_size=512,
                rows_per_sample=seq_len,
            )
            _matmul_small_gate[
                (
                    (rows + self.decoder_ffn_gate_block_n - 1) // self.decoder_ffn_gate_block_n,
                    (4096 + self.decoder_ffn_gate_block_m - 1) // self.decoder_ffn_gate_block_m,
                )
            ](
                buffers.x_normed,
                self.weights["ffn_gate_w"][layer_idx],
                self.weights["ffn_up_w"][layer_idx],
                buffers.hidden,
                rows,
                1024,
                4096,
                block_n=self.decoder_ffn_gate_block_n,
                block_m=self.decoder_ffn_gate_block_m,
                block_k=self.decoder_ffn_gate_block_k,
            )
            _matmul_small_res_gate_ffn_down[
                (
                    ((rows + self.decoder_ffn_down_block_n - 1) // self.decoder_ffn_down_block_n)
                    * (1024 // self.decoder_ffn_down_block_m),
                )
            ](
                buffers.hidden,
                self.weights["ffn_down_w"][layer_idx],
                buffers.x,
                buffers.x,
                buffers.gate,
                rows,
                4096,
                1024,
                block_n=self.decoder_ffn_down_block_n,
                block_m=self.decoder_ffn_down_block_m,
                block_k=self.decoder_ffn_down_block_k,
            )
        _adarms_norm_kernel[(rows,)](
            buffers.x,
            step_mods.final_modulation.reshape(-1),
            buffers.x_normed,
            buffers.gate,
            rows,
            1024,
            block_size=512,
            rows_per_sample=seq_len,
        )
        _matmul_small_bias_res[((rows + 15) // 16) * (32 // 16),](
            buffers.x_normed,
            final_w,
            buffers.noise,
            final_b,
            buffers.noise,
            rows,
            1024,
            32,
            block_n=16,
            block_m=16,
            block_k=256,
        )


class Pi05RealtimeExecutor:
    """Fixed-shape pi0.5 realtime executor for the experimental fast path.

    This keeps vLLM-Omni's worker, request, and serving shell, but removes the
    Python KV-list handoff from the captured hot path: the prefix encoder writes
    directly into the decoder's static K/V buffer and the decoder consumes that
    buffer in place.
    """

    def __init__(self, prefix_encoder: Pi05RealtimePrefixEncoder, decoder: Pi05RealtimeTritonDecoder):
        self.prefix_encoder = prefix_encoder
        self.decoder = decoder

    def __call__(
        self,
        *,
        prefix_embs: torch.Tensor,
        prefix_pad_masks: torch.Tensor,
        prefix_position_ids: torch.Tensor,
        valid_prefix_len: int,
        x_t: torch.Tensor,
        adarms_modulations: list[Any],
        num_steps: int,
    ) -> torch.Tensor:
        batch = int(prefix_embs.shape[0])
        prefix_len = int(prefix_embs.shape[1])
        buffers = self.decoder.prepare_prefix_buffers(
            prefix_len=prefix_len,
            valid_prefix_len=valid_prefix_len,
            dtype=x_t.dtype,
            prefix_pad_masks=prefix_pad_masks,
            batch=batch,
        )
        if batch == 1:
            # Prefix rows are the leading block of the K/V cache, so the encoder
            # can still write straight into it.
            self.prefix_encoder(
                prefix_embs=prefix_embs,
                prefix_pad_masks=prefix_pad_masks,
                prefix_position_ids=prefix_position_ids,
                valid_prefix_len=valid_prefix_len,
                output_k=buffers.k[:, :prefix_len],
                output_v=buffers.v[:, :prefix_len],
                return_prefix_kv=False,
            )
        else:
            # With B>1 the decoder interleaves each sample's prefix and suffix,
            # so the encoder's contiguous [B*prefix_len] block has to be copied
            # into place rather than written in situ.
            prefix_kv = self.prefix_encoder(
                prefix_embs=prefix_embs,
                prefix_pad_masks=prefix_pad_masks,
                prefix_position_ids=prefix_position_ids,
                valid_prefix_len=valid_prefix_len,
                return_prefix_kv=True,
            )
            self.decoder._copy_prefix_kv(prefix_kv, prefix_pad_masks, buffers, valid_prefix_len)
        return self.decoder(
            prefix_kv=None,
            prefix_pad_masks=prefix_pad_masks,
            valid_prefix_len=valid_prefix_len,
            x_t=x_t,
            adarms_modulations=adarms_modulations,
            num_steps=num_steps,
            decoder_buffers=buffers,
            prefix_len=prefix_len,
        )
