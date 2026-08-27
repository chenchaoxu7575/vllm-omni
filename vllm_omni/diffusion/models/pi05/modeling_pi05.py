# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inference-only pi0.5 VLA math kernel for vllm-omni.

pi0.5 keeps the same high-level shape as pi0: PaliGemma prefix, Gemma action
expert, and flow-matching denoising. The key architectural differences are:

* robot state is serialized into the prompt by ``processor_pi05``;
* suffix tokens are action tokens only;
* the action expert uses AdaRMS conditioning from the timestep embedding.
"""

from __future__ import annotations

import logging
import math
import os
import re
import time
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.cache_utils import DynamicCache
from transformers.masking_utils import create_causal_mask
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.auto import CONFIG_MAPPING
from transformers.models.gemma.modeling_gemma import (
    GemmaAttention,
    GemmaConfig,
    GemmaForCausalLM,
    GemmaMLP,
    GemmaModel,
    apply_rotary_pos_emb,
)
from transformers.models.paligemma.modeling_paligemma import (
    PaliGemmaForConditionalGeneration,
    PaliGemmaModel,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader

from vllm_omni.diffusion.models.pi0.modeling_pi0 import (
    DEFAULT_ACTION_DIM,
    DEFAULT_ACTION_HORIZON,
    DEFAULT_NUM_INFERENCE_STEPS,
    _apply_norm,
    _build_norm_buffers,
    create_sinusoidal_pos_embedding,
    get_gemma_config,
    make_att_2d_masks,
    prepare_attention_masks_4d,
)

logger = logging.getLogger(__name__)

try:
    import triton
    import triton.language as tl
    from triton.language.extra import libdevice
except ImportError:  # pragma: no cover - optional CUDA fast path
    triton = None
    tl = None
    libdevice = None


def _sync_for_timing(device: torch.device) -> None:
    if device.type == "cuda":
        torch.accelerator.synchronize(device)


def _per_sample_valid_prefix_len(
    prefix_pad_masks: torch.Tensor,
    valid_prefix_len=None,
) -> torch.Tensor:
    """Valid prefix length per sample, shape ``[B]``.

    The realtime path used to collapse this to a single Python int, which is
    only correct when the batch holds one sample: pi0.5 packs the discretised
    state into the language prompt, so the number of valid prefix tokens varies
    from sample to sample.
    """
    batch = int(prefix_pad_masks.shape[0])
    if valid_prefix_len is None:
        return prefix_pad_masks.reshape(batch, -1).sum(dim=1).to(torch.int32)
    if isinstance(valid_prefix_len, torch.Tensor):
        return valid_prefix_len.reshape(-1).to(torch.int32)
    return torch.full(
        (batch,), int(valid_prefix_len), dtype=torch.int32, device=prefix_pad_masks.device
    )


def _compile_call(fn, *, fullgraph: bool = False):
    return torch.compile(fn, mode="max-autotune-no-cudagraphs", fullgraph=fullgraph)


class _TimingBlock:
    def __init__(self, timing: dict[str, Any] | None, key: str, device: torch.device):
        self.timing = timing
        self.key = key
        self.device = device
        self.start = 0.0

    def __enter__(self):
        if self.timing is not None:
            _sync_for_timing(self.device)
            self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.timing is not None:
            _sync_for_timing(self.device)
            self.timing[self.key] = self.timing.get(self.key, 0.0) + (time.perf_counter() - self.start) * 1000.0


class _ProfileRange:
    def __init__(self, name: str, enabled: bool = False):
        self.name = name
        self.enabled = enabled
        self.record = None
        self.use_nvtx = False

    def __enter__(self):
        if not self.enabled:
            return self
        self.record = torch.profiler.record_function(self.name)
        self.record.__enter__()
        self.use_nvtx = torch.cuda.is_available()
        if self.use_nvtx:
            torch.cuda.nvtx.range_push(self.name)
        return self

    def __exit__(self, exc_type, exc, tb):
        if not self.enabled:
            return
        if self.use_nvtx:
            torch.cuda.nvtx.range_pop()
        if self.record is not None:
            self.record.__exit__(exc_type, exc, tb)


@dataclass
class _DenoiseAdarmsModulations:
    layer_modulations: list[tuple[torch.Tensor | None, torch.Tensor | None]]
    final_modulation: torch.Tensor | None


@dataclass
class _DenoiseStaticContext:
    attention_mask: torch.Tensor
    position_ids: torch.Tensor
    suffix_pad_masks: torch.Tensor
    suffix_att_masks: torch.Tensor
    time_tensors: list[torch.Tensor]
    time_conds: list[torch.Tensor]
    adarms_modulations: list[_DenoiseAdarmsModulations] | None
    rope_cos_sin: tuple[torch.Tensor, torch.Tensor] | None


@dataclass
class _DenoiseCudaGraphCache:
    graph: torch.cuda.CUDAGraph
    static_x: torch.Tensor
    static_prefix_pad_masks: torch.Tensor
    static_past_key_values: list[tuple[torch.Tensor, torch.Tensor]]
    static_context: _DenoiseStaticContext
    static_output: torch.Tensor


@dataclass
class _PrefixCudaGraphCache:
    graph: torch.cuda.CUDAGraph
    static_prefix_embs: torch.Tensor
    static_attention_mask: torch.Tensor
    static_position_ids: torch.Tensor
    static_hidden: torch.Tensor
    static_past_key_values: list[tuple[torch.Tensor, torch.Tensor]]


@dataclass
class _ImageCudaGraphCache:
    graph: torch.cuda.CUDAGraph
    static_pixels: torch.Tensor
    static_output: torch.Tensor


@dataclass
class _PrefixDenoiseCudaGraphCache:
    graph: torch.cuda.CUDAGraph
    static_prefix_embs: torch.Tensor
    static_prefix_attention_mask: torch.Tensor
    static_prefix_position_ids: torch.Tensor
    static_prefix_pad_masks: torch.Tensor
    static_x: torch.Tensor
    static_context: _DenoiseStaticContext
    static_output: torch.Tensor


@dataclass
class _RealtimePrefixKvCacheEntry:
    decoder_buffers: Any
    prefix_len: int
    valid_prefix_len: int
    prefix_pad_masks: torch.Tensor


@dataclass
class _RealtimeCachedDecoderCudaGraphCache:
    graph: torch.cuda.CUDAGraph
    static_x: torch.Tensor
    static_context: _DenoiseStaticContext
    static_output: torch.Tensor


@dataclass
class _RealtimePrefixEmbCacheEntry:
    prefix_embs: torch.Tensor
    prefix_pad_masks: torch.Tensor
    prefix_position_ids: torch.Tensor
    prefix_attention_mask: torch.Tensor
    valid_prefix_len: int


@dataclass
class _RealtimeImageEmbedCacheEntry:
    image_emb: torch.Tensor


def _gated_residual(
    x: torch.Tensor | None,
    y: torch.Tensor | None,
    gate: torch.Tensor | None,
) -> torch.Tensor | None:
    if x is None and y is None:
        return None
    if x is None or y is None:
        return x if x is not None else y
    if gate is None:
        return x + y
    return x + y * gate


def layernorm_forward(
    layernorm: nn.Module,
    x: torch.Tensor,
    cond: torch.Tensor | None = None,
    modulation: torch.Tensor | None = None,
):
    if modulation is not None:
        dtype = x.dtype
        normed = layernorm._norm(x)
        if x.ndim == 3 and modulation.ndim == 2:
            modulation = modulation.unsqueeze(1)
        scale, shift, gate = modulation.chunk(3, dim=-1)
        normed = normed * (1.0 + scale.float()) + shift.float()
        return normed.to(dtype), gate.to(dtype)
    if cond is not None:
        return layernorm(x, cond=cond)
    return layernorm(x)


class PiGemmaRMSNorm(nn.Module):
    """AdaRMS used by pi0.5.

    pi0.5 checkpoints use OpenPI's zero-initialized ``1 + weight`` RMSNorm
    parameterization, which is not layout-compatible with standard Gemma
    RMSNorm weights initialized around 1.0.
    """

    def __init__(self, dim: int, eps: float = 1e-6, cond_dim: int | None = None):
        super().__init__()
        self.eps = eps
        self.dim = dim
        self.cond_dim = cond_dim
        if cond_dim is not None:
            self.dense = nn.Linear(cond_dim, dim * 3, bias=True)
            nn.init.zeros_(self.dense.weight)
        else:
            self.weight = nn.Parameter(torch.zeros(dim))
            self.dense = None

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        var = torch.mean(torch.square(x.float()), dim=-1, keepdim=True)
        return x * torch.rsqrt(var + self.eps)

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        dtype = x.dtype
        normed = self._norm(x)
        if cond is None or self.dense is None:
            normed = normed * (1.0 + self.weight.float())
            return normed.to(dtype), None

        if cond.shape[-1] != self.cond_dim:
            raise ValueError(f"Expected cond dim {self.cond_dim}, got {cond.shape[-1]}")

        modulation = self.dense(cond)
        if x.ndim == 3:
            modulation = modulation.unsqueeze(1)
        scale, shift, gate = modulation.chunk(3, dim=-1)
        normed = normed * (1.0 + scale.float()) + shift.float()
        return normed.to(dtype), gate.to(dtype)


def _get_pi_gemma_decoder_layer_base():
    class _PiGemmaDecoderLayerBase(GradientCheckpointingLayer):
        def __init__(self, config: GemmaConfig, layer_idx: int):
            super().__init__()
            self.self_attn = GemmaAttention(config=config, layer_idx=layer_idx)
            self.mlp = GemmaMLP(config)
            cond_dim = getattr(config, "adarms_cond_dim", None) if getattr(config, "use_adarms", False) else None
            self.input_layernorm = PiGemmaRMSNorm(
                config.hidden_size,
                eps=config.rms_norm_eps,
                cond_dim=cond_dim,
            )
            self.post_attention_layernorm = PiGemmaRMSNorm(
                config.hidden_size,
                eps=config.rms_norm_eps,
                cond_dim=cond_dim,
            )

        def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: torch.Tensor | None = None,
            position_ids: torch.LongTensor | None = None,
            past_key_values=None,
            use_cache: bool = False,
            cache_position: torch.LongTensor | None = None,
            position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
            adarms_cond: torch.Tensor | None = None,
            **kwargs,
        ) -> torch.Tensor:
            residual = hidden_states
            hidden_states, gate = self.input_layernorm(hidden_states, cond=adarms_cond)
            hidden_states, _ = self.self_attn(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )
            hidden_states = _gated_residual(residual, hidden_states, gate)

            residual = hidden_states
            hidden_states, gate = self.post_attention_layernorm(hidden_states, cond=adarms_cond)
            hidden_states = self.mlp(hidden_states)
            hidden_states = _gated_residual(residual, hidden_states, gate)
            return hidden_states

    return _PiGemmaDecoderLayerBase


class PiGemmaModel(GemmaModel):  # type: ignore[misc]
    def __init__(self, config: GemmaConfig, **kwargs):
        super().__init__(config, **kwargs)
        cond_dim = getattr(config, "adarms_cond_dim", None)
        pi_layer = _get_pi_gemma_decoder_layer_base()
        self.layers = nn.ModuleList([pi_layer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)])
        self.norm = PiGemmaRMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            cond_dim=cond_dim,
        )

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: DynamicCache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        adarms_cond: torch.Tensor | None = None,
        **kwargs,
    ) -> BaseModelOutputWithPast:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        if attention_mask is not None and attention_mask.dim() == 4:
            causal_mask = attention_mask
        else:
            causal_mask = create_causal_mask(
                config=self.config,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                cache_position=cache_position,
                past_key_values=past_key_values,
                position_ids=position_ids,
            )

        hidden_states = inputs_embeds
        if len(self.layers) > 0 and self.layers[0].self_attn.q_proj.weight.dtype == torch.bfloat16:
            hidden_states = hidden_states.to(torch.bfloat16)

        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                adarms_cond=adarms_cond,
                **kwargs,
            )

        hidden_states, _ = self.norm(hidden_states, adarms_cond)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class PiGemmaForCausalLM(GemmaForCausalLM):  # type: ignore[misc]
    def __init__(self, config: GemmaConfig, **kwargs):
        super().__init__(config, **kwargs)
        self.model = PiGemmaModel(config)


class PaliGemmaModelWithPiGemma(PaliGemmaModel):
    def __init__(self, config):
        super().__init__(config)
        self.language_model = PiGemmaModel(config.text_config)


class PaliGemmaForConditionalGenerationWithPiGemma(PaliGemmaForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        self.model = PaliGemmaModelWithPiGemma(config)

    @property
    def language_model(self):
        return self.model.language_model


def _repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return hidden_states
    bsize, num_kv_heads, seq_len, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(bsize, num_kv_heads, n_rep, seq_len, head_dim)
    return hidden_states.reshape(bsize, num_kv_heads * n_rep, seq_len, head_dim)


def _attend(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: torch.Tensor | None,
    *,
    num_kv_groups: int,
    scaling: float,
) -> torch.Tensor:
    key_states = _repeat_kv(key_states, num_kv_groups)
    value_states = _repeat_kv(value_states, num_kv_groups)
    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask[:, :, :, : key_states.shape[-2]]
    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    return torch.matmul(attn_weights, value_states)


if triton is not None and tl is not None:

    @triton.jit
    def _pi05_final_head_kernel(
        hidden,
        modulation,
        weight,
        bias,
        output,
        hidden_stride_b: tl.constexpr,
        hidden_stride_s: tl.constexpr,
        hidden_stride_h: tl.constexpr,
        mod_stride_b: tl.constexpr,
        mod_stride_h: tl.constexpr,
        weight_stride_o: tl.constexpr,
        weight_stride_h: tl.constexpr,
        out_stride_b: tl.constexpr,
        out_stride_s: tl.constexpr,
        out_stride_o: tl.constexpr,
        hidden_size: tl.constexpr,
        action_dim: tl.constexpr,
        eps: tl.constexpr,
        block_h: tl.constexpr,
        block_o: tl.constexpr,
    ):
        row = tl.program_id(0)
        batch = tl.program_id(1)
        offs_h = tl.arange(0, block_h)
        offs_o = tl.arange(0, block_o)

        x = tl.load(
            hidden + batch * hidden_stride_b + row * hidden_stride_s + offs_h * hidden_stride_h,
            mask=offs_h < hidden_size,
            other=0.0,
        ).to(tl.float32)
        var = tl.sum(x * x, axis=0) / hidden_size
        normed = x * tl.rsqrt(var + eps)
        scale = tl.load(
            modulation + batch * mod_stride_b + offs_h * mod_stride_h,
            mask=offs_h < hidden_size,
            other=0.0,
        ).to(tl.float32)
        shift = tl.load(
            modulation + batch * mod_stride_b + (hidden_size + offs_h) * mod_stride_h,
            mask=offs_h < hidden_size,
            other=0.0,
        ).to(tl.float32)
        normed = normed * (1.0 + scale) + shift

        w = tl.load(
            weight + offs_o[:, None] * weight_stride_o + offs_h[None, :] * weight_stride_h,
            mask=(offs_o[:, None] < action_dim) & (offs_h[None, :] < hidden_size),
            other=0.0,
        )
        acc = tl.sum(normed[None, :] * w.to(tl.float32), axis=1)
        b = tl.load(bias + offs_o, mask=offs_o < action_dim, other=0.0).to(tl.float32)
        acc += b
        tl.store(
            output + batch * out_stride_b + row * out_stride_s + offs_o * out_stride_o,
            acc,
            mask=offs_o < action_dim,
        )

    @triton.jit
    def _pi05_final_head_euler_kernel(
        hidden,
        modulation,
        weight,
        bias,
        euler_base,
        output,
        hidden_stride_b: tl.constexpr,
        hidden_stride_s: tl.constexpr,
        hidden_stride_h: tl.constexpr,
        mod_stride_b: tl.constexpr,
        mod_stride_h: tl.constexpr,
        weight_stride_o: tl.constexpr,
        weight_stride_h: tl.constexpr,
        base_stride_b: tl.constexpr,
        base_stride_s: tl.constexpr,
        base_stride_o: tl.constexpr,
        out_stride_b: tl.constexpr,
        out_stride_s: tl.constexpr,
        out_stride_o: tl.constexpr,
        hidden_size: tl.constexpr,
        action_dim: tl.constexpr,
        eps: tl.constexpr,
        dt: tl.constexpr,
        block_h: tl.constexpr,
        block_o: tl.constexpr,
    ):
        row = tl.program_id(0)
        batch = tl.program_id(1)
        offs_h = tl.arange(0, block_h)
        offs_o = tl.arange(0, block_o)

        x = tl.load(
            hidden + batch * hidden_stride_b + row * hidden_stride_s + offs_h * hidden_stride_h,
            mask=offs_h < hidden_size,
            other=0.0,
        ).to(tl.float32)
        var = tl.sum(x * x, axis=0) / hidden_size
        normed = x * tl.rsqrt(var + eps)
        scale = tl.load(
            modulation + batch * mod_stride_b + offs_h * mod_stride_h,
            mask=offs_h < hidden_size,
            other=0.0,
        ).to(tl.float32)
        shift = tl.load(
            modulation + batch * mod_stride_b + (hidden_size + offs_h) * mod_stride_h,
            mask=offs_h < hidden_size,
            other=0.0,
        ).to(tl.float32)
        normed = normed * (1.0 + scale) + shift

        w = tl.load(
            weight + offs_o[:, None] * weight_stride_o + offs_h[None, :] * weight_stride_h,
            mask=(offs_o[:, None] < action_dim) & (offs_h[None, :] < hidden_size),
            other=0.0,
        )
        acc = tl.sum(normed[None, :] * w.to(tl.float32), axis=1)
        b = tl.load(bias + offs_o, mask=offs_o < action_dim, other=0.0).to(tl.float32)
        acc += b
        base = tl.load(
            euler_base + batch * base_stride_b + row * base_stride_s + offs_o * base_stride_o,
            mask=offs_o < action_dim,
            other=0.0,
        ).to(tl.float32)
        tl.store(
            output + batch * out_stride_b + row * out_stride_s + offs_o * out_stride_o,
            base + dt * acc,
            mask=offs_o < action_dim,
        )


def _triton_final_head(
    hidden_states: torch.Tensor,
    final_modulation: torch.Tensor | None,
    norm: nn.Module,
    action_out_proj: nn.Linear,
) -> torch.Tensor | None:
    if triton is None or hidden_states.device.type != "cuda" or final_modulation is None:
        return None
    if hidden_states.ndim != 3 or final_modulation.ndim not in (2, 3):
        return None
    if hidden_states.dtype != torch.bfloat16 or action_out_proj.weight.dtype != torch.bfloat16:
        return None
    if action_out_proj.bias is None:
        return None
    batch, seq_len, hidden_size = hidden_states.shape
    action_dim = action_out_proj.out_features
    if hidden_size != 1024 or action_dim > 64:
        return None
    if final_modulation.ndim == 3:
        if final_modulation.shape[1] != 1:
            return None
        final_modulation = final_modulation[:, 0, :]
    if final_modulation.shape[-1] != hidden_size * 3:
        return None

    output = torch.empty(
        batch,
        seq_len,
        action_dim,
        device=hidden_states.device,
        dtype=action_out_proj.weight.dtype,
    )
    grid = (seq_len, batch)
    _pi05_final_head_kernel[grid](
        hidden_states,
        final_modulation,
        action_out_proj.weight,
        action_out_proj.bias,
        output,
        hidden_states.stride(0),
        hidden_states.stride(1),
        hidden_states.stride(2),
        final_modulation.stride(0),
        final_modulation.stride(1),
        action_out_proj.weight.stride(0),
        action_out_proj.weight.stride(1),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        hidden_size,
        action_dim,
        float(norm.eps),
        block_h=1024,
        block_o=triton.next_power_of_2(action_dim),
        num_warps=8,
    )
    return output


def _triton_final_head_euler(
    hidden_states: torch.Tensor,
    final_modulation: torch.Tensor | None,
    norm: nn.Module,
    action_out_proj: nn.Linear,
    euler_base: torch.Tensor,
    dt: float,
) -> torch.Tensor | None:
    if triton is None or hidden_states.device.type != "cuda" or final_modulation is None:
        return None
    if hidden_states.ndim != 3 or final_modulation.ndim not in (2, 3):
        return None
    if hidden_states.dtype != torch.bfloat16 or action_out_proj.weight.dtype != torch.bfloat16:
        return None
    if action_out_proj.bias is None:
        return None
    batch, seq_len, hidden_size = hidden_states.shape
    action_dim = action_out_proj.out_features
    if euler_base.shape != (batch, seq_len, action_dim):
        return None
    if hidden_size != 1024 or action_dim > 64:
        return None
    if final_modulation.ndim == 3:
        if final_modulation.shape[1] != 1:
            return None
        final_modulation = final_modulation[:, 0, :]
    if final_modulation.shape[-1] != hidden_size * 3:
        return None

    output = torch.empty_like(euler_base)
    grid = (seq_len, batch)
    _pi05_final_head_euler_kernel[grid](
        hidden_states,
        final_modulation,
        action_out_proj.weight,
        action_out_proj.bias,
        euler_base,
        output,
        hidden_states.stride(0),
        hidden_states.stride(1),
        hidden_states.stride(2),
        final_modulation.stride(0),
        final_modulation.stride(1),
        action_out_proj.weight.stride(0),
        action_out_proj.weight.stride(1),
        euler_base.stride(0),
        euler_base.stride(1),
        euler_base.stride(2),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        hidden_size,
        action_dim,
        float(norm.eps),
        float(dt),
        block_h=1024,
        block_o=triton.next_power_of_2(action_dim),
        num_warps=8,
    )
    return output


if triton is not None:

    @triton.jit
    def _pi05_rope_qk_layout_kernel(
        q_flat,
        k_flat,
        cos,
        sin,
        q_out,
        k_out,
        q_flat_stride_b: tl.constexpr,
        q_flat_stride_s: tl.constexpr,
        q_flat_stride_h: tl.constexpr,
        k_flat_stride_b: tl.constexpr,
        k_flat_stride_s: tl.constexpr,
        k_flat_stride_h: tl.constexpr,
        cos_stride_b: tl.constexpr,
        cos_stride_s: tl.constexpr,
        cos_stride_h: tl.constexpr,
        q_stride_b: tl.constexpr,
        q_stride_n: tl.constexpr,
        q_stride_s: tl.constexpr,
        q_stride_h: tl.constexpr,
        k_stride_b: tl.constexpr,
        k_stride_n: tl.constexpr,
        k_stride_s: tl.constexpr,
        k_stride_h: tl.constexpr,
        seq_len: tl.constexpr,
        head_dim: tl.constexpr,
        num_heads: tl.constexpr,
        block_rows: tl.constexpr,
        block_half: tl.constexpr,
    ):
        pid_s = tl.program_id(0)
        pid_h = tl.program_id(1)
        pid_b = tl.program_id(2)
        offs_s = pid_s * block_rows + tl.arange(0, block_rows)
        offs_d0 = tl.arange(0, block_half)
        offs_d1 = offs_d0 + block_half
        is_q = pid_h < num_heads
        q_base = pid_h * head_dim

        if is_q:
            x0 = tl.load(
                q_flat
                + pid_b * q_flat_stride_b
                + offs_s[:, None] * q_flat_stride_s
                + (q_base + offs_d0)[None, :] * q_flat_stride_h,
                mask=offs_s[:, None] < seq_len,
                other=0.0,
            ).to(tl.float32)
            x1 = tl.load(
                q_flat
                + pid_b * q_flat_stride_b
                + offs_s[:, None] * q_flat_stride_s
                + (q_base + offs_d1)[None, :] * q_flat_stride_h,
                mask=offs_s[:, None] < seq_len,
                other=0.0,
            ).to(tl.float32)
        else:
            x0 = tl.load(
                k_flat
                + pid_b * k_flat_stride_b
                + offs_s[:, None] * k_flat_stride_s
                + offs_d0[None, :] * k_flat_stride_h,
                mask=offs_s[:, None] < seq_len,
                other=0.0,
            ).to(tl.float32)
            x1 = tl.load(
                k_flat
                + pid_b * k_flat_stride_b
                + offs_s[:, None] * k_flat_stride_s
                + offs_d1[None, :] * k_flat_stride_h,
                mask=offs_s[:, None] < seq_len,
                other=0.0,
            ).to(tl.float32)

        cos0 = tl.load(
            cos + pid_b * cos_stride_b + offs_s[:, None] * cos_stride_s + offs_d0[None, :] * cos_stride_h,
            mask=offs_s[:, None] < seq_len,
            other=1.0,
        ).to(tl.float32)
        cos1 = tl.load(
            cos + pid_b * cos_stride_b + offs_s[:, None] * cos_stride_s + offs_d1[None, :] * cos_stride_h,
            mask=offs_s[:, None] < seq_len,
            other=1.0,
        ).to(tl.float32)
        sin0 = tl.load(
            sin + pid_b * cos_stride_b + offs_s[:, None] * cos_stride_s + offs_d0[None, :] * cos_stride_h,
            mask=offs_s[:, None] < seq_len,
            other=0.0,
        ).to(tl.float32)
        sin1 = tl.load(
            sin + pid_b * cos_stride_b + offs_s[:, None] * cos_stride_s + offs_d1[None, :] * cos_stride_h,
            mask=offs_s[:, None] < seq_len,
            other=0.0,
        ).to(tl.float32)
        rot0 = x0 * cos0 - x1 * sin0
        rot1 = x1 * cos1 + x0 * sin1

        if is_q:
            tl.store(
                q_out
                + pid_b * q_stride_b
                + pid_h * q_stride_n
                + offs_s[:, None] * q_stride_s
                + offs_d0[None, :] * q_stride_h,
                rot0.to(tl.bfloat16),
                mask=offs_s[:, None] < seq_len,
            )
            tl.store(
                q_out
                + pid_b * q_stride_b
                + pid_h * q_stride_n
                + offs_s[:, None] * q_stride_s
                + offs_d1[None, :] * q_stride_h,
                rot1.to(tl.bfloat16),
                mask=offs_s[:, None] < seq_len,
            )
        else:
            tl.store(
                k_out
                + pid_b * k_stride_b
                + 0 * k_stride_n
                + offs_s[:, None] * k_stride_s
                + offs_d0[None, :] * k_stride_h,
                rot0.to(tl.bfloat16),
                mask=offs_s[:, None] < seq_len,
            )
            tl.store(
                k_out
                + pid_b * k_stride_b
                + 0 * k_stride_n
                + offs_s[:, None] * k_stride_s
                + offs_d1[None, :] * k_stride_h,
                rot1.to(tl.bfloat16),
                mask=offs_s[:, None] < seq_len,
            )

    @triton.jit
    def _pi05_suffix_attn_no_cat_kernel(
        q,
        k_prefix,
        v_prefix,
        k_suffix,
        v_suffix,
        attention_mask,
        out,
        q_stride_b: tl.constexpr,
        q_stride_h: tl.constexpr,
        q_stride_s: tl.constexpr,
        q_stride_d: tl.constexpr,
        kp_stride_b: tl.constexpr,
        kp_stride_h: tl.constexpr,
        kp_stride_s: tl.constexpr,
        kp_stride_d: tl.constexpr,
        vp_stride_b: tl.constexpr,
        vp_stride_h: tl.constexpr,
        vp_stride_s: tl.constexpr,
        vp_stride_d: tl.constexpr,
        ks_stride_b: tl.constexpr,
        ks_stride_h: tl.constexpr,
        ks_stride_s: tl.constexpr,
        ks_stride_d: tl.constexpr,
        vs_stride_b: tl.constexpr,
        vs_stride_h: tl.constexpr,
        vs_stride_s: tl.constexpr,
        vs_stride_d: tl.constexpr,
        mask_stride_b: tl.constexpr,
        mask_stride_h: tl.constexpr,
        mask_stride_q: tl.constexpr,
        mask_stride_k: tl.constexpr,
        out_stride_b: tl.constexpr,
        out_stride_h: tl.constexpr,
        out_stride_s: tl.constexpr,
        out_stride_d: tl.constexpr,
        prefix_len: tl.constexpr,
        suffix_len: tl.constexpr,
        head_dim: tl.constexpr,
        scaling: tl.constexpr,
        block_m: tl.constexpr,
        block_n: tl.constexpr,
        block_d: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_h = tl.program_id(1)
        pid_b = tl.program_id(2)
        offs_m = pid_m * block_m + tl.arange(0, block_m)
        offs_n = tl.arange(0, block_n)
        offs_d = tl.arange(0, block_d)
        total_len: tl.constexpr = prefix_len + suffix_len

        q_block = tl.load(
            q + pid_b * q_stride_b + pid_h * q_stride_h + offs_m[:, None] * q_stride_s + offs_d[None, :] * q_stride_d,
            mask=(offs_m[:, None] < suffix_len) & (offs_d[None, :] < head_dim),
            other=0.0,
        )
        m_i = tl.full((block_m,), -float("inf"), tl.float32)
        l_i = tl.zeros((block_m,), tl.float32)
        acc = tl.zeros((block_m, block_d), tl.float32)

        for start_n in tl.range(0, total_len, block_n):
            cur_n = start_n + offs_n
            prefix_mask = cur_n < prefix_len
            suffix_n = cur_n - prefix_len

            kp = tl.load(
                k_prefix
                + pid_b * kp_stride_b
                + 0 * kp_stride_h
                + cur_n[None, :] * kp_stride_s
                + offs_d[:, None] * kp_stride_d,
                mask=(prefix_mask[None, :]) & (offs_d[:, None] < head_dim),
                other=0.0,
            )
            ks = tl.load(
                k_suffix
                + pid_b * ks_stride_b
                + 0 * ks_stride_h
                + suffix_n[None, :] * ks_stride_s
                + offs_d[:, None] * ks_stride_d,
                mask=(~prefix_mask[None, :]) & (cur_n[None, :] < total_len) & (offs_d[:, None] < head_dim),
                other=0.0,
            )
            k_block = kp + ks
            qk = tl.dot(q_block, k_block) * scaling
            mask_vals = tl.load(
                attention_mask
                + pid_b * mask_stride_b
                + 0 * mask_stride_h
                + offs_m[:, None] * mask_stride_q
                + cur_n[None, :] * mask_stride_k,
                mask=(offs_m[:, None] < suffix_len) & (cur_n[None, :] < total_len),
                other=-float("inf"),
            ).to(tl.float32)
            qk = qk + mask_vals
            qk = tl.where((offs_m[:, None] < suffix_len) & (cur_n[None, :] < total_len), qk, -float("inf"))

            m_new = tl.maximum(m_i, tl.max(qk, axis=1))
            alpha = tl.exp(m_i - m_new)
            p = tl.exp(qk - m_new[:, None])
            l_new = l_i * alpha + tl.sum(p, axis=1)

            vp = tl.load(
                v_prefix
                + pid_b * vp_stride_b
                + 0 * vp_stride_h
                + cur_n[:, None] * vp_stride_s
                + offs_d[None, :] * vp_stride_d,
                mask=(prefix_mask[:, None]) & (offs_d[None, :] < head_dim),
                other=0.0,
            )
            vs = tl.load(
                v_suffix
                + pid_b * vs_stride_b
                + 0 * vs_stride_h
                + suffix_n[:, None] * vs_stride_s
                + offs_d[None, :] * vs_stride_d,
                mask=(~prefix_mask[:, None]) & (cur_n[:, None] < total_len) & (offs_d[None, :] < head_dim),
                other=0.0,
            )
            v_block = vp + vs
            acc = acc * alpha[:, None] + tl.dot(p.to(v_block.dtype), v_block)
            m_i = m_new
            l_i = l_new

        acc = acc / l_i[:, None]
        tl.store(
            out
            + pid_b * out_stride_b
            + pid_h * out_stride_h
            + offs_m[:, None] * out_stride_s
            + offs_d[None, :] * out_stride_d,
            acc,
            mask=(offs_m[:, None] < suffix_len) & (offs_d[None, :] < head_dim),
        )


def _triton_rope_qk_layout(
    attn: nn.Module,
    q_flat: torch.Tensor,
    k_flat: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    if triton is None or q_flat.device.type != "cuda":
        return None
    if q_flat.ndim != 3 or k_flat.ndim != 3:
        return None
    if q_flat.dtype != torch.bfloat16 or k_flat.dtype != torch.bfloat16:
        return None
    batch, seq_len, q_size = q_flat.shape
    head_dim = int(attn.head_dim)
    num_heads = int(attn.config.num_attention_heads)
    num_kv_heads = int(attn.config.num_key_value_heads)
    if head_dim != 256 or num_heads != 8 or num_kv_heads != 1:
        return None
    if q_size != num_heads * head_dim or k_flat.shape != (batch, seq_len, head_dim):
        return None
    if cos.shape[-2:] != (seq_len, head_dim) or sin.shape[-2:] != (seq_len, head_dim):
        return None
    q_flat = q_flat.contiguous()
    k_flat = k_flat.contiguous()
    cos = cos.contiguous()
    sin = sin.contiguous()
    q = torch.empty(batch, num_heads, seq_len, head_dim, device=q_flat.device, dtype=q_flat.dtype)
    k = torch.empty(batch, num_kv_heads, seq_len, head_dim, device=k_flat.device, dtype=k_flat.dtype)
    grid = (triton.cdiv(seq_len, 8), num_heads + num_kv_heads, batch)
    _pi05_rope_qk_layout_kernel[grid](
        q_flat,
        k_flat,
        cos,
        sin,
        q,
        k,
        q_flat.stride(0),
        q_flat.stride(1),
        q_flat.stride(2),
        k_flat.stride(0),
        k_flat.stride(1),
        k_flat.stride(2),
        cos.stride(0),
        cos.stride(1),
        cos.stride(2),
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        seq_len,
        head_dim,
        num_heads,
        block_rows=8,
        block_half=128,
        num_warps=4,
    )
    return q, k


def _triton_suffix_attend_no_cat(
    query_states: torch.Tensor,
    k_prefix: torch.Tensor,
    v_prefix: torch.Tensor,
    k_suffix: torch.Tensor,
    v_suffix: torch.Tensor,
    attention_mask: torch.Tensor | None,
    *,
    scaling: float,
) -> torch.Tensor | None:
    if triton is None or query_states.device.type != "cuda" or attention_mask is None:
        return None
    if query_states.ndim != 4 or k_prefix.ndim != 4 or k_suffix.ndim != 4:
        return None
    if query_states.dtype != torch.bfloat16 or k_suffix.dtype != torch.bfloat16 or v_suffix.dtype != torch.bfloat16:
        return None
    if k_prefix.dtype != torch.bfloat16 or v_prefix.dtype != torch.bfloat16:
        return None
    batch, num_heads, suffix_len, head_dim = query_states.shape
    if k_prefix.shape[0] != batch or k_suffix.shape[0] != batch:
        return None
    if k_prefix.shape[1] != 1 or v_prefix.shape[1] != 1 or k_suffix.shape[1] != 1 or v_suffix.shape[1] != 1:
        return None
    prefix_len = k_prefix.shape[2]
    if k_suffix.shape[2] != suffix_len or v_suffix.shape[2] != suffix_len:
        return None
    if k_prefix.shape[-1] != head_dim or v_prefix.shape[-1] != head_dim or v_suffix.shape[-1] != head_dim:
        return None
    if head_dim != 256 or suffix_len > 64 or prefix_len + suffix_len > 2048:
        return None
    if attention_mask.ndim != 4 or attention_mask.shape[0] != batch:
        return None
    if attention_mask.shape[-2] < suffix_len or attention_mask.shape[-1] < prefix_len + suffix_len:
        return None

    output = torch.empty_like(query_states)
    grid = (triton.cdiv(suffix_len, 2), num_heads, batch)
    _pi05_suffix_attn_no_cat_kernel[grid](
        query_states,
        k_prefix,
        v_prefix,
        k_suffix,
        v_suffix,
        attention_mask,
        output,
        query_states.stride(0),
        query_states.stride(1),
        query_states.stride(2),
        query_states.stride(3),
        k_prefix.stride(0),
        k_prefix.stride(1),
        k_prefix.stride(2),
        k_prefix.stride(3),
        v_prefix.stride(0),
        v_prefix.stride(1),
        v_prefix.stride(2),
        v_prefix.stride(3),
        k_suffix.stride(0),
        k_suffix.stride(1),
        k_suffix.stride(2),
        k_suffix.stride(3),
        v_suffix.stride(0),
        v_suffix.stride(1),
        v_suffix.stride(2),
        v_suffix.stride(3),
        attention_mask.stride(0),
        attention_mask.stride(1),
        attention_mask.stride(2),
        attention_mask.stride(3),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        output.stride(3),
        prefix_len,
        suffix_len,
        head_dim,
        float(scaling),
        block_m=2,
        block_n=32,
        block_d=256,
        num_warps=4,
        num_stages=1,
    )
    return output


def _suffix_attend(
    query_states: torch.Tensor,
    k_prefix: torch.Tensor,
    v_prefix: torch.Tensor,
    k_suffix: torch.Tensor,
    v_suffix: torch.Tensor,
    attention_mask: torch.Tensor | None,
    *,
    num_kv_groups: int,
    scaling: float,
) -> torch.Tensor:
    k = torch.cat([k_prefix.to(k_suffix.dtype), k_suffix], dim=2)
    v = torch.cat([v_prefix.to(v_suffix.dtype), v_suffix], dim=2)
    return _attend(
        query_states,
        k,
        v,
        attention_mask,
        num_kv_groups=num_kv_groups,
        scaling=scaling,
    )


def _suffix_attend_reuse_kv(
    query_states: torch.Tensor,
    k_prefix: torch.Tensor,
    v_prefix: torch.Tensor,
    k_suffix: torch.Tensor,
    v_suffix: torch.Tensor,
    attention_mask: torch.Tensor | None,
    kv_buffer: tuple[torch.Tensor, torch.Tensor] | None,
    *,
    num_kv_groups: int,
    scaling: float,
) -> torch.Tensor | None:
    if kv_buffer is None:
        return None
    k_full, v_full = kv_buffer
    prefix_len = k_prefix.shape[-2]
    suffix_len = k_suffix.shape[-2]
    expected_shape = (*k_prefix.shape[:2], prefix_len + suffix_len, k_prefix.shape[-1])
    if k_full.shape != expected_shape or v_full.shape != expected_shape:
        return None
    if k_full.dtype != k_suffix.dtype or v_full.dtype != v_suffix.dtype:
        return None

    k_full[:, :, prefix_len : prefix_len + suffix_len, :].copy_(k_suffix)
    v_full[:, :, prefix_len : prefix_len + suffix_len, :].copy_(v_suffix)
    return _attend(
        query_states,
        k_full,
        v_full,
        attention_mask,
        num_kv_groups=num_kv_groups,
        scaling=scaling,
    )


def _suffix_attend_no_cat(
    query_states: torch.Tensor,
    k_prefix: torch.Tensor,
    v_prefix: torch.Tensor,
    k_suffix: torch.Tensor,
    v_suffix: torch.Tensor,
    attention_mask: torch.Tensor | None,
    *,
    scaling: float,
) -> torch.Tensor | None:
    if k_prefix.shape[1] != 1 or k_suffix.shape[1] != 1:
        return None
    triton_out = _triton_suffix_attend_no_cat(
        query_states,
        k_prefix,
        v_prefix,
        k_suffix,
        v_suffix,
        attention_mask,
        scaling=scaling,
    )
    if triton_out is not None:
        return triton_out
    prefix_len = k_prefix.shape[-2]
    suffix_len = k_suffix.shape[-2]
    k_prefix = k_prefix.to(k_suffix.dtype)
    v_prefix = v_prefix.to(v_suffix.dtype)
    prefix_weights = torch.matmul(query_states, k_prefix.transpose(2, 3)) * scaling
    suffix_weights = torch.matmul(query_states, k_suffix.transpose(2, 3)) * scaling
    if attention_mask is not None:
        prefix_weights = prefix_weights + attention_mask[:, :, :, :prefix_len]
        suffix_weights = suffix_weights + attention_mask[:, :, :, prefix_len : prefix_len + suffix_len]
    attn_weights = torch.cat([prefix_weights, suffix_weights], dim=-1)
    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    prefix_probs, suffix_probs = attn_weights.split((prefix_len, suffix_len), dim=-1)
    return torch.matmul(prefix_probs, v_prefix) + torch.matmul(suffix_probs, v_suffix)


def _prepare_packed_gemma_mlp(mlp: nn.Module) -> None:
    if not all(hasattr(mlp, name) for name in ("gate_proj", "up_proj", "down_proj")):
        return
    gate_proj = mlp.gate_proj
    up_proj = mlp.up_proj
    gate_bias = getattr(gate_proj, "bias", None)
    up_bias = getattr(up_proj, "bias", None)
    key = (
        gate_proj.weight.data_ptr(),
        up_proj.weight.data_ptr(),
        gate_proj.weight.dtype,
        gate_proj.weight.device,
        gate_bias.data_ptr() if gate_bias is not None else 0,
        up_bias.data_ptr() if up_bias is not None else 0,
    )
    if getattr(mlp, "_pi05_packed_gate_up_key", None) == key:
        return
    if gate_bias is None and up_bias is not None or gate_bias is not None and up_bias is None:
        return
    mlp._pi05_packed_gate_up_key = key
    mlp._pi05_packed_gate_up_weight = torch.cat([gate_proj.weight, up_proj.weight], dim=0).contiguous()
    mlp._pi05_packed_gate_up_bias = None if gate_bias is None else torch.cat([gate_bias, up_bias], dim=0).contiguous()


def _gemma_mlp_forward(
    mlp: nn.Module,
    x: torch.Tensor,
    *,
    use_packed_mlp: bool = False,
) -> torch.Tensor:
    if not use_packed_mlp:
        return mlp(x)
    packed_weight = getattr(mlp, "_pi05_packed_gate_up_weight", None)
    if packed_weight is None:
        _prepare_packed_gemma_mlp(mlp)
        packed_weight = getattr(mlp, "_pi05_packed_gate_up_weight", None)
    if packed_weight is None:
        return mlp(x)
    packed_bias = getattr(mlp, "_pi05_packed_gate_up_bias", None)
    gate, up = F.linear(x, packed_weight, packed_bias).chunk(2, dim=-1)
    return mlp.down_proj(mlp.act_fn(gate) * up)


def _prepare_packed_qkv(attn: nn.Module) -> None:
    if not all(hasattr(attn, name) for name in ("q_proj", "k_proj", "v_proj")):
        return
    q_proj = attn.q_proj
    k_proj = attn.k_proj
    v_proj = attn.v_proj
    q_bias = getattr(q_proj, "bias", None)
    k_bias = getattr(k_proj, "bias", None)
    v_bias = getattr(v_proj, "bias", None)
    key = (
        q_proj.weight.data_ptr(),
        k_proj.weight.data_ptr(),
        v_proj.weight.data_ptr(),
        q_proj.weight.dtype,
        q_proj.weight.device,
        q_bias.data_ptr() if q_bias is not None else 0,
        k_bias.data_ptr() if k_bias is not None else 0,
        v_bias.data_ptr() if v_bias is not None else 0,
    )
    if getattr(attn, "_pi05_packed_qkv_key", None) == key:
        return
    if (q_bias is None) != (k_bias is None) or (q_bias is None) != (v_bias is None):
        return
    attn._pi05_packed_qkv_key = key
    packed_weight = torch.cat(
        [q_proj.weight, k_proj.weight, v_proj.weight],
        dim=0,
    ).contiguous()
    attn._pi05_packed_qkv_weight = packed_weight
    attn._pi05_packed_qkv_bias = None if q_bias is None else torch.cat([q_bias, k_bias, v_bias], dim=0).contiguous()
    attn._pi05_packed_qkv_splits = (
        q_proj.out_features,
        k_proj.out_features,
        v_proj.out_features,
    )


def _qkv_projection(attn: nn.Module, x: torch.Tensor, *, use_packed_qkv: bool = False):
    if not use_packed_qkv:
        return attn.q_proj(x), attn.k_proj(x), attn.v_proj(x)
    packed_weight = getattr(attn, "_pi05_packed_qkv_weight", None)
    if packed_weight is None:
        _prepare_packed_qkv(attn)
        packed_weight = getattr(attn, "_pi05_packed_qkv_weight", None)
    if packed_weight is None:
        return attn.q_proj(x), attn.k_proj(x), attn.v_proj(x)
    packed_bias = getattr(attn, "_pi05_packed_qkv_bias", None)
    q_size, k_size, v_size = attn._pi05_packed_qkv_splits
    return F.linear(x, packed_weight, packed_bias).split((q_size, k_size, v_size), dim=-1)


def _compute_layer_prefix_only(
    layer_idx: int,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor | None,
    position_ids: torch.Tensor,
    paligemma: PaliGemmaForConditionalGeneration,
    use_packed_qkv: bool = False,
    use_packed_mlp: bool = False,
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    model = paligemma.model.language_model
    layer = model.layers[layer_idx]
    residual = hidden_states
    x, gate = layernorm_forward(layer.input_layernorm, hidden_states)

    hidden_shape = (*x.shape[:-1], -1, layer.self_attn.head_dim)
    q, k, v = _qkv_projection(layer.self_attn, x, use_packed_qkv=use_packed_qkv)
    q = q.view(hidden_shape).transpose(1, 2)
    k = k.view(hidden_shape).transpose(1, 2)
    v = v.view(hidden_shape).transpose(1, 2)

    cos, sin = model.rotary_emb(v, position_ids)
    q, k = apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1)

    att = _attend(
        q,
        k,
        v,
        attention_mask,
        num_kv_groups=layer.self_attn.num_key_value_groups,
        scaling=1.0 / math.sqrt(layer.self_attn.head_dim),
    )
    att = att.transpose(1, 2).reshape(q.shape[0], -1, q.shape[1] * layer.self_attn.head_dim)

    out = layer.self_attn.o_proj(att)
    out = _gated_residual(residual, out, gate)
    after_first_residual = out
    out, gate = layernorm_forward(layer.post_attention_layernorm, out)
    out = _gemma_mlp_forward(layer.mlp, out, use_packed_mlp=use_packed_mlp)
    out = _gated_residual(after_first_residual, out, gate)
    return out, (k, v)


def _compute_layer_suffix_only(
    layer_idx: int,
    hidden_states: torch.Tensor,
    prefix_kv: tuple[torch.Tensor, torch.Tensor],
    attention_mask: torch.Tensor | None,
    position_ids: torch.Tensor,
    gemma_expert: GemmaForCausalLM,
    adarms_cond: torch.Tensor | None,
    adarms_modulations: tuple[torch.Tensor | None, torch.Tensor | None] | None = None,
    rope_cos_sin: tuple[torch.Tensor, torch.Tensor] | None = None,
    profile_nvtx: bool = False,
    use_packed_qkv: bool = False,
    use_packed_mlp: bool = False,
    use_triton_qkv_rope: bool = False,
    use_no_cat_suffix_attn: bool = False,
    suffix_attn_kv_buffer: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> torch.Tensor:
    layer = gemma_expert.model.layers[layer_idx]
    residual = hidden_states
    input_mod = adarms_modulations[0] if adarms_modulations is not None else None
    post_mod = adarms_modulations[1] if adarms_modulations is not None else None

    if not profile_nvtx:
        if rope_cos_sin is None:
            cos, sin = gemma_expert.model.rotary_emb(hidden_states, position_ids)
        else:
            cos, sin = rope_cos_sin
        x, gate = layernorm_forward(layer.input_layernorm, hidden_states, adarms_cond, modulation=input_mod)

        hidden_shape = (*x.shape[:-1], -1, layer.self_attn.head_dim)
        q, k_suffix, v_suffix = _qkv_projection(layer.self_attn, x, use_packed_qkv=use_packed_qkv)
        v_suffix = v_suffix.view(hidden_shape).transpose(1, 2)
        fused_qk = _triton_rope_qk_layout(layer.self_attn, q, k_suffix, cos, sin) if use_triton_qkv_rope else None
        if fused_qk is None:
            q = q.view(hidden_shape).transpose(1, 2)
            k_suffix = k_suffix.view(hidden_shape).transpose(1, 2)
            q, k_suffix = apply_rotary_pos_emb(q, k_suffix, cos, sin, unsqueeze_dim=1)
        else:
            q, k_suffix = fused_qk

        k_prefix, v_prefix = prefix_kv
        scaling = 1.0 / math.sqrt(layer.self_attn.head_dim)
        att = (
            _suffix_attend_reuse_kv(
                q,
                k_prefix,
                v_prefix,
                k_suffix,
                v_suffix,
                attention_mask,
                suffix_attn_kv_buffer,
                num_kv_groups=layer.self_attn.num_key_value_groups,
                scaling=scaling,
            )
            if use_no_cat_suffix_attn
            else None
        )
        if att is None:
            att = (
                _suffix_attend_no_cat(
                    q,
                    k_prefix,
                    v_prefix,
                    k_suffix,
                    v_suffix,
                    attention_mask,
                    scaling=scaling,
                )
                if use_no_cat_suffix_attn and suffix_attn_kv_buffer is None
                else None
            )
        if att is None:
            att = _suffix_attend(
                q,
                k_prefix,
                v_prefix,
                k_suffix,
                v_suffix,
                attention_mask,
                num_kv_groups=layer.self_attn.num_key_value_groups,
                scaling=scaling,
            )
        att = att.transpose(1, 2).reshape(q.shape[0], -1, q.shape[1] * layer.self_attn.head_dim)

        out = layer.self_attn.o_proj(att)
        out = _gated_residual(residual, out, gate)
        after_first_residual = out
        out, gate = layernorm_forward(layer.post_attention_layernorm, out, adarms_cond, modulation=post_mod)
        if out.dtype != layer.mlp.up_proj.weight.dtype:
            out = out.to(dtype=layer.mlp.up_proj.weight.dtype)
        out = _gemma_mlp_forward(
            layer.mlp,
            out,
            use_packed_mlp=use_packed_mlp,
        )
        return _gated_residual(after_first_residual, out, gate)

    with _ProfileRange(f"pi05.denoise.layer_{layer_idx}.input_norm", profile_nvtx):
        if rope_cos_sin is None:
            cos, sin = gemma_expert.model.rotary_emb(hidden_states, position_ids)
        else:
            cos, sin = rope_cos_sin
        x, gate = layernorm_forward(layer.input_layernorm, hidden_states, adarms_cond, modulation=input_mod)

    with _ProfileRange(f"pi05.denoise.layer_{layer_idx}.qkv_rope", profile_nvtx):
        hidden_shape = (*x.shape[:-1], -1, layer.self_attn.head_dim)
        q, k_suffix, v_suffix = _qkv_projection(layer.self_attn, x, use_packed_qkv=use_packed_qkv)
        v_suffix = v_suffix.view(hidden_shape).transpose(1, 2)
        fused_qk = _triton_rope_qk_layout(layer.self_attn, q, k_suffix, cos, sin) if use_triton_qkv_rope else None
        if fused_qk is None:
            q = q.view(hidden_shape).transpose(1, 2)
            k_suffix = k_suffix.view(hidden_shape).transpose(1, 2)
            q, k_suffix = apply_rotary_pos_emb(q, k_suffix, cos, sin, unsqueeze_dim=1)
        else:
            q, k_suffix = fused_qk

    with _ProfileRange(f"pi05.denoise.layer_{layer_idx}.kv_prepare", profile_nvtx):
        k_prefix, v_prefix = prefix_kv

    with _ProfileRange(f"pi05.denoise.layer_{layer_idx}.attention", profile_nvtx):
        scaling = 1.0 / math.sqrt(layer.self_attn.head_dim)
        att = (
            _suffix_attend_reuse_kv(
                q,
                k_prefix,
                v_prefix,
                k_suffix,
                v_suffix,
                attention_mask,
                suffix_attn_kv_buffer,
                num_kv_groups=layer.self_attn.num_key_value_groups,
                scaling=scaling,
            )
            if use_no_cat_suffix_attn
            else None
        )
        if att is None:
            att = (
                _suffix_attend_no_cat(
                    q,
                    k_prefix,
                    v_prefix,
                    k_suffix,
                    v_suffix,
                    attention_mask,
                    scaling=scaling,
                )
                if use_no_cat_suffix_attn and suffix_attn_kv_buffer is None
                else None
            )
        if att is None:
            att = _suffix_attend(
                q,
                k_prefix,
                v_prefix,
                k_suffix,
                v_suffix,
                attention_mask,
                num_kv_groups=layer.self_attn.num_key_value_groups,
                scaling=scaling,
            )
        att = att.transpose(1, 2).reshape(q.shape[0], -1, q.shape[1] * layer.self_attn.head_dim)

    with _ProfileRange(f"pi05.denoise.layer_{layer_idx}.attn_out", profile_nvtx):
        out = layer.self_attn.o_proj(att)
    out = _gated_residual(residual, out, gate)
    after_first_residual = out
    with _ProfileRange(f"pi05.denoise.layer_{layer_idx}.post_norm", profile_nvtx):
        out, gate = layernorm_forward(layer.post_attention_layernorm, out, adarms_cond, modulation=post_mod)
        if out.dtype != layer.mlp.up_proj.weight.dtype:
            out = out.to(dtype=layer.mlp.up_proj.weight.dtype)
    with _ProfileRange(f"pi05.denoise.layer_{layer_idx}.mlp", profile_nvtx):
        out = _gemma_mlp_forward(
            layer.mlp,
            out,
            use_packed_mlp=use_packed_mlp,
        )
    out = _gated_residual(after_first_residual, out, gate)
    return out


class PaliGemmaWithActionExpertPi05(nn.Module):
    def __init__(
        self,
        vlm_config,
        action_expert_config,
        *,
        image_size: int = 224,
        vocab_size: int = 257152,
        image_token_index: int = 257152,
    ):
        super().__init__()

        vlm_config_hf = CONFIG_MAPPING["paligemma"]()
        vlm_config_hf._vocab_size = vocab_size
        vlm_config_hf.image_token_index = image_token_index
        vlm_config_hf.text_config.hidden_size = vlm_config.width
        vlm_config_hf.text_config.intermediate_size = vlm_config.mlp_dim
        vlm_config_hf.text_config.num_attention_heads = vlm_config.num_heads
        vlm_config_hf.text_config.head_dim = vlm_config.head_dim
        vlm_config_hf.text_config.num_hidden_layers = vlm_config.depth
        vlm_config_hf.text_config.num_key_value_heads = vlm_config.num_kv_heads
        vlm_config_hf.text_config.hidden_activation = "gelu_pytorch_tanh"
        vlm_config_hf.text_config.dtype = "float32"
        vlm_config_hf.text_config.vocab_size = vocab_size
        vlm_config_hf.text_config.use_adarms = False
        vlm_config_hf.text_config.adarms_cond_dim = None
        vlm_config_hf.vision_config.image_size = image_size
        vlm_config_hf.vision_config.intermediate_size = 4304
        vlm_config_hf.vision_config.projection_dim = 2048
        vlm_config_hf.vision_config.projector_hidden_act = "gelu_fast"
        vlm_config_hf.vision_config.dtype = "float32"

        action_expert_config_hf = CONFIG_MAPPING["gemma"](
            head_dim=action_expert_config.head_dim,
            hidden_size=action_expert_config.width,
            intermediate_size=action_expert_config.mlp_dim,
            num_attention_heads=action_expert_config.num_heads,
            num_hidden_layers=action_expert_config.depth,
            num_key_value_heads=action_expert_config.num_kv_heads,
            vocab_size=vocab_size,
            hidden_activation="gelu_pytorch_tanh",
            dtype="float32",
            use_adarms=True,
            adarms_cond_dim=action_expert_config.width,
        )

        self.paligemma = PaliGemmaForConditionalGenerationWithPiGemma(config=vlm_config_hf)
        self.gemma_expert = PiGemmaForCausalLM(config=action_expert_config_hf)
        self.gemma_expert.model.embed_tokens = None

    @staticmethod
    def _module_param_dtype(module: nn.Module, fallback: torch.dtype) -> torch.dtype:
        try:
            return next(module.parameters()).dtype
        except StopIteration:
            return fallback

    def embed_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        vision_dtype = self._module_param_dtype(
            self.paligemma.model.vision_tower,
            pixel_values.dtype,
        )
        if pixel_values.dtype != vision_dtype:
            pixel_values = pixel_values.to(dtype=vision_dtype)
        if hasattr(self.paligemma, "get_image_features"):
            image_outputs = self.paligemma.get_image_features(pixel_values)
            if hasattr(image_outputs, "pooler_output"):
                features = image_outputs.pooler_output
            elif isinstance(image_outputs, (tuple, list)):
                features = image_outputs[0]
            else:
                features = image_outputs
        else:
            vision_outputs = self.paligemma.model.vision_tower(pixel_values)
            features = (
                vision_outputs.last_hidden_state if hasattr(vision_outputs, "last_hidden_state") else vision_outputs[0]
            )
            features = self.paligemma.model.multi_modal_projector(features)
        features = features * (self.paligemma.config.text_config.hidden_size**0.5)
        return features

    def embed_language_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        embed_tokens = self.paligemma.model.language_model.embed_tokens
        lang_emb = embed_tokens(tokens)
        if getattr(embed_tokens, "embed_scale", None) is None:
            lang_emb = lang_emb * math.sqrt(lang_emb.shape[-1])
        return lang_emb

    @staticmethod
    def _precompute_norm_modulation(
        norm: nn.Module,
        adarms_cond: torch.Tensor | None,
    ) -> torch.Tensor | None:
        dense = getattr(norm, "dense", None)
        if adarms_cond is None or dense is None:
            return None
        return dense(adarms_cond)

    def precompute_suffix_adarms(self, adarms_cond: torch.Tensor | None) -> _DenoiseAdarmsModulations:
        expert_lm = self.gemma_expert.model
        layer_modulations: list[tuple[torch.Tensor | None, torch.Tensor | None]] = []
        for layer in expert_lm.layers:
            layer_modulations.append(
                (
                    self._precompute_norm_modulation(layer.input_layernorm, adarms_cond),
                    self._precompute_norm_modulation(layer.post_attention_layernorm, adarms_cond),
                )
            )
        return _DenoiseAdarmsModulations(
            layer_modulations=layer_modulations,
            final_modulation=self._precompute_norm_modulation(expert_lm.norm, adarms_cond),
        )

    @staticmethod
    def _packed_qkv_ready(layers) -> bool:
        return all(getattr(layer.self_attn, "_pi05_packed_qkv_weight", None) is not None for layer in layers)

    @staticmethod
    def _packed_mlp_ready(layers) -> bool:
        return all(getattr(layer.mlp, "_pi05_packed_gate_up_weight", None) is not None for layer in layers)

    def reset_packed_weight_cache(self) -> None:
        for name in (
            "_pi05_packed_suffix_mlp_ready",
            "_pi05_packed_suffix_qkv_ready",
            "_pi05_packed_prefix_mlp_ready",
            "_pi05_packed_prefix_qkv_ready",
        ):
            if hasattr(self, name):
                delattr(self, name)

    def prepare_packed_suffix_mlp(self) -> None:
        if getattr(self, "_pi05_packed_suffix_mlp_ready", False):
            return
        layers = self.gemma_expert.model.layers
        for layer in layers:
            _prepare_packed_gemma_mlp(layer.mlp)
        self._pi05_packed_suffix_mlp_ready = self._packed_mlp_ready(layers)

    def prepare_packed_suffix_qkv(self) -> None:
        if getattr(self, "_pi05_packed_suffix_qkv_ready", False):
            return
        layers = self.gemma_expert.model.layers
        for layer in layers:
            _prepare_packed_qkv(layer.self_attn)
        self._pi05_packed_suffix_qkv_ready = self._packed_qkv_ready(layers)

    def prepare_packed_prefix_qkv(self) -> None:
        if getattr(self, "_pi05_packed_prefix_qkv_ready", False):
            return
        layers = self.paligemma.model.language_model.layers
        for layer in layers:
            _prepare_packed_qkv(layer.self_attn)
        self._pi05_packed_prefix_qkv_ready = self._packed_qkv_ready(layers)

    def prepare_packed_prefix_mlp(self) -> None:
        if getattr(self, "_pi05_packed_prefix_mlp_ready", False):
            return
        layers = self.paligemma.model.language_model.layers
        for layer in layers:
            _prepare_packed_gemma_mlp(layer.mlp)
        self._pi05_packed_prefix_mlp_ready = self._packed_mlp_ready(layers)

    def forward_suffix_only(
        self,
        *,
        suffix_embs: torch.Tensor,
        attention_mask: torch.Tensor | None,
        position_ids: torch.LongTensor | None,
        past_key_values,
        adarms_cond: torch.Tensor | None = None,
        adarms_modulations: _DenoiseAdarmsModulations | None = None,
        profile_nvtx: bool = False,
        use_packed_qkv: bool = False,
        use_packed_mlp: bool = False,
        use_triton_qkv_rope: bool = False,
        use_no_cat_suffix_attn: bool = False,
        suffix_attn_kv_buffers: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
        rope_cos_sin: tuple[torch.Tensor, torch.Tensor] | None = None,
        skip_final_norm: bool = False,
    ) -> torch.Tensor:
        if not isinstance(past_key_values, list):
            raise TypeError(
                "suffix_only forward expects past_key_values to be the "
                "list[(k, v)] produced by prefix_only forward; "
                f"got {type(past_key_values)}"
            )

        num_layers = self.paligemma.config.text_config.num_hidden_layers
        expert_lm = self.gemma_expert.model
        hidden_states = suffix_embs
        for layer_idx in range(num_layers):
            layer_mods = None if adarms_modulations is None else adarms_modulations.layer_modulations[layer_idx]
            hidden_states = _compute_layer_suffix_only(
                layer_idx,
                hidden_states,
                past_key_values[layer_idx],
                attention_mask,
                position_ids,
                gemma_expert=self.gemma_expert,
                adarms_cond=adarms_cond,
                adarms_modulations=layer_mods,
                rope_cos_sin=rope_cos_sin,
                profile_nvtx=profile_nvtx,
                use_packed_qkv=use_packed_qkv,
                use_packed_mlp=use_packed_mlp,
                use_triton_qkv_rope=use_triton_qkv_rope,
                use_no_cat_suffix_attn=use_no_cat_suffix_attn,
                suffix_attn_kv_buffer=(None if suffix_attn_kv_buffers is None else suffix_attn_kv_buffers[layer_idx]),
            )
        if skip_final_norm:
            return hidden_states
        final_mod = None if adarms_modulations is None else adarms_modulations.final_modulation
        hidden_states, _ = layernorm_forward(expert_lm.norm, hidden_states, adarms_cond, modulation=final_mod)
        return hidden_states

    def forward_prefix_only(
        self,
        *,
        prefix_embs: torch.Tensor,
        attention_mask: torch.Tensor | None,
        position_ids: torch.LongTensor,
        use_cache: bool = True,
        use_packed_prefix_qkv: bool = False,
        use_packed_prefix_mlp: bool = False,
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]] | None]:
        num_layers = self.paligemma.config.text_config.num_hidden_layers
        pali_lm = self.paligemma.model.language_model
        hidden_states = prefix_embs
        kv_list: list[tuple[torch.Tensor, torch.Tensor]] = []
        for layer_idx in range(num_layers):
            hidden_states, kv = _compute_layer_prefix_only(
                layer_idx,
                hidden_states,
                attention_mask,
                position_ids,
                paligemma=self.paligemma,
                use_packed_qkv=use_packed_prefix_qkv,
                use_packed_mlp=use_packed_prefix_mlp,
            )
            kv_list.append(kv)
        hidden_states, _ = layernorm_forward(pali_lm.norm, hidden_states)
        return hidden_states, (kv_list if use_cache else None)

    def forward(
        self,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values=None,
        inputs_embeds: list[torch.Tensor | None] | None = None,
        use_cache: bool | None = None,
        adarms_cond: list[torch.Tensor | None] | None = None,
    ):
        if adarms_cond is None:
            adarms_cond = [None, None]

        if inputs_embeds[1] is None:
            hidden_states, kv_list = self.forward_prefix_only(
                prefix_embs=inputs_embeds[0],
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=bool(use_cache),
            )
            return [hidden_states, None], kv_list

        if inputs_embeds[0] is None:
            hidden_states = self.forward_suffix_only(
                suffix_embs=inputs_embeds[1],
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                adarms_cond=adarms_cond[1],
            )
            return [None, hidden_states], None

        raise ValueError(
            "PaliGemmaWithActionExpertPi05.forward only supports prefix-only "
            "or suffix-only dispatch; got both inputs_embeds populated."
        )


class Pi05ForActionPrediction(nn.Module):
    """pi0.5 VLA model for robot action prediction via flow matching."""

    def __init__(
        self,
        config,
        quant_config: Any | None = None,
        prefix: str = "",
    ):
        super().__init__()
        del quant_config, prefix
        self.config = config

        self.action_dim = getattr(config, "max_action_dim", DEFAULT_ACTION_DIM)
        self.action_horizon = getattr(config, "chunk_size", DEFAULT_ACTION_HORIZON)
        self.num_inference_steps = getattr(config, "num_inference_steps", DEFAULT_NUM_INFERENCE_STEPS)

        paligemma_variant = getattr(config, "paligemma_variant", "gemma_2b")
        action_expert_variant = getattr(config, "action_expert_variant", "gemma_300m")
        vlm_config = get_gemma_config(paligemma_variant)
        expert_config = get_gemma_config(action_expert_variant)
        self.expert_width = expert_config.width

        image_resolution = getattr(config, "image_resolution", (224, 224))
        image_size = image_resolution[0] if isinstance(image_resolution, (tuple, list)) else 224
        vocab_size = getattr(config, "vocab_size", 257152)
        image_token_index = getattr(config, "image_token_index", vocab_size)
        self.paligemma_with_expert = PaliGemmaWithActionExpertPi05(
            vlm_config,
            expert_config,
            image_size=image_size,
            vocab_size=vocab_size,
            image_token_index=image_token_index,
        )

        self.action_in_proj = nn.Linear(self.action_dim, self.expert_width)
        self.action_out_proj = nn.Linear(self.expert_width, self.action_dim)
        self.time_mlp_in = nn.Linear(self.expert_width, self.expert_width)
        self.time_mlp_out = nn.Linear(self.expert_width, self.expert_width)

        self._action_norm = _build_norm_buffers(getattr(config, "norm_stats", None), "action")
        if self._action_norm is None:
            logger.info(
                "pi0.5: no action normalization stats on config.norm_stats; "
                "returned actions are in the model's normalized space."
            )

    def _unnormalize_actions(self, actions: torch.Tensor) -> torch.Tensor:
        return _apply_norm(actions, self._action_norm, inverse=True)

    def _image_embed_callable(
        self,
        torch_compile_image_embed: bool,
        *,
        torch_compile_fullgraph: bool = False,
    ):
        if not torch_compile_image_embed:
            return self.paligemma_with_expert.embed_image
        key = bool(torch_compile_fullgraph)
        compiled_variants = getattr(self, "_compiled_image_embed_variants", {})
        compiled = compiled_variants.get(key)
        if compiled is None:
            compiled = _compile_call(
                self.paligemma_with_expert.embed_image,
                fullgraph=torch_compile_fullgraph,
            )
            compiled_variants[key] = compiled
            self._compiled_image_embed_variants = compiled_variants
        return compiled

    def _prefix_forward_callable(
        self,
        torch_compile_prefix: bool,
        *,
        torch_compile_fullgraph: bool = False,
    ):
        if not torch_compile_prefix:
            return self.paligemma_with_expert.forward_prefix_only
        key = bool(torch_compile_fullgraph)
        compiled_variants = getattr(self, "_compiled_prefix_forward_variants", {})
        compiled = compiled_variants.get(key)
        if compiled is None:
            compiled = _compile_call(
                self.paligemma_with_expert.forward_prefix_only,
                fullgraph=torch_compile_fullgraph,
            )
            compiled_variants[key] = compiled
            self._compiled_prefix_forward_variants = compiled_variants
        return compiled

    def _suffix_forward_callable(
        self,
        torch_compile_suffix: bool,
        *,
        torch_compile_fullgraph: bool = False,
    ):
        if not torch_compile_suffix:
            return self.paligemma_with_expert.forward_suffix_only
        key = bool(torch_compile_fullgraph)
        compiled_variants = getattr(self, "_compiled_suffix_forward_variants", {})
        compiled = compiled_variants.get(key)
        if compiled is None:
            compiled = _compile_call(
                self.paligemma_with_expert.forward_suffix_only,
                fullgraph=torch_compile_fullgraph,
            )
            compiled_variants[key] = compiled
            self._compiled_suffix_forward_variants = compiled_variants
        return compiled

    def _denoise_loop_callable(self, torch_compile_denoise_loop: bool):
        if not torch_compile_denoise_loop:
            return self._run_denoise_loop
        compiled = getattr(self, "_compiled_denoise_loop", None)
        if compiled is None:
            compiled = _compile_call(self._run_denoise_loop)
            self._compiled_denoise_loop = compiled
        return compiled

    def _run_prefix_forward(
        self,
        *,
        prefix_embs: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        torch_compile_prefix: bool = False,
        torch_compile_prefix_fullgraph: bool = False,
        use_packed_prefix_qkv: bool = False,
        use_packed_prefix_mlp: bool = False,
    ):
        prefix_forward = self._prefix_forward_callable(
            torch_compile_prefix,
            torch_compile_fullgraph=torch_compile_prefix_fullgraph,
        )
        return prefix_forward(
            prefix_embs=prefix_embs,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=True,
            use_packed_prefix_qkv=use_packed_prefix_qkv,
            use_packed_prefix_mlp=use_packed_prefix_mlp,
        )

    @staticmethod
    def _image_cuda_graph_key(
        *,
        pixel_values: torch.Tensor,
        torch_compile_image_embed: bool,
        torch_compile_image_embed_fullgraph: bool,
    ) -> tuple[Any, ...]:
        return (
            bool(torch_compile_image_embed),
            bool(torch_compile_image_embed_fullgraph),
            str(pixel_values.device),
            pixel_values.dtype,
            tuple(pixel_values.shape),
        )

    def _run_image_embed(
        self,
        pixel_values: torch.Tensor,
        *,
        cuda_graph_image_embed: bool,
        torch_compile_image_embed: bool,
        torch_compile_image_embed_fullgraph: bool = False,
        timing: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        if not cuda_graph_image_embed:
            image_embed = self._image_embed_callable(
                torch_compile_image_embed,
                torch_compile_fullgraph=torch_compile_image_embed_fullgraph,
            )
            return image_embed(pixel_values)
        if pixel_values.device.type != "cuda":
            raise RuntimeError("cuda_graph_image_embed requires CUDA tensors")

        key = self._image_cuda_graph_key(
            pixel_values=pixel_values,
            torch_compile_image_embed=torch_compile_image_embed,
            torch_compile_image_embed_fullgraph=torch_compile_image_embed_fullgraph,
        )
        caches: dict[tuple[Any, ...], _ImageCudaGraphCache] = getattr(self, "_image_cuda_graph_caches", {})
        cache = caches.get(key)
        if cache is None:
            device = pixel_values.device
            capture_t0 = time.perf_counter() if timing is not None else 0.0
            image_embed = self._image_embed_callable(
                torch_compile_image_embed,
                torch_compile_fullgraph=torch_compile_image_embed_fullgraph,
            )
            cache = _ImageCudaGraphCache(
                graph=torch.cuda.CUDAGraph(),
                static_pixels=torch.empty_like(pixel_values),
                static_output=torch.empty(0, device=device, dtype=pixel_values.dtype),
            )
            cache.static_pixels.copy_(pixel_values)
            with torch.cuda.device(device):
                warmup_stream = torch.cuda.Stream()
                warmup_stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(warmup_stream):
                    for _ in range(3):
                        cache.static_output = image_embed(cache.static_pixels)
                torch.cuda.current_stream().wait_stream(warmup_stream)

                cache.static_pixels.copy_(pixel_values)
                with torch.cuda.graph(cache.graph):
                    cache.static_output = image_embed(cache.static_pixels)
            caches[key] = cache
            self._image_cuda_graph_caches = caches
            if timing is not None:
                _sync_for_timing(device)
                timing["image_cuda_graph_capture_ms"] = (
                    timing.get("image_cuda_graph_capture_ms", 0.0) + (time.perf_counter() - capture_t0) * 1000.0
                )

        with _TimingBlock(timing, "image_cuda_graph_copy_ms", pixel_values.device):
            cache.static_pixels.copy_(pixel_values)
        with _TimingBlock(timing, "image_cuda_graph_replay_ms", pixel_values.device):
            cache.graph.replay()
        return cache.static_output

    @staticmethod
    def _prefix_cuda_graph_key(
        *,
        prefix_embs: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        torch_compile_prefix: bool,
        torch_compile_prefix_fullgraph: bool,
        use_packed_prefix_qkv: bool,
        use_packed_prefix_mlp: bool,
    ) -> tuple[Any, ...]:
        return (
            bool(torch_compile_prefix),
            bool(torch_compile_prefix_fullgraph),
            bool(use_packed_prefix_qkv),
            bool(use_packed_prefix_mlp),
            str(prefix_embs.device),
            prefix_embs.dtype,
            tuple(prefix_embs.shape),
            attention_mask.dtype,
            tuple(attention_mask.shape),
            position_ids.dtype,
            tuple(position_ids.shape),
        )

    @staticmethod
    def _copy_prefix_graph_inputs(
        cache: _PrefixCudaGraphCache,
        *,
        prefix_embs: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> None:
        cache.static_prefix_embs.copy_(prefix_embs)
        cache.static_attention_mask.copy_(attention_mask)
        cache.static_position_ids.copy_(position_ids)

    def _get_or_create_prefix_cuda_graph(
        self,
        *,
        prefix_embs: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        torch_compile_prefix: bool,
        torch_compile_prefix_fullgraph: bool,
        use_packed_prefix_qkv: bool,
        use_packed_prefix_mlp: bool,
        timing: dict[str, Any] | None = None,
    ) -> _PrefixCudaGraphCache:
        if prefix_embs.device.type != "cuda":
            raise RuntimeError("cuda_graph_prefix requires CUDA tensors")

        key = self._prefix_cuda_graph_key(
            prefix_embs=prefix_embs,
            attention_mask=attention_mask,
            position_ids=position_ids,
            torch_compile_prefix=torch_compile_prefix,
            torch_compile_prefix_fullgraph=torch_compile_prefix_fullgraph,
            use_packed_prefix_qkv=use_packed_prefix_qkv,
            use_packed_prefix_mlp=use_packed_prefix_mlp,
        )
        caches: dict[tuple[Any, ...], _PrefixCudaGraphCache] = getattr(self, "_prefix_cuda_graph_caches", {})
        cache = caches.get(key)
        if cache is not None:
            return cache

        device = prefix_embs.device
        capture_t0 = time.perf_counter() if timing is not None else 0.0
        cache = _PrefixCudaGraphCache(
            graph=torch.cuda.CUDAGraph(),
            static_prefix_embs=torch.empty_like(prefix_embs),
            static_attention_mask=torch.empty_like(attention_mask),
            static_position_ids=torch.empty_like(position_ids),
            static_hidden=torch.empty_like(prefix_embs),
            static_past_key_values=[],
        )
        self._copy_prefix_graph_inputs(
            cache,
            prefix_embs=prefix_embs,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )

        with torch.cuda.device(device):
            warmup_stream = torch.cuda.Stream()
            warmup_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(warmup_stream):
                for _ in range(3):
                    self._run_prefix_forward(
                        prefix_embs=cache.static_prefix_embs,
                        attention_mask=cache.static_attention_mask,
                        position_ids=cache.static_position_ids,
                        torch_compile_prefix=torch_compile_prefix,
                        torch_compile_prefix_fullgraph=torch_compile_prefix_fullgraph,
                        use_packed_prefix_qkv=use_packed_prefix_qkv,
                        use_packed_prefix_mlp=use_packed_prefix_mlp,
                    )
            torch.cuda.current_stream().wait_stream(warmup_stream)

            self._copy_prefix_graph_inputs(
                cache,
                prefix_embs=prefix_embs,
                attention_mask=attention_mask,
                position_ids=position_ids,
            )
            with torch.cuda.graph(cache.graph):
                cache.static_hidden, cache.static_past_key_values = self._run_prefix_forward(
                    prefix_embs=cache.static_prefix_embs,
                    attention_mask=cache.static_attention_mask,
                    position_ids=cache.static_position_ids,
                    torch_compile_prefix=torch_compile_prefix,
                    torch_compile_prefix_fullgraph=torch_compile_prefix_fullgraph,
                    use_packed_prefix_qkv=use_packed_prefix_qkv,
                    use_packed_prefix_mlp=use_packed_prefix_mlp,
                )

        caches[key] = cache
        self._prefix_cuda_graph_caches = caches
        if timing is not None:
            _sync_for_timing(device)
            timing["prefix_cuda_graph_capture_ms"] = (
                timing.get("prefix_cuda_graph_capture_ms", 0.0) + (time.perf_counter() - capture_t0) * 1000.0
            )
        return cache

    def _run_prefix_cuda_graph(
        self,
        *,
        prefix_embs: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        torch_compile_prefix: bool,
        torch_compile_prefix_fullgraph: bool,
        use_packed_prefix_qkv: bool,
        use_packed_prefix_mlp: bool,
        timing: dict[str, Any] | None = None,
    ):
        cache = self._get_or_create_prefix_cuda_graph(
            prefix_embs=prefix_embs,
            attention_mask=attention_mask,
            position_ids=position_ids,
            torch_compile_prefix=torch_compile_prefix,
            torch_compile_prefix_fullgraph=torch_compile_prefix_fullgraph,
            use_packed_prefix_qkv=use_packed_prefix_qkv,
            use_packed_prefix_mlp=use_packed_prefix_mlp,
            timing=timing,
        )
        with _TimingBlock(timing, "prefix_cuda_graph_copy_ms", prefix_embs.device):
            self._copy_prefix_graph_inputs(
                cache,
                prefix_embs=prefix_embs,
                attention_mask=attention_mask,
                position_ids=position_ids,
            )
        with _TimingBlock(timing, "prefix_cuda_graph_replay_ms", prefix_embs.device):
            cache.graph.replay()
        return cache.static_hidden, cache.static_past_key_values

    def embed_prefix(
        self,
        images: list[torch.Tensor],
        image_masks: list[torch.Tensor],
        tokens: torch.Tensor,
        masks: torch.Tensor,
        timing: dict[str, Any] | None = None,
        cuda_graph_image_embed: bool = False,
        torch_compile_image_embed: bool = False,
        torch_compile_image_embed_fullgraph: bool = False,
        use_realtime_image_embed_cache: bool = False,
        realtime_image_embed_cache_keys: list[tuple[Any, ...]] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        embs = []
        pad_masks = []
        att_masks = []
        device = tokens.device

        for img_idx, (img, img_mask) in enumerate(zip(images, image_masks)):
            with _TimingBlock(timing, f"image_embed_{img_idx}_ms", device):
                cache_key = (
                    realtime_image_embed_cache_keys[img_idx]
                    if (
                        use_realtime_image_embed_cache
                        and realtime_image_embed_cache_keys is not None
                        and img_idx < len(realtime_image_embed_cache_keys)
                    )
                    else None
                )
                cached_img_emb = None if cache_key is None else self._get_realtime_image_embed_cache_entry(cache_key)
                if cached_img_emb is not None:
                    img_emb = cached_img_emb.image_emb
                    if timing is not None:
                        timing["realtime_image_embed_cache_hits"] = timing.get("realtime_image_embed_cache_hits", 0) + 1
                else:
                    img_emb = self._run_image_embed(
                        img,
                        cuda_graph_image_embed=cuda_graph_image_embed,
                        torch_compile_image_embed=torch_compile_image_embed,
                        torch_compile_image_embed_fullgraph=torch_compile_image_embed_fullgraph,
                        timing=timing,
                    )
                    if cache_key is not None:
                        self._store_realtime_image_embed_cache_entry(
                            cache_key=cache_key,
                            image_emb=img_emb,
                        )
                        if timing is not None:
                            timing["realtime_image_embed_cache_misses"] = (
                                timing.get("realtime_image_embed_cache_misses", 0) + 1
                            )
            bsize, num_img_embs = img_emb.shape[:2]
            embs.append(img_emb)
            pad_masks.append(img_mask[:, None].expand(bsize, num_img_embs))
            att_masks += [0] * num_img_embs

        with _TimingBlock(timing, "language_embed_ms", device):
            lang_emb = self.paligemma_with_expert.embed_language_tokens(tokens)
        embs.append(lang_emb)
        pad_masks.append(masks)
        att_masks += [0] * lang_emb.shape[1]

        with _TimingBlock(timing, "prefix_concat_ms", device):
            embs = torch.cat(embs, dim=1)
            pad_masks = torch.cat(pad_masks, dim=1)
            att_masks = torch.tensor(att_masks, dtype=torch.bool, device=embs.device)
            att_masks = att_masks[None, :].expand(pad_masks.shape[0], -1)
        return embs, pad_masks, att_masks

    def embed_suffix(
        self,
        noisy_actions: torch.Tensor,
        timestep: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        time_cond = self.embed_timestep(timestep)
        action_emb = self.embed_action_tokens(noisy_actions, time_cond)
        bsize, action_len = action_emb.shape[:2]
        pad_masks = torch.ones(bsize, action_len, dtype=torch.bool, device=action_emb.device)
        att_masks = self.build_suffix_attention_mask(bsize, action_emb.device, action_emb.dtype)
        return action_emb, pad_masks, att_masks, time_cond

    def embed_timestep(self, timestep: torch.Tensor) -> torch.Tensor:
        model_dtype = self.action_in_proj.weight.dtype
        time_emb = create_sinusoidal_pos_embedding(
            timestep,
            self.action_in_proj.out_features,
            min_period=getattr(self.config, "min_period", 4e-3),
            max_period=getattr(self.config, "max_period", 4.0),
            device=timestep.device,
        ).to(dtype=model_dtype)
        time_cond = self.time_mlp_in(time_emb)
        time_cond = F.silu(time_cond)
        time_cond = self.time_mlp_out(time_cond)
        return F.silu(time_cond)

    def embed_action_tokens(self, noisy_actions: torch.Tensor, time_cond: torch.Tensor) -> torch.Tensor:
        del time_cond
        model_dtype = self.action_in_proj.weight.dtype
        noisy_actions = noisy_actions.to(dtype=model_dtype)
        return self.action_in_proj(noisy_actions)

    def build_suffix_attention_mask(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        att_masks = torch.tensor(
            [1] + [0] * (self.action_horizon - 1),
            dtype=dtype,
            device=device,
        )
        return att_masks[None, :].expand(batch_size, -1)

    def _get_or_create_denoise_schedule(
        self,
        *,
        num_steps: int,
        batch_size: int,
        device: torch.device,
        precompute_adarms: bool,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], list[_DenoiseAdarmsModulations] | None]:
        key = (
            int(num_steps),
            int(batch_size),
            str(device),
            self.action_in_proj.weight.dtype,
            bool(precompute_adarms),
            getattr(self.config, "min_period", 4e-3),
            getattr(self.config, "max_period", 4.0),
        )
        caches = getattr(self, "_denoise_schedule_caches", {})
        cached = caches.get(key)
        if cached is not None:
            return cached

        time_tensors: list[torch.Tensor] = []
        time_conds: list[torch.Tensor] = []
        adarms_modulations: list[_DenoiseAdarmsModulations] | None = [] if precompute_adarms else None
        dt = -1.0 / num_steps
        for step in range(num_steps):
            time_val = 1.0 + step * dt
            time_tensor = torch.full((batch_size,), time_val, dtype=torch.float32, device=device)
            time_cond = self.embed_timestep(time_tensor)
            time_tensors.append(time_tensor)
            time_conds.append(time_cond)
            if adarms_modulations is not None:
                adarms_modulations.append(self.paligemma_with_expert.precompute_suffix_adarms(time_cond))

        cached = (time_tensors, time_conds, adarms_modulations)
        caches[key] = cached
        self._denoise_schedule_caches = caches
        return cached

    def prepare_denoise_static_context(
        self,
        prefix_pad_masks: torch.Tensor,
        num_steps: int,
        batch_size: int,
        device: torch.device,
        *,
        timing: dict[str, Any] | None = None,
        profile_nvtx: bool = False,
        precompute_adarms: bool = False,
        precompute_rope: bool = False,
    ) -> _DenoiseStaticContext:
        with _ProfileRange("pi05.denoise.static_context", profile_nvtx):
            suffix_pad_masks = torch.ones(
                batch_size,
                self.action_horizon,
                dtype=torch.bool,
                device=device,
            )
            suffix_att_masks = self.build_suffix_attention_mask(
                batch_size,
                device,
                self.action_in_proj.weight.dtype,
            )

            with _TimingBlock(timing, "denoise_static_mask_ms", device):
                suffix_len = suffix_pad_masks.shape[1]
                prefix_len = prefix_pad_masks.shape[1]
                prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)
                suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
                full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)

                prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
                position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1
                attention_mask = prepare_attention_masks_4d(full_att_2d_masks)

            with _TimingBlock(timing, "denoise_static_time_cond_ms", device):
                time_tensors, time_conds, adarms_modulations = self._get_or_create_denoise_schedule(
                    num_steps=num_steps,
                    batch_size=batch_size,
                    device=device,
                    precompute_adarms=precompute_adarms,
                )
            rope_cos_sin = None
            if precompute_rope:
                with _TimingBlock(timing, "denoise_static_rope_ms", device):
                    rope_input = torch.empty(
                        batch_size,
                        self.action_horizon,
                        self.expert_width,
                        dtype=self.action_in_proj.weight.dtype,
                        device=device,
                    )
                    rope_cos_sin = self.paligemma_with_expert.gemma_expert.model.rotary_emb(
                        rope_input,
                        position_ids,
                    )

        return _DenoiseStaticContext(
            attention_mask=attention_mask,
            position_ids=position_ids,
            suffix_pad_masks=suffix_pad_masks,
            suffix_att_masks=suffix_att_masks,
            time_tensors=time_tensors,
            time_conds=time_conds,
            adarms_modulations=adarms_modulations,
            rope_cos_sin=rope_cos_sin,
        )

    def denoise_step(
        self,
        prefix_pad_masks: torch.Tensor,
        past_key_values,
        x_t: torch.Tensor,
        timestep: torch.Tensor,
        timing: dict[str, Any] | None = None,
        step_index: int | None = None,
        static_context: _DenoiseStaticContext | None = None,
        direct_suffix: bool = False,
        torch_compile_suffix: bool = False,
        torch_compile_suffix_fullgraph: bool = False,
        profile_nvtx: bool = False,
        use_packed_qkv: bool = False,
        use_packed_mlp: bool = False,
        use_triton_qkv_rope: bool = False,
        use_no_cat_suffix_attn: bool = False,
        suffix_attn_kv_buffers: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
        use_triton_final_head: bool = False,
        use_triton_final_head_euler: bool = False,
        euler_dt: float | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, bool]:
        device = x_t.device
        step_timing: dict[str, Any] | None = {} if timing is not None else None
        with _ProfileRange("pi05.denoise.step", profile_nvtx):
            with _TimingBlock(step_timing, "suffix_embed_ms", device):
                if static_context is not None and step_index is not None:
                    adarms_cond = static_context.time_conds[step_index]
                    suffix_embs = self.embed_action_tokens(x_t, adarms_cond)
                    attention_mask = static_context.attention_mask
                    position_ids = static_context.position_ids
                    adarms_modulations = (
                        None
                        if static_context.adarms_modulations is None
                        else static_context.adarms_modulations[step_index]
                    )
                    rope_cos_sin = static_context.rope_cos_sin
                else:
                    suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(x_t, timestep)
                    adarms_modulations = None
                    rope_cos_sin = None

            if static_context is None or step_index is None:
                with _TimingBlock(step_timing, "suffix_mask_ms", device):
                    suffix_len = suffix_pad_masks.shape[1]
                    batch_size = prefix_pad_masks.shape[0]
                    prefix_len = prefix_pad_masks.shape[1]
                    prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)
                    suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
                    full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)

                    prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
                    position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1
                    attention_mask = prepare_attention_masks_4d(full_att_2d_masks)

            with _TimingBlock(step_timing, "suffix_forward_ms", device):
                skip_final_norm = False
                if direct_suffix:
                    suffix_forward = self._suffix_forward_callable(
                        torch_compile_suffix,
                        torch_compile_fullgraph=torch_compile_suffix_fullgraph,
                    )
                    skip_final_norm = bool(
                        use_triton_final_head
                        and adarms_modulations is not None
                        and adarms_modulations.final_modulation is not None
                    )
                    suffix_hidden = suffix_forward(
                        suffix_embs=suffix_embs,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        past_key_values=past_key_values,
                        adarms_cond=adarms_cond,
                        adarms_modulations=adarms_modulations,
                        profile_nvtx=profile_nvtx,
                        use_packed_qkv=use_packed_qkv,
                        use_packed_mlp=use_packed_mlp,
                        use_triton_qkv_rope=use_triton_qkv_rope,
                        use_no_cat_suffix_attn=use_no_cat_suffix_attn,
                        suffix_attn_kv_buffers=suffix_attn_kv_buffers,
                        rope_cos_sin=rope_cos_sin,
                        skip_final_norm=skip_final_norm,
                    )
                else:
                    outputs_embeds, _ = self.paligemma_with_expert.forward(
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        past_key_values=past_key_values,
                        inputs_embeds=[None, suffix_embs],
                        use_cache=False,
                        adarms_cond=[None, adarms_cond],
                    )
                    suffix_hidden = outputs_embeds[1]

            with _TimingBlock(step_timing, "action_out_proj_ms", device):
                suffix_out = suffix_hidden[:, -self.action_horizon :]
                out = None
                euler_applied = False
                if (
                    direct_suffix
                    and use_triton_final_head_euler
                    and use_triton_final_head
                    and adarms_modulations is not None
                    and euler_dt is not None
                ):
                    out = _triton_final_head_euler(
                        suffix_out,
                        adarms_modulations.final_modulation,
                        self.paligemma_with_expert.gemma_expert.model.norm,
                        self.action_out_proj,
                        x_t,
                        euler_dt,
                    )
                    euler_applied = out is not None
                if direct_suffix and use_triton_final_head and adarms_modulations is not None:
                    if out is None:
                        out = _triton_final_head(
                            suffix_out,
                            adarms_modulations.final_modulation,
                            self.paligemma_with_expert.gemma_expert.model.norm,
                            self.action_out_proj,
                        )
                if out is None:
                    if skip_final_norm:
                        final_norm = self.paligemma_with_expert.gemma_expert.model.norm
                        suffix_out, _ = layernorm_forward(
                            final_norm,
                            suffix_out,
                            adarms_cond,
                            modulation=adarms_modulations.final_modulation,
                        )
                    suffix_out = suffix_out.to(dtype=self.action_out_proj.weight.dtype)
                    out = self.action_out_proj(suffix_out)

        if timing is not None and step_timing is not None:
            timing.setdefault("denoise_steps", []).append({"step": step_index, **step_timing})
            for key, value in step_timing.items():
                timing[f"denoise_{key}"] = timing.get(f"denoise_{key}", 0.0) + value
        if use_triton_final_head_euler:
            return out, euler_applied
        return out

    def _run_denoise_loop(
        self,
        *,
        prefix_pad_masks: torch.Tensor,
        past_key_values,
        x_t: torch.Tensor,
        num_steps: int,
        static_context: _DenoiseStaticContext | None,
        direct_suffix: bool,
        torch_compile_suffix: bool = False,
        torch_compile_suffix_fullgraph: bool = False,
        torch_compile_denoise_loop: bool = False,
        timing: dict[str, Any] | None = None,
        profile_nvtx: bool = False,
        use_packed_qkv: bool = False,
        use_packed_mlp: bool = False,
        use_triton_qkv_rope: bool = False,
        use_no_cat_suffix_attn: bool = False,
        use_triton_final_head: bool = False,
        use_triton_final_head_euler: bool = False,
    ) -> torch.Tensor:
        if torch_compile_denoise_loop:
            denoise_loop = self._denoise_loop_callable(True)
            return denoise_loop(
                prefix_pad_masks=prefix_pad_masks,
                past_key_values=past_key_values,
                x_t=x_t,
                num_steps=num_steps,
                static_context=static_context,
                direct_suffix=direct_suffix,
                torch_compile_suffix=torch_compile_suffix,
                torch_compile_suffix_fullgraph=torch_compile_suffix_fullgraph,
                torch_compile_denoise_loop=False,
                timing=timing,
                profile_nvtx=profile_nvtx,
                use_packed_qkv=use_packed_qkv,
                use_packed_mlp=use_packed_mlp,
                use_triton_qkv_rope=use_triton_qkv_rope,
                use_no_cat_suffix_attn=use_no_cat_suffix_attn,
                use_triton_final_head=use_triton_final_head,
                use_triton_final_head_euler=use_triton_final_head_euler,
            )

        dt = -1.0 / num_steps
        bsize = x_t.shape[0]
        device = x_t.device
        suffix_attn_kv_buffers = None
        if use_no_cat_suffix_attn and direct_suffix:
            suffix_len = x_t.shape[1]
            suffix_attn_kv_buffers = []
            for k_prefix, v_prefix in past_key_values:
                full_shape = (
                    k_prefix.shape[0],
                    k_prefix.shape[1],
                    k_prefix.shape[2] + suffix_len,
                    k_prefix.shape[3],
                )
                k_full = torch.empty(full_shape, device=k_prefix.device, dtype=k_prefix.dtype)
                v_full = torch.empty(full_shape, device=v_prefix.device, dtype=v_prefix.dtype)
                k_full[:, :, : k_prefix.shape[2], :].copy_(k_prefix)
                v_full[:, :, : v_prefix.shape[2], :].copy_(v_prefix)
                suffix_attn_kv_buffers.append((k_full, v_full))
        for step in range(num_steps):
            if static_context is None:
                with _TimingBlock(timing, "time_tensor_ms", device):
                    time_val = 1.0 + step * dt
                    time_tensor = torch.full((bsize,), time_val, dtype=torch.float32, device=device)
            else:
                time_tensor = static_context.time_tensors[step]
            step_out = self.denoise_step(
                prefix_pad_masks=prefix_pad_masks,
                past_key_values=past_key_values,
                x_t=x_t,
                timestep=time_tensor,
                timing=timing,
                step_index=step,
                static_context=static_context,
                direct_suffix=direct_suffix,
                torch_compile_suffix=torch_compile_suffix,
                torch_compile_suffix_fullgraph=torch_compile_suffix_fullgraph,
                profile_nvtx=profile_nvtx,
                use_packed_qkv=use_packed_qkv,
                use_packed_mlp=use_packed_mlp,
                use_triton_qkv_rope=use_triton_qkv_rope,
                use_no_cat_suffix_attn=use_no_cat_suffix_attn,
                suffix_attn_kv_buffers=suffix_attn_kv_buffers,
                use_triton_final_head=use_triton_final_head,
                use_triton_final_head_euler=use_triton_final_head_euler,
                euler_dt=dt,
            )
            with _TimingBlock(timing, "euler_update_ms", device):
                if use_triton_final_head_euler:
                    v_t, euler_applied = step_out
                    x_t = v_t if euler_applied else x_t + dt * v_t
                else:
                    x_t = x_t + dt * step_out
        return x_t

    def _get_or_create_realtime_triton_decoder(self):
        decoder = getattr(self, "_pi05_realtime_triton_decoder", None)
        if decoder is not None:
            return decoder
        from vllm_omni.diffusion.models.pi05.realtime_triton import (
            Pi05RealtimeTritonDecoder,
        )
        from vllm_omni.diffusion.models.pi05.realtime_triton import (
            is_available as realtime_triton_available,
        )

        if not realtime_triton_available():
            raise RuntimeError("use_realtime_triton_decoder requires CUDA and Triton")
        decoder = Pi05RealtimeTritonDecoder(self)
        self._pi05_realtime_triton_decoder = decoder
        return decoder

    def _get_or_create_realtime_triton_prefix_encoder(self):
        encoder = getattr(self, "_pi05_realtime_triton_prefix_encoder", None)
        if encoder is not None:
            return encoder
        from vllm_omni.diffusion.models.pi05.realtime_triton import (
            Pi05RealtimePrefixEncoder,
        )
        from vllm_omni.diffusion.models.pi05.realtime_triton import (
            is_available as realtime_triton_available,
        )

        if not realtime_triton_available():
            raise RuntimeError("use_realtime_triton_prefix_encoder requires CUDA and Triton")
        encoder = Pi05RealtimePrefixEncoder(self)
        self._pi05_realtime_triton_prefix_encoder = encoder
        return encoder

    def _get_or_create_realtime_triton_executor(self):
        executor = getattr(self, "_pi05_realtime_triton_executor", None)
        if executor is not None:
            return executor
        from vllm_omni.diffusion.models.pi05.realtime_triton import (
            Pi05RealtimeExecutor,
        )

        executor = Pi05RealtimeExecutor(
            self._get_or_create_realtime_triton_prefix_encoder(),
            self._get_or_create_realtime_triton_decoder(),
        )
        self._pi05_realtime_triton_executor = executor
        return executor

    def _get_realtime_contiguous_prefix_position_ids(
        self,
        *,
        prefix_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        caches: dict[tuple[str, int], torch.Tensor] = getattr(
            self,
            "_realtime_contiguous_prefix_position_id_caches",
            {},
        )
        key = (str(device), int(prefix_len))
        cached = caches.get(key)
        if cached is not None:
            return cached
        cached = torch.arange(prefix_len, dtype=torch.long, device=device).unsqueeze(0)
        caches[key] = cached
        self._realtime_contiguous_prefix_position_id_caches = caches
        return cached

    def _run_realtime_triton_prefix_encoder(
        self,
        *,
        prefix_embs: torch.Tensor,
        prefix_pad_masks: torch.Tensor,
        prefix_position_ids: torch.Tensor | None = None,
        valid_prefix_len: int | None = None,
        output_k: torch.Tensor | None = None,
        output_v: torch.Tensor | None = None,
        timing: dict[str, Any] | None = None,
    ):
        with _TimingBlock(timing, "realtime_triton_prefix_encoder_ms", prefix_embs.device):
            encoder = self._get_or_create_realtime_triton_prefix_encoder()
            return encoder(
                prefix_embs=prefix_embs,
                prefix_pad_masks=prefix_pad_masks,
                prefix_position_ids=prefix_position_ids,
                valid_prefix_len=valid_prefix_len,
                output_k=output_k,
                output_v=output_v,
            )

    def _run_realtime_triton_decoder(
        self,
        *,
        prefix_pad_masks: torch.Tensor,
        past_key_values,
        x_t: torch.Tensor,
        static_context: _DenoiseStaticContext,
        num_steps: int,
        valid_prefix_len: int | None = None,
        decoder_buffers: Any | None = None,
        timing: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        if static_context.adarms_modulations is None:
            raise RuntimeError("use_realtime_triton_decoder requires precompute_adarms=True")
        with _TimingBlock(timing, "realtime_triton_decoder_ms", x_t.device):
            decoder = self._get_or_create_realtime_triton_decoder()
            return decoder(
                prefix_kv=past_key_values,
                prefix_pad_masks=prefix_pad_masks,
                valid_prefix_len=_per_sample_valid_prefix_len(prefix_pad_masks, valid_prefix_len),
                x_t=x_t,
                adarms_modulations=static_context.adarms_modulations,
                num_steps=num_steps,
                decoder_buffers=decoder_buffers,
            )

    def _get_or_create_realtime_prefix_kv_cache_entry(
        self,
        *,
        cache_key: tuple[Any, ...],
        prefix_embs: torch.Tensor,
        prefix_pad_masks: torch.Tensor,
        prefix_position_ids: torch.Tensor,
        valid_prefix_len: int,
    ) -> tuple[_RealtimePrefixKvCacheEntry, bool]:
        caches: dict[tuple[Any, ...], _RealtimePrefixKvCacheEntry] = getattr(self, "_realtime_prefix_kv_caches", {})
        cached = caches.get(cache_key)
        if cached is not None:
            return cached, True

        max_entries = int(os.environ.get("PI05_REALTIME_PREFIX_KV_CACHE_MAX_ENTRIES", "8"))
        if max_entries <= 0:
            raise RuntimeError("PI05_REALTIME_PREFIX_KV_CACHE_MAX_ENTRIES must be positive")
        if len(caches) >= max_entries:
            caches.clear()
            self._realtime_cached_decoder_cuda_graph_caches = {}

        executor = self._get_or_create_realtime_triton_executor()
        prefix_len = int(prefix_embs.shape[1])
        decoder_buffers = executor.decoder.prepare_prefix_buffers(
            prefix_len=prefix_len,
            valid_prefix_len=int(valid_prefix_len),
            dtype=prefix_embs.dtype,
            prefix_pad_masks=prefix_pad_masks,
        )
        executor.prefix_encoder(
            prefix_embs=prefix_embs,
            prefix_pad_masks=prefix_pad_masks,
            prefix_position_ids=prefix_position_ids,
            valid_prefix_len=int(valid_prefix_len),
            output_k=decoder_buffers.k[:, :prefix_len],
            output_v=decoder_buffers.v[:, :prefix_len],
            return_prefix_kv=False,
        )
        cached = _RealtimePrefixKvCacheEntry(
            decoder_buffers=decoder_buffers,
            prefix_len=prefix_len,
            valid_prefix_len=int(valid_prefix_len),
            prefix_pad_masks=prefix_pad_masks.detach().clone(),
        )
        caches[cache_key] = cached
        self._realtime_prefix_kv_caches = caches
        return cached, False

    def has_realtime_prefix_kv_cache(self, cache_key: tuple[Any, ...]) -> bool:
        caches: dict[tuple[Any, ...], _RealtimePrefixKvCacheEntry] = getattr(self, "_realtime_prefix_kv_caches", {})
        return cache_key in caches

    def clear_realtime_prefix_kv_cache(self) -> None:
        self._realtime_prefix_kv_caches = {}
        self._realtime_cached_decoder_cuda_graph_caches = {}

    def has_realtime_prefix_emb_cache(self, cache_key: tuple[Any, ...]) -> bool:
        caches: dict[tuple[Any, ...], _RealtimePrefixEmbCacheEntry] = getattr(
            self,
            "_realtime_prefix_emb_caches",
            {},
        )
        return cache_key in caches

    def clear_realtime_prefix_emb_cache(self) -> None:
        self._realtime_prefix_emb_caches = {}

    def clear_realtime_prefix_caches(self) -> None:
        self.clear_realtime_prefix_kv_cache()
        self.clear_realtime_prefix_emb_cache()
        self.clear_realtime_image_embed_cache()

    def clear_realtime_image_embed_cache(self) -> None:
        self._realtime_image_embed_caches = {}

    def _get_realtime_image_embed_cache_entry(
        self,
        cache_key: tuple[Any, ...],
    ) -> _RealtimeImageEmbedCacheEntry | None:
        caches: dict[tuple[Any, ...], _RealtimeImageEmbedCacheEntry] = getattr(
            self,
            "_realtime_image_embed_caches",
            {},
        )
        return caches.get(cache_key)

    def _store_realtime_image_embed_cache_entry(
        self,
        *,
        cache_key: tuple[Any, ...],
        image_emb: torch.Tensor,
    ) -> None:
        caches: dict[tuple[Any, ...], _RealtimeImageEmbedCacheEntry] = getattr(
            self,
            "_realtime_image_embed_caches",
            {},
        )
        if cache_key in caches:
            return
        max_entries = int(os.environ.get("PI05_REALTIME_IMAGE_EMB_CACHE_MAX_ENTRIES", "8"))
        if max_entries <= 0:
            raise RuntimeError("PI05_REALTIME_IMAGE_EMB_CACHE_MAX_ENTRIES must be positive")
        if len(caches) >= max_entries:
            caches.clear()
        caches[cache_key] = _RealtimeImageEmbedCacheEntry(image_emb=image_emb.detach().clone())
        self._realtime_image_embed_caches = caches

    def _get_realtime_prefix_emb_cache_entry(
        self,
        cache_key: tuple[Any, ...],
    ) -> _RealtimePrefixEmbCacheEntry:
        caches: dict[tuple[Any, ...], _RealtimePrefixEmbCacheEntry] = getattr(
            self,
            "_realtime_prefix_emb_caches",
            {},
        )
        cached = caches.get(cache_key)
        if cached is None:
            raise KeyError("realtime prefix embedding cache entry not found")
        return cached

    def _store_realtime_prefix_emb_cache_entry(
        self,
        *,
        cache_key: tuple[Any, ...],
        prefix_embs: torch.Tensor,
        prefix_pad_masks: torch.Tensor,
        prefix_position_ids: torch.Tensor,
        prefix_attention_mask: torch.Tensor,
        valid_prefix_len: int,
    ) -> None:
        caches: dict[tuple[Any, ...], _RealtimePrefixEmbCacheEntry] = getattr(
            self,
            "_realtime_prefix_emb_caches",
            {},
        )
        if cache_key in caches:
            return
        max_entries = int(os.environ.get("PI05_REALTIME_PREFIX_EMB_CACHE_MAX_ENTRIES", "8"))
        if max_entries <= 0:
            raise RuntimeError("PI05_REALTIME_PREFIX_EMB_CACHE_MAX_ENTRIES must be positive")
        if len(caches) >= max_entries:
            caches.clear()
        caches[cache_key] = _RealtimePrefixEmbCacheEntry(
            prefix_embs=prefix_embs.detach().clone(),
            prefix_pad_masks=prefix_pad_masks.detach().clone(),
            prefix_position_ids=prefix_position_ids.detach().clone(),
            prefix_attention_mask=prefix_attention_mask.detach().clone(),
            valid_prefix_len=int(valid_prefix_len),
        )
        self._realtime_prefix_emb_caches = caches

    def _get_realtime_prefix_kv_cache_entry(
        self,
        cache_key: tuple[Any, ...],
    ) -> _RealtimePrefixKvCacheEntry:
        caches: dict[tuple[Any, ...], _RealtimePrefixKvCacheEntry] = getattr(self, "_realtime_prefix_kv_caches", {})
        cached = caches.get(cache_key)
        if cached is None:
            raise KeyError("realtime prefix KV cache entry not found")
        return cached

    def _get_or_create_realtime_cached_decoder_cuda_graph(
        self,
        *,
        cache_entry: _RealtimePrefixKvCacheEntry,
        x_t: torch.Tensor,
        num_steps: int,
        static_context: _DenoiseStaticContext,
        timing: dict[str, Any] | None = None,
    ) -> _RealtimeCachedDecoderCudaGraphCache:
        if static_context.adarms_modulations is None:
            raise RuntimeError("realtime prefix KV cache requires precompute_adarms=True")
        key = (
            int(num_steps),
            str(x_t.device),
            x_t.dtype,
            tuple(x_t.shape),
            cache_entry.prefix_len,
            cache_entry.valid_prefix_len,
            id(cache_entry.decoder_buffers),
            static_context.adarms_modulations is not None,
        )
        caches: dict[tuple[Any, ...], _RealtimeCachedDecoderCudaGraphCache] = getattr(
            self, "_realtime_cached_decoder_cuda_graph_caches", {}
        )
        cached = caches.get(key)
        if cached is not None:
            return cached

        device = x_t.device
        capture_t0 = time.perf_counter() if timing is not None else 0.0
        static_context_for_graph = _DenoiseStaticContext(
            attention_mask=torch.empty_like(static_context.attention_mask),
            position_ids=torch.empty_like(static_context.position_ids),
            suffix_pad_masks=torch.empty_like(static_context.suffix_pad_masks),
            suffix_att_masks=torch.empty_like(static_context.suffix_att_masks),
            time_tensors=[torch.empty_like(tensor) for tensor in static_context.time_tensors],
            time_conds=[torch.empty_like(tensor) for tensor in static_context.time_conds],
            adarms_modulations=self._empty_adarms_modulations_like(static_context.adarms_modulations),
            rope_cos_sin=(
                None
                if static_context.rope_cos_sin is None
                else (
                    torch.empty_like(static_context.rope_cos_sin[0]),
                    torch.empty_like(static_context.rope_cos_sin[1]),
                )
            ),
        )
        cached = _RealtimeCachedDecoderCudaGraphCache(
            graph=torch.cuda.CUDAGraph(),
            static_x=torch.empty_like(x_t),
            static_context=static_context_for_graph,
            static_output=torch.empty_like(x_t),
        )
        cached.static_x.copy_(x_t)
        self._copy_static_context(cached.static_context, static_context, copy_time=True)
        decoder = self._get_or_create_realtime_triton_decoder()

        def _run_decoder_body():
            return decoder(
                prefix_kv=None,
                prefix_pad_masks=cache_entry.prefix_pad_masks,
                valid_prefix_len=cache_entry.valid_prefix_len,
                x_t=cached.static_x,
                adarms_modulations=cached.static_context.adarms_modulations,
                num_steps=num_steps,
                decoder_buffers=cache_entry.decoder_buffers,
                prefix_len=cache_entry.prefix_len,
            )

        with torch.cuda.device(device):
            warmup_stream = torch.cuda.Stream()
            warmup_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(warmup_stream):
                for _ in range(3):
                    cached.static_output = _run_decoder_body()
            torch.cuda.current_stream().wait_stream(warmup_stream)

            cached.static_x.copy_(x_t)
            self._copy_static_context(cached.static_context, static_context, copy_time=True)
            with torch.cuda.graph(cached.graph):
                cached.static_output = _run_decoder_body()

        caches[key] = cached
        self._realtime_cached_decoder_cuda_graph_caches = caches
        if timing is not None:
            _sync_for_timing(device)
            timing["realtime_prefix_kv_decoder_cuda_graph_capture_ms"] = (
                timing.get("realtime_prefix_kv_decoder_cuda_graph_capture_ms", 0.0)
                + (time.perf_counter() - capture_t0) * 1000.0
            )
        return cached

    def _run_realtime_prefix_kv_cached_decoder_cuda_graph(
        self,
        *,
        prefix_cache_key: tuple[Any, ...],
        prefix_embs: torch.Tensor,
        prefix_pad_masks: torch.Tensor,
        prefix_position_ids: torch.Tensor,
        valid_prefix_len: int,
        x_t: torch.Tensor,
        num_steps: int,
        static_context: _DenoiseStaticContext,
        timing: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        cache_entry, hit = self._get_or_create_realtime_prefix_kv_cache_entry(
            cache_key=prefix_cache_key,
            prefix_embs=prefix_embs,
            prefix_pad_masks=prefix_pad_masks,
            prefix_position_ids=prefix_position_ids,
            valid_prefix_len=valid_prefix_len,
        )
        if timing is not None:
            timing["realtime_prefix_kv_cache_hit"] = bool(hit)
        graph_cache = self._get_or_create_realtime_cached_decoder_cuda_graph(
            cache_entry=cache_entry,
            x_t=x_t,
            num_steps=num_steps,
            static_context=static_context,
            timing=timing,
        )
        with _TimingBlock(timing, "realtime_prefix_kv_decoder_cuda_graph_copy_ms", x_t.device):
            graph_cache.static_x.copy_(x_t)
            self._copy_static_context(graph_cache.static_context, static_context, copy_time=False)
        with _TimingBlock(timing, "realtime_prefix_kv_decoder_cuda_graph_replay_ms", x_t.device):
            graph_cache.graph.replay()
        return graph_cache.static_output

    def _run_realtime_prefix_kv_cache_hit_decoder_cuda_graph(
        self,
        *,
        prefix_cache_key: tuple[Any, ...],
        x_t: torch.Tensor,
        num_steps: int,
        static_context: _DenoiseStaticContext,
        timing: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        cache_entry = self._get_realtime_prefix_kv_cache_entry(prefix_cache_key)
        if timing is not None:
            timing["realtime_prefix_kv_cache_hit"] = True
        graph_cache = self._get_or_create_realtime_cached_decoder_cuda_graph(
            cache_entry=cache_entry,
            x_t=x_t,
            num_steps=num_steps,
            static_context=static_context,
            timing=timing,
        )
        with _TimingBlock(timing, "realtime_prefix_kv_decoder_cuda_graph_copy_ms", x_t.device):
            graph_cache.static_x.copy_(x_t)
            self._copy_static_context(graph_cache.static_context, static_context, copy_time=False)
        with _TimingBlock(timing, "realtime_prefix_kv_decoder_cuda_graph_replay_ms", x_t.device):
            graph_cache.graph.replay()
        return graph_cache.static_output

    @staticmethod
    def _copy_static_context(
        dst: _DenoiseStaticContext,
        src: _DenoiseStaticContext,
        *,
        copy_time: bool = True,
    ) -> None:
        dst.attention_mask.copy_(src.attention_mask)
        dst.position_ids.copy_(src.position_ids)
        dst.suffix_pad_masks.copy_(src.suffix_pad_masks)
        dst.suffix_att_masks.copy_(src.suffix_att_masks)
        if copy_time:
            for dst_tensor, src_tensor in zip(dst.time_tensors, src.time_tensors):
                dst_tensor.copy_(src_tensor)
            for dst_tensor, src_tensor in zip(dst.time_conds, src.time_conds):
                dst_tensor.copy_(src_tensor)
            if dst.adarms_modulations is not None and src.adarms_modulations is not None:
                for dst_step, src_step in zip(dst.adarms_modulations, src.adarms_modulations):
                    for (dst_in, dst_post), (src_in, src_post) in zip(
                        dst_step.layer_modulations,
                        src_step.layer_modulations,
                    ):
                        if dst_in is not None and src_in is not None:
                            dst_in.copy_(src_in)
                        if dst_post is not None and src_post is not None:
                            dst_post.copy_(src_post)
                    if dst_step.final_modulation is not None and src_step.final_modulation is not None:
                        dst_step.final_modulation.copy_(src_step.final_modulation)
        if dst.rope_cos_sin is not None and src.rope_cos_sin is not None:
            dst.rope_cos_sin[0].copy_(src.rope_cos_sin[0])
            dst.rope_cos_sin[1].copy_(src.rope_cos_sin[1])

    @staticmethod
    def _empty_adarms_modulations_like(
        adarms_modulations: list[_DenoiseAdarmsModulations] | None,
    ) -> list[_DenoiseAdarmsModulations] | None:
        if adarms_modulations is None:
            return None
        empty_modulations: list[_DenoiseAdarmsModulations] = []
        for step_mod in adarms_modulations:
            empty_modulations.append(
                _DenoiseAdarmsModulations(
                    layer_modulations=[
                        (
                            None if input_mod is None else torch.empty_like(input_mod),
                            None if post_mod is None else torch.empty_like(post_mod),
                        )
                        for input_mod, post_mod in step_mod.layer_modulations
                    ],
                    final_modulation=(
                        None if step_mod.final_modulation is None else torch.empty_like(step_mod.final_modulation)
                    ),
                )
            )
        return empty_modulations

    def _copy_denoise_graph_inputs(
        self,
        cache: _DenoiseCudaGraphCache,
        *,
        prefix_pad_masks: torch.Tensor,
        past_key_values,
        x_t: torch.Tensor,
        static_context: _DenoiseStaticContext,
        copy_time: bool = True,
    ) -> None:
        cache.static_x.copy_(x_t)
        cache.static_prefix_pad_masks.copy_(prefix_pad_masks)
        for (static_k, static_v), (current_k, current_v) in zip(cache.static_past_key_values, past_key_values):
            static_k.copy_(current_k)
            static_v.copy_(current_v)
        self._copy_static_context(cache.static_context, static_context, copy_time=copy_time)

    @staticmethod
    def _denoise_cuda_graph_key(
        *,
        num_steps: int,
        prefix_pad_masks: torch.Tensor,
        past_key_values,
        x_t: torch.Tensor,
        static_context: _DenoiseStaticContext,
        torch_compile_suffix: bool,
        torch_compile_suffix_fullgraph: bool,
        torch_compile_denoise_loop: bool,
        use_packed_qkv: bool,
        use_packed_mlp: bool,
        use_triton_qkv_rope: bool,
        use_no_cat_suffix_attn: bool,
        use_triton_final_head: bool,
        use_triton_final_head_euler: bool,
    ) -> tuple[Any, ...]:
        return (
            int(num_steps),
            bool(torch_compile_suffix),
            bool(torch_compile_suffix_fullgraph),
            bool(torch_compile_denoise_loop),
            bool(use_packed_qkv),
            bool(use_packed_mlp),
            bool(use_triton_qkv_rope),
            bool(use_no_cat_suffix_attn),
            bool(use_triton_final_head),
            bool(use_triton_final_head_euler),
            static_context.adarms_modulations is not None,
            static_context.rope_cos_sin is not None,
            str(x_t.device),
            x_t.dtype,
            tuple(x_t.shape),
            tuple(prefix_pad_masks.shape),
            tuple(static_context.attention_mask.shape),
            static_context.attention_mask.dtype,
            tuple(static_context.position_ids.shape),
            static_context.position_ids.dtype,
            tuple((tuple(k.shape), k.dtype, tuple(v.shape), v.dtype) for k, v in past_key_values),
        )

    def _get_or_create_denoise_cuda_graph(
        self,
        *,
        prefix_pad_masks: torch.Tensor,
        past_key_values,
        x_t: torch.Tensor,
        num_steps: int,
        static_context: _DenoiseStaticContext,
        torch_compile_suffix: bool,
        torch_compile_suffix_fullgraph: bool,
        torch_compile_denoise_loop: bool,
        use_packed_qkv: bool,
        use_packed_mlp: bool,
        use_triton_qkv_rope: bool,
        use_no_cat_suffix_attn: bool,
        use_triton_final_head: bool,
        use_triton_final_head_euler: bool,
        timing: dict[str, Any] | None = None,
    ) -> _DenoiseCudaGraphCache:
        if x_t.device.type != "cuda":
            raise RuntimeError("cuda_graph_denoise requires CUDA tensors")

        key = self._denoise_cuda_graph_key(
            num_steps=num_steps,
            prefix_pad_masks=prefix_pad_masks,
            past_key_values=past_key_values,
            x_t=x_t,
            static_context=static_context,
            torch_compile_suffix=torch_compile_suffix,
            torch_compile_suffix_fullgraph=torch_compile_suffix_fullgraph,
            torch_compile_denoise_loop=torch_compile_denoise_loop,
            use_packed_qkv=use_packed_qkv,
            use_packed_mlp=use_packed_mlp,
            use_triton_qkv_rope=use_triton_qkv_rope,
            use_no_cat_suffix_attn=use_no_cat_suffix_attn,
            use_triton_final_head=use_triton_final_head,
            use_triton_final_head_euler=use_triton_final_head_euler,
        )
        caches: dict[tuple[Any, ...], _DenoiseCudaGraphCache] = getattr(self, "_denoise_cuda_graph_caches", {})
        cache = caches.get(key)
        if cache is not None:
            return cache

        device = x_t.device
        capture_t0 = time.perf_counter() if timing is not None else 0.0
        static_context_for_graph = _DenoiseStaticContext(
            attention_mask=torch.empty_like(static_context.attention_mask),
            position_ids=torch.empty_like(static_context.position_ids),
            suffix_pad_masks=torch.empty_like(static_context.suffix_pad_masks),
            suffix_att_masks=torch.empty_like(static_context.suffix_att_masks),
            time_tensors=[torch.empty_like(tensor) for tensor in static_context.time_tensors],
            time_conds=[torch.empty_like(tensor) for tensor in static_context.time_conds],
            adarms_modulations=self._empty_adarms_modulations_like(static_context.adarms_modulations),
            rope_cos_sin=(
                None
                if static_context.rope_cos_sin is None
                else (
                    torch.empty_like(static_context.rope_cos_sin[0]),
                    torch.empty_like(static_context.rope_cos_sin[1]),
                )
            ),
        )
        cache = _DenoiseCudaGraphCache(
            graph=torch.cuda.CUDAGraph(),
            static_x=torch.empty_like(x_t),
            static_prefix_pad_masks=torch.empty_like(prefix_pad_masks),
            static_past_key_values=[(torch.empty_like(k), torch.empty_like(v)) for k, v in past_key_values],
            static_context=static_context_for_graph,
            static_output=torch.empty_like(x_t),
        )
        self._copy_denoise_graph_inputs(
            cache,
            prefix_pad_masks=prefix_pad_masks,
            past_key_values=past_key_values,
            x_t=x_t,
            static_context=static_context,
        )

        with torch.cuda.device(device):
            warmup_stream = torch.cuda.Stream()
            warmup_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(warmup_stream):
                for _ in range(3):
                    cache.static_output = self._run_denoise_loop(
                        prefix_pad_masks=cache.static_prefix_pad_masks,
                        past_key_values=cache.static_past_key_values,
                        x_t=cache.static_x,
                        num_steps=num_steps,
                        static_context=cache.static_context,
                        direct_suffix=True,
                        torch_compile_suffix=torch_compile_suffix,
                        torch_compile_suffix_fullgraph=torch_compile_suffix_fullgraph,
                        torch_compile_denoise_loop=torch_compile_denoise_loop,
                        use_packed_qkv=use_packed_qkv,
                        use_packed_mlp=use_packed_mlp,
                        use_triton_qkv_rope=use_triton_qkv_rope,
                        use_no_cat_suffix_attn=use_no_cat_suffix_attn,
                        use_triton_final_head=use_triton_final_head,
                        use_triton_final_head_euler=use_triton_final_head_euler,
                    )
            torch.cuda.current_stream().wait_stream(warmup_stream)

            self._copy_denoise_graph_inputs(
                cache,
                prefix_pad_masks=prefix_pad_masks,
                past_key_values=past_key_values,
                x_t=x_t,
                static_context=static_context,
                copy_time=True,
            )
            with torch.cuda.graph(cache.graph):
                cache.static_output = self._run_denoise_loop(
                    prefix_pad_masks=cache.static_prefix_pad_masks,
                    past_key_values=cache.static_past_key_values,
                    x_t=cache.static_x,
                    num_steps=num_steps,
                    static_context=cache.static_context,
                    direct_suffix=True,
                    torch_compile_suffix=torch_compile_suffix,
                    torch_compile_suffix_fullgraph=torch_compile_suffix_fullgraph,
                    torch_compile_denoise_loop=torch_compile_denoise_loop,
                    use_packed_qkv=use_packed_qkv,
                    use_packed_mlp=use_packed_mlp,
                    use_triton_qkv_rope=use_triton_qkv_rope,
                    use_no_cat_suffix_attn=use_no_cat_suffix_attn,
                    use_triton_final_head=use_triton_final_head,
                    use_triton_final_head_euler=use_triton_final_head_euler,
                )

        caches[key] = cache
        self._denoise_cuda_graph_caches = caches
        if timing is not None:
            _sync_for_timing(device)
            timing["denoise_cuda_graph_capture_ms"] = (
                timing.get("denoise_cuda_graph_capture_ms", 0.0) + (time.perf_counter() - capture_t0) * 1000.0
            )
        return cache

    def _run_denoise_cuda_graph(
        self,
        *,
        prefix_pad_masks: torch.Tensor,
        past_key_values,
        x_t: torch.Tensor,
        num_steps: int,
        static_context: _DenoiseStaticContext,
        torch_compile_suffix: bool,
        torch_compile_suffix_fullgraph: bool,
        torch_compile_denoise_loop: bool,
        use_packed_qkv: bool,
        use_packed_mlp: bool,
        use_triton_qkv_rope: bool,
        use_no_cat_suffix_attn: bool,
        use_triton_final_head: bool,
        use_triton_final_head_euler: bool,
        timing: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        cache = self._get_or_create_denoise_cuda_graph(
            prefix_pad_masks=prefix_pad_masks,
            past_key_values=past_key_values,
            x_t=x_t,
            num_steps=num_steps,
            static_context=static_context,
            torch_compile_suffix=torch_compile_suffix,
            torch_compile_suffix_fullgraph=torch_compile_suffix_fullgraph,
            torch_compile_denoise_loop=torch_compile_denoise_loop,
            use_packed_qkv=use_packed_qkv,
            use_packed_mlp=use_packed_mlp,
            use_triton_qkv_rope=use_triton_qkv_rope,
            use_no_cat_suffix_attn=use_no_cat_suffix_attn,
            use_triton_final_head=use_triton_final_head,
            use_triton_final_head_euler=use_triton_final_head_euler,
            timing=timing,
        )
        with _TimingBlock(timing, "denoise_cuda_graph_copy_ms", x_t.device):
            self._copy_denoise_graph_inputs(
                cache,
                prefix_pad_masks=prefix_pad_masks,
                past_key_values=past_key_values,
                x_t=x_t,
                static_context=static_context,
                copy_time=False,
            )
        with _TimingBlock(timing, "denoise_cuda_graph_replay_ms", x_t.device):
            cache.graph.replay()
        return cache.static_output

    @staticmethod
    def _prefix_denoise_cuda_graph_key(
        *,
        prefix_embs: torch.Tensor,
        prefix_attention_mask: torch.Tensor,
        prefix_position_ids: torch.Tensor,
        prefix_pad_masks: torch.Tensor,
        x_t: torch.Tensor,
        num_steps: int,
        static_context: _DenoiseStaticContext,
        torch_compile_prefix: bool,
        torch_compile_suffix: bool,
        torch_compile_prefix_fullgraph: bool,
        torch_compile_suffix_fullgraph: bool,
        torch_compile_denoise_loop: bool,
        use_packed_prefix_qkv: bool,
        use_packed_prefix_mlp: bool,
        use_packed_qkv: bool,
        use_packed_mlp: bool,
        use_triton_qkv_rope: bool,
        use_no_cat_suffix_attn: bool,
        use_triton_final_head: bool,
        use_triton_final_head_euler: bool,
    ) -> tuple[Any, ...]:
        return (
            int(num_steps),
            bool(torch_compile_prefix),
            bool(torch_compile_suffix),
            bool(torch_compile_prefix_fullgraph),
            bool(torch_compile_suffix_fullgraph),
            bool(torch_compile_denoise_loop),
            bool(use_packed_prefix_qkv),
            bool(use_packed_prefix_mlp),
            bool(use_packed_qkv),
            bool(use_packed_mlp),
            bool(use_triton_qkv_rope),
            bool(use_no_cat_suffix_attn),
            bool(use_triton_final_head),
            bool(use_triton_final_head_euler),
            static_context.adarms_modulations is not None,
            static_context.rope_cos_sin is not None,
            str(x_t.device),
            x_t.dtype,
            tuple(x_t.shape),
            prefix_embs.dtype,
            tuple(prefix_embs.shape),
            prefix_attention_mask.dtype,
            tuple(prefix_attention_mask.shape),
            prefix_position_ids.dtype,
            tuple(prefix_position_ids.shape),
            tuple(prefix_pad_masks.shape),
            tuple(static_context.attention_mask.shape),
            static_context.attention_mask.dtype,
        )

    @staticmethod
    def _copy_prefix_denoise_graph_inputs(
        cache: _PrefixDenoiseCudaGraphCache,
        *,
        prefix_embs: torch.Tensor,
        prefix_attention_mask: torch.Tensor,
        prefix_position_ids: torch.Tensor,
        prefix_pad_masks: torch.Tensor,
        x_t: torch.Tensor,
    ) -> None:
        cache.static_prefix_embs.copy_(prefix_embs)
        cache.static_prefix_attention_mask.copy_(prefix_attention_mask)
        cache.static_prefix_position_ids.copy_(prefix_position_ids)
        cache.static_prefix_pad_masks.copy_(prefix_pad_masks)
        cache.static_x.copy_(x_t)

    def _get_or_create_prefix_denoise_cuda_graph(
        self,
        *,
        prefix_embs: torch.Tensor,
        prefix_attention_mask: torch.Tensor,
        prefix_position_ids: torch.Tensor,
        prefix_pad_masks: torch.Tensor,
        x_t: torch.Tensor,
        num_steps: int,
        static_context: _DenoiseStaticContext,
        torch_compile_prefix: bool,
        torch_compile_suffix: bool,
        torch_compile_prefix_fullgraph: bool,
        torch_compile_suffix_fullgraph: bool,
        torch_compile_denoise_loop: bool,
        use_packed_prefix_qkv: bool,
        use_packed_prefix_mlp: bool,
        use_packed_qkv: bool,
        use_packed_mlp: bool,
        use_triton_qkv_rope: bool,
        use_no_cat_suffix_attn: bool,
        use_triton_final_head: bool,
        use_triton_final_head_euler: bool,
        timing: dict[str, Any] | None = None,
    ) -> _PrefixDenoiseCudaGraphCache:
        if x_t.device.type != "cuda":
            raise RuntimeError("cuda_graph_prefix_denoise requires CUDA tensors")

        key = self._prefix_denoise_cuda_graph_key(
            prefix_embs=prefix_embs,
            prefix_attention_mask=prefix_attention_mask,
            prefix_position_ids=prefix_position_ids,
            prefix_pad_masks=prefix_pad_masks,
            x_t=x_t,
            num_steps=num_steps,
            static_context=static_context,
            torch_compile_prefix=torch_compile_prefix,
            torch_compile_suffix=torch_compile_suffix,
            torch_compile_prefix_fullgraph=torch_compile_prefix_fullgraph,
            torch_compile_suffix_fullgraph=torch_compile_suffix_fullgraph,
            torch_compile_denoise_loop=torch_compile_denoise_loop,
            use_packed_prefix_qkv=use_packed_prefix_qkv,
            use_packed_prefix_mlp=use_packed_prefix_mlp,
            use_packed_qkv=use_packed_qkv,
            use_packed_mlp=use_packed_mlp,
            use_triton_qkv_rope=use_triton_qkv_rope,
            use_no_cat_suffix_attn=use_no_cat_suffix_attn,
            use_triton_final_head=use_triton_final_head,
            use_triton_final_head_euler=use_triton_final_head_euler,
        )
        caches: dict[tuple[Any, ...], _PrefixDenoiseCudaGraphCache] = getattr(
            self, "_prefix_denoise_cuda_graph_caches", {}
        )
        cache = caches.get(key)
        if cache is not None:
            return cache

        device = x_t.device
        capture_t0 = time.perf_counter() if timing is not None else 0.0
        static_context_for_graph = _DenoiseStaticContext(
            attention_mask=torch.empty_like(static_context.attention_mask),
            position_ids=torch.empty_like(static_context.position_ids),
            suffix_pad_masks=torch.empty_like(static_context.suffix_pad_masks),
            suffix_att_masks=torch.empty_like(static_context.suffix_att_masks),
            time_tensors=[torch.empty_like(tensor) for tensor in static_context.time_tensors],
            time_conds=[torch.empty_like(tensor) for tensor in static_context.time_conds],
            adarms_modulations=self._empty_adarms_modulations_like(static_context.adarms_modulations),
            rope_cos_sin=(
                None
                if static_context.rope_cos_sin is None
                else (
                    torch.empty_like(static_context.rope_cos_sin[0]),
                    torch.empty_like(static_context.rope_cos_sin[1]),
                )
            ),
        )
        cache = _PrefixDenoiseCudaGraphCache(
            graph=torch.cuda.CUDAGraph(),
            static_prefix_embs=torch.empty_like(prefix_embs),
            static_prefix_attention_mask=torch.empty_like(prefix_attention_mask),
            static_prefix_position_ids=torch.empty_like(prefix_position_ids),
            static_prefix_pad_masks=torch.empty_like(prefix_pad_masks),
            static_x=torch.empty_like(x_t),
            static_context=static_context_for_graph,
            static_output=torch.empty_like(x_t),
        )
        self._copy_prefix_denoise_graph_inputs(
            cache,
            prefix_embs=prefix_embs,
            prefix_attention_mask=prefix_attention_mask,
            prefix_position_ids=prefix_position_ids,
            prefix_pad_masks=prefix_pad_masks,
            x_t=x_t,
        )
        self._copy_static_context(cache.static_context, static_context, copy_time=True)

        def _run_graph_body():
            _, past_key_values = self._run_prefix_forward(
                prefix_embs=cache.static_prefix_embs,
                attention_mask=cache.static_prefix_attention_mask,
                position_ids=cache.static_prefix_position_ids,
                torch_compile_prefix=torch_compile_prefix,
                torch_compile_prefix_fullgraph=torch_compile_prefix_fullgraph,
                use_packed_prefix_qkv=use_packed_prefix_qkv,
                use_packed_prefix_mlp=use_packed_prefix_mlp,
            )
            return self._run_denoise_loop(
                prefix_pad_masks=cache.static_prefix_pad_masks,
                past_key_values=past_key_values,
                x_t=cache.static_x,
                num_steps=num_steps,
                static_context=cache.static_context,
                direct_suffix=True,
                torch_compile_suffix=torch_compile_suffix,
                torch_compile_suffix_fullgraph=torch_compile_suffix_fullgraph,
                torch_compile_denoise_loop=torch_compile_denoise_loop,
                use_packed_qkv=use_packed_qkv,
                use_packed_mlp=use_packed_mlp,
                use_triton_qkv_rope=use_triton_qkv_rope,
                use_no_cat_suffix_attn=use_no_cat_suffix_attn,
                use_triton_final_head=use_triton_final_head,
                use_triton_final_head_euler=use_triton_final_head_euler,
            )

        with torch.cuda.device(device):
            warmup_stream = torch.cuda.Stream()
            warmup_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(warmup_stream):
                for _ in range(3):
                    cache.static_output = _run_graph_body()
            torch.cuda.current_stream().wait_stream(warmup_stream)

            self._copy_prefix_denoise_graph_inputs(
                cache,
                prefix_embs=prefix_embs,
                prefix_attention_mask=prefix_attention_mask,
                prefix_position_ids=prefix_position_ids,
                prefix_pad_masks=prefix_pad_masks,
                x_t=x_t,
            )
            self._copy_static_context(cache.static_context, static_context, copy_time=True)
            with torch.cuda.graph(cache.graph):
                cache.static_output = _run_graph_body()

        caches[key] = cache
        self._prefix_denoise_cuda_graph_caches = caches
        if timing is not None:
            _sync_for_timing(device)
            timing["prefix_denoise_cuda_graph_capture_ms"] = (
                timing.get("prefix_denoise_cuda_graph_capture_ms", 0.0) + (time.perf_counter() - capture_t0) * 1000.0
            )
        return cache

    def _run_prefix_denoise_cuda_graph(
        self,
        *,
        prefix_embs: torch.Tensor,
        prefix_attention_mask: torch.Tensor,
        prefix_position_ids: torch.Tensor,
        prefix_pad_masks: torch.Tensor,
        x_t: torch.Tensor,
        num_steps: int,
        static_context: _DenoiseStaticContext,
        torch_compile_prefix: bool,
        torch_compile_suffix: bool,
        torch_compile_prefix_fullgraph: bool,
        torch_compile_suffix_fullgraph: bool,
        torch_compile_denoise_loop: bool,
        use_packed_prefix_qkv: bool,
        use_packed_prefix_mlp: bool,
        use_packed_qkv: bool,
        use_packed_mlp: bool,
        use_triton_qkv_rope: bool,
        use_no_cat_suffix_attn: bool,
        use_triton_final_head: bool,
        use_triton_final_head_euler: bool,
        timing: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        cache = self._get_or_create_prefix_denoise_cuda_graph(
            prefix_embs=prefix_embs,
            prefix_attention_mask=prefix_attention_mask,
            prefix_position_ids=prefix_position_ids,
            prefix_pad_masks=prefix_pad_masks,
            x_t=x_t,
            num_steps=num_steps,
            static_context=static_context,
            torch_compile_prefix=torch_compile_prefix,
            torch_compile_suffix=torch_compile_suffix,
            torch_compile_prefix_fullgraph=torch_compile_prefix_fullgraph,
            torch_compile_suffix_fullgraph=torch_compile_suffix_fullgraph,
            torch_compile_denoise_loop=torch_compile_denoise_loop,
            use_packed_prefix_qkv=use_packed_prefix_qkv,
            use_packed_prefix_mlp=use_packed_prefix_mlp,
            use_packed_qkv=use_packed_qkv,
            use_packed_mlp=use_packed_mlp,
            use_triton_qkv_rope=use_triton_qkv_rope,
            use_no_cat_suffix_attn=use_no_cat_suffix_attn,
            use_triton_final_head=use_triton_final_head,
            use_triton_final_head_euler=use_triton_final_head_euler,
            timing=timing,
        )
        with _TimingBlock(timing, "prefix_denoise_cuda_graph_copy_ms", x_t.device):
            self._copy_prefix_denoise_graph_inputs(
                cache,
                prefix_embs=prefix_embs,
                prefix_attention_mask=prefix_attention_mask,
                prefix_position_ids=prefix_position_ids,
                prefix_pad_masks=prefix_pad_masks,
                x_t=x_t,
            )
            self._copy_static_context(cache.static_context, static_context, copy_time=False)
        with _TimingBlock(timing, "prefix_denoise_cuda_graph_replay_ms", x_t.device):
            cache.graph.replay()
        return cache.static_output

    @staticmethod
    def _prefix_realtime_triton_cuda_graph_key(
        *,
        prefix_embs: torch.Tensor,
        prefix_attention_mask: torch.Tensor,
        prefix_position_ids: torch.Tensor,
        prefix_pad_masks: torch.Tensor,
        valid_prefix_len: int | None,
        x_t: torch.Tensor,
        num_steps: int,
        static_context: _DenoiseStaticContext,
        torch_compile_prefix: bool,
        torch_compile_prefix_fullgraph: bool,
        use_packed_prefix_qkv: bool,
        use_packed_prefix_mlp: bool,
        use_realtime_triton_prefix_encoder: bool,
    ) -> tuple[Any, ...]:
        return (
            int(num_steps),
            bool(torch_compile_prefix),
            bool(torch_compile_prefix_fullgraph),
            bool(use_packed_prefix_qkv),
            bool(use_packed_prefix_mlp),
            bool(use_realtime_triton_prefix_encoder),
            static_context.adarms_modulations is not None,
            str(x_t.device),
            x_t.dtype,
            tuple(x_t.shape),
            prefix_embs.dtype,
            tuple(prefix_embs.shape),
            prefix_attention_mask.dtype,
            tuple(prefix_attention_mask.shape),
            prefix_position_ids.dtype,
            tuple(prefix_position_ids.shape),
            tuple(prefix_pad_masks.shape),
            # Per sample, not summed: two different splits can share a total.
            tuple(_per_sample_valid_prefix_len(prefix_pad_masks, valid_prefix_len).tolist()),
        )

    def _get_or_create_prefix_realtime_triton_cuda_graph(
        self,
        *,
        prefix_embs: torch.Tensor,
        prefix_attention_mask: torch.Tensor,
        prefix_position_ids: torch.Tensor,
        prefix_pad_masks: torch.Tensor,
        valid_prefix_len: int | None,
        x_t: torch.Tensor,
        num_steps: int,
        static_context: _DenoiseStaticContext,
        torch_compile_prefix: bool,
        torch_compile_prefix_fullgraph: bool,
        use_packed_prefix_qkv: bool,
        use_packed_prefix_mlp: bool,
        use_realtime_triton_prefix_encoder: bool,
        timing: dict[str, Any] | None = None,
    ) -> _PrefixDenoiseCudaGraphCache:
        if x_t.device.type != "cuda":
            raise RuntimeError("cuda_graph_prefix_denoise realtime backend requires CUDA tensors")
        if static_context.adarms_modulations is None:
            raise RuntimeError("realtime Triton graph requires precompute_adarms=True")

        key = self._prefix_realtime_triton_cuda_graph_key(
            prefix_embs=prefix_embs,
            prefix_attention_mask=prefix_attention_mask,
            prefix_position_ids=prefix_position_ids,
            prefix_pad_masks=prefix_pad_masks,
            valid_prefix_len=valid_prefix_len,
            x_t=x_t,
            num_steps=num_steps,
            static_context=static_context,
            torch_compile_prefix=torch_compile_prefix,
            torch_compile_prefix_fullgraph=torch_compile_prefix_fullgraph,
            use_packed_prefix_qkv=use_packed_prefix_qkv,
            use_packed_prefix_mlp=use_packed_prefix_mlp,
            use_realtime_triton_prefix_encoder=use_realtime_triton_prefix_encoder,
        )
        caches: dict[tuple[Any, ...], _PrefixDenoiseCudaGraphCache] = getattr(
            self, "_prefix_realtime_triton_cuda_graph_caches", {}
        )
        cache = caches.get(key)
        if cache is not None:
            return cache

        device = x_t.device
        capture_t0 = time.perf_counter() if timing is not None else 0.0
        static_context_for_graph = _DenoiseStaticContext(
            attention_mask=torch.empty_like(static_context.attention_mask),
            position_ids=torch.empty_like(static_context.position_ids),
            suffix_pad_masks=torch.empty_like(static_context.suffix_pad_masks),
            suffix_att_masks=torch.empty_like(static_context.suffix_att_masks),
            time_tensors=[torch.empty_like(tensor) for tensor in static_context.time_tensors],
            time_conds=[torch.empty_like(tensor) for tensor in static_context.time_conds],
            adarms_modulations=self._empty_adarms_modulations_like(static_context.adarms_modulations),
            rope_cos_sin=(
                None
                if static_context.rope_cos_sin is None
                else (
                    torch.empty_like(static_context.rope_cos_sin[0]),
                    torch.empty_like(static_context.rope_cos_sin[1]),
                )
            ),
        )
        cache = _PrefixDenoiseCudaGraphCache(
            graph=torch.cuda.CUDAGraph(),
            static_prefix_embs=torch.empty_like(prefix_embs),
            static_prefix_attention_mask=torch.empty_like(prefix_attention_mask),
            static_prefix_position_ids=torch.empty_like(prefix_position_ids),
            static_prefix_pad_masks=torch.empty_like(prefix_pad_masks),
            static_x=torch.empty_like(x_t),
            static_context=static_context_for_graph,
            static_output=torch.empty_like(x_t),
        )
        self._copy_prefix_denoise_graph_inputs(
            cache,
            prefix_embs=prefix_embs,
            prefix_attention_mask=prefix_attention_mask,
            prefix_position_ids=prefix_position_ids,
            prefix_pad_masks=prefix_pad_masks,
            x_t=x_t,
        )
        self._copy_static_context(cache.static_context, static_context, copy_time=True)
        valid_prefix_len = _per_sample_valid_prefix_len(prefix_pad_masks, valid_prefix_len)

        def _run_graph_body():
            if use_realtime_triton_prefix_encoder:
                executor = self._get_or_create_realtime_triton_executor()
                return executor(
                    prefix_embs=cache.static_prefix_embs,
                    prefix_pad_masks=cache.static_prefix_pad_masks,
                    prefix_position_ids=cache.static_prefix_position_ids,
                    valid_prefix_len=valid_prefix_len,
                    x_t=cache.static_x,
                    adarms_modulations=cache.static_context.adarms_modulations,
                    num_steps=num_steps,
                )
            _, past_key_values = self._run_prefix_forward(
                prefix_embs=cache.static_prefix_embs,
                attention_mask=cache.static_prefix_attention_mask,
                position_ids=cache.static_prefix_position_ids,
                torch_compile_prefix=torch_compile_prefix,
                torch_compile_prefix_fullgraph=torch_compile_prefix_fullgraph,
                use_packed_prefix_qkv=use_packed_prefix_qkv,
                use_packed_prefix_mlp=use_packed_prefix_mlp,
            )
            return self._run_realtime_triton_decoder(
                prefix_pad_masks=cache.static_prefix_pad_masks,
                past_key_values=past_key_values,
                x_t=cache.static_x,
                static_context=cache.static_context,
                num_steps=num_steps,
                valid_prefix_len=valid_prefix_len,
            )

        with torch.cuda.device(device):
            warmup_stream = torch.cuda.Stream()
            warmup_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(warmup_stream):
                for _ in range(3):
                    cache.static_output = _run_graph_body()
            torch.cuda.current_stream().wait_stream(warmup_stream)

            self._copy_prefix_denoise_graph_inputs(
                cache,
                prefix_embs=prefix_embs,
                prefix_attention_mask=prefix_attention_mask,
                prefix_position_ids=prefix_position_ids,
                prefix_pad_masks=prefix_pad_masks,
                x_t=x_t,
            )
            self._copy_static_context(cache.static_context, static_context, copy_time=True)
            with torch.cuda.graph(cache.graph):
                cache.static_output = _run_graph_body()

        caches[key] = cache
        self._prefix_realtime_triton_cuda_graph_caches = caches
        if timing is not None:
            _sync_for_timing(device)
            timing["prefix_realtime_triton_cuda_graph_capture_ms"] = (
                timing.get("prefix_realtime_triton_cuda_graph_capture_ms", 0.0)
                + (time.perf_counter() - capture_t0) * 1000.0
            )
            timing["prefix_realtime_triton_valid_prefix_len"] = valid_prefix_len
            timing["use_realtime_triton_prefix_encoder"] = bool(use_realtime_triton_prefix_encoder)
        return cache

    def _run_prefix_realtime_triton_cuda_graph(
        self,
        *,
        prefix_embs: torch.Tensor,
        prefix_attention_mask: torch.Tensor,
        prefix_position_ids: torch.Tensor,
        prefix_pad_masks: torch.Tensor,
        valid_prefix_len: int | None,
        x_t: torch.Tensor,
        num_steps: int,
        static_context: _DenoiseStaticContext,
        torch_compile_prefix: bool,
        torch_compile_prefix_fullgraph: bool,
        use_packed_prefix_qkv: bool,
        use_packed_prefix_mlp: bool,
        use_realtime_triton_prefix_encoder: bool,
        timing: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        cache = self._get_or_create_prefix_realtime_triton_cuda_graph(
            prefix_embs=prefix_embs,
            prefix_attention_mask=prefix_attention_mask,
            prefix_position_ids=prefix_position_ids,
            prefix_pad_masks=prefix_pad_masks,
            valid_prefix_len=valid_prefix_len,
            x_t=x_t,
            num_steps=num_steps,
            static_context=static_context,
            torch_compile_prefix=torch_compile_prefix,
            torch_compile_prefix_fullgraph=torch_compile_prefix_fullgraph,
            use_packed_prefix_qkv=use_packed_prefix_qkv,
            use_packed_prefix_mlp=use_packed_prefix_mlp,
            use_realtime_triton_prefix_encoder=use_realtime_triton_prefix_encoder,
            timing=timing,
        )
        with _TimingBlock(timing, "prefix_realtime_triton_cuda_graph_copy_ms", x_t.device):
            self._copy_prefix_denoise_graph_inputs(
                cache,
                prefix_embs=prefix_embs,
                prefix_attention_mask=prefix_attention_mask,
                prefix_position_ids=prefix_position_ids,
                prefix_pad_masks=prefix_pad_masks,
                x_t=x_t,
            )
            self._copy_static_context(cache.static_context, static_context, copy_time=False)
        with _TimingBlock(timing, "prefix_realtime_triton_cuda_graph_replay_ms", x_t.device):
            cache.graph.replay()
        return cache.static_output

    @torch.no_grad()
    def sample_actions_from_realtime_prefix_kv_cache(
        self,
        *,
        realtime_prefix_cache_key: tuple[Any, ...],
        noise: torch.Tensor | None = None,
        num_steps: int | None = None,
        timing: dict[str, Any] | None = None,
        profile_nvtx: bool = False,
        precompute_rope: bool = False,
    ) -> torch.Tensor:
        if num_steps is None:
            num_steps = self.num_inference_steps
        cache_entry = self._get_realtime_prefix_kv_cache_entry(realtime_prefix_cache_key)
        device = cache_entry.prefix_pad_masks.device
        if timing is not None:
            timing["num_inference_steps"] = int(num_steps)
            timing["model_dtype"] = str(self.action_in_proj.weight.dtype)
            timing["use_realtime_triton_decoder"] = True
            timing["use_realtime_triton_prefix_encoder"] = True
            timing["use_realtime_triton_prefix_kv_cache"] = True
            timing["realtime_prefix_kv_cache_hit"] = True
        if noise is None:
            with _TimingBlock(timing, "noise_init_ms", device):
                noise = torch.randn(
                    1,
                    self.action_horizon,
                    self.action_dim,
                    dtype=torch.float32,
                    device=device,
                )
        else:
            noise = noise.to(device=device, dtype=torch.float32)
        with _TimingBlock(timing, "denoise_static_context_ms", device):
            static_context = self.prepare_denoise_static_context(
                prefix_pad_masks=cache_entry.prefix_pad_masks,
                num_steps=int(num_steps),
                batch_size=1,
                device=device,
                timing=timing,
                profile_nvtx=profile_nvtx,
                precompute_adarms=True,
                precompute_rope=precompute_rope,
            )
        with _TimingBlock(timing, "realtime_prefix_kv_cache_hit_loop_ms", device):
            return self._run_realtime_prefix_kv_cache_hit_decoder_cuda_graph(
                prefix_cache_key=realtime_prefix_cache_key,
                x_t=noise,
                num_steps=int(num_steps),
                static_context=static_context,
                timing=timing,
            )

    @torch.no_grad()
    def sample_actions_from_realtime_prefix_emb_cache(
        self,
        *,
        realtime_prefix_cache_key: tuple[Any, ...],
        noise: torch.Tensor | None = None,
        num_steps: int | None = None,
        timing: dict[str, Any] | None = None,
        profile_nvtx: bool = False,
        precompute_rope: bool = False,
    ) -> torch.Tensor:
        if num_steps is None:
            num_steps = self.num_inference_steps
        cache_entry = self._get_realtime_prefix_emb_cache_entry(realtime_prefix_cache_key)
        device = cache_entry.prefix_embs.device
        if timing is not None:
            timing["num_inference_steps"] = int(num_steps)
            timing["model_dtype"] = str(self.action_in_proj.weight.dtype)
            timing["use_realtime_triton_decoder"] = True
            timing["use_realtime_triton_prefix_encoder"] = True
            timing["use_realtime_triton_prefix_emb_cache"] = True
            timing["realtime_prefix_emb_cache_hit"] = True
        if noise is None:
            with _TimingBlock(timing, "noise_init_ms", device):
                noise = torch.randn(
                    1,
                    self.action_horizon,
                    self.action_dim,
                    dtype=torch.float32,
                    device=device,
                )
        else:
            noise = noise.to(device=device, dtype=torch.float32)
        with _TimingBlock(timing, "denoise_static_context_ms", device):
            static_context = self.prepare_denoise_static_context(
                prefix_pad_masks=cache_entry.prefix_pad_masks,
                num_steps=int(num_steps),
                batch_size=1,
                device=device,
                timing=timing,
                profile_nvtx=profile_nvtx,
                precompute_adarms=True,
                precompute_rope=precompute_rope,
            )
        with _TimingBlock(timing, "realtime_prefix_emb_cache_hit_loop_ms", device):
            return self._run_prefix_realtime_triton_cuda_graph(
                prefix_embs=cache_entry.prefix_embs,
                prefix_attention_mask=cache_entry.prefix_attention_mask,
                prefix_position_ids=cache_entry.prefix_position_ids,
                prefix_pad_masks=cache_entry.prefix_pad_masks,
                valid_prefix_len=cache_entry.valid_prefix_len,
                x_t=noise,
                num_steps=int(num_steps),
                static_context=static_context,
                torch_compile_prefix=False,
                torch_compile_prefix_fullgraph=False,
                use_packed_prefix_qkv=False,
                use_packed_prefix_mlp=False,
                use_realtime_triton_prefix_encoder=True,
                timing=timing,
            )

    @torch.no_grad()
    def sample_actions(
        self,
        images: list[torch.Tensor],
        image_masks: list[torch.Tensor],
        tokens: torch.Tensor,
        masks: torch.Tensor,
        noise: torch.Tensor | None = None,
        num_steps: int | None = None,
        timing: dict[str, Any] | None = None,
        direct_suffix: bool = False,
        static_denoise: bool = False,
        cuda_graph_denoise: bool = False,
        cuda_graph_image_embed: bool = False,
        cuda_graph_prefix: bool = False,
        cuda_graph_prefix_denoise: bool = False,
        torch_compile_image_embed: bool = False,
        torch_compile_image_embed_fullgraph: bool = False,
        torch_compile_prefix: bool = False,
        torch_compile_prefix_fullgraph: bool = False,
        torch_compile_suffix: bool = False,
        torch_compile_suffix_fullgraph: bool = False,
        torch_compile_denoise_loop: bool = False,
        use_packed_prefix_qkv: bool = False,
        use_packed_prefix_mlp: bool = False,
        use_packed_qkv: bool = False,
        use_packed_mlp: bool = False,
        use_triton_qkv_rope: bool = False,
        use_no_cat_suffix_attn: bool = False,
        use_triton_final_head: bool = False,
        use_triton_final_head_euler: bool = False,
        use_realtime_triton_decoder: bool = False,
        use_realtime_triton_prefix_encoder: bool = False,
        use_realtime_triton_prefix_emb_cache: bool = False,
        use_realtime_triton_prefix_kv_cache: bool = False,
        use_realtime_image_embed_cache: bool = False,
        realtime_prefix_cache_key: tuple[Any, ...] | None = None,
        realtime_image_embed_cache_keys: list[tuple[Any, ...]] | None = None,
        profile_nvtx: bool = False,
        precompute_adarms: bool = False,
        precompute_rope: bool = False,
        prefix_valid_len: int | None = None,
        prefix_masks_contiguous: bool = False,
    ) -> torch.Tensor:
        if num_steps is None:
            num_steps = self.num_inference_steps

        bsize = tokens.shape[0]
        device = tokens.device
        effective_use_packed_mlp = bool(use_packed_mlp)
        if timing is not None:
            timing["num_inference_steps"] = int(num_steps)
            timing["model_dtype"] = str(self.action_in_proj.weight.dtype)
            timing["cuda_graph_denoise"] = bool(cuda_graph_denoise)
            timing["cuda_graph_image_embed"] = bool(cuda_graph_image_embed)
            timing["cuda_graph_prefix"] = bool(cuda_graph_prefix)
            timing["cuda_graph_prefix_denoise"] = bool(cuda_graph_prefix_denoise)
            timing["torch_compile_image_embed"] = bool(torch_compile_image_embed)
            timing["torch_compile_image_embed_fullgraph"] = bool(torch_compile_image_embed_fullgraph)
            timing["torch_compile_prefix"] = bool(torch_compile_prefix)
            timing["torch_compile_prefix_fullgraph"] = bool(torch_compile_prefix_fullgraph)
            timing["torch_compile_suffix"] = bool(torch_compile_suffix)
            timing["torch_compile_suffix_fullgraph"] = bool(torch_compile_suffix_fullgraph)
            timing["torch_compile_denoise_loop"] = bool(torch_compile_denoise_loop)
            timing["use_packed_prefix_qkv"] = bool(use_packed_prefix_qkv)
            timing["use_packed_prefix_mlp"] = bool(use_packed_prefix_mlp)
            timing["use_packed_qkv"] = bool(use_packed_qkv)
            timing["use_packed_mlp"] = bool(use_packed_mlp)
            timing["effective_use_packed_mlp"] = effective_use_packed_mlp
            timing["use_triton_qkv_rope"] = bool(use_triton_qkv_rope)
            timing["use_no_cat_suffix_attn"] = bool(use_no_cat_suffix_attn)
            timing["use_triton_final_head"] = bool(use_triton_final_head)
            timing["use_triton_final_head_euler"] = bool(use_triton_final_head_euler)
            timing["use_realtime_triton_decoder"] = bool(use_realtime_triton_decoder)
            timing["use_realtime_triton_prefix_encoder"] = bool(use_realtime_triton_prefix_encoder)
            timing["use_realtime_triton_prefix_emb_cache"] = bool(use_realtime_triton_prefix_emb_cache)
            timing["use_realtime_triton_prefix_kv_cache"] = bool(use_realtime_triton_prefix_kv_cache)
            timing["use_realtime_image_embed_cache"] = bool(use_realtime_image_embed_cache)
            timing["precompute_rope"] = bool(precompute_rope)
        if noise is None:
            with _TimingBlock(timing, "noise_init_ms", device):
                noise = torch.randn(
                    bsize,
                    self.action_horizon,
                    self.action_dim,
                    dtype=torch.float32,
                    device=device,
                )

        with _TimingBlock(timing, "embed_prefix_total_ms", device):
            prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
                images,
                image_masks,
                tokens,
                masks,
                timing=timing,
                cuda_graph_image_embed=cuda_graph_image_embed,
                torch_compile_image_embed=torch_compile_image_embed,
                torch_compile_image_embed_fullgraph=torch_compile_image_embed_fullgraph,
                use_realtime_image_embed_cache=use_realtime_image_embed_cache,
                realtime_image_embed_cache_keys=realtime_image_embed_cache_keys,
            )
        compress_prefix_padding = os.environ.get("PI05_REALTIME_COMPRESS_PREFIX_PADDING", "1").lower() not in {
            "0",
            "false",
            "no",
        }
        if compress_prefix_padding and use_realtime_triton_prefix_encoder and bsize == 1:
            with _TimingBlock(timing, "prefix_trim_padding_ms", device):
                original_prefix_len = int(prefix_pad_masks.shape[1])
                if prefix_masks_contiguous and prefix_valid_len is not None:
                    valid_prefix_len_for_trim = int(prefix_valid_len)
                else:
                    valid_prefix_len_for_trim = int(prefix_pad_masks[0].sum().item())
                if 0 < valid_prefix_len_for_trim < prefix_pad_masks.shape[1]:
                    if prefix_masks_contiguous and prefix_valid_len is not None:
                        prefix_embs = prefix_embs[:, :valid_prefix_len_for_trim]
                        prefix_pad_masks = prefix_pad_masks[:, :valid_prefix_len_for_trim]
                        prefix_att_masks = prefix_att_masks[:, :valid_prefix_len_for_trim]
                        prefix_valid_len = valid_prefix_len_for_trim
                    elif not bool(prefix_pad_masks[0, valid_prefix_len_for_trim:].any().item()):
                        prefix_embs = prefix_embs[:, :valid_prefix_len_for_trim]
                        prefix_pad_masks = prefix_pad_masks[:, :valid_prefix_len_for_trim]
                        prefix_att_masks = prefix_att_masks[:, :valid_prefix_len_for_trim]
                        prefix_valid_len = valid_prefix_len_for_trim
                    else:
                        keep_prefix = prefix_pad_masks[0].bool()
                        prefix_embs = prefix_embs[:, keep_prefix].contiguous()
                        prefix_pad_masks = prefix_pad_masks[:, keep_prefix].contiguous()
                        prefix_att_masks = prefix_att_masks[:, keep_prefix].contiguous()
                        prefix_valid_len = int(prefix_pad_masks.shape[1])
                    if timing is not None:
                        timing["prefix_trimmed_tokens"] = int(original_prefix_len - valid_prefix_len_for_trim)
                elif prefix_masks_contiguous and prefix_valid_len is not None:
                    prefix_valid_len = valid_prefix_len_for_trim
        with _TimingBlock(timing, "prefix_mask_ms", device):
            realtime_prefix_graph = (
                cuda_graph_prefix_denoise and use_realtime_triton_decoder and use_realtime_triton_prefix_encoder
            )
            contiguous_realtime_prefix = (
                realtime_prefix_graph
                and prefix_masks_contiguous
                and prefix_valid_len is not None
                and int(prefix_valid_len) == int(prefix_pad_masks.shape[1])
            )
            if contiguous_realtime_prefix:
                prefix_position_ids = self._get_realtime_contiguous_prefix_position_ids(
                    prefix_len=int(prefix_pad_masks.shape[1]),
                    device=device,
                )
                prefix_att_2d_masks_4d = torch.empty(
                    (0,),
                    dtype=self.action_in_proj.weight.dtype,
                    device=device,
                )
            else:
                prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
                prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
                prefix_att_2d_masks_4d = prepare_attention_masks_4d(prefix_att_2d_masks)
            if timing is not None:
                timing["realtime_prefix_fast_position_ids"] = bool(contiguous_realtime_prefix)
        if use_realtime_triton_prefix_emb_cache:
            if not use_realtime_triton_prefix_encoder:
                raise RuntimeError(
                    "use_realtime_triton_prefix_emb_cache requires use_realtime_triton_prefix_encoder=True"
                )
            if realtime_prefix_cache_key is None:
                raise RuntimeError("use_realtime_triton_prefix_emb_cache requires realtime_prefix_cache_key")
            self._store_realtime_prefix_emb_cache_entry(
                cache_key=realtime_prefix_cache_key,
                prefix_embs=prefix_embs,
                prefix_pad_masks=prefix_pad_masks,
                prefix_position_ids=prefix_position_ids,
                prefix_attention_mask=prefix_att_2d_masks_4d,
                valid_prefix_len=(
                    int(prefix_pad_masks[0].sum().item()) if prefix_valid_len is None else int(prefix_valid_len)
                ),
            )
            if timing is not None:
                timing["realtime_prefix_emb_cache_hit"] = False

        x_t = noise
        if cuda_graph_prefix_denoise:
            direct_suffix = True
            static_denoise = True
        if cuda_graph_denoise:
            direct_suffix = True
            static_denoise = True
        if torch_compile_suffix:
            direct_suffix = True
        if torch_compile_denoise_loop:
            direct_suffix = True
            static_denoise = True
        if use_packed_prefix_qkv:
            with _TimingBlock(timing, "packed_prefix_qkv_prepare_ms", device):
                self.paligemma_with_expert.prepare_packed_prefix_qkv()
        if use_packed_prefix_mlp:
            with _TimingBlock(timing, "packed_prefix_mlp_prepare_ms", device):
                self.paligemma_with_expert.prepare_packed_prefix_mlp()
        if use_packed_qkv:
            direct_suffix = True
            with _TimingBlock(timing, "packed_qkv_prepare_ms", device):
                self.paligemma_with_expert.prepare_packed_suffix_qkv()
        if use_triton_qkv_rope:
            direct_suffix = True
            with _TimingBlock(timing, "packed_qkv_prepare_ms", device):
                self.paligemma_with_expert.prepare_packed_suffix_qkv()
        if use_no_cat_suffix_attn:
            direct_suffix = True
        if effective_use_packed_mlp:
            direct_suffix = True
            with _TimingBlock(timing, "packed_mlp_prepare_ms", device):
                self.paligemma_with_expert.prepare_packed_suffix_mlp()
        if use_triton_final_head:
            direct_suffix = True
        if use_triton_final_head_euler:
            direct_suffix = True
        if use_realtime_triton_decoder:
            direct_suffix = True
            static_denoise = True
            precompute_adarms = True
        if use_realtime_triton_prefix_encoder and not use_realtime_triton_decoder:
            raise RuntimeError("use_realtime_triton_prefix_encoder requires use_realtime_triton_decoder=True")
        if use_realtime_triton_prefix_emb_cache and not use_realtime_triton_prefix_encoder:
            raise RuntimeError("use_realtime_triton_prefix_emb_cache requires use_realtime_triton_prefix_encoder=True")
        if use_realtime_triton_prefix_kv_cache:
            if not use_realtime_triton_prefix_encoder:
                raise RuntimeError(
                    "use_realtime_triton_prefix_kv_cache requires use_realtime_triton_prefix_encoder=True"
                )
            if realtime_prefix_cache_key is None:
                raise RuntimeError("use_realtime_triton_prefix_kv_cache requires realtime_prefix_cache_key")
        static_context = (
            self.prepare_denoise_static_context(
                prefix_pad_masks=prefix_pad_masks,
                num_steps=num_steps,
                batch_size=bsize,
                device=device,
                timing=timing,
                profile_nvtx=profile_nvtx,
                precompute_adarms=precompute_adarms,
                precompute_rope=precompute_rope,
            )
            if static_denoise
            else None
        )
        if cuda_graph_prefix_denoise:
            if static_context is None:
                raise RuntimeError("cuda_graph_prefix_denoise requires static denoise context")
            if use_realtime_triton_decoder:
                with _TimingBlock(timing, "prefix_realtime_triton_loop_ms", device):
                    if use_realtime_triton_prefix_kv_cache:
                        if static_context is None:
                            raise RuntimeError("realtime prefix KV cache requires static denoise context")
                        return self._run_realtime_prefix_kv_cached_decoder_cuda_graph(
                            prefix_cache_key=realtime_prefix_cache_key,
                            prefix_embs=prefix_embs,
                            prefix_pad_masks=prefix_pad_masks,
                            prefix_position_ids=prefix_position_ids,
                            valid_prefix_len=(
                                int(prefix_pad_masks[0].sum().item())
                                if prefix_valid_len is None
                                else int(prefix_valid_len)
                            ),
                            x_t=x_t,
                            num_steps=num_steps,
                            static_context=static_context,
                            timing=timing,
                        )
                    return self._run_prefix_realtime_triton_cuda_graph(
                        prefix_embs=prefix_embs,
                        prefix_attention_mask=prefix_att_2d_masks_4d,
                        prefix_position_ids=prefix_position_ids,
                        prefix_pad_masks=prefix_pad_masks,
                        x_t=x_t,
                        num_steps=num_steps,
                        static_context=static_context,
                        torch_compile_prefix=torch_compile_prefix,
                        torch_compile_prefix_fullgraph=torch_compile_prefix_fullgraph,
                        use_packed_prefix_qkv=use_packed_prefix_qkv,
                        use_packed_prefix_mlp=use_packed_prefix_mlp,
                        use_realtime_triton_prefix_encoder=use_realtime_triton_prefix_encoder,
                        valid_prefix_len=prefix_valid_len,
                        timing=timing,
                    )
            with _TimingBlock(timing, "prefix_denoise_loop_ms", device):
                return self._run_prefix_denoise_cuda_graph(
                    prefix_embs=prefix_embs,
                    prefix_attention_mask=prefix_att_2d_masks_4d,
                    prefix_position_ids=prefix_position_ids,
                    prefix_pad_masks=prefix_pad_masks,
                    x_t=x_t,
                    num_steps=num_steps,
                    static_context=static_context,
                    torch_compile_prefix=torch_compile_prefix,
                    torch_compile_suffix=torch_compile_suffix,
                    torch_compile_prefix_fullgraph=torch_compile_prefix_fullgraph,
                    torch_compile_suffix_fullgraph=torch_compile_suffix_fullgraph,
                    torch_compile_denoise_loop=torch_compile_denoise_loop,
                    use_packed_prefix_qkv=use_packed_prefix_qkv,
                    use_packed_prefix_mlp=use_packed_prefix_mlp,
                    use_packed_qkv=use_packed_qkv,
                    use_packed_mlp=effective_use_packed_mlp,
                    use_triton_qkv_rope=use_triton_qkv_rope,
                    use_no_cat_suffix_attn=use_no_cat_suffix_attn,
                    use_triton_final_head=use_triton_final_head,
                    use_triton_final_head_euler=use_triton_final_head_euler,
                    timing=timing,
                )

        with _TimingBlock(timing, "prefix_forward_ms", device):
            if cuda_graph_prefix:
                _, past_key_values = self._run_prefix_cuda_graph(
                    prefix_embs=prefix_embs,
                    attention_mask=prefix_att_2d_masks_4d,
                    position_ids=prefix_position_ids,
                    torch_compile_prefix=torch_compile_prefix,
                    torch_compile_prefix_fullgraph=torch_compile_prefix_fullgraph,
                    use_packed_prefix_qkv=use_packed_prefix_qkv,
                    use_packed_prefix_mlp=use_packed_prefix_mlp,
                    timing=timing,
                )
            else:
                _, past_key_values = self._run_prefix_forward(
                    prefix_embs=prefix_embs,
                    attention_mask=prefix_att_2d_masks_4d,
                    position_ids=prefix_position_ids,
                    torch_compile_prefix=torch_compile_prefix,
                    torch_compile_prefix_fullgraph=torch_compile_prefix_fullgraph,
                    use_packed_prefix_qkv=use_packed_prefix_qkv,
                    use_packed_prefix_mlp=use_packed_prefix_mlp,
                )

        with _TimingBlock(timing, "denoise_loop_ms", device):
            if use_realtime_triton_decoder:
                if static_context is None:
                    raise RuntimeError("use_realtime_triton_decoder requires static denoise context")
                x_t = self._run_realtime_triton_decoder(
                    prefix_pad_masks=prefix_pad_masks,
                    past_key_values=past_key_values,
                    x_t=x_t,
                    static_context=static_context,
                    num_steps=num_steps,
                    timing=timing,
                )
            elif cuda_graph_denoise:
                if static_context is None:
                    raise RuntimeError("cuda_graph_denoise requires static denoise context")
                x_t = self._run_denoise_cuda_graph(
                    prefix_pad_masks=prefix_pad_masks,
                    past_key_values=past_key_values,
                    x_t=x_t,
                    num_steps=num_steps,
                    static_context=static_context,
                    torch_compile_suffix=torch_compile_suffix,
                    torch_compile_suffix_fullgraph=torch_compile_suffix_fullgraph,
                    torch_compile_denoise_loop=torch_compile_denoise_loop,
                    use_packed_qkv=use_packed_qkv,
                    use_packed_mlp=effective_use_packed_mlp,
                    use_triton_qkv_rope=use_triton_qkv_rope,
                    use_no_cat_suffix_attn=use_no_cat_suffix_attn,
                    use_triton_final_head=use_triton_final_head,
                    use_triton_final_head_euler=use_triton_final_head_euler,
                    timing=timing,
                )
            else:
                x_t = self._run_denoise_loop(
                    prefix_pad_masks=prefix_pad_masks,
                    past_key_values=past_key_values,
                    x_t=x_t,
                    num_steps=num_steps,
                    static_context=static_context,
                    direct_suffix=direct_suffix,
                    torch_compile_suffix=torch_compile_suffix,
                    torch_compile_suffix_fullgraph=torch_compile_suffix_fullgraph,
                    torch_compile_denoise_loop=torch_compile_denoise_loop,
                    timing=timing,
                    profile_nvtx=profile_nvtx,
                    use_packed_qkv=use_packed_qkv,
                    use_packed_mlp=effective_use_packed_mlp,
                    use_triton_qkv_rope=use_triton_qkv_rope,
                    use_no_cat_suffix_attn=use_no_cat_suffix_attn,
                    use_triton_final_head=use_triton_final_head,
                    use_triton_final_head_euler=use_triton_final_head_euler,
                )
        return x_t

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        params_dict = dict(self.named_parameters())
        buffers_dict = dict(self.named_buffers())
        paligemma_submodules = ("vision_tower", "multi_modal_projector", "language_model")

        def _remap(name: str) -> str:
            if name.startswith("model."):
                name = name[len("model.") :]
            if name.startswith("action_time_mlp_in."):
                name = "time_mlp_in." + name[len("action_time_mlp_in.") :]
            elif name.startswith("action_time_mlp_out."):
                name = "time_mlp_out." + name[len("action_time_mlp_out.") :]

            for sub in paligemma_submodules:
                flat = f"paligemma_with_expert.paligemma.{sub}."
                nested = f"paligemma_with_expert.paligemma.model.{sub}."
                if name.startswith(flat) and not name.startswith(nested):
                    return nested + name[len(flat) :]

            if name == "paligemma_with_expert.paligemma.lm_head.weight":
                return "paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight"
            return name

        def _fix_vision_tower(name: str) -> str:
            vt = "paligemma_with_expert.paligemma.model.vision_tower."
            if not name.startswith(vt):
                return name
            rest = name[len(vt) :]
            if name in params_dict or name in buffers_dict:
                return name
            if rest.startswith("vision_model."):
                candidate = vt + rest[len("vision_model.") :]
            else:
                candidate = vt + "vision_model." + rest
            return candidate if (candidate in params_dict or candidate in buffers_dict) else name

        loaded = 0
        skipped: list[str] = []
        filled_params: set[str] = set()

        for name, loaded_weight in weights:
            mapped = _fix_vision_tower(_remap(name))
            if mapped.startswith("state_proj."):
                continue
            if re.match(
                r"paligemma_with_expert\.gemma_expert\.model\.layers\.\d+\.(input_layernorm|post_attention_layernorm)\.weight",
                mapped,
            ):
                continue
            if re.match(r"paligemma_with_expert\.gemma_expert\.model\.norm\.weight", mapped):
                continue

            if mapped in params_dict:
                param = params_dict[mapped]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
                loaded += 1
                filled_params.add(mapped)
            elif mapped in buffers_dict:
                buffers_dict[mapped].copy_(loaded_weight)
                loaded += 1
                filled_params.add(mapped)
            else:
                skipped.append(mapped)

        # LeRobot pi0.5 stores PaliGemma's tied text embedding as lm_head.weight.
        # The action path consumes embed_tokens, but keeping lm_head filled avoids
        # a misleading missing-weight warning and preserves the tied-weight state.
        paligemma_embed = "paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight"
        paligemma_lm_head = "paligemma_with_expert.paligemma.lm_head.weight"
        if (
            paligemma_embed in filled_params
            and paligemma_lm_head in params_dict
            and paligemma_lm_head not in filled_params
        ):
            params_dict[paligemma_lm_head].data.copy_(params_dict[paligemma_embed].data)
            filled_params.add(paligemma_lm_head)

        self.paligemma_with_expert.reset_packed_weight_cache()

        missing_params: list[str] = []
        for pname in params_dict:
            if pname in filled_params:
                continue
            if "rotary_emb" in pname or pname.endswith(".inv_freq"):
                continue
            missing_params.append(pname)

        if missing_params or skipped:
            parts: list[str] = []
            if skipped:
                parts.append(f"{len(skipped)} checkpoint key(s) had no home in the model (first 5: {skipped[:5]})")
            if missing_params:
                parts.append(f"{len(missing_params)} model param(s) received NO weight (first 5: {missing_params[:5]})")
            logger.warning("pi0.5 load_weights: %d tensors loaded — %s.", loaded, "; ".join(parts))
        else:
            logger.info("pi0.5 load_weights: %d tensors loaded, 0 skipped, 0 missing.", loaded)
        return filled_params


EntryClass = Pi05ForActionPrediction
