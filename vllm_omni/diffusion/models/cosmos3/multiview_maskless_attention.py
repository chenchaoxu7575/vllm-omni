# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Maskless Cosmos3 multiview attention: unmasked varlen passes merged by log-sum-exp.

Planning is host-side (:mod:`.multiview_maskless_plan`).  This module owns the
pass kernels, the FP32 merge and the custom op that keeps both opaque to
Dynamo so the GEN layers compile around a single call.

Pass-kernel contract (``PASS_KERNELS``): packed ``[total, H, D]`` queries and
``[total, H_kv, D]`` keys/values with ``H_kv | H``, int32 cumulative sequence
offsets, no causal masking, no dropout, no zero-length segment; returns the
attention output in the query dtype and the per-row log-sum-exp as natural-log
FP32 ``[total_q, H]``.  Only vLLM's bundled FlashAttention is registered for
production; a pure-torch kernel serves CPU tests.
"""

import logging
import math
from collections.abc import Callable
from dataclasses import dataclass, replace
from functools import cache, lru_cache
from typing import Any

import torch
import torch.nn.functional as F

from vllm_omni.diffusion.forward_context import get_forward_context, is_forward_context_available

from .multiview_flex_attention import MultiviewAttentionContext
from .multiview_maskless_plan import (
    META_GEN_TOKENS,
    META_K_START,
    META_KEY_STREAM,
    META_MAX_SEQLEN_K,
    META_MAX_SEQLEN_Q,
    META_Q_START,
    META_SIZE,
    SAME_VIEW_GROUP_FIELDS,
    STAGING_FIELDS,
    TENSORS_PER_PASS,
    MultiviewMasklessPlan,
    build_multiview_maskless_plan,
)

logger = logging.getLogger(__name__)

#: Rows merged per step: bounds the FP32 temporaries to a few hundred MiB.
MERGE_CHUNK_SIZE = 8192
_INT32_LIMIT = 2**31

PassKernel = Callable[..., tuple[torch.Tensor, torch.Tensor]]
PASS_KERNELS: dict[str, Callable[[], PassKernel]] = {}


def register_pass_kernel(name: str, factory: Callable[[], PassKernel]) -> None:
    PASS_KERNELS[name] = factory


@cache
def get_pass_kernel(name: str) -> PassKernel:
    factory = PASS_KERNELS.get(name)
    if factory is None:
        raise ValueError(f"Unknown Cosmos3 maskless pass kernel {name!r}; expected one of {sorted(PASS_KERNELS)}.")
    return factory()


def validate_indexing(lengths: list[int], heads: int, head_dim: int) -> None:
    """Check before allocating indices or converting cumulative offsets to int32."""
    if heads <= 0 or head_dim <= 0 or any(length < 0 for length in lengths):
        raise ValueError("Maskless attention requires nonnegative lengths and positive head geometry.")
    if sum(lengths) >= _INT32_LIMIT or any(length * heads * head_dim >= _INT32_LIMIT for length in lengths):
        raise ValueError(
            "Maskless attention exceeds int32 indexing (length × local heads × head_dim or cumulative offsets). "
            "Reduce frames/resolution/views or increase supported TP/Ulysses head partitioning."
        )


def normalize_varlen_lse(lse: torch.Tensor, tokens: int, query_heads: int) -> torch.Tensor:
    """vLLM FA2/FA3/FA4 varlen return LSE as ``[heads, total_q]``; merge wants ``[total_q, heads]`` FP32."""
    if lse.ndim != 2 or lse.shape != (query_heads, tokens):
        raise ValueError(f"Expected FlashAttention LSE [heads,tokens]={query_heads, tokens}, got {tuple(lse.shape)}.")
    return lse.transpose(0, 1).float().contiguous()


def _make_vllm_flash_attn_kernel(fa_version: int, num_splits: int = 0) -> PassKernel:
    def kernel(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        max_seqlen_q: int,
        max_seqlen_k: int,
        softmax_scale: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        from vllm_omni.diffusion.attention.backends.utils.fa import vllm_flash_attn_varlen_with_lse

        out, lse = vllm_flash_attn_varlen_with_lse(
            q,
            k,
            v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            softmax_scale=softmax_scale,
            causal=False,
            fa_version=fa_version,
            num_splits=num_splits,
        )
        return out, normalize_varlen_lse(lse, q.shape[0], q.shape[1])

    return kernel


def torch_reference_pass_kernel(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    softmax_scale: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The pass-kernel contract as per-segment dense softmax, for any device and dtype."""
    compute_dtype = torch.float64 if q.dtype == torch.float64 else torch.float32
    scale = softmax_scale if softmax_scale is not None else 1.0 / math.sqrt(q.shape[-1])
    ratio = q.shape[1] // k.shape[1]
    out = torch.empty_like(q)
    lse = torch.empty(q.shape[0], q.shape[1], dtype=torch.float32, device=q.device)
    bounds = zip(
        cu_seqlens_q[:-1].tolist(), cu_seqlens_q[1:].tolist(), cu_seqlens_k[:-1].tolist(), cu_seqlens_k[1:].tolist()
    )
    for qa, qb, ka, kb in bounds:
        keys = k[ka:kb].to(compute_dtype).repeat_interleave(ratio, 1)
        values = v[ka:kb].to(compute_dtype).repeat_interleave(ratio, 1)
        scores = torch.einsum("qhd,khd->hqk", q[qa:qb].to(compute_dtype), keys) * scale
        out[qa:qb] = torch.einsum("hqk,khd->qhd", scores.softmax(-1), values).to(q.dtype)
        lse[qa:qb] = scores.logsumexp(-1).transpose(0, 1).float()
    return out, lse


for _version in (2, 3, 4):
    register_pass_kernel(f"fa{_version}", lambda version=_version: _make_vllm_flash_attn_kernel(version))
register_pass_kernel("torch", lambda: torch_reference_pass_kernel)
register_pass_kernel("fa3_single_split", lambda: _make_vllm_flash_attn_kernel(3, num_splits=1))


@cache
def resolve_maskless_kernel() -> str:
    """Pin the worker's vLLM-bundled FlashAttention once and log the choice."""
    from vllm_omni.diffusion.attention.backends.utils.fa import resolve_vllm_flash_attn_version

    fa_version = resolve_vllm_flash_attn_version()
    logger.info(
        "Cosmos3 maskless: GPU=%s FA=%s merge=streaming-fp32 PyTorch=%s CUDA=%s",
        torch.cuda.get_device_name() if torch.cuda.is_available() else "none",
        fa_version,
        torch.__version__,
        torch.version.cuda,
    )
    return f"fa{fa_version}"


def merge_step(
    acc: torch.Tensor, acc_lse: torch.Tensor, out: torch.Tensor, lse: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fold one branch into the FP32 accumulator.

    This is the sequential sigmoid/logsigmoid recurrence of NATTEN 0.21.6's
    ``merge_attentions``: ``acc`` becomes the softmax-weighted average over
    branches seen so far and ``acc_lse`` their ``logaddexp``.  LSE is natural
    log; both accumulators stay FP32 until the final cast.
    """
    weight = torch.sigmoid(lse.float() - acc_lse).unsqueeze(-1)
    acc = acc - weight * (acc - out.float())
    acc_lse = acc_lse - F.logsigmoid(acc_lse - lse.float())
    return acc, acc_lse


def merge_rows(
    acc: torch.Tensor, acc_lse: torch.Tensor, out: torch.Tensor, lse: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """:func:`merge_step` on an FP32 copy of ``acc``, returned in ``acc``'s dtype.

    With an FP32 accumulator this is ``merge_step``; with output rows in the
    query dtype it is the same FP32 arithmetic followed by the final cast.
    """
    merged, merged_lse = merge_step(acc.float(), acc_lse, out, lse)
    return merged.to(acc.dtype), merged_lse


def _merge_attention_outputs(outputs: list[torch.Tensor], lse_tensors: list[torch.Tensor]) -> torch.Tensor:
    """All-branches reference merge with ``[..., tokens, heads]`` FP32 LSE (tests and oracles)."""
    if not outputs or len(outputs) != len(lse_tensors):
        raise ValueError("Maskless merge requires matching nonempty output and LSE lists.")
    acc, acc_lse = outputs[0].float(), lse_tensors[0].float()
    for out, lse in zip(outputs[1:], lse_tensors[1:], strict=True):
        acc, acc_lse = merge_step(acc, acc_lse, out, lse)
    return acc.to(outputs[0].dtype)


_compiled_merge_rows: Callable[..., tuple[torch.Tensor, torch.Tensor]] | None = None


def _merge_rows_for(device: torch.device) -> Callable[..., tuple[torch.Tensor, torch.Tensor]]:
    # Elementwise over chunk rows: one dynamic-shape graph per accumulator dtype
    # serves every chunk length and every geometry.  CPU (tests) runs eager.
    if device.type != "cuda":
        return merge_rows
    global _compiled_merge_rows
    if _compiled_merge_rows is None:
        _compiled_merge_rows = torch.compile(merge_rows, fullgraph=True, dynamic=True)
    return _compiled_merge_rows


Rows = slice | torch.Tensor


def _pass_rows(start: int, index: torch.Tensor) -> Rows:
    """Stream rows a pass reads: a slice when the planner marked them contiguous, else the index."""
    return slice(start, start + index.numel()) if start >= 0 else index


def _gather(tensor: torch.Tensor, rows: Rows) -> torch.Tensor:
    return tensor[rows] if isinstance(rows, slice) else tensor.index_select(0, rows)


def _scatter(tensor: torch.Tensor, rows: Rows, value: torch.Tensor) -> None:
    if isinstance(rows, slice):
        tensor[rows].copy_(value)
    else:
        tensor.index_copy_(0, rows, value.to(tensor.dtype))


def _chunk_rows(rows: Rows, chunk: slice) -> Rows:
    if isinstance(rows, slice):
        return slice(rows.start + chunk.start, min(rows.start + chunk.stop, rows.stop))
    return rows[chunk]


def _merge_pass(
    acc: torch.Tensor,
    acc_lse: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    rows: Rows,
    first_pass: bool,
    result: torch.Tensor | None = None,
) -> None:
    """Fuse row gathering, the existing FP32 recurrence, and output writes."""
    if first_pass:
        merged, merged_lse = out, lse
    else:
        merged, merged_lse = merge_rows(_gather(acc, rows), _gather(acc_lse, rows), out, lse)
    _scatter(acc, rows, merged)
    _scatter(acc_lse, rows, merged_lse)
    if result is not None:
        # The final pass need not cover all queries. Copy the full accumulator,
        # including rows that were last updated by an earlier pass.
        result.copy_(acc)


_compiled_merge_pass = torch.compile(_merge_pass, fullgraph=True, dynamic=True)


@lru_cache(maxsize=16)
def _same_view_groups(
    cu_q: torch.Tensor, cu_k: torch.Tensor, descriptors: tuple[int, ...]
) -> tuple[tuple[slice, torch.Tensor, torch.Tensor, int], ...]:
    """Prepare rebased device offsets once per plan; cache size bounds retention."""
    groups = []
    for i in range(0, len(descriptors), SAME_VIEW_GROUP_FIELDS):
        first, last, start, stop, maximum = descriptors[i : i + SAME_VIEW_GROUP_FIELDS]
        groups.append((slice(start, stop), cu_q[first : last + 1] - start, cu_k[first : last + 1] - start, maximum))
    return tuple(groups)


def _grouped_same_view(
    kernel: PassKernel,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_q: torch.Tensor,
    cu_k: torch.Tensor,
    descriptors: tuple[int, ...],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Camera/LiDAR queries are disjoint: concatenate their outputs, never LSE-merge them."""
    outputs, lses = [], []
    for rows, group_cu_q, group_cu_k, maximum in _same_view_groups(cu_q, cu_k, descriptors):
        out, lse = kernel(
            q[rows],
            k[rows],
            v[rows],
            cu_seqlens_q=group_cu_q,
            cu_seqlens_k=group_cu_k,
            max_seqlen_q=maximum,
            max_seqlen_k=maximum,
        )
        outputs.append(out)
        lses.append(lse)
    return torch.cat(outputs), torch.cat(lses)


def _append_staged_stream(
    key_streams: list[torch.Tensor], value_streams: list[torch.Tensor], staging: torch.Tensor
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Concatenate the plan's ``(stream, start, length)`` ranges into one more key/value stream."""
    if staging.ndim != 2 or staging.shape[1] != STAGING_FIELDS or staging.shape[0] == 0:
        raise ValueError(f"Maskless staging must be a non-empty [R, {STAGING_FIELDS}] tensor.")
    ranges = staging.tolist()
    for stream, start, length in ranges:
        if not 0 <= stream < len(key_streams) or start < 0 or length <= 0:
            raise ValueError(f"Maskless staging range {(stream, start, length)} is invalid.")
        if start + length > key_streams[stream].shape[0]:
            raise ValueError(
                f"Maskless staging range {(stream, start, length)} exceeds key stream {stream} "
                f"of {key_streams[stream].shape[0]} tokens."
            )
    keys = torch.cat([key_streams[stream][start : start + length] for stream, start, length in ranges])
    values = torch.cat([value_streams[stream][start : start + length] for stream, start, length in ranges])
    return [*key_streams, keys], [*value_streams, values]


def _maskless_attention_impl(
    q: torch.Tensor,
    key_streams: list[torch.Tensor],
    value_streams: list[torch.Tensor],
    plan: list[torch.Tensor],
    num_passes: int,
    kernel_name: str,
) -> torch.Tensor:
    """Run the planned passes over ``q[0]`` and merge them.

    ``key_streams[i]``/``value_streams[i]`` are packed ``[tokens, H_kv, D]``
    tensors; each pass's ``META_KEY_STREAM`` selects the stream it reads keys
    from. The bidirectional multiview op passes ``[GEN, UND]``; causal variants
    append further streams such as retained KV history. A trailing staging
    tensor in ``plan`` (see ``STAGING_FIELDS``) appends one more stream, built
    once per call from contiguous ranges of the others.
    """
    if q.ndim != 4 or q.shape[0] != 1:
        raise ValueError("Maskless multiview attention requires B == 1.")
    staged = len(plan) - num_passes * TENSORS_PER_PASS
    if num_passes <= 0 or staged not in (0, 1):
        raise ValueError(f"Maskless plan carries {len(plan)} tensors for {num_passes} passes.")
    if not key_streams or len(key_streams) != len(value_streams):
        raise ValueError("Maskless attention needs matching non-empty key and value stream lists.")
    first_meta = plan[TENSORS_PER_PASS - 1]
    planned_tokens = int(first_meta[META_GEN_TOKENS])
    if q.shape[1] != planned_tokens or plan[0].numel() > planned_tokens:
        raise ValueError(
            "Cosmos3 maskless packed GEN length does not match the request plan: "
            f"attention={q.shape[1]}, plan={planned_tokens}."
        )
    heads, head_dim = q.shape[2], q.shape[3]
    kv_heads = key_streams[0].shape[1] if key_streams[0].ndim == 3 else -1
    for stream_keys, stream_values in zip(key_streams, value_streams, strict=True):
        if stream_keys.ndim != 3 or stream_keys.shape != stream_values.shape:
            raise ValueError("Maskless key/value streams must be matching [tokens, H_kv, D] tensors.")
        if stream_keys.shape[1] != kv_heads or stream_keys.shape[2] != head_dim:
            raise ValueError("Maskless key streams must share the query head size and one KV head count.")
    if kv_heads <= 0 or heads % kv_heads:
        raise ValueError("Maskless attention requires an integral GQA ratio.")
    kernel = get_pass_kernel(kernel_name)
    config = get_forward_context().omni_diffusion_config if is_forward_context_available() else None
    options = (config.custom_pipeline_args if config is not None else None) or {}
    same_view_single_split = options.get("cosmos3_same_view_single_split", False)
    if not isinstance(same_view_single_split, bool):
        raise ValueError("cosmos3_same_view_single_split must be a bool")
    same_view_kernel = (
        get_pass_kernel("fa3_single_split") if same_view_single_split and kernel_name == "fa3" else kernel
    )
    lse_fusion = options.get("cosmos3_lse_fusion", False)
    same_view_grouping = options.get("cosmos3_same_view_grouping", False)
    for name, enabled in (("cosmos3_lse_fusion", lse_fusion), ("cosmos3_same_view_grouping", same_view_grouping)):
        if not isinstance(enabled, bool):
            raise ValueError(f"custom_pipeline_args[{name!r}] must be a bool.")
    direct_store_enabled = options.get("cosmos3_lse_direct_store", False)
    if not isinstance(direct_store_enabled, bool):
        raise ValueError("cosmos3_lse_direct_store must be a bool")
    if direct_store_enabled and not lse_fusion:
        raise ValueError("Direct LSE store requires cosmos3_lse_fusion.")
    direct_merge = None
    if direct_store_enabled and q.device.type == "cuda":
        from .multiview_lse_store import direct_store

        direct_merge = direct_store
    trimmed_queries = plan[0].numel() < planned_tokens
    if trimmed_queries and not (
        options.get("cosmos3_last_layer_same_view_trim", False) and same_view_single_split and lse_fusion
    ):
        raise ValueError("A trimmed last-layer plan requires LSE fusion and same-view single split.")
    fused_merge = lse_fusion and q.device.type == "cuda"
    merge = _merge_rows_for(q.device)
    # Preserve the caller's tensor mode: HSDP/offload use no_grad() and need
    # outputs with version counters.  Only the passes and the accumulators are
    # inference-mode temporaries.
    result = torch.empty_like(q)
    result_rows = result[0]
    with torch.inference_mode():
        if staged:
            key_streams, value_streams = _append_staged_stream(key_streams, value_streams, plan[-1])
        # The first pass covers every row and seeds the accumulators. With at
        # most one merge, rows go straight to the output: the merge runs in FP32
        # and rounds once, exactly as an FP32 accumulator would. Longer plans
        # keep FP32 rows between merges.
        if num_passes <= 2:
            acc = result_rows
        else:
            acc = torch.empty((planned_tokens, heads, head_dim), dtype=torch.float32, device=q.device)
        acc_lse = torch.empty((planned_tokens, heads), dtype=torch.float32, device=q.device)
        if trimmed_queries:
            # Omitted final-layer control rows are dead after this layer. Give
            # their later passes finite seeds; target rows are overwritten by
            # the unchanged first-pass output before any LSE merge.
            acc.zero_()
            acc_lse.zero_()
        for index in range(num_passes):
            q_index, k_index, cu_seqlens_q, cu_seqlens_k, meta = plan[
                index * TENSORS_PER_PASS : (index + 1) * TENSORS_PER_PASS
            ]
            meta_values = meta.tolist()
            max_seqlen_q, max_seqlen_k = int(meta_values[META_MAX_SEQLEN_Q]), int(meta_values[META_MAX_SEQLEN_K])
            stream = int(meta_values[META_KEY_STREAM])
            if not 0 <= stream < len(key_streams):
                raise ValueError(f"Maskless pass addresses key stream {stream}, but {len(key_streams)} were given.")
            q_rows = _pass_rows(int(meta_values[META_Q_START]), q_index)
            k_rows = _pass_rows(int(meta_values[META_K_START]), k_index)
            if isinstance(q_rows, slice) and q_rows.stop > planned_tokens:
                raise ValueError(f"Maskless pass reads query rows {q_rows} of {planned_tokens}.")
            if isinstance(k_rows, slice) and k_rows.stop > key_streams[stream].shape[0]:
                raise ValueError(f"Maskless pass reads key rows {k_rows} of {key_streams[stream].shape[0]}.")
            # Actual heads here are already partitioned by TP and Ulysses.
            validate_indexing([max_seqlen_q], heads, head_dim)
            validate_indexing([max_seqlen_k], kv_heads, head_dim)
            queries = _gather(q[0], q_rows)
            keys, values = _gather(key_streams[stream], k_rows), _gather(value_streams[stream], k_rows)
            pass_kernel = same_view_kernel if index == 0 else kernel
            if same_view_grouping and index == 0 and len(meta_values) > META_SIZE:
                out, lse = _grouped_same_view(
                    pass_kernel, queries, keys, values, cu_seqlens_q, cu_seqlens_k, tuple(meta_values[META_SIZE:])
                )
            else:
                out, lse = pass_kernel(
                    queries,
                    keys,
                    values,
                    cu_seqlens_q=cu_seqlens_q,
                    cu_seqlens_k=cu_seqlens_k,
                    max_seqlen_q=max_seqlen_q,
                    max_seqlen_k=max_seqlen_k,
                )
            del queries, keys, values
            rows = q_index.numel()
            if out.shape != (rows, heads, head_dim) or lse.shape != (rows, heads):
                raise ValueError(
                    f"Maskless pass kernel returned output {tuple(out.shape)} and LSE {tuple(lse.shape)} "
                    f"for {rows} rows × {heads} heads × {head_dim}."
                )
            final_result = result_rows if index == num_passes - 1 and acc is not result_rows else None
            stored = direct_merge is not None and direct_merge(
                acc, acc_lse, out, lse, q_index,
                start=int(meta_values[META_Q_START]), seed=index == 0, result=final_result,
            )
            if stored:
                pass
            elif fused_merge:
                _compiled_merge_pass(
                    acc,
                    acc_lse,
                    out,
                    lse,
                    q_rows,
                    index == 0,
                    result_rows if index == num_passes - 1 and acc is not result_rows else None,
                )
            else:
                for start in range(0, rows, MERGE_CHUNK_SIZE):
                    chunk = slice(start, start + MERGE_CHUNK_SIZE)
                    target = _chunk_rows(q_rows, chunk)
                    if index == 0:
                        _scatter(acc, target, out[chunk])
                        _scatter(acc_lse, target, lse[chunk])
                    else:
                        merged, merged_lse = merge(
                            _gather(acc, target), _gather(acc_lse, target), out[chunk], lse[chunk]
                        )
                        _scatter(acc, target, merged)
                        _scatter(acc_lse, target, merged_lse)
            del out, lse
        if acc is not result_rows and not fused_merge:
            for start in range(0, planned_tokens, MERGE_CHUNK_SIZE):
                chunk = slice(start, start + MERGE_CHUNK_SIZE)
                result_rows[chunk].copy_(acc[chunk])
    return result


# Dynamo sees one opaque call per layer; the guard preserves registration
# across module re-imports in tests.
if not hasattr(torch.ops.vllm_omni, "cosmos3_maskless_attention"):

    @torch.library.custom_op("vllm_omni::cosmos3_maskless_attention", mutates_args=())
    def _cosmos3_maskless_attention_op(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        k_und: torch.Tensor,
        v_und: torch.Tensor,
        plan: list[torch.Tensor],
        num_passes: int,
        kernel: str,
    ) -> torch.Tensor:
        if q.ndim != 4 or q.shape[0] != 1:
            raise ValueError("Maskless multiview attention requires B == 1.")
        if k.shape != v.shape or k_und.shape != v_und.shape or k.shape[:2] != q.shape[:2]:
            raise ValueError("Maskless Q/K/V geometry mismatch.")
        if k_und.shape[0] != 1 or k_und.shape[2:] != k.shape[2:] or q.shape[-1] != k.shape[-1]:
            raise ValueError("Maskless GEN/UND head geometry mismatch.")
        return _maskless_attention_impl(q, [k[0], k_und[0]], [v[0], v_und[0]], plan, num_passes, kernel)

    @_cosmos3_maskless_attention_op.register_fake
    def _(q, k, v, k_und, v_und, plan, num_passes, kernel):
        return torch.empty_like(q)


if not hasattr(torch.ops.vllm_omni, "cosmos3_maskless_attention_streams"):

    @torch.library.custom_op("vllm_omni::cosmos3_maskless_attention_streams", mutates_args=())
    def _cosmos3_maskless_attention_streams_op(
        q: torch.Tensor,
        key_streams: list[torch.Tensor],
        value_streams: list[torch.Tensor],
        plan: list[torch.Tensor],
        num_passes: int,
        kernel: str,
    ) -> torch.Tensor:
        """Generalized executor: ``META_KEY_STREAM`` selects among packed ``[tokens, H_kv, D]`` streams."""
        return _maskless_attention_impl(q, list(key_streams), list(value_streams), plan, num_passes, kernel)

    @_cosmos3_maskless_attention_streams_op.register_fake
    def _(q, key_streams, value_streams, plan, num_passes, kernel):
        return torch.empty_like(q)


_cosmos3_maskless_attention_op = torch.ops.vllm_omni.cosmos3_maskless_attention
maskless_attention_streams = torch.ops.vllm_omni.cosmos3_maskless_attention_streams


@dataclass(frozen=True, eq=False)
class MasklessRuntime:
    """Flattened plan plus the worker's pinned kernel, carried on the attention context."""

    plan: list[torch.Tensor]
    num_passes: int
    kernel: str


def get_maskless_plan(
    context: MultiviewAttentionContext, *, num_und_tokens: int, device: torch.device | str
) -> MultiviewMasklessPlan:
    """Build or retrieve the request-local plan for one CFG text length."""
    device = torch.device(device)
    key = ("maskless", context.layout, num_und_tokens, device)
    plan = context.mask_cache.get(key)
    if plan is None:
        plan = build_multiview_maskless_plan(context.layout, num_und_tokens, device)
        context.mask_cache[key] = plan
    return plan


def prepare_maskless_context(
    context: MultiviewAttentionContext,
    *,
    num_und_tokens: int,
    device: torch.device | str,
    kernel: str | None = None,
) -> MultiviewAttentionContext:
    """Attach the plan and kernel before the compiled GEN layers run."""
    plan = get_maskless_plan(context, num_und_tokens=num_und_tokens, device=device)
    runtime = MasklessRuntime(plan.flatten(), len(plan.passes), kernel or resolve_maskless_kernel())
    return replace(context, maskless=runtime)



def prepare_long_first_same_view_context(
    context: MultiviewAttentionContext, *, head_dim: int, dtype: torch.dtype
) -> MultiviewAttentionContext:
    """Schedule longer independent same-view sequences first, preserving KV order."""
    config = get_forward_context().omni_diffusion_config if is_forward_context_available() else None
    options = (config.custom_pipeline_args if config is not None else None) or {}
    enabled = options.get("cosmos3_same_view_long_first", False)
    if not isinstance(enabled, bool):
        raise ValueError("cosmos3_same_view_long_first must be a bool")
    if not enabled:
        return context
    if not all(options.get(name, False) is True for name in ("cosmos3_lse_fusion", "cosmos3_same_view_single_split")):
        raise ValueError("Long-first same-view scheduling requires LSE fusion and same-view single split.")
    runtime = context.maskless
    if runtime is None or runtime.kernel != "fa3" or head_dim != 128 or dtype != torch.bfloat16:
        return context
    device = runtime.plan[0].device
    if device.type != "cuda" or torch.cuda.get_device_capability(device) != (9, 0):
        return context
    key = ("same_view_long_first", context.layout, runtime.plan[0].data_ptr())
    cached = context.mask_cache.get(key)
    if cached is None:
        cached = _sort_same_view_sequences(runtime)
        context.mask_cache[key] = cached
    return replace(context, maskless=cached)


def _sort_same_view_sequences(runtime: MasklessRuntime) -> MasklessRuntime:
    """Reorder complete descriptors; never reorder a sequence's key/value rows."""
    from .multiview_maskless_plan import contiguous_start

    q_index, k_index = (tensor.cpu() for tensor in runtime.plan[:2])
    cu_q, cu_k = (tensor.cpu().tolist() for tensor in runtime.plan[2:4])
    order = sorted(range(len(cu_k) - 1), key=lambda i: cu_k[i + 1] - cu_k[i], reverse=True)
    if order == list(range(len(order))):
        return runtime
    device = runtime.plan[0].device
    plan = list(runtime.plan)
    for slot, indices, offsets, start_meta in (
        (0, q_index, cu_q, META_Q_START),
        (1, k_index, cu_k, META_K_START),
    ):
        selected = torch.cat([indices[offsets[i] : offsets[i + 1]] for i in order])
        lengths = [offsets[i + 1] - offsets[i] for i in order]
        plan[slot] = selected.to(device)
        plan[slot + 2] = torch.tensor(
            [0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int32, device=device
        )
        if slot == 0:
            plan[4] = plan[4].clone()
        plan[4][start_meta] = contiguous_start(selected)
    return MasklessRuntime(plan, runtime.num_passes, runtime.kernel)


def prepare_last_layer_same_view_context(
    context: MultiviewAttentionContext, *, head_dim: int, dtype: torch.dtype
) -> MultiviewAttentionContext | None:
    """Drop only whole control query tiles in the terminal GEN layer.

    K/V and subsequent pass ordering stay intact. Align in 128-token units,
    which contain whole FA3 d128 query tiles for every supported GQA ratio.
    Keeping a mixed boundary tile preserves each retained row's MMA position.
    Planning uses the layout, independent of camera count and CP/CFG rank.
    """
    config = get_forward_context().omni_diffusion_config if is_forward_context_available() else None
    options = (config.custom_pipeline_args if config is not None else None) or {}
    enabled = options.get("cosmos3_last_layer_same_view_trim", False)
    if not isinstance(enabled, bool):
        raise ValueError("cosmos3_last_layer_same_view_trim must be a bool")
    if not enabled:
        return None
    if not all(options.get(name, False) is True for name in ("cosmos3_lse_fusion", "cosmos3_same_view_single_split")):
        raise ValueError("Last-layer same-view trimming requires LSE fusion and same-view single split.")
    runtime = context.maskless
    if runtime is None or runtime.kernel != "fa3" or head_dim != 128 or dtype != torch.bfloat16:
        return None
    device = runtime.plan[0].device
    if device.type != "cuda" or torch.cuda.get_device_capability(device) != (9, 0):
        return None
    key = ("last_layer_same_view_trim", context.layout, runtime.plan[0].data_ptr())
    cached = context.mask_cache.get(key)
    if cached is not None:
        return replace(context, maskless=cached)
    cached = _trim_same_view_queries(runtime, context.layout)
    context.mask_cache[key] = cached
    return replace(context, maskless=cached)


def _trim_same_view_queries(runtime: MasklessRuntime, layout: Any) -> MasklessRuntime:
    """Host planner; its CPU form also validates arbitrary layout geometry."""
    device = runtime.plan[0].device
    target = torch.empty(layout.gen_tokens, dtype=torch.bool)
    start = 0
    for item in layout.items:
        target[start : start + item.num_tokens] = not item.is_control
        start += item.num_tokens
    q_index = runtime.plan[0].cpu()
    offsets = runtime.plan[2].cpu().tolist()
    selected, lengths = [], []
    for lo, hi in zip(offsets, offsets[1:]):
        segment = q_index[lo:hi]
        keep = []
        for begin in range(0, hi - lo, 128):
            tile = segment[begin : begin + 128]
            if target[tile].any():
                keep.append(tile)
        # Control-only view: preserve it to keep the varlen KV pairing valid.
        retained = torch.cat(keep) if keep else segment
        selected.append(retained)
        lengths.append(retained.numel())
    if sum(lengths) == q_index.numel() or max(lengths) <= 64:
        return runtime
    from .multiview_maskless_plan import contiguous_start

    plan = list(runtime.plan)
    selected_index = torch.cat(selected)
    plan[0] = selected_index.to(device)
    plan[2] = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int32, device=device)
    plan[4] = plan[4].clone()
    plan[4][META_MAX_SEQLEN_Q] = max(lengths)
    plan[4][META_Q_START] = contiguous_start(selected_index)
    cached = MasklessRuntime(plan, runtime.num_passes, runtime.kernel)
    return cached


def maskless_multiview_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    k_und: torch.Tensor,
    v_und: torch.Tensor,
    context: MultiviewAttentionContext,
) -> torch.Tensor:
    runtime: Any = context.maskless
    if runtime is None:
        raise RuntimeError(
            "Cosmos3 maskless attention needs a prepared plan; call prepare_maskless_context before the GEN layers."
        )
    return _cosmos3_maskless_attention_op(q, k, v, k_und, v_und, runtime.plan, runtime.num_passes, runtime.kernel)
