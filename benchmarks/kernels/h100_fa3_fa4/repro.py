#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Standalone H100 same-view attention: FA3 auto, FA3 split=1, FA4 auto.

Synthetic BF16 inputs match the production shapes and contiguous layout.
No model, checkpoint, Omni pipeline, torch.compile, or CUDA graph is required.
"""

import argparse
import hashlib
import importlib.metadata
import itertools
import json
import math
import random
import statistics
import sys
from pathlib import Path

CASES = {"fa3_auto": (3, 0), "fa3_split1": (3, 1), "fa4_auto": (4, 0)}
LENGTHS = [39780] * 11 + [121136]
TOTAL = sum(LENGTHS)
HQ, HKV, DIM = 4, 1, 128
USEFUL_FLOPS = 2 * HQ * (DIM + DIM) * sum(length * length for length in LENGTHS)


def compare(actual, expected, *, atol, rtol):
    """Check every element, including nonfinites, without a full FP32 copy."""
    maximum = torch.zeros((), device="cuda")
    squared_error = torch.zeros((), device="cuda", dtype=torch.float64)
    passed = torch.ones((), device="cuda", dtype=torch.bool)
    actual, expected = actual.reshape(-1), expected.reshape(-1)
    for start in range(0, actual.numel(), 1 << 20):
        a, b = actual[start : start + (1 << 20)].float(), expected[start : start + (1 << 20)].float()
        error = (a - b).abs()
        passed &= (torch.isfinite(a) & torch.isfinite(b) & (error <= atol + rtol * b.abs())).all()
        maximum = torch.maximum(maximum, error.max())
        squared_error += error.square().sum(dtype=torch.float64)
    max_error, rmse = maximum.item(), (squared_error / actual.numel()).sqrt().item()
    return {
        "passed": passed.item(),
        "elements": actual.numel(),
        "atol": atol,
        "rtol": rtol,
        "max_abs_error": max_error if math.isfinite(max_error) else None,
        "rmse": rmse if math.isfinite(rmse) else None,
    }


def sampled_reference(q, k, v):
    """Eight rows per segment, all four Q heads, and every key in that segment."""
    indices, outputs, lses = [], [], []
    start = 0
    for length in LENGTHS:
        # Cover boundaries, the first tile boundary, interior, and the tail.
        rows = [0, 1, 127, 128, length // 3, length // 2, length - 2, length - 1]
        index = torch.tensor([start + row for row in rows], device="cuda")
        query = q.index_select(0, index).float().transpose(0, 1)
        key = k[start : start + length].float().transpose(0, 1)
        value = v[start : start + length].float().transpose(0, 1)
        scores = torch.matmul(query, key.transpose(-2, -1)) * (DIM**-0.5)
        outputs.append(torch.matmul(scores.softmax(-1), value).transpose(0, 1))
        lses.append(scores.logsumexp(-1).transpose(0, 1))
        indices.append(index)
        start += length
    return torch.cat(indices), torch.cat(outputs), torch.cat(lses)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("check", "bench", "profile"), default="check")
    parser.add_argument("--cases", choices=tuple(CASES), nargs="+", default=list(CASES))
    parser.add_argument("--warmup", type=int, default=10, help="Untimed calls per selected case, including FA4 JIT")
    parser.add_argument("--samples", type=int, default=30, help="Bench samples per case; multiple of case permutations")
    parser.add_argument("--seed", type=int, default=20261013)
    parser.add_argument("--output", type=Path, help="Write result JSON here; otherwise print it")
    parser.add_argument(
        "--dry-run", action="store_true", help="Print workload and validate CLI without importing torch"
    )
    args = parser.parse_args()
    if len(set(args.cases)) != len(args.cases) or args.warmup < 3:
        parser.error("cases must be unique and warmup must be at least 3")
    if args.mode == "check" and args.cases != list(CASES):
        parser.error("check requires all three cases in the default order")
    if args.mode == "profile" and len(args.cases) != 1:
        parser.error("profile requires exactly one --cases value")
    permutations = list(itertools.permutations(args.cases))
    if args.mode == "bench" and (args.samples < 1 or args.samples % len(permutations)):
        parser.error("samples must be a positive multiple of the number of case permutations")
    result = {
        "status": "dry_run" if args.dry_run else "running",
        "mode": args.mode,
        "cases": {case: {"fa_version": CASES[case][0], "num_splits": CASES[case][1]} for case in args.cases},
        "workload": {
            "source": "synthetic independent N(0,1) BF16; production shapes/layout, not captured model values",
            "seed": args.seed,
            "lengths_q": LENGTHS,
            "lengths_k": LENGTHS,
            "q_shape": [TOTAL, HQ, DIM],
            "kv_shape": [TOTAL, HKV, DIM],
            "causal": False,
            "softmax_scale": DIM**-0.5,
            "return_softmax_lse": True,
            "useful_qk_pv_flops": USEFUL_FLOPS,
            "flops_scope": "Shape-derived QK^T + PV only; excludes tile padding, softmax, exp, and reductions",
        },
        "warmup_calls_per_case": args.warmup,
    }
    if args.dry_run:
        print(json.dumps(result, indent=2))
        return

    global torch
    import torch
    from vllm.vllm_flash_attn import flash_attn_interface

    properties = torch.cuda.get_device_properties(0)
    if (
        "H100" not in properties.name
        or torch.cuda.get_device_capability() != (9, 0)
        or properties.multi_processor_count != 132
    ):
        raise RuntimeError("This reproduction requires a full 132-SM H100 SXM; its nominal BF16 peak is 989 TFLOP/s")
    if not all(flash_attn_interface.is_fa_version_supported(CASES[case][0]) for case in args.cases):
        raise RuntimeError("The installed vLLM must support every requested FA version; no fallback is used")
    versions = {name: importlib.metadata.version(name) for name in ("torch", "vllm", "triton", "nvidia-cutlass-dsl")}
    source = Path(flash_attn_interface.__file__)
    result["environment"] = {
        "versions": versions,
        "python": sys.version,
        "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(),
        "capability": list(torch.cuda.get_device_capability()),
        "sm_count": properties.multi_processor_count,
        "dispatcher_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    }
    torch.backends.cuda.matmul.allow_tf32 = False
    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    q = torch.randn(TOTAL, HQ, DIM, dtype=torch.bfloat16, device="cuda", generator=generator)
    k = torch.randn(TOTAL, HKV, DIM, dtype=torch.bfloat16, device="cuda", generator=generator)
    v = torch.randn(TOTAL, HKV, DIM, dtype=torch.bfloat16, device="cuda", generator=generator)
    offsets = [0, *itertools.accumulate(LENGTHS)]
    cu_seqlens = torch.tensor(offsets, dtype=torch.int32, device="cuda")
    result["workload"].update(q_stride=list(q.stride()), k_stride=list(k.stride()), v_stride=list(v.stride()))

    def call(case):
        version, splits = CASES[case]
        output, lse = flash_attn_interface.flash_attn_varlen_func(
            q,
            k,
            v,
            max(LENGTHS),
            cu_seqlens,
            max(LENGTHS),
            cu_seqlens_k=cu_seqlens,
            dropout_p=0.0,
            softmax_scale=None,
            causal=False,
            window_size=None,
            softcap=0.0,
            deterministic=False,
            return_softmax_lse=True,
            num_splits=splits,
            fa_version=version,
        )
        # Match the production pass API, including its LSE layout conversion.
        return output, lse.transpose(0, 1).float().contiguous()

    with torch.inference_mode():
        for iteration in range(args.warmup):
            for case in args.cases if iteration % 2 == 0 else reversed(args.cases):
                output = call(case)
                del output
        torch.accelerator.synchronize()
        if args.mode == "check":
            indices, ref_output, ref_lse = sampled_reference(q, k, v)
            baseline = call("fa3_auto")
            checks = {}
            for case in args.cases:
                output, lse = baseline if case == "fa3_auto" else call(case)
                if output.shape != q.shape or output.dtype != torch.bfloat16:
                    raise AssertionError("Unexpected attention output shape or dtype")
                if lse.shape != q.shape[:2] or lse.dtype != torch.float32:
                    raise AssertionError("Unexpected LSE shape or dtype")
                checks[case] = {
                    "all_output_vs_fa3_auto": compare(output, baseline[0], atol=0.0002, rtol=0.02),
                    "all_lse_vs_fa3_auto": compare(lse, baseline[1], atol=0.001, rtol=1e-5),
                    "sampled_output_vs_fp32": compare(output[indices], ref_output, atol=0.0002, rtol=0.02),
                    "sampled_lse_vs_fp32": compare(lse[indices], ref_lse, atol=0.001, rtol=1e-5),
                }
                del output, lse
            result["checks"] = checks
            result["reference"] = {
                "sampled_query_indices": indices.cpu().tolist(),
                "keys": "Every key in the corresponding segment; no key subsampling",
                "precision": "FP32 matmul, softmax, PV and logsumexp; TF32 disabled",
                "scope": "Kernel numerical check on synthetic inputs; not model-quality validation",
            }
            result["status"] = (
                "passed" if all(c["passed"] for case in checks.values() for c in case.values()) else "failed"
            )
        elif args.mode == "bench":
            rng = random.Random(args.seed)
            orders = []
            for _ in range(args.samples // len(permutations)):
                cycle = permutations.copy()
                rng.shuffle(cycle)
                orders.extend(cycle)
            events = [
                (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
                for _ in range(args.samples * len(args.cases))
            ]
            samples = {case: [] for case in args.cases}
            case_order = [case for order in orders for case in order]
            torch.accelerator.synchronize()
            for case, (start, end) in zip(case_order, events):
                start.record()
                output = call(case)
                end.record()
                del output
            torch.accelerator.synchronize()
            for case, (start, end) in zip(case_order, events):
                samples[case].append(start.elapsed_time(end))
            result["timing"] = {
                case: {"mean_ms": statistics.mean(values), "samples_ms": values} for case, values in samples.items()
            }
            result["orders"] = orders
            result["timing_scope"] = (
                "CUDA events around full varlen API plus LSE transpose/contiguous, including prepare/main/combine; "
                "excludes input generation, warmup/JIT and validation; no profiler and no CUDA graphs"
            )
            result["status"] = "passed"
        else:
            case = args.cases[0]
            label = f"h100_same_view_{case}"
            torch.cuda.cudart().cudaProfilerStart()
            try:
                with torch.cuda.nvtx.range(label):
                    output = call(case)
                torch.accelerator.synchronize()
            finally:
                torch.cuda.cudart().cudaProfilerStop()
            result["profile"] = {"nvtx_push_pop_range": label, "calls": 1, "warmup_outside_capture": True}
            result["status"] = "passed"

    encoded = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded)
        print(json.dumps({"status": result["status"], "output": str(args.output)}))
    else:
        print(encoded, end="")
    if result["status"] == "failed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
