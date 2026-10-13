#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Reproduce kernel throughput from a single-launch NCU raw CSV (no GPU needed).

Useful FLOPs count QK^T and PV only, excluding padded tiles and scalar operations.
The denominator includes the entire selected attention kernel's elapsed time.
Neither percentage is whole-model MFU. The clock-adjusted denominator is a
theoretical BF16 peak, not measured achieved FLOP/s. NCU's SM clock metric averages
across SM instances over the measured interval, not across benchmark repetitions.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

DENSE = "sm__ops_path_tensor_op_hgmma_src_bf16_dst_fp32_sparsity_off.sum"
DURATION = "gpu__time_duration.sum"
CLOCK = "sm__cycles_elapsed.avg.per_second"
REFERENCE = Path(__file__).with_name("reference")


def read_launch(path: Path) -> dict:
    """Accept NCU wide raw exports or long Metric Name/Unit/Value exports."""
    with path.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.reader(stream))
    index = next(
        (i for i, row in enumerate(rows) if "Kernel Name" in row and "ID" in row),
        None,
    )
    if index is None:
        raise ValueError(f"{path.name}: no NCU raw CSV header")
    header = rows[index]
    launches = {}
    if "Metric Name" in header:
        for row in rows[index + 1 :]:
            if not row or row[0].startswith("=="):
                continue
            if len(row) != len(header):
                raise ValueError(f"{path.name}: malformed metric row")
            item = dict(zip(header, row))
            identity = tuple(item.get(key, "") for key in ("ID", "Process ID", "Context", "Stream", "Kernel Name"))
            launch = launches.setdefault(
                identity,
                {"id": item["ID"], "kernel": item["Kernel Name"], "metrics": {}},
            )
            name = item["Metric Name"]
            metric = {"value": item["Metric Value"], "unit": item["Metric Unit"]}
            previous = launch["metrics"].get(name)
            if previous is not None and previous != metric:
                raise ValueError(f"{path.name}: conflicting duplicate metric {name}")
            launch["metrics"][name] = metric
    else:
        if len(rows) <= index + 1 or len(rows[index + 1]) != len(header):
            raise ValueError(f"{path.name}: missing wide-export unit row")
        units = rows[index + 1]
        for row in rows[index + 2 :]:
            if not row or row[0].startswith("=="):
                continue
            if len(row) != len(header):
                raise ValueError(f"{path.name}: malformed launch row")
            item = dict(zip(header, row))
            # Preserve every row as a launch, even if two rows repeat the same ID.
            launches[len(launches)] = {
                "id": item["ID"],
                "kernel": item["Kernel Name"],
                "metrics": {name: {"value": value, "unit": unit} for name, unit, value in zip(header, units, row)},
            }
    if len(launches) != 1:
        raise ValueError(
            f"{path.name}: expected exactly one selected main-kernel launch; "
            f"found {len(launches)}. Export/filter one launch before analysis."
        )
    launch = next(iter(launches.values()))
    if not any(name in launch["kernel"] for name in ("FlashAttnFwdSm90", "FlashAttentionForwardSm90")):
        raise ValueError(f"{path.name}: selected launch is not an FA3/FA4 SM90 main kernel")
    return launch


def metric_value(metrics: dict, name: str, kind: str) -> float:
    if name not in metrics:
        raise ValueError(f"Missing required metric: {name}")
    metric = metrics[name]
    value = float(metric["value"].replace(",", ""))
    unit = metric["unit"].strip().lower()
    scales = {
        "time": {"ns": 1e-9, "us": 1e-6, "µs": 1e-6, "ms": 1e-3, "s": 1.0, "second": 1.0},
        "clock": {"hz": 1.0, "khz": 1e3, "mhz": 1e6, "ghz": 1e9, "cycle/second": 1.0},
        "ops": {"": 1.0, "op": 1.0, "ops": 1.0, "flop": 1.0, "flops": 1.0},
        "ops_cycle": {"": 1.0, "op/cycle": 1.0, "ops/cycle": 1.0},
    }
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"Invalid {name}: {metric['value']}")
    if unit not in scales[kind]:
        raise ValueError(f"Unsupported unit for {name}: {metric['unit']!r}")
    return value * scales[kind][unit]


def analyze(path: Path, args: argparse.Namespace) -> dict:
    launch = read_launch(path)
    metrics = launch["metrics"]
    seconds = metric_value(metrics, DURATION, "time")
    clock_hz = metric_value(metrics, CLOCK, "clock")
    peak_per_cycle = metric_value(metrics, DENSE + ".peak_sustained", "ops_cycle")
    useful = (
        2
        * args.query_heads
        * (args.head_dim + args.value_dim)
        * (args.cameras * args.camera_tokens**2 + args.lidar_tokens**2)
    )
    useful_tflops = useful / seconds / 1e12
    clock_peak = peak_per_cycle * clock_hz / 1e12
    result = {
        "source": path.name,
        "launch_id": launch["id"],
        "kernel": launch["kernel"],
        "kernel_ms": seconds * 1000,
        "useful_qk_pv_flops": useful,
        "useful_tflops_per_second": useful_tflops,
        "nominal_bf16_peak_tflops": args.nominal_tflops,
        "useful_over_nominal_peak_pct": 100 * useful_tflops / args.nominal_tflops,
        "sm_clock_hz": clock_hz,
        "dense_bf16_peak_ops_per_cycle": peak_per_cycle,
        "clock_adjusted_bf16_peak_tflops": clock_peak,
        "useful_over_clock_adjusted_peak_pct": 100 * useful_tflops / clock_peak,
    }
    if DENSE in metrics:
        native = metric_value(metrics, DENSE, "ops")
        result.update(
            native_dense_bf16_flops=int(native),
            native_dense_bf16_tflops_per_second=native / seconds / 1e12,
            native_padding_over_useful_pct=100 * (native / useful - 1),
            native_over_clock_adjusted_peak_pct=100 * native / seconds / 1e12 / clock_peak,
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_files", nargs="*", type=Path)
    parser.add_argument("--json", action="store_true", help="Print full results as JSON")
    parser.add_argument("--verify-reference", action="store_true", help="Check bundled values offline")
    parser.add_argument("--cameras", type=int, default=11)
    parser.add_argument("--camera-tokens", type=int, default=39780)
    parser.add_argument("--lidar-tokens", type=int, default=121136)
    parser.add_argument("--query-heads", type=int, default=4)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--value-dim", type=int, default=128)
    parser.add_argument("--nominal-tflops", type=float, default=989.0)
    args = parser.parse_args()
    numeric = ("cameras", "camera_tokens", "lidar_tokens", "query_heads", "head_dim", "value_dim", "nominal_tflops")
    if any(not math.isfinite(getattr(args, key)) or getattr(args, key) <= 0 for key in numeric):
        parser.error("Workload dimensions and nominal peak must be positive and finite")
    if args.verify_reference and args.csv_files:
        parser.error("Use either CSV paths or --verify-reference")
    if not args.verify_reference and not args.csv_files:
        parser.error("Specify one or more raw CSV files, or --verify-reference")
    try:
        if args.verify_reference:
            reference = json.loads((REFERENCE / "reference.json").read_text())
            results = []
            for case in reference["cases"]:
                result = analyze(REFERENCE / case["raw_csv"], args)
                for key, expected in case["expected"].items():
                    if not math.isclose(result[key], expected, rel_tol=1e-12, abs_tol=1e-9):
                        raise ValueError(f"{case['case']}: {key}: {result[key]} != {expected}")
                results.append(result)
        else:
            results = [analyze(path, args) for path in args.csv_files]
    except (ValueError, KeyError, OSError) as error:
        parser.error(str(error))
    if args.json:
        print(json.dumps(results, indent=2, allow_nan=False))
    else:
        print("NCU main kernel | ms | useful TFLOP/s | % nominal BF16 peak | % clock-adjusted BF16 peak")
        for result in results:
            print(
                f"{result['source']} | {result['kernel_ms']:.6f} | "
                f"{result['useful_tflops_per_second']:.6f} | "
                f"{result['useful_over_nominal_peak_pct']:.6f} | "
                f"{result['useful_over_clock_adjusted_peak_pct']:.6f}"
            )
        if args.verify_reference:
            print("Reference values verified; no GPU or profiler required.")


if __name__ == "__main__":
    main()
