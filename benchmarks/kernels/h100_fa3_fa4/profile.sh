#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# Capture one warmed-up attention main kernel. Usage: ./profile.sh CASE [OUTPUT_DIR]
set -euo pipefail

case_name=${1:?Usage: profile.sh fa3_auto|fa3_split1|fa4_auto [OUTPUT_DIR]}
output_dir=${2:-results}
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
case "$case_name" in
    fa3_auto|fa3_split1) kernel=FlashAttnFwdSm90 ;;
    fa4_auto) kernel=FlashAttentionForwardSm90 ;;
    *) echo "Unknown case: $case_name" >&2; exit 2 ;;
esac
mkdir -p "$output_dir"
stem="$output_dir/h100_${case_name}_same_view"

metrics=gpu__time_duration.sum,sm__cycles_elapsed.avg.per_second
metrics+=,sm__ops_path_tensor_op_hgmma_src_bf16_dst_fp32_sparsity_off.sum
metrics+=,sm__ops_path_tensor_op_hgmma_src_bf16_dst_fp32_sparsity_off.sum.peak_sustained

"${NCU:-ncu}" --profile-from-start off --nvtx \
    --nvtx-include "h100_same_view_${case_name}/" \
    --kernel-name-base demangled --kernel-name "regex:${kernel}" \
    --launch-count 1 --replay-mode kernel --clock-control none --cache-control all \
    --section SpeedOfLight --section SpeedOfLight_HierarchicalTensorRooflineChart \
    --section LaunchStats --section Occupancy --metrics "$metrics" \
    --export "$stem" \
    "${PYTHON:-python}" "$script_dir/repro.py" --mode profile --cases "$case_name" \
    --output "${stem}.json"

test -s "${stem}.ncu-rep"
"${NCU:-ncu}" --import "${stem}.ncu-rep" --page raw --csv > "${stem}.raw.csv"
"${PYTHON:-python}" "$script_dir/analyze_ncu.py" "${stem}.raw.csv"
