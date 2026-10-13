# H100 FA3 / FA4 same-view attention repro

Compare **FA3 auto**, **FA3 `num_splits=1`**, and **FA4 auto** on the same
Cosmos3-Nano-Transfer-Auto same-view workload. One H100 is sufficient; no model,
checkpoint, or input dataset is required.

## Workload and environment

| Setting | Value |
|---|---|
| GPU | H100 SXM 80 GB, 132 SMs |
| Sequences | 11 cameras × 39,780 tokens + 1 LiDAR × 121,136 tokens |
| LiDAR / camera length | 3.05× |
| Q / KV heads; head dimension | 4 / 1; 128 |
| Inputs / accumulation | BF16 / FP32 |
| Attention | Noncausal, dense within each sequence, no sliding window or dropout |
| Outputs | Attention output and natural-log LSE |

The three cases explicitly select `fa_version=3/3/4` and `num_splits=0/1/0`
through `vllm.vllm_flash_attn.flash_attn_varlen_func`. Zero means automatic
selection. FA4 here is the **varlen same-view kernel**.

The original profiles used the following environment:

| Component | Version |
|---|---|
| Python | 3.13.15 |
| PyTorch / CUDA | 2.13.0+cu130 / 13.0 |
| vLLM | 0.31.0 |
| Triton | 3.7.1 |
| CUTLASS DSL | 4.7.1 |
| Nsight Compute | 2025.3.1.0 |
| vLLM FlashAttention source | `9cd61de38763d712bb6ce56e2a02cc2bf718c89f` |

Use a matching CUDA 13 vLLM environment with its FA3 and FA4 implementations
available. The script imports vLLM directly; installing vLLM-Omni or the separate
`flash-attn-4` package is unnecessary. It records the installed versions in its
output and fails if a requested backend is unavailable.

## Run

From this directory:

```bash
# Validate all three outputs and LSEs, including sampled FP32 reference queries.
python repro.py --mode check --output results/check.json

# Warmed-up CUDA-event timings for the same-view API.
python repro.py --mode bench --output results/bench.json

# Capture one warmed-up main kernel per case and print its utilization.
bash profile.sh fa3_auto
bash profile.sh fa3_split1
bash profile.sh fa4_auto

# Recalculate the original published numbers without a GPU or NCU installation.
python analyze_ncu.py --verify-reference
```

`NCU=/path/to/ncu` and `PYTHON=/path/to/python` select executables for profiling.
NCU requires permission to access GPU performance counters. Profiling uses
`--clock-control none --cache-control all`, matching the original reports.

The repro creates seeded synthetic Q/K/V with the original shapes and contiguous
strides. The original NCU reports used captured model activations, so exact timing
and numerical-error values need not match the synthetic run.
Benchmark timing includes the attention API and LSE layout conversion; NCU selects
only the main attention kernel. Compilation and warmup precede both measurements.

## Original NCU results and calculation

| Main kernel | Time | Useful TFLOP/s | Nominal-peak utilization | Clock-adjusted utilization |
|---|---:|---:|---:|---:|
| FA3 auto | 96.21 ms | 682.90 | 69.0% | 76.2% |
| FA3 `num_splits=1` | 86.34 ms | 760.96 | 76.9% | 87.6% |
| FA4 auto | 87.32 ms | 752.42 | 76.1% | 86.9% |

Both utilization columns use the **same useful QKᵀ + PV FLOPs and full main-kernel
duration**. Only the peak denominator changes:

```text
F = 2 × Hq × (Dqk + Dv) × sum(Lq_i × Lk_i)
  = 2 × 4 × 256 × (11 × 39780² + 121136²)
  = 65,701,607,211,008 FLOPs

Useful throughput       = F / NCU kernel duration
Nominal utilization     = useful throughput / (989 × 10¹² FLOP/s)
Clock-adjusted peak     = NCU dense BF16 peak FLOPs/cycle × measured SM cycles/s
Clock-adjusted utilization = useful throughput / clock-adjusted peak
```

The clock-adjusted peak is `132 × 4096 × measured SM cycles/s` for this H100.
`sm__cycles_elapsed.avg.per_second` is averaged across SM instances over the
profile interval. The numerator excludes padding and softmax operations; the
kernel duration includes softmax, loads/stores, and synchronization, including
their overlap. These are **kernel-level** ratios. The earlier 87.6% figure used
the clock-adjusted denominator; its nominal-peak equivalent is 76.9%.

[reference/](reference/) contains the original exported metrics and report
checksums. `analyze_ncu.py` also reports NCU-counted dense BF16 FLOPs, which include
padding, separately from the useful FLOPs above. The three original `.ncu-rep`
files can be shared as separate attachments.

## Why `num_splits=1` matters for this shape

The largest LiDAR K+V segment exceeds FA3's 50 MiB heuristic threshold, selecting
a host split cap of 2. The per-sequence scheduler nevertheless assigns one actual
split to each of the 12 sequences. The split-enabled main-kernel implementation
still differs: forcing `num_splits=1` changes the Q loading and producer-thread
path, and removes the unnecessary combine launch. This is a shape-specific
comparison; the faster setting must be checked for other sequence lengths.

The workload was captured from vLLM-Omni `0ed322e39d0d6bac1f933b692aa85bfeffdd1fc1`
with `nvidia/Cosmos3-Nano-Transfer-Auto` revision
`da7c96b894a2911e4e379d44e5de0412bb4f7cf2`.
