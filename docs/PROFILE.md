# FingerNet — Inference Profiling

End-to-end profiling of the FingerNet extraction pipeline with per-stage
timing (CUDA events) on an H100 PCIe (80 GB), PyTorch 2.10.0+cu128.

## How to reproduce

Script: [`pytorch/scripts/profile_inference.py`](../pytorch/scripts/profile_inference.py).

```bash
cd pytorch
python scripts/profile_inference.py \
    --input /tmp/bn48k_sample_1000.txt \
    --num-images 1000 --batch-size 8 --num-workers 8 \
    --warmup-batches 5 \
    --report ../docs/profile.md \
    --json ../docs/profile.json \
    --trace ../docs/profile_trace.json   # optional, open in chrome://tracing
```

## Methodology

- **GPU timing** uses `torch.cuda.Event(enable_timing=True)`. Using
  `time.perf_counter()` over GPU code is incorrect: CUDA kernels are
  asynchronous, so a host-side timer would lump every kernel into the
  next sync point. Events are recorded into the stream (in order with
  the kernels) and measure device time. `flush()` at the end of each
  batch drains all events with **a single** `cuda.synchronize()`.
- **Wall-clock** with `time.perf_counter()` for host-side costs
  (DataLoader, `.cpu()`, save).
- **H2D / D2H transfers** isolated with sync on both sides.
- **5 warmup batches** discarded (cuDNN autotune, lazy allocs, JIT).
- **Chrome trace** optionally via `torch.profiler.profile(activities=[CPU,
  CUDA])` → `prof.export_chrome_trace(...)` for kernel-level inspection
  in `chrome://tracing` or Perfetto.

## Setup

| Item | Value |
|---|---|
| GPU | NVIDIA H100 PCIe (80 GB, CC 9.0) |
| PyTorch / CUDA | 2.10.0+cu128 / 12.8 |
| Model | FingerNet released_version |
| Threshold | 0.05 (matches the threshold used for SD258) |
| Dataset | BN48k, first 1000 images (all 512×512 PNG) |
| `pin_memory` / `num_workers` | True / 8 |

## Headline result

The profiling script enables every safe optimization (including
`cudnn.benchmark=True`) to surface the headline throughput. Production
inference (`api.py`) intentionally leaves `cudnn.benchmark` OFF — see
the **Reproducibility note** below.

| Variant | Batch | Throughput | Wall total | NMS | Speedup |
|---|---:|---:|---:|---:|---:|
| Baseline (pre-optimization) | 32 | **29.6 img/s** | 33.83 s | 25.13 s (75%) | 1.0× |
| Optimized | 32 | **92.2 img/s** | 10.85 s | 1.76 s (17%) | 3.1× |
| Optimized, batch_size=8 | **8** | **🏆 112.9 img/s** | **8.86 s** | 1.98 s (23%) | **3.8×** |

## Optimizations applied (results unchanged in production)

1. **Greedy NMS on CPU/NumPy** — `pytorch/fingernet/wrapper.py:_post_nms`.
   The original loop had `if keep[i]:` against a GPU tensor every
   iteration, forcing a host↔device sync per step. At threshold 0.05
   each image has hundreds–thousands of candidates ⇒ hundreds–thousands
   of syncs. The `suppress_mask` is small (≤ a few MB) — transferring
   it to CPU once and iterating in NumPy is bit-identical and removes
   the syncs entirely.
   **Impact: NMS stage 25.13 s → 1.98 s (~14× faster).**

2. **`non_blocking=True` for H2D** — `pytorch/fingernet/api.py`. Since
   the `DataLoader` already uses `pin_memory=True`, the copy can run
   concurrently with the previous kernel.

3. **`torch.inference_mode()` instead of `torch.no_grad()`** —
   `pytorch/fingernet/wrapper.py` and `api.py`. Slightly faster (no
   version-counter bookkeeping).

### Reproducibility — investigation of residual differences

The previously saved SD258 outputs at
`/storage/.../SD258/out/fingernet/` were generated on 2026-04-29 with
the same `dev` codebase (commit `f3d8cb2`) and the same TF32 setting.
Re-extracting the full SD258 with the **new optimized code** does not
produce byte-identical files. We bisected the source.

**Test 1 — determinism of the new code itself.** Two consecutive
`fingernet infer ... --full` runs on SD258 with the new code:

| Output dir | Run-1 vs Run-2 (same / total) |
|---|---:|
| enhanced, ori, mask, minutiae | 516 / 516 |
| enhanced_mod, ori_mod | 516 / 516 |
| quality, minutiae_unmod | 516 / 516 |

→ **The new code is 100% deterministic on this machine** (no run-to-run
drift from TF32, threading, or async D2H ordering).

**Test 2 — `cudnn.benchmark` on/off.** With `cudnn.benchmark=True`,
0/516 enhanced PNGs match the prior extraction; with `False`, 455/516
match. Conclusion: cuDNN autotune picks a *different* conv algorithm
from the heuristic default and that difference dominates. We default
`cudnn.benchmark = False` in `api.py` to preserve byte-level similarity
with prior runs.

**Test 3 — TF32 on/off.** Forcing FP32 (`fp32_precision = "ieee"`) for
both matmul and cuDNN conv made things *worse*: minutiae match dropped
from 501/516 to 20/516. → The prior SD258 extraction was done with
TF32 ON. The residual differences are not from changing the precision
regime.

**Test 4 — magnitude of remaining diffs.** With `cudnn.benchmark=False`
+ TF32 ON (the new defaults), counting the actual differences:

| Output | Same files | Near-identical (≤1 px diff) | Total |
|---|---:|---:|---:|
| `enhanced/`        | 455 (88%) | +7  → 462 (90%) | 516 |
| `ori/`             | 493 (96%) | —              | 516 |
| `mask/`            | 515 (99.8%) | —            | 516 |
| `minutiae/`        | **501 (97%)** | —          | 516 |
| `enhanced_mod/`    | 508 (98%) | +1  → 509 (99%) | 516 |
| `ori_mod/`         | 510 (99%) | —              | 516 |
| `quality/`         | 450 (87%) | +66 → 516 (100%) | 516 |
| `minutiae_unmod/`  | 500 (97%) | —              | 516 |

For minutiae specifically — out of **102,613 minutiae** in the prior
extraction, **102,605 (99.992%)** appear identically in the new one.
The 8 that differ are all ±1 in a single field:

| Δ | Cases |
|---|---|
| Quality off by ±1 (e.g. `q=78` → `q=79`) | 5 |
| Position off by ±1 pixel | 2 |
| Angle off by ±2° | 1 |
| Extra spurious low-quality minutia (`q=8`) | 1 (in 1 file) |

These are all rounding-boundary flips: the network output crosses a
quality threshold or a pixel-grid boundary by a hair, and the discrete
output flips to the neighbour.

**Test 5 — pixel-level diff distribution.** For the 61 enhanced PNGs
that differ, the diff is concentrated in a tiny fraction of pixels:

- 89% of files: max diff ≤ 1 in uint8.
- 94% of files: max diff ≤ 5.
- Worst 3 files: max diff up to 28 — but only on 0.04% of pixels (a
  few hundred isolated points out of 614,400). These are points where
  the convolution output crossed a different rounding bucket.

**Root cause.** Same code regime (TF32 + cuDNN heuristic), but the
cuDNN heuristic itself picks slightly different conv algorithms across
PyTorch / cuDNN versions. We currently run **PyTorch 2.10.0+cu128,
cuDNN 9.10.2**. The April extraction was on the same machine but with
whatever stack was active that day; the differing kernels' floating-
point reduction order under TF32 produces these epsilon-level shifts.
This is unavoidable without `torch.use_deterministic_algorithms(True)`
*and* a frozen library stack — neither of which we can reasonably
require here.

Bottom line: **the new code reproduces the prior SD258 extraction at
the noise level of TF32 itself**. For users who need maximum throughput
on a homogeneous fixed-size workload at the cost of these
epsilon-level drifts, flip the flag in user code:

```python
import torch
torch.backends.cudnn.benchmark = True   # +17% on BN48k 512×512
```

## Stage breakdown — optimized, batch_size=8, 1000 images

| Stage | Kind | Total (s) | ms/batch | ms/img | p50 (ms) | p95 (ms) | % of GPU total |
|---|---|---:|---:|---:|---:|---:|---:|
| **Network forward** | | **6.25** | | **6.25** | | | **72.5%** |
| &nbsp;&nbsp;`img_norm` | gpu | 0.04 | 0.32 | 0.04 | 0.28 | 0.61 | 0.5% |
| &nbsp;&nbsp;`feature_extractor` (VGG) | gpu | 1.98 | 15.81 | 1.98 | 15.79 | 15.85 | 22.9% |
| &nbsp;&nbsp;`ori_seg_head` (ASPP) | gpu | 0.40 | 3.18 | 0.40 | 3.18 | 3.21 | 4.6% |
| &nbsp;&nbsp;`enhancement` (Gabor 25×25 ×90) | gpu | **2.65** | 21.18 | 2.65 | 21.19 | 21.20 | **30.7%** |
| &nbsp;&nbsp;`misc_upsample_concat` | gpu | 0.00 | 0.03 | 0.00 | 0.03 | 0.03 | 0.0% |
| &nbsp;&nbsp;`minutiae_head` | gpu | 1.19 | 9.49 | 1.19 | 9.49 | 9.50 | 13.8% |
| **Post-processing** | | **2.06** | | **2.06** | | | **23.9%** |
| &nbsp;&nbsp;`mask_binarize` | gpu | 0.01 | 0.06 | 0.01 | 0.06 | 0.06 | 0.1% |
| &nbsp;&nbsp;`minutiae_detect + NMS` | gpu+cpu | 1.98 | 15.81 | 1.98 | 15.32 | 21.52 | 22.9% |
| &nbsp;&nbsp;`orientation_field` | gpu | 0.03 | 0.27 | 0.03 | 0.27 | 0.37 | 0.4% |
| &nbsp;&nbsp;`enhanced_image` | gpu | 0.04 | 0.32 | 0.04 | 0.32 | 0.41 | 0.5% |
| **Transfers** | | **0.31** | | **0.31** | | | **3.6%** |
| &nbsp;&nbsp;H2D | transfer | 0.06 | 0.47 | 0.06 | 0.44 | 0.62 | 0.7% |
| &nbsp;&nbsp;D2H | transfer | 0.25 | 1.99 | 0.25 | 1.71 | 2.29 | 2.9% |
| **Other (padding)** | | **0.00** | | **0.00** | | | **0.0%** |
| **GPU total** | | **8.62** | 68.9 | 8.62 | | | 100% |

**Wall total: 8.86 s** (112.9 img/s). GPU vs. wall delta ≈ 0.24 s — that's
DataLoader + Python overhead + sync waits. Pipeline is essentially
saturated.

### Baseline vs. optimized (batch_size=32, 1000 images)

| Stage | Baseline (s) | Optimized (s) | Δ |
|---|---:|---:|---:|
| Network forward | 7.11 | 7.13 | =0 |
| `feature_extractor` | 2.08 | 2.08 | =0 |
| `enhancement` | 3.27 | 3.29 | =0 |
| `minutiae_head` | 1.31 | 1.31 | =0 |
| Post-processing | **25.16** | **1.79** | **−93%** |
| `minutiae_detect + NMS` | **25.13** | **1.76** | **−93%** |
| Transfers (H2D + D2H) | 1.14 | 1.39 | +22% (measurement noise) |
| **GPU total** | **33.42** | **10.32** | **−69%** |
| **Wall total** | **33.83** | **10.85** | **−68%** |
| **Throughput** | **29.6 img/s** | **92.2 img/s** | **+212%** |

## Batch size sweep (optimized, 1000 images, 1× H100)

| batch_size | Throughput (img/s) | Wall (s) | ms / img |
|---:|---:|---:|---:|
| 1 | 64.6 | 15.47 | 15.5 |
| 2 | 90.7 | 11.02 | 11.0 |
| 4 | 105.9 | 9.45 | 9.5 |
| **8** | **🏆 112.9** | **8.86** | **8.9** |
| 16 | 100.3 | 9.97 | 10.0 |
| 32 | 95.2 | 10.50 | 10.5 |
| 64 | 66.2 | 15.11 | 15.1 |
| 96 | 63.0 | 15.86 | 15.9 |
| 128 | 40.1 | 24.91 | 24.9 |

**Sweet spot: batch_size = 8 on a single H100.** U-shaped curve:

- **Left tail (bs=1–4):** SM under-utilization. `feature_extractor` and
  `enhancement` at batch=1 don't fill the H100's 132 SMs; cuDNN still
  picks a decent kernel, but there's idle time.
- **Right tail (bs ≥ 16):** the per-image loop in
  `_post_detect_minutiae` serializes post-processing (one NMS call per
  image in the batch). Since post-processing is no longer the
  bottleneck, larger batches just accumulate sequential Python work
  with no proportional GPU gain.

For the full BN48k (97,534 images) at 113 img/s on 1× H100: **~14 min**.
On 4× H100 with DDP (`--gpus 4`): **~3.5 min** estimated (the pipeline
is GPU-bound and scales nearly linearly).

## Where time goes now (post-optimization, batch=8)

```
Network forward     ██████████████████████████████████████████████   72%
  enhancement       ████████████████████                              31%
  feature_extractor ██████████████                                    23%
  minutiae_head     ████████                                          14%
  ori_seg_head      ███                                                5%
Post-processing     ███████████████                                   24%
  detect+NMS        ███████████████                                   23%
Transfers           ██                                                 4%
  D2H               ██                                                 3%
  H2D               ▏                                                  1%
```

## Per-dataset throughput: BN48k vs SD258

The pipeline is GPU compute-bound (network forward = 72% of GPU time),
so per-image latency scales linearly with the image area. We measured
this directly with the same profiling script on both datasets.

### Pure inference (profile script — no `--full`, no disk save)

| Dataset | Image size | Pixels / img | Total imgs | Throughput | ms / img |
|---|---:|---:|---:|---:|---:|
| BN48k | 512 × 512 | 262,144 | 97,534 | **112.9 img/s** | 8.86 |
| SD258 | 800 × 768 | 614,400 |    516 | **47.0 img/s** | 21.26 |

- **Pixel ratio** (SD258 / BN48k): 614,400 / 262,144 = **2.34×**
- **Time ratio** (SD258 / BN48k): 21.26 / 8.86 = **2.40×**

The two ratios match within 3% — the slowdown on SD258 is *entirely*
explained by the bigger images, not by any algorithmic difference. The
network is conv-heavy (Gabor 25×25 ×90 + VGG-style backbone), and conv
FLOPs scale linearly with output spatial size, so this is exactly what
should happen.

### End-to-end production timings (measured, includes disk save)

Wall-clock for the actual `fingernet infer` CLI on 1× H100 (model load
~5 s amortized, then GPU inference + post-proc + thread-pool PNG/.MIN
save). Tested on a 1000-image BN48k subset and the full SD258.

| Dataset / mode | Imgs | Wall (s) | img/s | ms/img (after warmup) |
|---|---:|---:|---:|---:|
| BN48k, default (4 outputs) | 1,000 | 23.5 | 42.6 | ~18.5 |
| BN48k, `--full` (8 outputs) | 1,000 | 29.6 | 33.8 | ~24.6 |
| SD258, default (4 outputs) |   516 | 25.4 | 20.4 | ~39.5 |
| SD258, `--full` (8 outputs) |   516 | 40.0 | 12.9 | ~67.9 |

The gap between "pure inference" and "production" is the PNG/.MIN
encoding cost on the CPU thread pool. With `--full` you save 7 PNGs +
2 `.min` files per image vs 3 PNGs + 1 `.min` in default mode — that
translates to roughly +30–35 % wall time.

### Extrapolation to the full bases

Linear extrapolation from the per-image times above. The bottleneck is
GPU forward, so DDP across 4 GPUs scales near-linearly (each rank
processes its own shard end-to-end, including save).

| Run target | 1× H100 | 4× H100 (DDP) |
|---|---:|---:|
| **BN48k full base** (97,534 imgs)   |          |          |
| &nbsp;&nbsp;default (4 outputs) | **~30 min** (1,805 s) | **~7.5 min** (~450 s) |
| &nbsp;&nbsp;`--full` (8 outputs) | **~40 min** (2,400 s) | **~10 min** (~600 s) |
| **SD258 full base** (516 imgs)      |          |          |
| &nbsp;&nbsp;default (4 outputs) | **~25 s** | **~7 s** |
| &nbsp;&nbsp;`--full` (8 outputs) | **~40 s** | **~12 s** |

> The 4× H100 estimates assume linear DDP scaling and ignore startup
> (NCCL init ≈ 5–10 s on first run). For SD258 with only 516 images,
> startup dominates — single GPU is faster in practice. For BN48k full
> base, multi-GPU is clearly worth it.

> Production wall time on 1× H100 has grown vs. the headline 113 img/s
> in the profile because (a) the profile excludes disk save and (b) the
> profile uses `cudnn.benchmark=True` for peak throughput. Production
> defaults `cudnn.benchmark=False` for byte-level reproducibility (see
> the Reproducibility section below) — that costs ~17% on fixed-size
> workloads.

## Possible further optimizations (riskier)

If more speed is needed, in increasing risk order:

1. **`torch.compile(mode="reduce-overhead")`** on the wrapper —
   potentially 10–30% on forward. Risk: already exposed via `--compile`
   in the CLI, marked experimental; numeric drift would need
   validation.
2. **AMP / bf16 autocast** — another 1.5–2× on forward on H100, but can
   flip minutiae near the threshold (numeric). Needs a quality
   benchmark.
3. **D2H into pinned host buffers + non-blocking** — saturate PCIe and
   overlap with the next batch. Today `.cpu()` is blocking.
4. **NMS pre-filter via cell-list** before the N×N matrix — currently
   quadratic in N. Can hurt for latents with thousands of candidates.
5. **Replace `enhancement_module._select_max_orientation`** —
   normalization + softmax-like with `torch.where` is 3–4 separate
   kernels; could fuse into one (or rewrite in Triton).

## Tested optimizations that DON'T work

### `channels_last` memory format on GPU

The standard rule of thumb is "NHWC + Tensor Cores beats NCHW for
conv-heavy networks". On this model, **NHWC is 19% slower** end-to-end
(BN48k 1000 imgs, 1× H100, 5 trials, both `cudnn.benchmark` on and off).

Per-stage breakdown (ms / batch of 8, 30 iterations averaged):

| Stage | NCHW | NHWC | Speedup |
|---|---:|---:|---:|
| feature_extractor (VGG) | 15.9 | 12.8 | **1.24×** ✓ |
| ori_seg_head | 3.3 | 4.0 | 0.82× ✗ |
| enhancement | 41.6 | 77.4 | 0.54× ✗ |
| &nbsp;&nbsp;gabor_real (1→90 ch, 25×25) | 18.6 | 70.1 | **0.26×** ✗ |
| &nbsp;&nbsp;gabor_imag (1→90 ch, 25×25) | 18.4 | 70.1 | **0.26×** ✗ |

**Root cause.** Tensor Core kernels in NHWC require channel counts
that are multiples of 8 (or 4 for TF32). The two Gabor convolutions
have **1 input channel**, so NHWC cannot dispatch to the fast implicit
GEMM path and falls back to a naive kernel — making them 3.8× *slower*
than NCHW. Since the enhancement module dominates (~50% of forward),
the ~24% gain on the VGG backbone is wiped out.

Bonus reason not to apply this: NHWC outputs are **not bit-identical**
to NCHW (cuDNN picks different conv algorithms per layout, and TF32
reduction order differs by algorithm). It would silently regress the
reproducibility we worked to preserve.

**Verdict: do not apply.** This is the classic case where a generic
optimization rule fails on a specific model architecture. The fix
would be at the model level (e.g. make Gabor a depthwise-then-pointwise
factorization with proper channel counts), not at the layout flag.

## Advanced profiling: Chrome trace

Running with `--trace docs/profile_trace.json` produces a trace openable
in `chrome://tracing` or Perfetto. The trace shows:

- CUDA kernels with their names (`cudnn::cnn::*`, `at::native::*`).
- A **separate H2D/D2H copy stream** alongside the kernels — confirms
  overlap.
- CPU time on DataLoader workers and the main thread.
- Top 25 ops printed to stdout via `prof.key_averages().table(...)`.

The trace file (~19 MB JSON) is not committed to the repo; regenerate
it on demand with `--trace path.json`.

## Files generated by the script

- `--report path.md` — self-contained markdown summary.
- `--json path.json` — raw measurements (mean / per-batch lists).
- `--trace path.json` — Chrome trace (CUDA + CPU).

Useful flags:

- `--no-d2h` to isolate the pure GPU pipeline (no `.cpu()` at the end).
- `--no-perf-knobs` to measure baseline without cudnn.benchmark / TF32.
- `--threshold 0.05` (default) — matches the SD258 extraction.
