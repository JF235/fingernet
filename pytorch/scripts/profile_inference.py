"""
Profile FingerNet inference end-to-end with stage-by-stage GPU and CPU timing.

Methodology
-----------
- GPU timing uses CUDA events (``torch.cuda.Event(enable_timing=True)``).
  ``time.perf_counter()`` is unsafe for GPU work because CUDA kernels are
  asynchronous: a host-side timer would lump every kernel into the next sync
  point. CUDA events are recorded into the stream, run in-order with the
  kernels, and report device-side elapsed time.
- Wall-clock timing uses ``time.perf_counter()`` to capture host-side cost
  (data loading, ``.cpu()`` calls, save-to-disk).
- H2D / D2H transfers are isolated by inserting ``torch.cuda.synchronize()``
  on both sides and timing the explicit copies.
- A first short window is dropped as warmup (cuDNN autotune, lazy allocs,
  weight DMA, kernel JIT).
- Optionally records a Chrome trace via ``torch.profiler`` for kernel-level
  inspection.

Output
------
Prints a markdown summary to stdout (per-stage mean ms, % of total, GPU vs
CPU). Optionally writes a Chrome trace (``--trace out/trace.json``) and a
JSON dump of the raw measurements (``--json out/profile.json``).

Usage
-----
    python scripts/profile_inference.py \\
        --input /storage/.../BN48k/images \\
        --num-images 1000 --batch-size 32 --num-workers 8 \\
        --trace out/trace.json --json out/profile.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from fingernet.api import (
    FingerprintDataset,
    dynamic_padding_collate,
    find_image_paths,
)
from fingernet.fnet_utils import DEFAULT_WEIGHTS_PATH, get_fingernet_logger
from fingernet.wrapper import (
    _post_binarize_mask_fast,
    _post_detect_minutiae,
    get_fingernet,
)

logger = get_fingernet_logger("fingernet.profile", level=logging.INFO)


# --------------------------------------------------------------------------- #
# Timing utilities                                                            #
# --------------------------------------------------------------------------- #


class CudaStageTimer:
    """Accumulate per-stage GPU time using CUDA events.

    Events are recorded inline (no host sync). ``flush()`` is called once
    per batch to drain the queue with a single ``cuda.synchronize()`` and
    materialize ``elapsed_time`` for each pair.
    """

    def __init__(self):
        self._pending: list[tuple[str, torch.cuda.Event, torch.cuda.Event]] = []
        self.totals_ms: dict[str, float] = defaultdict(float)
        self.counts: dict[str, int] = defaultdict(int)
        self.per_batch_ms: dict[str, list[float]] = defaultdict(list)

    @contextmanager
    def stage(self, name: str):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        try:
            yield
        finally:
            end.record()
            self._pending.append((name, start, end))

    def flush(self):
        if not self._pending:
            return
        torch.cuda.synchronize()
        # Aggregate everything recorded since the last flush as one batch.
        batch_acc: dict[str, float] = defaultdict(float)
        for name, start, end in self._pending:
            ms = start.elapsed_time(end)
            self.totals_ms[name] += ms
            self.counts[name] += 1
            batch_acc[name] += ms
        for name, ms in batch_acc.items():
            self.per_batch_ms[name].append(ms)
        self._pending.clear()


class WallStageTimer:
    """Accumulate host-side wall-clock per-stage time."""

    def __init__(self):
        self.totals_s: dict[str, float] = defaultdict(float)
        self.counts: dict[str, int] = defaultdict(int)
        self.per_batch_s: dict[str, list[float]] = defaultdict(list)

    @contextmanager
    def stage(self, name: str):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            dt = time.perf_counter() - t0
            self.totals_s[name] += dt
            self.counts[name] += 1
            self.per_batch_s[name].append(dt)


# --------------------------------------------------------------------------- #
# Instrumented forward pass (mirrors FingerNet.forward + postprocess)         #
# --------------------------------------------------------------------------- #


def instrumented_step(
    model_wrapper,
    batch_tensors: torch.Tensor,
    batch_orig_shapes,
    threshold: float,
    device: str,
    gtimer: CudaStageTimer,
    wtimer: WallStageTimer,
    do_save_d2h: bool = True,
):
    """Run one batch through the full pipeline, instrumented end-to-end.

    Stages mirror the production ``_run_full_gpu`` path:
      preprocess -> H2D -> img_norm -> feature_extractor -> ori_seg_head ->
      enhancement_module -> minutiae_head -> postprocess (mask, minutiae,
      orientation, enhanced) -> D2H.
    """
    fnet = model_wrapper.fingernet  # core nn.Module

    # ------------------- H2D transfer (timed in isolation) ------------------ #
    with wtimer.stage("h2d_wall"):
        with gtimer.stage("h2d_transfer"):
            x = batch_tensors.to(device, non_blocking=True)

    # ------------------- Preprocess (padding) ------------------------------- #
    with gtimer.stage("preprocess_pad"):
        _, _, h, w = x.shape
        pad_h = (8 - h % 8) % 8
        pad_w = (8 - w % 8) % 8
        x_padded = F.pad(x, (0, pad_w, 0, pad_h), mode="constant", value=0)

    # ------------------- Network ------------------------------------------- #
    with torch.inference_mode():
        with gtimer.stage("net_img_norm"):
            x_norm = fnet.img_norm(x_padded)

        with gtimer.stage("net_feature_extractor"):
            features = fnet.feature_extractor(x_norm)

        with gtimer.stage("net_ori_seg_head"):
            ori_map, seg_map = fnet.ori_seg_head(features)

        with gtimer.stage("net_enhancement"):
            enh_real, enh_phase, upsampled_ori_map = fnet.enhancement_module(
                x_padded, ori_map
            )

        with gtimer.stage("net_misc_upsample_concat"):
            upsampled_seg = F.interpolate(
                F.softsign(seg_map), scale_factor=8, mode="nearest"
            )
            minutiae_input = torch.cat([enh_phase, upsampled_seg], dim=1)

        with gtimer.stage("net_minutiae_head"):
            mnt_o, mnt_w, mnt_h, mnt_s = fnet.minutiae_head(minutiae_input, ori_map)

    # This script drives the blocks by hand to time them, so it has to reassemble
    # what FingerNet.forward returns -- including the decode's argmax (see OUTPUTS).
    bin_of = lambda t: torch.argmax(t, dim=1).to(torch.int32)
    raw_outputs = {
        "segmentation": seg_map,
        "orientation_index": bin_of(ori_map),
        "enhanced_real": enh_real,
        "minutiae_orientation_index": bin_of(mnt_o),
        "minutiae_x_index": bin_of(mnt_w),
        "minutiae_y_index": bin_of(mnt_h),
        "minutiae_score": mnt_s,
    }

    # ------------------- Post-processing (on GPU) -------------------------- #
    with gtimer.stage("post_mask_binarize"):
        cleaned_mask = _post_binarize_mask_fast(raw_outputs["segmentation"])
        cleaned_mask_up = F.interpolate(
            cleaned_mask.unsqueeze(1).float(), scale_factor=8, mode="nearest"
        ).squeeze(1)

    with gtimer.stage("post_minutiae_detect_nms"):
        final_minutiae_list = _post_detect_minutiae(
            raw_outputs, threshold, cleaned_mask
        )

    with gtimer.stage("post_orientation_field"):
        ori_idx = torch.argmax(raw_outputs["orientation"], dim=1)
        ori_idx_up = F.interpolate(
            ori_idx.unsqueeze(1).float(), scale_factor=8, mode="nearest"
        ).squeeze(1)
        orientation_field = (ori_idx_up * 2.0 - 89.0) * torch.pi / 180.0
        orientation_field = orientation_field * cleaned_mask_up

    with gtimer.stage("post_enhanced_image"):
        enh = raw_outputs["enhanced_real"].squeeze(1) * cleaned_mask_up
        b, hh, ww = enh.shape
        enh_flat = enh.view(b, -1)
        enh_min = enh_flat.min(dim=1, keepdim=True)[0]
        enh_max = enh_flat.max(dim=1, keepdim=True)[0]
        enh_norm = (enh_flat - enh_min) / (enh_max - enh_min + 1e-8)
        enh_visual = (enh_norm.view(b, hh, ww) * 255).byte()
        seg_mask_u8 = (cleaned_mask_up * 255).byte()

    # ------------------- D2H transfer -------------------------------------- #
    if do_save_d2h:
        with wtimer.stage("d2h_wall"):
            with gtimer.stage("d2h_transfer"):
                # Single batched copy of the four heavy outputs. .cpu() is
                # blocking (waits for GPU) so timing is well-defined.
                _ = enh_visual.cpu()
                _ = seg_mask_u8.cpu()
                _ = orientation_field.cpu()
                # Minutiae is a list of small tensors per image; copy each.
                _ = [m.cpu() for m in final_minutiae_list]


# --------------------------------------------------------------------------- #
# Optimization knobs                                                          #
# --------------------------------------------------------------------------- #


def apply_perf_knobs():
    """Enable safe perf knobs.

    cudnn.benchmark gives a measurable speed-up on fixed-size workloads but
    differs in algorithm choice vs. cuDNN's heuristic default; combined
    with TF32 non-associativity this flips a small fraction of pixels near
    the threshold/NMS boundary. We enable it for *profiling* only, where
    headline throughput matters; production inference (api.py) leaves it
    OFF for byte-level reproducibility across runs and machines.
    """
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.fp32_precision = "tf32"
    torch.backends.cudnn.conv.fp32_precision = "tf32"


# --------------------------------------------------------------------------- #
# Reporting                                                                   #
# --------------------------------------------------------------------------- #


STAGE_KIND = {
    "h2d_transfer": "transfer",
    "preprocess_pad": "gpu",
    "net_img_norm": "gpu",
    "net_feature_extractor": "gpu",
    "net_ori_seg_head": "gpu",
    "net_enhancement": "gpu",
    "net_misc_upsample_concat": "gpu",
    "net_minutiae_head": "gpu",
    "post_mask_binarize": "gpu",
    "post_minutiae_detect_nms": "gpu",
    "post_orientation_field": "gpu",
    "post_enhanced_image": "gpu",
    "d2h_transfer": "transfer",
}

NETWORK_STAGES = [
    "net_img_norm",
    "net_feature_extractor",
    "net_ori_seg_head",
    "net_enhancement",
    "net_misc_upsample_concat",
    "net_minutiae_head",
]
POSTPROC_STAGES = [
    "post_mask_binarize",
    "post_minutiae_detect_nms",
    "post_orientation_field",
    "post_enhanced_image",
]
TRANSFER_STAGES = ["h2d_transfer", "d2h_transfer"]


def percentile(values, q):
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * q
    f, c = int(k), min(int(k) + 1, len(s) - 1)
    if f == c:
        return s[f]
    return s[f] + (s[c] - s[f]) * (k - f)


def format_report(
    gtimer: CudaStageTimer,
    wtimer: WallStageTimer,
    num_images: int,
    num_batches: int,
    batch_size: int,
    total_wall_s: float,
    config: dict,
) -> str:
    lines = []
    lines.append("# FingerNet Inference Profile\n")
    lines.append("## Configuration\n")
    for k, v in config.items():
        lines.append(f"- **{k}**: `{v}`")
    lines.append("")
    lines.append("## Throughput\n")
    images_per_s = num_images / total_wall_s if total_wall_s > 0 else 0.0
    lines.append(f"- Images processed: **{num_images}**")
    lines.append(f"- Batches: **{num_batches}** (batch_size={batch_size})")
    lines.append(f"- Total wall time: **{total_wall_s:.2f} s**")
    lines.append(f"- Throughput: **{images_per_s:.1f} images/s**")
    lines.append(
        f"- Mean per-batch wall time: **{1000 * total_wall_s / num_batches:.2f} ms/batch** "
        f"({1000 * total_wall_s / num_images:.2f} ms/image)"
    )
    lines.append("")

    # ---- Per-stage table (GPU events) ----
    total_gpu_ms = sum(gtimer.totals_ms.get(s, 0.0) for s in STAGE_KIND)
    lines.append("## Stage breakdown (CUDA events)\n")
    lines.append(
        "| Stage | Kind | Total (s) | Mean / batch (ms) | Mean / image (ms) | "
        "p50 (ms) | p95 (ms) | % of GPU total |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|")

    def emit_row(name: str, kind: str):
        total_ms = gtimer.totals_ms.get(name, 0.0)
        per_batch = gtimer.per_batch_ms.get(name, [])
        mean_b = total_ms / max(num_batches, 1)
        mean_i = total_ms / max(num_images, 1)
        p50 = percentile(per_batch, 0.5)
        p95 = percentile(per_batch, 0.95)
        pct = 100 * total_ms / total_gpu_ms if total_gpu_ms > 0 else 0.0
        lines.append(
            f"| `{name}` | {kind} | {total_ms / 1000:.3f} | {mean_b:.3f} | "
            f"{mean_i:.3f} | {p50:.3f} | {p95:.3f} | {pct:.1f}% |"
        )

    lines.append(f"| **Network forward** | gpu | | | | | | |")
    for s in NETWORK_STAGES:
        emit_row(s, STAGE_KIND[s])
    lines.append(f"| **Post-processing** | gpu | | | | | | |")
    for s in POSTPROC_STAGES:
        emit_row(s, STAGE_KIND[s])
    lines.append(f"| **Transfers** | transfer | | | | | | |")
    for s in TRANSFER_STAGES:
        emit_row(s, STAGE_KIND[s])
    lines.append(f"| **Other** | gpu | | | | | | |")
    emit_row("preprocess_pad", "gpu")
    lines.append("")

    # ---- Aggregate buckets ----
    net_total = sum(gtimer.totals_ms.get(s, 0.0) for s in NETWORK_STAGES)
    post_total = sum(gtimer.totals_ms.get(s, 0.0) for s in POSTPROC_STAGES)
    transfer_total = sum(gtimer.totals_ms.get(s, 0.0) for s in TRANSFER_STAGES)
    other_total = gtimer.totals_ms.get("preprocess_pad", 0.0)
    lines.append("## Aggregate buckets\n")
    lines.append(
        "| Bucket | Total (s) | Mean / image (ms) | % of GPU total |"
    )
    lines.append("|---|---:|---:|---:|")
    for label, total in [
        ("Network forward", net_total),
        ("Post-processing", post_total),
        ("H2D + D2H transfers", transfer_total),
        ("Other (padding)", other_total),
        ("**GPU total**", total_gpu_ms),
    ]:
        pct = 100 * total / total_gpu_ms if total_gpu_ms > 0 else 0.0
        lines.append(
            f"| {label} | {total / 1000:.3f} | "
            f"{total / max(num_images, 1):.3f} | {pct:.1f}% |"
        )
    lines.append("")
    lines.append(
        f"> **GPU total = {total_gpu_ms / 1000:.3f} s vs Wall total = {total_wall_s:.3f} s.** "
        "Difference is data loading + Python overhead + sync waits."
    )

    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True, help="Image directory or .txt list")
    p.add_argument("--num-images", type=int, default=1000)
    p.add_argument("-b", "--batch-size", type=int, default=32)
    p.add_argument("-j", "--num-workers", type=int, default=8)
    p.add_argument("--max-dim", type=int, default=1024)
    p.add_argument("--threshold", type=float, default=0.05)
    p.add_argument("--warmup-batches", type=int, default=5)
    p.add_argument("--weights", default=DEFAULT_WEIGHTS_PATH)
    p.add_argument("--device", default="cuda:0")
    p.add_argument(
        "--no-d2h",
        action="store_true",
        help="Skip D2H copy (isolate pure GPU pipeline)",
    )
    p.add_argument(
        "--no-perf-knobs",
        action="store_true",
        help="Disable cudnn.benchmark / TF32 to measure baseline",
    )
    p.add_argument("--trace", default=None, help="torch.profiler Chrome trace path")
    p.add_argument(
        "--trace-batches",
        type=int,
        default=5,
        help="How many batches to capture in the Chrome trace",
    )
    p.add_argument("--json", default=None, help="Dump raw measurements as JSON")
    p.add_argument("--report", default=None, help="Write markdown report to this file")
    args = p.parse_args()

    if not args.no_perf_knobs:
        apply_perf_knobs()

    # ------------------- Build dataset / dataloader ----------------------- #
    paths = find_image_paths(args.input, recursive=True)
    if args.num_images and args.num_images < len(paths):
        paths = paths[: args.num_images]
    logger.info(f"Profiling on {len(paths)} images")

    dataset = FingerprintDataset(paths, max_dim=args.max_dim)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=(args.num_workers > 0),
        collate_fn=dynamic_padding_collate,
        shuffle=False,
    )

    # ------------------- Load model ---------------------------------------- #
    model = get_fingernet(weights_path=args.weights, device=args.device)
    model.eval()

    # ------------------- Warmup ------------------------------------------- #
    logger.info(f"Warming up: {args.warmup_batches} batches")
    warm_iter = iter(dataloader)
    with torch.inference_mode():
        for _ in range(args.warmup_batches):
            try:
                bt, _, _ = next(warm_iter)
            except StopIteration:
                break
            if bt is None:
                continue
            x = bt.to(args.device, non_blocking=True)
            _ = model(x, minutiae_threshold=args.threshold)
    torch.cuda.synchronize()

    # Optional Chrome trace pass on a small window. Captures kernels +
    # memcopies separately so you can confirm overlap in chrome://tracing.
    if args.trace:
        Path(os.path.dirname(args.trace) or ".").mkdir(parents=True, exist_ok=True)
        logger.info(f"Recording torch.profiler trace -> {args.trace}")
        trace_iter = iter(dataloader)
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=True,
            with_stack=False,
            profile_memory=False,
        ) as prof:
            with torch.inference_mode():
                for _ in range(args.trace_batches):
                    try:
                        bt, _, shp = next(trace_iter)
                    except StopIteration:
                        break
                    if bt is None:
                        continue
                    x = bt.to(args.device, non_blocking=True)
                    out = model(x, minutiae_threshold=args.threshold)
                    # force D2H so the copy is in the trace
                    _ = out["enhanced_image"].cpu()
            torch.cuda.synchronize()
        prof.export_chrome_trace(args.trace)
        # Also print top kernels
        print("\n--- top 25 CUDA ops ---")
        print(
            prof.key_averages().table(
                sort_by="self_cuda_time_total", row_limit=25
            )
        )

    # ------------------- Profiled run ------------------------------------- #
    gtimer = CudaStageTimer()
    wtimer = WallStageTimer()
    num_images = 0
    num_batches = 0

    torch.cuda.synchronize()
    t_start = time.perf_counter()

    for batch_tensors, batch_paths, batch_orig_shapes in dataloader:
        if batch_tensors is None:
            continue
        instrumented_step(
            model,
            batch_tensors,
            batch_orig_shapes,
            threshold=args.threshold,
            device=args.device,
            gtimer=gtimer,
            wtimer=wtimer,
            do_save_d2h=not args.no_d2h,
        )
        gtimer.flush()
        num_images += batch_tensors.shape[0]
        num_batches += 1

    torch.cuda.synchronize()
    total_wall = time.perf_counter() - t_start

    # ------------------- Report ------------------------------------------- #
    config = {
        "device": torch.cuda.get_device_name(args.device),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "threshold": args.threshold,
        "perf_knobs": (not args.no_perf_knobs),
        "d2h_included": (not args.no_d2h),
        "warmup_batches": args.warmup_batches,
    }
    report = format_report(
        gtimer, wtimer, num_images, num_batches, args.batch_size, total_wall, config
    )
    print("\n" + report)
    if args.report:
        Path(os.path.dirname(args.report) or ".").mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(report)
        logger.info(f"Report written -> {args.report}")

    if args.json:
        Path(os.path.dirname(args.json) or ".").mkdir(parents=True, exist_ok=True)
        out = {
            "config": config,
            "num_images": num_images,
            "num_batches": num_batches,
            "total_wall_s": total_wall,
            "gpu_totals_ms": dict(gtimer.totals_ms),
            "gpu_per_batch_ms": {k: list(v) for k, v in gtimer.per_batch_ms.items()},
            "wall_totals_s": dict(wtimer.totals_s),
        }
        Path(args.json).write_text(json.dumps(out, indent=2))
        logger.info(f"Raw JSON -> {args.json}")


if __name__ == "__main__":
    main()
