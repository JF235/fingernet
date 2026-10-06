#!/usr/bin/env python3
"""Test whether sub-pixel parabolic peak refinement de-grids the minutiae coords
WITHOUT retraining — on the stitchv2_b (bilinear) mnt_h heatmaps that show the
worst 4 px lattice.

Mechanism under test: the decode (postprocess.extract_minutiae) takes integer
local-max pixel positions. The heatmap's real information sits on a coarse ~4 px
node lattice; a smooth (bilinear) upsample puts every local maximum exactly on a
node, so integer NMS snaps every minutia onto the 4 px grid. A quadratic fit on
the 3x3 neighbourhood of each peak recovers the continuous sub-pixel peak between
nodes, de-snapping the coordinates. We compare the placement-density grid of the
INTEGER coords vs the SUB-PIXEL-then-rounded coords over the same peaks.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import maximum_filter

sys.path.insert(0, "/home/jcontreras/work/mntstitch/inference/python/tests")
from minutiae_density import detect_grid, detect_peaks  # noqa: E402

HEAT = "/home/jcontreras/work/mntstitch/training/datasets/BN48k/out/stitchv2_b/mnt_h"
SIZE = 512
LIMIT = 3000
THRESH = 0.2 * 255  # PNGs are 0..255; extract_minutiae uses 0.2 on [0,1]
MINDIST = 11


def parabolic_offset(a, b, c):
    """Sub-pixel offset of the peak of the parabola through (-1,a),(0,b),(1,c)."""
    denom = a - 2.0 * b + c
    off = np.where(np.abs(denom) > 1e-6, 0.5 * (a - c) / denom, 0.0)
    return np.clip(off, -0.5, 0.5)


def main() -> None:
    paths = sorted(Path(HEAT).rglob("*.png"))[:LIMIT]
    print(f"{len(paths)} heatmaps from {HEAT}")
    f_int = np.zeros((SIZE, SIZE), dtype=np.float64)
    f_sub = np.zeros((SIZE, SIZE), dtype=np.float64)
    ns = MINDIST | 1
    n = 0
    for p in paths:
        h = cv2.imread(str(p), cv2.IMREAD_UNCHANGED).astype(np.float32)
        if h.shape != (SIZE, SIZE):
            continue
        hmax = maximum_filter(h, size=ns, mode="nearest")
        ys, xs = np.nonzero((h == hmax) & (h > THRESH))
        m = (ys > 0) & (ys < SIZE - 1) & (xs > 0) & (xs < SIZE - 1)
        ys, xs = ys[m], xs[m]
        if ys.size == 0:
            continue
        # integer coords
        for y, x in zip(ys, xs):
            f_int[y, x] += 1.0
        # sub-pixel parabolic refine along x and y, then round to int
        dx = parabolic_offset(h[ys, xs - 1], h[ys, xs], h[ys, xs + 1])
        dy = parabolic_offset(h[ys - 1, xs], h[ys, xs], h[ys + 1, xs])
        rx = np.clip(np.round(xs + dx).astype(int), 0, SIZE - 1)
        ry = np.clip(np.round(ys + dy).astype(int), 0, SIZE - 1)
        for y, x in zip(ry, rx):
            f_sub[y, x] += 1.0
        n += 1

    for name, field in (("INTEGER (current decode)", f_int),
                        ("SUB-PIXEL parabolic", f_sub)):
        info, spec = detect_grid(field)
        peaks = detect_peaks(spec, field.shape, prominence=6.0, max_peaks=4)
        pk = ", ".join(f"{next((q for q in p_['period_px'] if q),'?')}px@{p_['prominence']}x"
                       for p_ in peaks) or "none"
        print(f"\n{name}: {n} imgs")
        print(f"  max_modulation = {info['max_modulation']:.5f}  grid={info['is_grid']}")
        print(f"  y={info['y_axis']['modulation']:.5f}({info['y_axis']['grid_px']}px) "
              f"x={info['x_axis']['modulation']:.5f}({info['x_axis']['grid_px']}px)")
        print(f"  peaks: {pk}")


if __name__ == "__main__":
    main()
