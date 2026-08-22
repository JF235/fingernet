#!/usr/bin/env python3
"""Does FingerNet's minutiae output carry a periodic grid, and how many .min
files are needed before it emerges from the placement-density spectrum?

FingerNet decodes minutiae coords as (wrapper.py _post_detect_minutiae_single):
    x = col*8 + argmax(x_offset[0..7]),  y = row*8 + argmax(y_offset[0..7])
i.e. a coarse stride-8 grid cell plus an integer sub-cell offset chosen by an
argmax over 8 discrete bins. Any non-uniformity in the offset distribution is a
period-8 lattice (with a 4 px harmonic) baked into the coordinates by the
architecture -- before any NMS. This script measures it on BN48k.

Reuses detect_grid / detect_peaks from the mntstitch density test (the NIST
MINEX III grid detector), accumulating the density field once and snapshotting
the spectrum at log-spaced file counts to locate the emergence threshold.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

MNT = "/home/jcontreras/work/mntstitch/inference/python/src"
TST = "/home/jcontreras/work/mntstitch/inference/python/tests"
sys.path.insert(0, MNT)
sys.path.insert(0, TST)
from mntstitch.io import load_minutiae  # noqa: E402
from minutiae_density import detect_grid, detect_peaks  # noqa: E402

MIN_ROOT = "/home/jcontreras/work/mntstitch/training/datasets/BN48k/out/fingernet/minutiae"
SIZE = 512
CHECKPOINTS = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 30000]


def main() -> None:
    paths = sorted(Path(MIN_ROOT).rglob("*.min"))
    print(f"found {len(paths)} .min files under {MIN_ROOT}")
    limit = CHECKPOINTS[-1]
    paths = paths[:limit]

    field = np.zeros((SIZE, SIZE), dtype=np.float64)
    # modulo accumulators over ALL loaded minutiae (architectural fingerprint)
    mod8x = np.zeros(8, dtype=np.int64); mod8y = np.zeros(8, dtype=np.int64)
    n_min = 0
    cp_set = set(CHECKPOINTS)

    print(f"\n{'files':>7} {'minutiae':>10} {'max_mod':>9} {'grid?':>6} "
          f"{'y_mod':>8} {'y_px':>6} {'x_mod':>8} {'x_px':>6} {'peaks(period px)'}")
    for i, p in enumerate(paths, 1):
        for m in load_minutiae(p):
            if 0 <= m.x < SIZE and 0 <= m.y < SIZE:
                field[m.y, m.x] += 1.0
                mod8x[m.x % 8] += 1; mod8y[m.y % 8] += 1
                n_min += 1
        if i in cp_set:
            info, spec = detect_grid(field)
            peaks = detect_peaks(spec, field.shape, prominence=6.0, max_peaks=6)
            pk = ",".join(
                str(next((q for q in p_["period_px"] if q), "?")) for p_ in peaks
            ) or "-"
            print(f"{i:>7} {n_min:>10} {info['max_modulation']:>9.5f} "
                  f"{str(info['is_grid']):>6} "
                  f"{info['y_axis']['modulation']:>8.5f} "
                  f"{str(info['y_axis']['grid_px']):>6} "
                  f"{info['x_axis']['modulation']:>8.5f} "
                  f"{str(info['x_axis']['grid_px']):>6} {pk}")

    print(f"\n--- coordinate modulo-8 histogram over {n_min} minutiae ---")
    print(f"  x%8: {(mod8x / mod8x.mean()).round(3).tolist()}  (max/min {mod8x.max()/mod8x.min():.2f})")
    print(f"  y%8: {(mod8y / mod8y.mean()).round(3).tolist()}  (max/min {mod8y.max()/mod8y.min():.2f})")
    m4x = mod8x.reshape(2, 4).sum(0); m4y = mod8y.reshape(2, 4).sum(0)
    print(f"  x%4: {(m4x / m4x.mean()).round(3).tolist()}  (max/min {m4x.max()/m4x.min():.2f})")
    print(f"  y%4: {(m4y / m4y.mean()).round(3).tolist()}  (max/min {m4y.max()/m4y.min():.2f})")


if __name__ == "__main__":
    main()
