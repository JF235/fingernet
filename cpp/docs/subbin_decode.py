#!/usr/bin/env python3
"""Textbook diagram: how FingerNet turns dense stride-8 heads into one minutia.

Three steps, left to right: WHICH block (score) -> WHERE inside it (two 8-bin
offset heads) -> WHICH direction (180 orientation bins). This is Block E of
[cpp/include/fingernet/postproc.hpp](../include/fingernet/postproc.hpp), mirroring
`_post_detect_minutiae_single` in [pytorch/fingernet/wrapper.py](../../pytorch/fingernet/wrapper.py):

    x   = col*8 + argmax(minutiae_x_offset[:, row, col])              # 8 bins
    y   = row*8 + argmax(minutiae_y_offset[:, row, col])              # 8 bins
    ang = (argmax(minutiae_orientation[:, row, col])*2 - 89) * pi/180  # 180 bins

The worked example is real, not invented: the numbers below were measured by
running `fingernet_mnt_trimmed_dyn.onnx` (ONNX, CPU) on SD258 186/01, taking the
highest-scoring surviving minutia. Embedding them keeps this script a pure figure
generator -- matplotlib + numpy, no model, no dataset.

Usage:  python3 subbin_decode.py [-o subbin_decode.png]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle, Wedge

# --- worked example, measured (see docstring) --------------------------------
ROW, COL = 35, 53
SCORES3 = [[0.236, 0.537, 0.075],          # the 3x3 of minutiae_score around the cell
           [0.982, 0.988, 0.099],
           [0.981, 0.972, 0.068]]
XOFF = [0.735, 0.830, 0.794, 0.656, 0.447, 0.260, 0.144, 0.072]
YOFF = [0.151, 0.257, 0.394, 0.562, 0.684, 0.736, 0.717, 0.596]
ORI = [                                     # all 180 bins, 2 degrees each
    0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000,
    0.000, 0.000, 0.001, 0.001, 0.002, 0.002, 0.003, 0.004, 0.005, 0.005, 0.004, 0.003,
    0.002, 0.001, 0.001, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000,
    0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000,
    0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000,
    0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000,
    0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000,
    0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000,
    0.000, 0.001, 0.001, 0.002, 0.004, 0.006, 0.012, 0.021, 0.048, 0.084, 0.147, 0.243,
    0.376, 0.463, 0.509, 0.522, 0.469, 0.417, 0.299, 0.239, 0.144, 0.100, 0.061, 0.034,
    0.019, 0.013, 0.008, 0.006, 0.004, 0.003, 0.002, 0.001, 0.001, 0.000, 0.000, 0.000,
    0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000,
    0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000,
    0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000,
    0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000,
]
ORI_BIN = int(np.argmax(ORI))
THRESHOLD = 0.05

# --- palette (validated: CVD ΔE 9.2 / normal-vision 24.0, all pairs, light) ---
C_X = "#2a78d6"     # everything about the x axis
C_Y = "#eb6834"     # everything about the y axis
C_A = "#1baf7a"     # everything about the angle
INK = "#0b0b0b"
INK_2 = "#52514e"
INK_3 = "#8a8984"
LINE = "#c9c8c3"
SURFACE = "#fcfcfb"


def tint(hex_color: str, a: float) -> tuple:
    """Blend a hex colour toward the surface -- a light fill of the same hue."""
    c = np.array([int(hex_color[i:i + 2], 16) for i in (1, 3, 5)]) / 255
    s = np.array([int(SURFACE[i:i + 2], 16) for i in (1, 3, 5)]) / 255
    return tuple(s + (c - s) * a)


def step_header(ax, x, y, n, title):
    ax.add_patch(Circle((x + 1.5, y + 0.4), 1.5, fc=INK, ec="none", zorder=3))
    ax.text(x + 1.5, y + 0.4, str(n), color=SURFACE, fontsize=11, fontweight="bold",
            ha="center", va="center", zorder=4)
    ax.text(x + 4.2, y + 0.4, title, color=INK, fontsize=12.5, fontweight="bold",
            ha="left", va="center")


def arrow(ax, x0, x1, y):
    ax.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle="-|>", mutation_scale=16,
                                 color=INK_3, lw=1.4, shrinkA=0, shrinkB=0))


# ---- layout: one place, so nothing collides by accident ---------------------
Y_LABEL, Y_LABEL_SUB = 77.0, 74.4      # the "<head name>" / caption line of each step
Y_ARROW = 52.0                         # the two arrows between steps
Y_FORMULA = 21.5                       # the result line of steps 2 and 3
S1_X, S2_X = 4.0, 52.0                 # left edge of steps 1 and 2
S3_CX, S3_CY, S3_R = 127.0, 57.5, 10.5  # dial centre and radius


# ---------------------------------------------------------------- step 1
def draw_step1(ax):
    """The coarse grid: one cell per 8x8 px block; the cell whose score fires."""
    cs, top = 7.2, 66.0
    for i in range(3):
        for j in range(3):
            v = SCORES3[i][j]
            ax.add_patch(Rectangle((S1_X + j * cs, top - (i + 1) * cs), cs, cs,
                                   fc=tint(C_X, v), ec=LINE, lw=1.0))
            ax.text(S1_X + j * cs + cs / 2, top - (i + 0.5) * cs, f"{v:.2f}",
                    ha="center", va="center", fontsize=8.5,
                    color=INK if v > THRESHOLD else INK_3)
    ax.add_patch(Rectangle((S1_X + cs, top - 2 * cs), cs, cs, fill=False, ec=C_X,
                           lw=2.6, zorder=3))
    ax.text(S1_X + 1.5 * cs, top - 3 * cs - 2.0, f"row {ROW}, col {COL}",
            ha="center", va="top", fontsize=9, color=C_X, fontweight="bold")

    ax.text(S1_X, Y_LABEL, "minutiae_score", fontsize=9.5, color=INK,
            family="monospace")
    ax.text(S1_X, Y_LABEL_SUB, f"> {THRESHOLD}  ·  one value per 8×8 px block",
            fontsize=8.5, color=INK_2, va="top")


# ---------------------------------------------------------------- step 2
def draw_step2(ax):
    """The 8x8 block: two argmaxes pick a column and a row -> one pixel."""
    px = 3.4                                   # one image pixel, in figure units
    bx, by = S2_X + 13.0, 58.0                 # top-left corner of the block
    xo, yo = int(np.argmax(XOFF)), int(np.argmax(YOFF))
    BH = 8.0                                   # bar length at activation 1.0

    # x_offset bars, one per column, standing on top of the block
    for k, v in enumerate(XOFF):
        ax.add_patch(Rectangle((bx + k * px + 0.5, by + 2.0), px - 1.0, v * BH,
                               fc=C_X if k == xo else tint(C_X, 0.30), ec="none"))
    ax.text(bx, Y_LABEL, "minutiae_x_offset", fontsize=9, color=INK, family="monospace")
    ax.text(bx, Y_LABEL_SUB, "8 bins, one per column", fontsize=8, color=INK_2, va="top")
    ax.text(bx + xo * px + px / 2, by + 2.0 + XOFF[xo] * BH + 0.5, f"argmax = {xo}",
            ha="center", va="bottom", fontsize=8.5, color=C_X, fontweight="bold")

    # y_offset bars, one per row, hanging off the left edge of the block
    for k, v in enumerate(YOFF):
        ax.add_patch(Rectangle((bx - 1.6 - v * BH, by - (k + 1) * px + 0.5), v * BH,
                               px - 1.0, fc=C_Y if k == yo else tint(C_Y, 0.30),
                               ec="none"))
    # caption in the empty corner left of the x-bars, right-aligned onto the strip
    ax.text(bx - 1.6, by + 9.4, "minutiae_y_offset", fontsize=9, color=INK,
            family="monospace", ha="right")
    ax.text(bx - 1.6, by + 6.9, "8 bins, one per row", fontsize=8, color=INK_2,
            ha="right")
    ax.text(bx - 1.6 - YOFF[yo] * BH - 0.8, by - (yo + 0.5) * px, f"argmax = {yo}",
            ha="right", va="center", fontsize=8.5, color=C_Y, fontweight="bold")

    # the block itself: the winning column and row, and the pixel they intersect
    ax.add_patch(Rectangle((bx + xo * px, by - 8 * px), px, 8 * px,
                           fc=tint(C_X, 0.22), ec="none"))
    ax.add_patch(Rectangle((bx, by - (yo + 1) * px), 8 * px, px,
                           fc=tint(C_Y, 0.22), ec="none"))
    for k in range(9):
        ax.plot([bx + k * px] * 2, [by, by - 8 * px], color=LINE, lw=0.9, zorder=2)
        ax.plot([bx, bx + 8 * px], [by - k * px] * 2, color=LINE, lw=0.9, zorder=2)
    ax.add_patch(Rectangle((bx + xo * px, by - (yo + 1) * px), px, px, fc=C_A,
                           ec="none", zorder=3))
    for k in range(8):
        ax.text(bx + k * px + px / 2, by - 8 * px - 1.2, str(k), ha="center", va="top",
                fontsize=7.5, color=C_X if k == xo else INK_3)
        ax.text(bx + 8 * px + 1.2, by - k * px - px / 2, str(k), ha="left", va="center",
                fontsize=7.5, color=C_Y if k == yo else INK_3)
    ax.text(bx + 4 * px, by - 8 * px - 4.8, "the 8×8 px block", ha="center", va="top",
            fontsize=8.5, color=INK_2)

    ax.text(S2_X + 6, Y_FORMULA,
            f"x = col·8 + {xo} = {COL * 8 + xo}\n"
            f"y = row·8 + {yo} = {ROW * 8 + yo}",
            fontsize=11.5, color=INK, family="monospace", va="top", linespacing=1.7)


# ---------------------------------------------------------------- step 3
def draw_step3(ax):
    """180 bins of 2 degrees; the -89 is half a bin, i.e. the bin centre."""
    cx, cy, R = S3_CX, S3_CY, S3_R
    deg = ORI_BIN * 2.0 - 89.0                     # bin centre, model's internal sense
    # The dial is drawn in the model's INTERNAL sense, which runs clockwise:
    # display angle = -value. (The .min writer negates this to reach the project's
    # CCW convention -- the seam where the mufis adapter has to convert.)
    # a ROTATIONAL HISTOGRAM: one 2-degree wedge per bin, length = its activation.
    # The dial runs CLOCKWISE because that is the model's internal sense (the .min
    # writer negates it to reach the project's CCW convention).
    r0 = 2.6                                        # inner hole, so the bins read as bars
    vmax = max(ORI)
    ax.add_patch(Circle((cx, cy), R, fc=SURFACE, ec=LINE, lw=1.0, zorder=1))
    ax.add_patch(Circle((cx, cy), r0, fc=SURFACE, ec=LINE, lw=0.8, zorder=4))
    for k, v in enumerate(ORI):
        rr = r0 + (v / vmax) * (R - r0)
        ax.add_patch(Wedge((cx, cy), rr, -(2 * k - 88.0), -(2 * k - 90.0), width=rr - r0,
                           fc=C_A if k == ORI_BIN else tint(C_A, 0.45), ec="none",
                           zorder=2 + (k == ORI_BIN)))
    for lab in (-89, 1, 91, 181):                   # a few bin centres, labelled
        a = np.deg2rad(-lab)
        ax.plot([cx + R * np.cos(a), cx + (R + 1.4) * np.cos(a)],
                [cy + R * np.sin(a), cy + (R + 1.4) * np.sin(a)], color=INK_3, lw=1.0)
        ax.text(cx + (R + 3.2) * np.cos(a), cy + (R + 3.2) * np.sin(a), f"{lab}°",
                ha="center", va="center", fontsize=7.5, color=INK_3)
    a = np.deg2rad(-deg)
    ax.annotate(f"argmax = bin {ORI_BIN}",
                (cx + R * np.cos(a), cy + R * np.sin(a)), textcoords="offset points",
                xytext=(-16, -12), ha="center", fontsize=8.5, color=C_A,
                fontweight="bold")
    ax.text(cx, Y_LABEL, "minutiae_orientation", fontsize=9, color=INK,
            family="monospace", ha="center")
    ax.text(cx, Y_LABEL_SUB, "180 bins × 2° = 360°", fontsize=8.5, color=INK_2,
            ha="center", va="top")

    # the number line: why the constant is -89 and not -90
    lx, ly, w = cx - 17.0, 34.0, 34.0
    lo = ORI_BIN - 2
    ax.plot([lx, lx + w], [ly, ly], color=INK_3, lw=1.1)
    for i in range(5):
        b0 = 2 * (lo + i) - 90.0
        px_ = lx + w * i / 4.0
        ax.plot([px_, px_], [ly - 1.0, ly + 1.0], color=INK_3, lw=1.1)
        ax.text(px_, ly - 1.8, f"{b0:.0f}°", ha="center", va="top", fontsize=7,
                color=INK_3)
        if i < 4:
            c = px_ + w / 8.0
            k = lo + i
            hit = k == ORI_BIN
            ax.plot([c], [ly], "|", ms=7, mew=1.6, color=C_A if hit else LINE)
            ax.text(c, ly + 1.6, f"{2 * k - 89:.0f}°", ha="center", va="bottom",
                    fontsize=8 if hit else 7, color=C_A if hit else INK_3,
                    fontweight="bold" if hit else "normal")
            if hit:
                ax.text(c, ly - 4.6, f"bin {k}", ha="center", va="top", fontsize=7.5,
                        color=C_A, fontweight="bold")
    ax.text(lx, ly + 6.0, "the −89 is the bin CENTRE", fontsize=8.5, color=INK_2,
            ha="left")

    ax.text(lx, Y_FORMULA,
            f"θ = (k·2 − 89)° = {deg:.0f}°  cw\n"
            f"                 = {(-deg) % 360:.0f}°  ccw",
            fontsize=11.5, color=INK, family="monospace", va="top", linespacing=1.7)


# ---------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--out",
                    default=str(Path(__file__).with_name("subbin_decode.png")))
    a = ap.parse_args()

    fig, ax = plt.subplots(figsize=(15.0, 9.6))
    fig.patch.set_facecolor(SURFACE)
    ax.set_xlim(0, 150)
    ax.set_ylim(6, 96)
    ax.set_aspect("equal")
    ax.axis("off")

    ax.text(2, 93.0, "FingerNet sub-bin decode", fontsize=17, color=INK,
            fontweight="bold", va="top")
    ax.text(2, 88.6, "three argmaxes rebuild one minutia from dense stride-8 maps",
            fontsize=11, color=INK_2, va="top")

    step_header(ax, 2, 82.0, 1, "Which block?")
    step_header(ax, S2_X, 82.0, 2, "Where inside the block?")
    step_header(ax, 106, 82.0, 3, "Which direction?")

    draw_step1(ax)
    draw_step2(ax)
    draw_step3(ax)

    arrow(ax, 42.0, 50.0, Y_ARROW)
    arrow(ax, 100.0, 108.0, Y_ARROW)

    ax.text(2, 9.0, "position is a classification over 8 offsets, not a regression — "
            "so every coordinate is an integer", fontsize=10, color=INK_2, va="top")

    fig.savefig(a.out, dpi=170, bbox_inches="tight", facecolor=SURFACE)
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
