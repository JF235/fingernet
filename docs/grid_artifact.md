# Grid artifact in minutiae outputs — findings & open hypotheses

Investigation of the periodic lattice (NIST MINEX III "grid detector") seen in
minutiae placement density, both in extractor labels (FingerNet) and in trained
mntstitch models. Detector + measurements: `mntstitch .../tests/minutiae_density.py`;
experiment scripts: `fingernet/docs/fingernet_grid.py`, `fingernet/docs/subpixel_fix.py`.

## What is measured

`max_modulation` = strongest non-DC peak of the FFT of the (x,y) placement-density
field, normalized by total mass. NIST flags a grid at ≥ 0.002. Period = pixel
spacing of the lattice. All on BN48k.

| source | max_mod | period | note |
|---|---|---|---|
| FingerNet labels | **~0.55** | **8 px** | hard; present at N=1 image |
| stitchv2 points (nearest upsample) | 0.072 | 4 px | |
| stitchv2_b points (bilinear upsample) | **0.334** | 4 px | ~5× WORSE than nearest |
| stitchv2 mnt_h heatmap (model side) | 0.0040 | 4 px | |
| stitchv2_b mnt_h heatmap (model side) | 0.0070 | 4 px | + new 2 px peak |
| bilinear + sub-pixel parabolic decode | 0.331 | — | no help |
| bilinear + Gaussian blur σ=2.5 before NMS | 0.111 | — | partial, kills precision, still > nearest |

## Confirmed so far

- **FingerNet decode is a hard period-8 grid.** Coords are `col*8 + argmax(x_off[0..7])`
  (stride-8 minutiae head + argmax over 8 discrete offset bins). Offset bins 0 and 7
  are never chosen. Deterministic per image (full strength at N=1, not a statistical
  aggregate). FingerNet is one of the label sources for stitch training.

- **Bilinear upsampling is strictly worse than nearest for the stitch model** (0.072 →
  0.334 in points; 0.0040 → 0.0070 in the heatmap). This refutes the in-code assumption
  ([tree_unet.py:147-154](../../mntstitch/training/training_template/src/model/tree_unet.py))
  that bilinear "removes the MINEX-like grid".

- **The grid is a model-side phase-lock, not a decode artifact.** mnt_h's real
  information sits on a coarse ~4 px node lattice. Bilinear is a faithful convex
  interpolator → every local max lands exactly on a node at a fixed sub-cell phase →
  integer-NMS coords snap hard to the 4 px lattice. Nearest makes flat plateaus and
  `_collapse_peaks` takes their centroid, which *dithers* coords off-node → weaker grid.
  Nearest is the "lesser evil" precisely because it is a worse interpolator.

- **Sub-pixel refinement cannot fix it.** Parabolic offsets are large (|dx|≈0.28) yet the
  *continuous* positions `(x+dx) mod 4` are still gridded — bilinear max/min 12.3 (refine
  concentrates MORE on the node), nearest 2.3. The periodicity is in the trained heatmap
  values themselves, anchored to input pixel 0 (so every image shares the phase and it
  aggregates into a clean lattice). Signature of the classic CNN upsampling
  gridding/checkerboard artifact (Odena et al. 2016).

- **Period is a fingerprint of origin:** FingerNet = 8 px (argmax of 8 offset bins);
  stitch model = 4 px (decoder upsampling phase). The mismatch + the rotation
  augmentation argue the model's 4 px grid is its own, not inherited from FingerNet's
  8 px labels — but this is not yet proven (see hypotheses).

## Open hypotheses (per pipeline stage) — NOT yet investigated

Pipeline: `extractors → per-source .min → fusion (fusemnt) → grid target H/U/V → patch
sampling+augmentation (dataloader) → TreeBranchingUNet → dense mnt_h output → extract_minutiae`.

1. **Annotators / extractors.** Each source extractor may quantize coords on its own
   lattice. FingerNet = 8 px (confirmed). Verifinger / Innovatrics / pyfing / manual:
   unknown — measure each `labels/minutiae/<src>` separately. If a source grid survives
   fusion, the target inherits it.

2. **Target-generation algorithm (fusemnt).** Minutiae are quantized to integer grid bins
   by `coords_to_idx` (`round(x/dx − 0.5)`) on a `(H_g, W_g, T_g)` lattice *before* the
   Gaussian is splatted ([fusemnt.py:43-46](../../mntstitch/training/lib/mntstitch_training/fusemnt.py)).
   If `(H_g, W_g)` = 128 (stride 4), every target peak sits on a 4 px node → the model is
   trained to reproduce a 4 px lattice. Period matches the observed 4 px. **Strong
   candidate.** Check the configured `grid_dims`.

3. **Target format.** Whether the coarse-grid target is upsampled to full res to supervise
   the full-res `mnt_h` (and with which interpolation), and the splat `sigma_bins`: a small
   sigma makes near-delta peaks pinned to nodes; a large sigma may wash the lattice. Also
   whether the H loss is effectively computed at grid res (the model code comments claim
   "loss avg-pooled to grid res" while [loss.py](../../mntstitch/training/training_template/src/objective/loss.py)
   only avg-pools when pred is coarser than target — reconcile).

4. **Training: sampling / dataloader / augmentation.** Patches are cut with a centered
   affine rotated by −phi (continuous degrees,
   [patch_ops.py](../../mntstitch/training/lib/mntstitch_training/patch_ops.py)), which
   *should* smear axis-aligned grids. Risks that defeat this: the warp's resample
   interpolation re-quantizing the H target onto the patch pixel lattice; rotation/offset
   distributions that are secretly quantized; crop offsets that are grid-aligned; the
   target re-rendered on the patch grid after warp (re-introducing #2 per patch).

5. **Model architecture.** Decoder upsampling phase-lock / checkerboard (Odena 2016):
   strided upsampling makes the response phase-dependent w.r.t. the input lattice. Confirmed
   mechanism on the model side; the 4 px period = the upsample stage spacing. Candidate
   fixes (need retrain): anti-aliased upsampling (BlurPool, Zhang 2019); resize-conv with
   kernel divisible by stride / ICNR-init PixelShuffle (Odena/Aitken).

6. **Model output type.** A dense full-res heatmap + integer argmax is inherently a
   position-quantizer whose sub-grid structure may be unsupervised (free to grid). Contrast
   a continuous coordinate/offset-regression head (node + Δ∈ℝ), which would emit off-node
   positions by construction.

7. **Post-processing (extract_minutiae).** Integer local-max NMS with no sub-pixel
   ([postprocess.py:140-150](../../mntstitch/inference/python/src/mntstitch/postprocess.py))
   is the final snap onto the pixel grid. `_collapse_peaks` centroid dithers (nearest) or
   not (bilinear). FP16 rounding can create tied plateaus. Confirmed contributor but not the
   root (the heatmap is already gridded).

8. **Other.**
   - **Mutual reinforcement:** a 4 px *target* grid (#2/#3) and a 4 px *decoder* grid (#5)
     would compound — same period, hard to separate without ablating each independently.
   - **Phase anchoring:** the grid is locked to input pixel 0; where the anchor is set
     (encoder stride, input resize) determines whether augmentation can break it.
   - **Input preprocessing:** fixed-kernel resize/normalization of source images to the
     working resolution could imprint a lattice upstream of everything.
