# IENF Preprocessing at a Glance: S1616-2_b

A picture-first summary: one figure per step, one or two lines of text each. For the full detail (numbers, code, checks, zoomed comparisons) see `results/ienf_preprocessing_walkthrough/ienf_preprocessing_walkthrough_S1616-2_b.md`.

Sample: **S1616-2_b**, a Control (normal) slide that the model classifies correctly. The figures contain patient images and stay local (not in git).

## Overview

![All steps on one page](figures/s0_overview.png)

**Pipeline:** image → ROI → mask → patch grid → patches → UNI features → CLAM → heatmap

---

## Step 1 · Original image

![Step 1](figures/s1_original.png)

- Fluorescence skin image, 3815 × 1039 px RGB, mostly dark.
- A magenta stripe runs along the lower part of the epidermis. Thin bright green fibres cross the band and the dermis below it.
- The dashed box is the region shown in all later figures.

## Step 2 · Epidermis ROI

![Step 2](figures/s2_roi.png)

- The epidermis was outlined by hand in ImageJ (`.roi`, 172 points). The upper edge is the skin surface; the lower edge is the epidermis–dermis junction.
- Only this region is used. Everything outside it is ignored.

## Step 3 · Mask + 5 px buffer

![Step 3](figures/s3_mask_buffer.png)

- The outline is filled into a black/white mask.
- The lower edge is moved 5 px (≈ 3 µm) toward the dermis (orange). This keeps fibres that cross the junction.

## Step 4 · Patch grid

![Step 4](figures/s4_grid.png)

- A square grid starts at the top-left corner of the ROI.
- A square is kept if at least 5% of it is ROI (red). The others are dropped (dotted).
- 224 px: 48 squares → **22 patches**. 128 px: 108 squares → **50 patches**.

## Step 5 · Cut out the patches

![Step 5](figures/s5_patches.png)

- Each kept square is read from the image.
- All patches of one slide together form one **bag**, the unit CLAM classifies.

## Step 6 · UNI features

![Step 6](figures/s6_features.png)

- Each patch is resized to 224 × 224 and passed through UNI, a pathology foundation model that stays frozen.
- Each patch becomes 1024 numbers. The slide becomes a 50 × 1024 matrix.
- From here on, the model sees only these numbers, not the pixels.

## Step 7 · CLAM: attention and prediction

![Step 7](figures/s7_attention.png)

- CLAM gives every patch an attention weight. Red means the patch counts more in the slide-level decision.
- The weighted average of the features is classified.
- This slide: P(disease) = 0.0012 → normal. Correct.
- Model: run `ienf_2cls_uni_p128_CLAM_B8_s1`, fold 0, where this slide was in the test set.

## Step 8 · Heatmap

![Step 8](figures/s8_heatmap.png)

- Attention is painted back onto the image inside the ROI.
- For a smooth map, patches are scored every 6 px (95% overlap, 19,458 patches), then blurred.
- Resolution is still about one patch, 128 px ≈ 83 µm, so the map shows regions, not single fibres.

---

## One-line summary per step

| Step | Input → output | S1616-2_b |
|---|---|---|
| 1 | TIFF image | 3815 × 1039 px RGB |
| 2 | hand-drawn ROI | 172-point polygon |
| 3 | mask + 5 px buffer | 0/1 mask |
| 4 | grid + ≥ 5% ROI rule | 22 (224 px) / 50 (128 px) patches |
| 5 | patches → bag | 50 images of 128 × 128 |
| 6 | UNI encoder | 50 × 1024 features |
| 7 | CLAM-SB | attention per patch, P(disease) = 0.0012 |
| 8 | heatmap | 95% overlap + blur |

## Reproduce

```bash
python heatmaps/ienf_walkthrough_simple.py \
    --slide S1616-2_b \
    --run ienf_2cls_uni_p128_CLAM_B8_s1 \
    --cache results/ienf_preprocessing_walkthrough/fine_scores_overlap95.npz \
    --out_dir results/ienf_preprocessing_walkthrough_simple
```

This needs the heatmap score cache written by `heatmaps/ienf_walkthrough.py`; run that script first. Numbers above are in `facts.json` next to this report.
