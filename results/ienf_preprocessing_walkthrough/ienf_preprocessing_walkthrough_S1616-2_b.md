# IENF → CLAM Pipeline, Step by Step: Sample S1616-2_b

This report follows one IENF skin-biopsy image, **S1616-2_b**, through every step of the pipeline. The steps are: preprocessing (ROI → mask → buffer → contour → patch grid → patch file), feature extraction (patch → encoder input → UNI features), the CLAM model (attention → prediction) and the heatmap. The last part shows zoomed details that compare the heatmap with the original image.

Every figure is computed by `heatmaps/ienf_walkthrough.py` with the same code the pipeline uses:
- `IENF_CLAM_Preprocessing/ienf_prep` for the ROI, buffer and patch grid;
- CLAM's own transform, UNI loader, model and `visHeatmap` for features, attention and heatmaps.

Two consistency checks pass:
- The patch coordinates in the saved patch files equal the recomputed ones (asserted in the script).
- The model's prediction for this slide equals the saved test result (asserted to 1e-4).

Related reports: `results/ienf_2cls_patch_size_B_report.md` (the 4 training runs and the heatmap selection). The figures contain patient images; they are kept local and are not in git (`.gitignore`).

## Contents

0. [The sample](#0-the-sample)
1. [Original image](#1-original-image)
2. [Epidermis ROI](#2-epidermis-roi)
3. [Rasterized mask](#3-rasterized-mask)
4. [Buffer toward the dermis](#4-buffer-toward-the-dermis)
5. [CLAM contour and grid anchor](#5-clam-contour-and-grid-anchor)
6. [Patch grid and accept rule](#6-patch-grid-and-accept-rule)
7. [Patch file and reading patches](#7-patch-file-and-reading-patches)
8. [Encoder input transform](#8-encoder-input-transform)
9. [UNI features](#9-uni-features)
10. [CLAM attention and prediction](#10-clam-attention-and-prediction)
11. [Heatmap](#11-heatmap)
12. [Zoomed details](#12-zoomed-details)
13. [What this sample shows](#13-what-this-sample-shows)
14. [Reproduction and files](#14-reproduction-and-files)

### Pipeline at a glance

| Step | What happens | Code | Output for S1616-2_b |
|---|---|---|---|
| 1 | Read the TIFF | `ienf_prep/roi.read_image_size`, `slide_reader.IENFSlide` | 3815 × 1039 px RGB |
| 2 | Read the epidermis ROI | `roi.read_roi_polygon` (roifile) | polygon, 172 vertices |
| 3 | Rasterize the polygon | `roi.polygon_to_mask` (`cv2.fillPoly`) | 391,294 ROI pixels |
| 4 | Extend 5 px toward the dermis | `roi.extend_mask_toward_dermis` (`cv2.dilate`) | +17,005 px → 408,299 px |
| 5 | Mask → CLAM contour | `roi.mask_to_clam_contours` (`cv2.findContours`) | 1 contour, 0 holes → `contours/S1616-2_b.pkl` |
| 6 | Patch grid + accept rule | `patching.generate_coords_roi_overlap` | 224 px: 22 patches; 128 px: 50 patches |
| 7 | Write the patch file | `patching.save_patch_h5` (CLAM's `save_hdf5`) | `patches/S1616-2_b.h5` |
| 8 | Read + transform patches | `slide_reader`, CLAM `get_eval_transforms` | 3 × 224 × 224 tensors |
| 9 | UNI features | CLAM `extract_features_fp.py` (`uni_v1`) | 50 × 1024 (128 px) |
| 10 | CLAM-SB | `models/model_clam.py` | attention per patch, P(disease) = 0.0012 |
| 11 | Heatmap | CLAM `visHeatmap` + ROI contour | blockmap, 50% and 95% overlap |

## 0. The sample

| | |
|---|---|
| Slide | `S1616-2_b` (patient `S1616-2`, side b; side a is `S1616-2_a`) |
| Label | Control → **normal** in the 2-class task |
| Split | test set of fold 0 in `splits/ienf_kfold5_seed1` (training set in folds 1–4) |
| Out-of-fold P(disease), 4 runs | 224 px B 4: 0.007 · 224 px B 8: 0.056 · 128 px B 8: 0.001 · 128 px B 16: 0.004 |
| Model used here | `ienf_2cls_uni_p128_CLAM_B8_s1`, fold 0 checkpoint (the model that tested this slide) |

It was chosen as a sample because all four models classify it correctly and with high confidence. It ranks 54th of 387 slides by mean confidence over the four runs, and 5th among Control slides. The image also contains typical features of the dataset: a thin, curved epidermis band, two round artefacts (air bubbles or similar) below the band on the left, and bright green fibre-like structures in the dermis. Step 6 shows both patch sizes (224 and 128 px). From step 8 on, the report follows the 128 px pipeline, because that run has the highest mean AUC and was used for the heatmaps.

## 1. Original image

![Original image](figures/01_raw_image.png)

*Figure 1. Top: the whole TIFF with a 500 µm scale bar. Middle: the three channels of the exported RGB image. Bottom: histogram per channel (log scale) and file facts.*

- **File.** Uncompressed, single-strip RGB TIFF, 8 bit per channel, 3815 × 1039 px, one resolution level. OpenSlide cannot open plain TIFFs like this. That is why the pipeline uses `ienf_prep/slide_reader.py`, a small stand-in that decodes the image once and serves `read_region` like OpenSlide. Areas outside the image are returned as black with alpha 0.
- **Pixel size.** The TIFF resolution tag says 150 dpi, which is a placeholder, not a calibration. The dataset-wide value from the preprocessing project is **1.54 px/µm (0.65 µm/px)**.
  - The image has its own burned-in scale bar (top right, labelled "200 µm"). It measures **309 px** (from x = 3302 to 3610 at y ≈ 271), i.e. 1.545 px/µm, so this image agrees with the dataset value.
  - The automatic scale-bar check in the preprocessing project (`verify_pixel_size_orientation.csv`) had not found this bar (it reported 28 px). I measured it here directly; it is not part of the pipeline.
  - At this scale the image is about 2.48 × 0.67 mm.
  - The original 20× export of the same section is 7629 × 2078 px (`labels/original_image_info.csv`), so the TIFF used here is that export downsampled about 2×.
- **Content.** Mostly dark: channel means are R 14.3, G 32.9, B 19.0 (of 255). A magenta (red + blue) stripe runs along the lower part of the epidermis; above it, the epidermis is darker. Thin bright green fibres are visible in and below the epidermis, and a fainter green texture fills the dermis. This report does not say which stain each channel shows; it only describes the RGB channels of the exported file.

## 2. Epidermis ROI

![ROI polygon](figures/02_roi_polygon.png)

*Figure 2. The epidermis ROI (`Epidermis_ROI/S1616-2_b.roi`, ImageJ polygon, name "Epider") on the image, and a zoom of the dashed box.*

- The ROI is a hand-drawn ImageJ polygon with **172 vertices**, read with `roifile.ImagejRoi.fromfile(...).coordinates()`.
- The vertices are 36 px apart (median), at most 122 px. The straight segments between them are what the mask follows (Step 3).
- The upper edge follows the skin surface and the lower edge the epidermis–dermis junction. The zoom shows that the lower edge runs along the lower border of the magenta stripe, with several green fibres crossing it.
- The ROI spans x = 148–3548 and y = 330–831; it covers the whole visible epidermis from the left tip to the right end.

## 3. Rasterized mask

![Mask](figures/03_mask.png)

*Figure 3. Top: the polygon filled with `cv2.fillPoly` into a binary mask of image size. Bottom: thickness of the band (mask pixels per image column) along x.*

- The mask has **391,294** pixels set to 1 (about 9.9% of the image).
- The band is **107 px thick at the median** (≈ 69 µm). The 10th–90th percentile range is 47–194 px (≈ 31–126 µm). It is thin on the left (about 40–70 px) and about twice as thick on the right (about 180–200 px).
- This variation matters for patching. A 128 px patch is thicker than the band on the left and about as thick as the band on the right, so the share of a patch inside the ROI differs a lot along the band (Step 6).

## 4. Buffer toward the dermis

![Buffer](figures/04_buffer.png)

*Figure 4. Top: the raw ROI (yellow) and the 5 px added below it (orange). Bottom: a 200 × 75 px zoom without and with the overlay.*

- **Why.** The nerve fibres that matter for IENF cross the epidermis–dermis junction. A hand-drawn lower edge may sit a few pixels above the junction, so the mask is extended slightly downward to include it.
- **How.** `cv2.dilate` with a 6 × 1 vertical kernel anchored at its bottom pixel: a pixel becomes part of the mask if any of the 5 pixels above it is in the mask. Only the lower edge moves; the upper edge (skin surface) is unchanged. For sections whose dermis lies above the band (only `S1357-2_b` in this dataset), the kernel is anchored at the top instead.
- **Effect on this slide.** +17,005 px (4.3% of the ROI); the buffered mask has 408,299 px. 5 px ≈ 3.2 µm, which is small next to a 128 px (83 µm) patch. The buffer mainly affects which patches pass the accept rule at the margin.

## 5. CLAM contour and grid anchor

![Contour](figures/05_contour_anchor.png)

*Figure 5. The buffered mask converted to a CLAM contour (orange) and its bounding box (dashed). The grid of every patch size starts at the top-left corner of the box.*

- `cv2.findContours(..., RETR_CCOMP, CHAIN_APPROX_NONE)` gives **one outer contour with 6885 points and no holes**. It is stored as `{"tissue": [contour], "holes": [[]]}` in `contours/S1616-2_b.pkl`, the same structure that CLAM's own tissue segmentation saves.
- CLAM's heatmap code loads this file with `WholeSlideImage.initSegmentation`. That is how the heatmaps in Step 11 are coloured only inside the ROI.
- The grid anchor is the top-left corner of the bounding box, **(148, 330)**, as in CLAM's `process_contour`. The bounding box is 3401 × 507 px.

## 6. Patch grid and accept rule

![Grid and accept rule](figures/06_grid_accept_rule.png)

*Figure 6. Every grid candidate (patch size = step, no overlap) inside the bounding box. Red outline and coloured fill: kept; dotted gray: rejected. The numbers in the 224 px panel are the share of the patch covered by the buffered ROI.*

**Grid.** Candidates are laid from the anchor in steps of the patch size, to the right and down, until the bounding box is covered (`use_padding=True`, so a patch may extend past the image edge; none does on this slide).

| | 224 px | 128 px |
|---|---|---|
| Grid candidates | 48 (16 columns × 3 rows) | 108 (27 × 4) |
| Kept (ROI ≥ 5% of the patch) | **22** | **50** |
| Smallest ROI share among kept | 5.5% | 8.4% |
| Kept with ROI share 5–20% | 4 | 8 |
| ROI pixels inside some kept patch | 99.05% | 99.37% |
| Patches partly outside the image | 0 | 0 |

**Accept rule `roi_overlap`.** A candidate is kept if the buffered ROI covers at least 5% of its area. The share is computed exactly with an integral image. This replaces CLAM's default rule `four_pt`, which tests 4 points around the patch centre. `four_pt` is designed for large tissue areas and misses thin bands:

![roi_overlap vs four_pt](figures/06b_roi_overlap_vs_four_pt.png)

*Figure 6b. 224 px patches selected by `roi_overlap` (top) and by CLAM's `four_pt` (bottom). Red = ROI pixels that are in no selected patch.*

| 224 px | patches | ROI coverage | ROI pixels missed |
|---|---|---|---|
| `roi_overlap ≥ 5%` (used) | 22 | 99.0% | 3,884 |
| CLAM `four_pt` | 18 | 94.5% | 22,659 |

At 128 px, `four_pt` keeps 39 patches and covers 92.3% of the ROI (`roi_overlap`: 50 patches, 99.4%). Most of what `four_pt` misses is the lower part of the band where it crosses a grid row boundary (x ≈ 2200–2600 in Figure 6b) and the right tip. Those are exactly the junction areas that matter for IENF.

**A property of the rule worth knowing.** The rule decides only which patches enter the bag. The patch content is the whole square, not just the ROI part. With 5% as the threshold, a kept patch can be mostly background or dermis: 20 of the 50 kept 128 px patches have an ROI share below 40%. This becomes important in Steps 10–12.

## 7. Patch file and reading patches

![224 px patches](figures/07_patches_224.png)

*Figure 7a. The 22 patches of the 224 px patch file, read back through the openslide stand-in, ordered left to right. Each title gives the index, the top-left (x, y) and the ROI share.*

![128 px patches](figures/07_patches_128.png)

*Figure 7b. The same for the 50 patches of the 128 px patch file.*

- **File.** `patches/S1616-2_b.h5` has a dataset `coords` (N × 2, int64, top-left at level 0), written with CLAM's own `save_hdf5`.
  - Attributes CLAM reads: `patch_size`, `patch_level = 0`, `downsample`, `level_dim`, `downsampled_level_dim`, `name`, `save_path`.
  - Extra provenance attributes that CLAM ignores: `step_size`, `contour_fn = roi_overlap`, `min_roi_overlap = 0.05`, `roi_buffer_down_px = 5`, `use_padding = True`.
- **What the patches contain** (Figure 7b):
  - Patches of the top grid row at the left (x < 1150) contain the magenta stripe with ROI shares of 37–52%.
  - From x ≈ 1300 to the right, the band runs lower, so the top-row patches there contain mostly the dark area above the skin surface (ROI 8–33%) and most of the band sits in the second row.
  - Patches of the lower rows on the right contain the band plus dermis with bright green fibres.
- The same 50 coordinates are in the feature file `ienf_uni_p128_s128/h5_files/S1616-2_b.h5`. Its row order (by x, then y) is the order of the feature rows.

## 8. Encoder input transform

![Encoder input](figures/08_encoder_input.png)

*Figure 8. The patch with the highest attention (index 19, at (1940, 330)) as read, after resizing to 224 px, after normalization, and the distribution of the normalized values.*

1. **Read.** A 128 × 128 RGBA region via `read_region`, converted to RGB.
2. **Resize.** `transforms.Resize(224)`, bilinear, ×1.75. At 128 px the encoder therefore sees the tissue at about 0.37 µm/px, larger than its native scale. 224 px patches are not resized.
3. **Normalize.** `ToTensor` (0–1) and `Normalize` with ImageNet mean (0.485, 0.456, 0.406) and std (0.229, 0.224, 0.225), the constants CLAM uses for `uni_v1`.

Because the fluorescence images are dark, almost all normalized values are strongly negative: about −2.1 to −1.0 for this patch, while a typical bright-field patch is centred near 0. UNI was trained on bright-field (H&E/IHC) images, so these inputs are far from its training distribution. The resulting features still separate normal from disease reasonably well (AUC ≈ 0.87, main report), but this domain gap is a likely limit.

## 9. UNI features

![UNI features](figures/09_uni_features.png)

*Figure 9. Top left: the 50 × 1024 feature matrix, z-scored per dimension, patches grouped by grid row (black lines). Top right: cosine similarity between all patches, same order (white lines = new grid row). Bottom: the first three principal components of the features mapped to RGB and painted at each patch location.*

- **Saved features.** `ienf_uni_p128_s128/pt_files/S1616-2_b.pt` has 50 × 1024 features. The mean vector length is 39.7, close to UNI features on Camelyon+ (38.7). Re-encoding patch 19 in this script gives a cosine similarity of 1.0000 to the saved vector, so the saved features match the transform above.
- **Structure.** Patches are similar to their neighbours and form groups that follow the image content, not the patch index:
  - the left top-row patches with the band (light green-grey in the PCA map);
  - the right top-row patches that are mostly above the skin surface (green);
  - second-row patches with the band (orange, and violet on the right);
  - the lowest rows with dermis (pink).
- The cosine similarity is 0.52 on average (minimum 0.10). PC1–3 explain 31%, 13% and 6% of the variance.

## 10. CLAM attention and prediction

![Attention and prediction](figures/10_attention_prediction.png)

*Figure 10. Top: attention of each bag patch as a percentile within the slide. Bottom left: the softmax attention weights, grouped by grid row (dotted lines). Bottom right: the prediction.*

**How CLAM-SB turns the bag into a prediction:**

1. Each feature vector is projected 1024 → 512: h = ReLU(W x), with dropout during training.
2. Gated attention: a = w · (tanh(V h) ⊙ σ(U h)), one raw score per patch.
3. Weights: softmax over the patches (sum = 1).
4. Slide vector: weighted sum Σ weight · h.
5. A linear classifier on the slide vector gives two logits; softmax gives P(disease).

**For this slide:**

- logits: normal +3.18, disease −3.53 → **P(disease) = 0.0012**, predicted normal (correct).
- The weights are concentrated: the largest is 0.114, 5.7 times the uniform weight of 0.020, and the top 5 patches hold 41% of the total. The smallest weight is 0.00003.
- **All 5 highest-attention patches are in the top grid row** at x = 1300–2452, where the patch is mostly the dark area above the skin surface (ROI share 22–33%).
- The 5 lowest-attention patches lie in the lower rows on the right (ROI share 9–45%). They contain the lower part of the band and dermis with bright fibres.

![Attention vs. patch content](figures/10b_attention_vs_content.png)

*Figure 10b. Attention percentile of each bag patch against its ROI share, its mean brightness and its mean green value.*

| Within this slide (n = 50 patches) | Spearman r with attention | p |
|---|---|---|
| ROI share of the patch | +0.21 | 0.15 |
| Mean pixel value (all channels) | **−0.47** | 0.0005 |
| Mean green value | −0.30 | 0.03 |

In this slide the model gives more attention to **darker** patches. It does not prefer patches that are more inside the ROI. This is one slide and one model, so it describes what happens here, not a general rule. It does raise the question of what the model actually uses, which the heatmap alone does not answer (Step 12).

## 11. Heatmap

![Heatmap stages](figures/11_heatmap_stages.png)

*Figure 11. Top: blockmap (the 50 bag patches, no overlap). Middle: 50% overlap (190 patches, step 64 px). Bottom: 95% overlap with Gaussian blur (19,458 patches, step 6 px), the setting of the original CLAM paper.*

How the heatmap is made (`heatmaps/ienf_heatmaps.py`, same steps as CLAM's `create_heatmaps.py`):

1. **Blockmap.** The attention scores of the bag patches are converted to percentiles within the slide and drawn as squares.
2. **Overlapping patches.** Patches are laid with a smaller step on the same ROI rule. Each one is read, transformed and encoded with UNI, and scored by the same CLAM model (`attention_only=True`).
3. **Reference percentiles.** Each new score is converted to its percentile among the 50 bag scores (`score2percentile`, CLAM's `use_ref_scores`). The colours are therefore comparable with the blockmap.
4. **Drawing.** `WholeSlideImage.visHeatmap` averages the overlapping scores per pixel and colours them with jet.
   - Only pixels inside the ROI contour are coloured (`segment=True`).
   - With `blur=True`, a Gaussian blur with a 13 px kernel (2 × step + 1) is applied to the coloured layer.
   - The coloured layer is blended with the original at alpha 0.4, so the original stays visible.

The three versions agree: the left part of the band (x < 1000) is low (blue), the upper part of the band from x ≈ 1300 to 2600 is high (red), and the right part has two local maxima. With more overlap the blocks disappear but the pattern stays. The 95% map is smoother, not more detailed: every value still describes a 128 px (≈ 83 µm) window around the point.

## 12. Zoomed details

The zoom windows were chosen automatically from the 95% heatmap. A 420 px wide window (≈ 270 µm) is moved along the band, and the mean attention percentile inside it gives a profile. The windows are centred at the profile's maximum ("high"), its minimum ("low") and the point closest to its median ("middle", not overlapping the others).

| Window | x range | y range | Mean percentile | Mean R / G / B inside the ROI |
|---|---|---|---|---|
| high | 1948–2368 | 355–631 | 64 | 34 / 42 / 41 |
| middle | 2388–2808 | 363–689 | 44 | 23 / 39 / 28 |
| low | 488–908 | 245–505 | 17 | 48 / 27 / 51 |

### 12.1 Heatmap vs. original

![Zoom: heatmap vs. original](figures/12_zoom_heatmap_vs_original.png)

*Figure 12. Top: the 95% heatmap with the three windows and the attention profile (white line, right axis). Rows: each window as original (left), heatmap (middle), and original with the 128 px bag patches outlined in their attention colour and labelled with the percentile (right; yellow = ROI contour).*

- **High window.** The band here is wide. Its upper part, toward the skin surface, is red; the lower part near the junction is yellow-green. The patch grid explains why: the top-row patches (percentiles 100, 56, 98, 92) cover the upper part of the band and the dark area above it. The second-row patches (58, 48, 74, 64) cover the lower band and the junction with bright fibres. Several fine green fibres run through the lower band and cross the junction, but they lie in the patches with lower attention.
- **Middle window.** A red spot in the upper band at the left (patches 92 and 96 above it, 66 in the band) turns to blue on the right. The bottom row of patches (2, 8, 6) lies mostly in the dermis below the band and has nearly zero weight.
- **Low window.** The thin left part of the band, a clearly visible magenta stripe with few green fibres. All patches here have percentiles 18–30, except one at the left edge with 76.

### 12.2 Highest- and lowest-attention patches

![Top and bottom patches](figures/13_top_bottom_patches.png)

*Figure 13. The 5 bag patches with the highest (top row) and lowest (bottom row) attention at their native 128 px (nearest-neighbour upscaling). Yellow line: the ROI boundary inside the patch.*

- **Highest.** All five look alike: a dark upper part (above the skin surface) and the top of the epidermis in the lower third. They have few or no visible fibres. The ROI line shows that 67–78% of each patch is outside the ROI.
- **Lowest.** All five are mostly dermis below the band, with bright green fibres, magenta structures and textured background. The ROI line is in their upper part.

The heatmap colours only the ROI part of a patch. So the red upper band in Figures 11 and 12 corresponds to patches whose content is largely the background above the tissue. That is not visible in the heatmap itself.

### 12.3 The ROI edge at pixel level

![ROI edge](figures/14_roi_edge_pixels.png)

*Figure 14. A 160 px wide strip (≈ 104 µm) through the high window, without smoothing: original pixels (left), raw ROI in yellow and the 5 px buffer in orange (middle), and the 95% heatmap (right). White dots: ROI vertices.*

- The upper edge runs along the faint upper border of the tissue, with dark background above it. The magenta stripe lies in the lower part of the band, and the lower edge follows its lower border; the buffer adds a thin strip below it.
- Fine green fibres (a few pixels wide) cross the band. At 0.65 µm/px they are 1–3 px thick, small compared with a 128 px patch.
- The heatmap fades from orange at the top to green-blue at the bottom of the band within about 100 px, the scale of one patch. This gradient is a result of which patches cover which part of the band. It is not a structure the model resolved inside a patch.

### 12.4 Channels in the high and low windows

![Channels](figures/15_channels_high_low.png)

*Figure 15. The high (top) and low (bottom) windows split into the R, G and B channels of the exported TIFF (display range 0–255); yellow = ROI boundary.*

- In the **high** window the green channel shows many fine, beaded fibres in the band and thick bright fibres just below the junction. Red and blue show the band as a faint layer.
- In the **low** window the band is mainly bright in red and blue (the magenta stripe), and the green channel shows only a few short fibre fragments.
- The mean values inside the ROI reflect this: G 42 in the high window vs. 27 in the low window; R 34 vs. 48. Whether the green fibres are intraepidermal nerve fibres would need the stain information and an expert; this report only describes intensities.

## 13. What this sample shows

1. **The preprocessing does what it is designed to do.**
   - The ROI is read exactly. The 5 px buffer moves only the lower edge.
   - The `roi_overlap` rule covers 99% of the ROI with 50 (128 px) or 22 (224 px) patches. CLAM's `four_pt` would miss 5.5% (224 px) to 7.7% (128 px) of the ROI, mainly along the junction.
   - Patch files, features and the prediction are all consistent with the pipeline outputs.
2. **The bag contains more than the epidermis.** Because the band is thin and patches are square, about half of each patch is on average outside the ROI (mean ROI share 0.50). In this slide that means background above the skin surface in the top row and dermis in the lower rows.
3. **In this slide, the model attends to the patches that are mostly above the skin surface.**
   - The 5 highest-attention patches have 67–78% of their area outside the ROI. Attention is negatively correlated with patch brightness (r = −0.47).
   - The heatmap, masked to the ROI, shows this as "high attention in the upper epidermis". The model may be using the skin-surface edge, the stratum corneum, or simply darkness; the heatmap cannot tell these apart.
   - It is one correctly classified normal slide. Whether the same happens in other slides, or in disease slides, has not been checked.
4. **The heatmap resolution is the patch size.** With 95% overlap the map is smooth, but the colour changes over about one patch width (83 µm), so it cannot point to individual nerve fibres (1–3 px).

### Suggested checks (not done)

- Compute the attention-vs-brightness and attention-vs-ROI-share correlations over all slides (and separately for normal and disease) to see whether point 3 is general.
- Mask the non-ROI part of each patch (e.g. set it to black or to the mean background) before feature extraction, and compare performance. This would show how much the model depends on what is outside the epidermis.
- Use smaller patches, or a higher accept threshold, so that patches are mostly epidermis.
- Compare the heatmaps with the IENF particle ROIs (`IENF_ROI/S1616-2_b.zip`) to see whether high attention coincides with annotated nerve fibres.

## 14. Reproduction and files

Patching (once for all slides; both patch sizes):

```bash
cd ~/projects/IENF_CLAM_Preprocessing
python scripts/create_patches_roi.py \
    --source /data/datasets/neuroimages/20250214 \
    --csv ~/projects/CLAM/dataset_csv/ienf_3cls.csv \
    --save_dir /var/tmp/shouteng_data/IENF_CLAM/clam_prep \
    --clam_root ~/projects/CLAM \
    --patch_size 224 128 --overlap 0
```

Feature extraction and training: `run_ienf_uni_features.sh` and `run_ienf_2cls.sh` in the CLAM repository (see the main IENF report).

All figures of this report:

```bash
cd ~/projects/CLAM
UNI_CKPT_PATH=~/projects/UNI/assets/ckpts/uni/pytorch_model.bin \
python heatmaps/ienf_walkthrough.py \
    --slide S1616-2_b \
    --run ienf_2cls_uni_p128_CLAM_B8_s1 \
    --out_dir results/ienf_preprocessing_walkthrough
```

The script takes a few minutes on the RTX 3090; most of the time goes into the 19,458 overlapping patches of the 95% heatmap. Their scores are cached in `fine_scores_overlap95.npz`, so later runs are faster. Any other slide can be shown with `--slide`; the zoom windows are chosen automatically.

| File | Content |
|---|---|
| `results/ienf_preprocessing_walkthrough/ienf_preprocessing_walkthrough_S1616-2_b.md` | this report |
| `results/ienf_preprocessing_walkthrough/figures/*.png` | the 18 figures (local only, contain patient images) |
| `results/ienf_preprocessing_walkthrough/facts.json` | every number in this report, as computed by the script |
| `results/ienf_preprocessing_walkthrough/fine_scores_overlap95.npz` | cached 95% overlap scores (local only) |
| `heatmaps/ienf_walkthrough.py` | the script |
| `/var/tmp/shouteng_data/IENF_CLAM/clam_prep/p{224_s224,128_s128}_bufdown5/` | contours, patch files and mask overviews of all slides |
| `/var/tmp/shouteng_data/CLAM_DATA/ienf_uni_p{224_s224,128_s128}/` | UNI features of all slides |
