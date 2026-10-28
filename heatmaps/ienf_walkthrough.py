"""Step-by-step figures of the IENF -> CLAM pipeline for one slide (default S1616-2_b).

Every figure is computed from the same code the pipeline uses (IENF_CLAM_Preprocessing for
ROI / patching, CLAM for features, model and heatmap), not redrawn by hand:

  01 raw image and channels           07 extracted patches (both patch sizes)
  02 ROI polygon (ImageJ .roi)         08 encoder input transform (resize + normalize)
  03 rasterized mask                   09 UNI features (matrix, PCA colours, similarity)
  04 5-px buffer toward the dermis     10 CLAM attention on the bag and prediction
  05 CLAM contour + grid anchor        11 heatmap 50% vs. 95% overlap
  06 candidate grid + accept rule      12-15 zoomed details (heatmap vs. original, top/bottom patches,
  06b roi_overlap vs. CLAM four_pt            ROI edge, channels)

Usage (repo root, clam env):
  UNI_CKPT_PATH=~/projects/UNI/assets/ckpts/uni/pytorch_model.bin \\
  python heatmaps/ienf_walkthrough.py \\
      --slide S1616-2_b --run ienf_2cls_uni_p128_CLAM_B8_s1 \\
      --out_dir results/ienf_preprocessing_walkthrough
"""
import argparse
import json
import os
import pickle
import sys
from argparse import Namespace
from pathlib import Path

IENF_PREP = Path.home() / 'projects' / 'IENF_CLAM_Preprocessing'
CLAM_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CLAM_ROOT))
sys.path.insert(0, str(IENF_PREP))
from ienf_prep import slide_reader  # noqa: E402
sys.modules['openslide'] = slide_reader

import cv2  # noqa: E402
import h5py  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from matplotlib.colors import Normalize  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402
from PIL import Image  # noqa: E402

from ienf_prep.patching import (count_partially_outside, generate_coords, generate_coords_roi_overlap,  # noqa: E402
                                load_clam, roi_coverage)
from ienf_prep.roi import (extend_mask_toward_dermis, mask_to_clam_contours, polygon_to_mask,  # noqa: E402
                           read_roi_polygon)
from models import get_encoder  # noqa: E402
from utils.constants import MODEL2CONSTANTS  # noqa: E402
from utils.eval_utils import initiate_model  # noqa: E402
from vis_utils.heatmap_utils import drawHeatmap, score2percentile  # noqa: E402
from wsi_core.WholeSlideImage import WholeSlideImage  # noqa: E402

DATA_ROOT = '/data/datasets/neuroimages/20250214'
PREP = '/var/tmp/shouteng_data/IENF_CLAM/clam_prep'
FEAT_ROOT = '/var/tmp/shouteng_data/CLAM_DATA'
UM_PER_PX = 1 / 1.54  # dataset-wide pixel size (IENF_CLAM_Preprocessing README); not verified for every image
BUFFER = 5
MIN_OVERLAP = 0.05
C_ROI, C_BUF, C_KEEP, C_DROP, C_GRID = '#ffd400', '#ff7a00', '#ff2d2d', '#8a8a8a', '#00e5ff'

plt.rcParams.update({'figure.facecolor': '#fcfcfb', 'savefig.facecolor': '#fcfcfb', 'font.size': 10,
                     'axes.titlesize': 11, 'axes.titleweight': 'semibold', 'axes.titlelocation': 'left',
                     'savefig.dpi': 150, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.15})


def off(ax):
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def note(fig, text, y=-0.01):
    fig.text(0.01, y, text, fontsize=8.5, color='#52514e', va='top', wrap=True)


def scalebar(ax, um=200, x=None, y=None, color='white', crop_x0=0):
    """Scale bar in data coordinates (pixels of the shown array)."""
    px = um / UM_PER_PX
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    x = x if x is not None else xlim[1] - px - 0.03 * (xlim[1] - xlim[0])
    y = y if y is not None else max(ylim) - 0.08 * abs(ylim[1] - ylim[0])
    ax.plot([x, x + px], [y, y], color=color, lw=3, solid_capstyle='butt')
    ax.text(x + px / 2, y - 0.03 * abs(ylim[1] - ylim[0]), f'{um} µm', color=color, ha='center', va='bottom', fontsize=8)


def save(fig, out, name):
    fig.savefig(out / name)
    plt.close(fig)
    print('wrote', name)


def patch_rects(ax, coords, ps, color, lw=1.2, x0=0, y0=0, ls='-'):
    for x, y in coords:
        ax.add_patch(Rectangle((x - x0, y - y0), ps, ps, fill=False, ec=color, lw=lw, ls=ls))


def coverage_fraction(mask, x, y, ps):
    h, w = mask.shape
    x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + ps, w), min(y + ps, h)
    return mask[y0:y1, x0:x1].sum() / float(ps * ps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--slide', default='S1616-2_b')
    ap.add_argument('--run', default='ienf_2cls_uni_p128_CLAM_B8_s1')
    ap.add_argument('--out_dir', default='results/ienf_preprocessing_walkthrough')
    ap.add_argument('--overlap', type=float, default=0.95)
    args = ap.parse_args()
    sid = args.slide
    out = Path(args.out_dir) / 'figures'
    out.mkdir(parents=True, exist_ok=True)
    facts = {}

    # ------------------------------------------------------------------ inputs
    img_path = f'{DATA_ROOT}/Image/Original/{sid}.tif'
    roi_path = f'{DATA_ROOT}/Epidermis_ROI/{sid}.roi'
    with Image.open(img_path) as im:
        size = im.size
        rgb = np.asarray(im.convert('RGB')).copy()
    W, H = size
    poly = read_roi_polygon(roi_path)
    raw = polygon_to_mask(poly, size)
    buf = extend_mask_toward_dermis(raw, BUFFER, dermis_up=False)
    tissue, holes = mask_to_clam_contours(buf)
    ys, xs = np.nonzero(buf)
    bx0, by0, bx1, by1 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
    # display crop around the band
    cx0, cy0 = max(0, bx0 - 120), max(0, by0 - 120)
    cx1, cy1 = min(W, bx1 + 120), min(H, by1 + 120)
    crop = rgb[cy0:cy1, cx0:cx1]
    ext = [cx0, cx1, cy1, cy0]
    facts.update(slide=sid, W=W, H=H, n_vertices=len(poly), raw_area=int(raw.sum()), buf_area=int(buf.sum()),
                 bbox=[bx0, by0, bx1, by1], n_tissue=len(tissue), n_holes=sum(len(h) for h in holes))

    # ------------------------------------------------------------------ 01 raw image + channels
    fig = plt.figure(figsize=(14, 8.2))
    gs = fig.add_gridspec(3, 3, height_ratios=[1.35, 0.8, 0.8])
    ax = fig.add_subplot(gs[0, :])
    ax.imshow(rgb)
    ax.set_title(f'{sid}: original TIFF, {W} × {H} px, 8-bit RGB, uncompressed')
    ax.set_xlabel('x (px)'); ax.set_ylabel('y (px)')
    scalebar(ax, 500)
    for i, (name, cmap) in enumerate([('Red channel', 'Reds_r'), ('Green channel', 'Greens_r'), ('Blue channel', 'Blues_r')]):
        a = fig.add_subplot(gs[1, i])
        a.imshow(rgb[..., i], cmap='gray', vmin=0, vmax=255)
        a.set_title(name); off(a)
    a = fig.add_subplot(gs[2, :2])
    for i, c in enumerate(['#d62728', '#2ca02c', '#1f77b4']):
        h = np.bincount(rgb[..., i].ravel(), minlength=256)
        a.plot(np.arange(256), h, color=c, lw=1.4)
    a.set_yscale('log'); a.set_xlabel('Pixel value'); a.set_ylabel('Pixel count (log)')
    a.set_title('Histogram per channel'); a.set_xlim(0, 255)
    a2 = fig.add_subplot(gs[2, 2])
    a2.axis('off')
    stats = [f'{n}: mean {rgb[..., i].mean():.1f}, 99th pct {np.percentile(rgb[..., i], 99):.0f}' for i, n in enumerate('RGB')]
    a2.text(0, 0.9, 'File facts\n' + '\n'.join([
        f'size {W} × {H} px', 'TIFF, RGB, 8 bit/channel, no compression,', 'single strip, 1 level (no pyramid)',
        'resolution tag 150 dpi (placeholder)', f'assumed 1.54 px/µm ≈ {UM_PER_PX:.2f} µm/px',
        f'image ≈ {W * UM_PER_PX / 1000:.2f} × {H * UM_PER_PX / 1000:.2f} mm', ''] + stats), va='top', fontsize=9, family='monospace')
    fig.tight_layout()
    note(fig, 'Scale bar uses the dataset-wide pixel size (1.54 px/µm); this image has no usable calibration of its own.')
    save(fig, out, '01_raw_image.png')
    facts['channel_means'] = [round(float(rgb[..., i].mean()), 1) for i in range(3)]

    # ------------------------------------------------------------------ 02 ROI polygon
    fig, axes = plt.subplots(2, 1, figsize=(14, 7.6), gridspec_kw={'height_ratios': [1, 1.05]})
    axes[0].imshow(crop, extent=ext)
    axes[0].plot(np.r_[poly[:, 0], poly[0, 0]], np.r_[poly[:, 1], poly[0, 1]], color=C_ROI, lw=1.2)
    axes[0].plot(poly[:, 0], poly[:, 1], 'o', ms=2.2, color=C_ROI)
    axes[0].set_title(f'Epidermis ROI ({Path(roi_path).name}): ImageJ polygon with {len(poly)} vertices')
    zx0, zx1 = 1500, 2100
    zy0, zy1 = int(poly[(poly[:, 0] > zx0) & (poly[:, 0] < zx1), 1].min()) - 60, int(poly[(poly[:, 0] > zx0) & (poly[:, 0] < zx1), 1].max()) + 60
    axes[0].add_patch(Rectangle((zx0, zy0), zx1 - zx0, zy1 - zy0, fill=False, ec='white', lw=1.2, ls='--'))
    axes[0].set_xlabel('x (px)'); axes[0].set_ylabel('y (px)')
    axes[1].imshow(rgb[zy0:zy1, zx0:zx1], extent=[zx0, zx1, zy1, zy0])
    sel = (poly[:, 0] >= zx0 - 50) & (poly[:, 0] <= zx1 + 50)
    axes[1].plot(np.r_[poly[:, 0], poly[0, 0]], np.r_[poly[:, 1], poly[0, 1]], color=C_ROI, lw=1.4)
    axes[1].plot(poly[sel, 0], poly[sel, 1], 'o', ms=4, color=C_ROI, mec='black', mew=0.5)
    axes[1].set_xlim(zx0, zx1); axes[1].set_ylim(zy1, zy0)
    axes[1].set_title('Zoom (dashed box): vertices are hand-placed points; the polygon is closed from the last vertex back to the first')
    seg = np.hypot(*np.diff(np.r_[poly, poly[:1]], axis=0).T)
    fig.tight_layout()
    note(fig, f'Vertex spacing: median {np.median(seg):.0f} px, max {seg.max():.0f} px. Upper edge = skin surface, lower edge = '
              'epidermis–dermis junction. Read with roifile.ImagejRoi.coordinates().')
    save(fig, out, '02_roi_polygon.png')
    facts['vertex_spacing_median'] = float(np.median(seg)); facts['vertex_spacing_max'] = float(seg.max())

    # ------------------------------------------------------------------ 03 rasterized mask
    thick = raw.sum(0).astype(float)
    cols = np.where(thick > 0)[0]
    fig, axes = plt.subplots(2, 1, figsize=(14, 6.6), gridspec_kw={'height_ratios': [1, 0.7]})
    axes[0].imshow(raw[cy0:cy1, cx0:cx1], cmap='gray', extent=ext, interpolation='nearest')
    axes[0].set_title(f'Rasterized ROI mask (cv2.fillPoly): {int(raw.sum()):,} pixels = 1, the rest = 0')
    axes[0].set_xlabel('x (px)'); axes[0].set_ylabel('y (px)')
    axes[1].plot(cols, thick[cols], color='#1f77b4', lw=1)
    axes[1].set_xlim(cx0, cx1)
    axes[1].set_xlabel('x (px)'); axes[1].set_ylabel('Band thickness (px)')
    sec = axes[1].secondary_yaxis('right', functions=(lambda v: v * UM_PER_PX, lambda v: v / UM_PER_PX))
    sec.set_ylabel('µm (assumed scale)')
    axes[1].axhline(np.median(thick[cols]), color='#888', lw=1, ls='--')
    axes[1].text(cx0 + 20, np.median(thick[cols]) + 4, f'median {np.median(thick[cols]):.0f} px', color='#555', fontsize=8)
    axes[1].set_title('Thickness of the epidermis band per image column (number of mask pixels)')
    fig.tight_layout()
    save(fig, out, '03_mask.png')
    facts['thick_median'] = float(np.median(thick[cols])); facts['thick_p10_p90'] = [float(np.percentile(thick[cols], 10)), float(np.percentile(thick[cols], 90))]
    facts['band_width_cols'] = int(len(cols))

    # ------------------------------------------------------------------ 04 buffer
    added = (buf > 0) & (raw == 0)
    lower = np.array([np.nonzero(raw[:, x])[0].max() if raw[:, x].any() else -1 for x in range(W)])
    zx0, zx1 = 2300, 2500
    yc = int(np.median(lower[zx0:zx1][lower[zx0:zx1] > 0]))
    zy0, zy1 = yc - 45, yc + 30
    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(2, 2, height_ratios=[0.8, 1.2])
    ax = fig.add_subplot(gs[0, :])
    ov = crop.copy()
    m_raw, m_add = raw[cy0:cy1, cx0:cx1] > 0, added[cy0:cy1, cx0:cx1]
    ov[m_raw] = (0.55 * ov[m_raw] + 0.45 * np.array([255, 212, 0])).astype(np.uint8)
    ov[m_add] = np.array([255, 122, 0])
    ax.imshow(ov, extent=ext)
    ax.add_patch(Rectangle((zx0, zy0), zx1 - zx0, zy1 - zy0, fill=False, ec='white', lw=1.2, ls='--'))
    ax.set_title(f'Buffer: lower edge moved {BUFFER} px toward the dermis (orange); upper edge unchanged')
    ax.set_xlabel('x (px)'); ax.set_ylabel('y (px)')
    for j, (title, show) in enumerate([('Zoom, original pixels', False), ('Zoom, raw ROI (yellow) + added pixels (orange)', True)]):
        a = fig.add_subplot(gs[1, j])
        z = rgb[zy0:zy1, zx0:zx1].copy()
        if show:
            mr, ma = raw[zy0:zy1, zx0:zx1] > 0, added[zy0:zy1, zx0:zx1]
            z[mr] = (0.6 * z[mr] + 0.4 * np.array([255, 212, 0])).astype(np.uint8)
            z[ma] = np.array([255, 122, 0])
        a.imshow(z, extent=[zx0, zx1, zy1, zy0], interpolation='nearest')
        a.set_title(title); a.set_xlabel('x (px)'); a.set_ylabel('y (px)')
    fig.tight_layout()
    note(fig, f'Implementation: cv2.dilate with a ({BUFFER + 1} × 1) vertical kernel anchored at its bottom, so a pixel becomes 1 when one of '
              f'the {BUFFER} pixels above it is 1. Added: {int(added.sum()):,} px ({added.sum() / raw.sum():.1%} of the ROI). '
              f'{BUFFER} px ≈ {BUFFER * UM_PER_PX:.1f} µm.')
    save(fig, out, '04_buffer.png')
    facts['buffer_added'] = int(added.sum())

    # ------------------------------------------------------------------ 05 CLAM contour + anchor
    fig, ax = plt.subplots(figsize=(14, 4.4))
    ax.imshow(crop, extent=ext)
    for c in tissue:
        ax.plot(np.r_[c[:, 0, 0], c[0, 0, 0]], np.r_[c[:, 0, 1], c[0, 0, 1]], color=C_BUF, lw=1.3)
    ax.add_patch(Rectangle((bx0, by0), bx1 - bx0 + 1, by1 - by0 + 1, fill=False, ec=C_GRID, lw=1.3, ls='--'))
    ax.plot(bx0, by0, 'o', ms=8, color=C_GRID, mec='black')
    ax.annotate(f'grid anchor ({bx0}, {by0})', (bx0, by0), xytext=(15, -18), textcoords='offset points', color=C_GRID, fontsize=9)
    ax.set_title(f'CLAM contour (cv2.findContours on the buffered mask): {len(tissue)} tissue contour with '
                 f'{len(tissue[0])} points, {facts["n_holes"]} holes; dashed = bounding box')
    ax.set_xlabel('x (px)'); ax.set_ylabel('y (px)')
    fig.tight_layout()
    note(fig, 'Saved as contours/<slide>.pkl in CLAM\'s format {"tissue": [...], "holes": [[...]]}; CLAM\'s heatmap code loads it with '
              'WholeSlideImage.initSegmentation. Patch grids start at the top-left corner of the bounding box.')
    save(fig, out, '05_contour_anchor.png')
    facts['contour_points'] = int(len(tissue[0]))

    # ------------------------------------------------------------------ 06 candidate grid + rule (both sizes)
    clam = load_clam(CLAM_ROOT)
    grids = {}
    fig, axes = plt.subplots(2, 1, figsize=(14, 8.6))
    for ax, ps in zip(axes, [224, 128]):
        cand = [(x, y) for x in range(bx0, bx1 + 1, ps) for y in range(by0, by1 + 1, ps)]
        fr = np.array([coverage_fraction(buf, x, y, ps) for x, y in cand])
        kept = generate_coords_roi_overlap(buf, ps, ps, MIN_OVERLAP, True)
        fp = np.concatenate([generate_coords(clam, c, h, size, ps, ps, 'four_pt', True) for c, h in zip(tissue, holes)])
        grids[ps] = dict(cand=np.array(cand), frac=fr, kept=kept, four_pt=fp)
        ax.imshow(crop, extent=ext)
        cm = plt.get_cmap('viridis')
        for (x, y), f in zip(cand, fr):
            if f >= MIN_OVERLAP:
                ax.add_patch(Rectangle((x, y), ps, ps, fill=True, fc=cm(f), alpha=0.35, ec=C_KEEP, lw=1.3))
            else:
                ax.add_patch(Rectangle((x, y), ps, ps, fill=False, ec=C_DROP, lw=0.8, ls=':'))
            if ps == 224:
                ax.text(x + ps / 2, y + ps / 2, f'{f:.0%}', ha='center', va='center', fontsize=6.5, color='white')
        ax.plot(np.r_[tissue[0][:, 0, 0], tissue[0][0, 0, 0]], np.r_[tissue[0][:, 0, 1], tissue[0][0, 0, 1]], color=C_ROI, lw=0.8)
        ax.set_xlim(cx0, cx1); ax.set_ylim(cy1, cy0)
        ax.set_title(f'{ps} px patches, step {ps}: {len(cand)} grid candidates → {len(kept)} kept (ROI covers ≥ {MIN_OVERLAP:.0%} of the patch)')
        ax.set_xlabel('x (px)'); ax.set_ylabel('y (px)')
        facts[f'p{ps}'] = dict(candidates=len(cand), kept=int(len(kept)), four_pt=int(len(fp)),
                               cov_roi_overlap=round(roi_coverage(buf, kept, ps), 4), cov_four_pt=round(roi_coverage(buf, fp, ps), 4),
                               partial=count_partially_outside(kept, size, ps),
                               frac_kept_min=float(fr[fr >= MIN_OVERLAP].min()), n_low=int(((fr >= MIN_OVERLAP) & (fr < 0.2)).sum()))
    axes[0].legend(handles=[Patch(fc='none', ec=C_KEEP, label='kept (fill colour = ROI share, viridis)'),
                            Patch(fc='none', ec=C_DROP, ls=':', label='rejected (< 5%)'),
                            Line2D([], [], color=C_ROI, label='buffered ROI contour')],
                   loc='upper center', bbox_to_anchor=(0.5, -0.17), ncol=3, frameon=False)
    fig.tight_layout()
    note(fig, 'Numbers in the 224 px panel: share of the patch area covered by the buffered ROI (exact, from an integral image).')
    save(fig, out, '06_grid_accept_rule.png')

    # ------------------------------------------------------------------ 06b roi_overlap vs four_pt (224)
    fig, axes = plt.subplots(2, 1, figsize=(14, 7.6))
    ps = 224
    for ax, (key, title) in zip(axes, [('kept', 'roi_overlap ≥ 5% (used)'), ('four_pt', "CLAM default four_pt (4 points around the centre)")]):
        co = grids[ps][key]
        cov = np.zeros(buf.shape, bool)
        for x, y in co:
            cov[max(y, 0):y + ps, max(x, 0):x + ps] = True
        miss = (buf > 0) & ~cov
        ov = crop.copy()
        mm = miss[cy0:cy1, cx0:cx1]
        ov[mm] = np.array([255, 45, 45])
        ax.imshow(ov, extent=ext)
        patch_rects(ax, co, ps, 'white', 1)
        ax.set_title(f'{title}: {len(co)} patches, ROI coverage {roi_coverage(buf, co, ps):.1%}; red = ROI pixels in no patch ({int(miss.sum()):,} px)')
        ax.set_xlabel('x (px)'); ax.set_ylabel('y (px)')
        facts.setdefault('missed_px', {})[key] = int(miss.sum())
    fig.tight_layout()
    save(fig, out, '06b_roi_overlap_vs_four_pt.png')

    # ------------------------------------------------------------------ 07 extracted patches
    slide = slide_reader.IENFSlide(img_path)
    for ps in [224, 128]:
        with h5py.File(f'{PREP}/p{ps}_s{ps}_bufdown5/patches/{sid}.h5') as f:
            co = f['coords'][:]
            attrs = dict(f['coords'].attrs)
        assert np.array_equal(np.sort(co, 0), np.sort(grids[ps]['kept'], 0)), 'h5 coords differ from recomputed'
        order = np.lexsort((co[:, 1], co[:, 0]))
        co = co[order]
        grids[ps]['h5'] = co
        n = len(co)
        ncol = 11 if ps == 224 else 13
        nrow = int(np.ceil(n / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(14, 1.75 * nrow + 0.8))
        for i, ax in enumerate(axes.ravel()):
            off(ax)
            if i < n:
                x, y = co[i]
                p = np.asarray(slide.read_region((int(x), int(y)), 0, (ps, ps)).convert('RGB'))
                ax.imshow(p)
                ax.set_title(f'#{i} ({x},{y})\nROI {coverage_fraction(buf, x, y, ps):.0%}', fontsize=6.5, loc='center', fontweight='normal')
        fig.suptitle(f'{ps} px: the {n} patches in patches/{sid}.h5, read through the openslide stand-in (ienf_prep/slide_reader.py), '
                     'ordered left to right', x=0.01, ha='left', fontsize=11, fontweight='semibold')
        fig.tight_layout()
        note(fig, f'h5 attrs: patch_size={attrs["patch_size"]}, patch_level={attrs["patch_level"]}, step_size={attrs["step_size"]}, '
                  f'contour_fn={attrs["contour_fn"]}, min_roi_overlap={attrs["min_roi_overlap"]}, roi_buffer_down_px={attrs["roi_buffer_down_px"]}, '
                  f'use_padding={attrs["use_padding"]}. Each tile shows #index, top-left (x,y) and the ROI share.')
        save(fig, out, f'07_patches_{ps}.png')

    # ------------------------------------------------------------------ model + attention (128 px run)
    run_dir = CLAM_ROOT / 'results' / args.run
    ps = int(args.run.split('_p')[1].split('_')[0])
    feat_dir = f'ienf_uni_p{ps}_s{ps}'
    fold = None
    for k in range(5):
        res = pickle.load(open(run_dir / f'split_{k}_results.pkl', 'rb'))
        if sid in res:
            fold, saved_p1 = k, float(res[sid]['prob'][0][1])
    model = initiate_model(Namespace(drop_out=0.25, n_classes=2, embed_dim=1024, model_size='small', model_type='clam_sb'),
                           str(run_dir / f's_{fold}_checkpoint.pt'))
    feats = torch.load(f'{FEAT_ROOT}/{feat_dir}/pt_files/{sid}.pt')
    with h5py.File(f'{FEAT_ROOT}/{feat_dir}/h5_files/{sid}.h5') as f:
        fco = f['coords'][:]
    with torch.inference_mode():
        logits, Y_prob, _, A_raw, _ = model(feats.cuda())
        A = model(feats.cuda(), attention_only=True).view(-1).cpu().numpy()
    p1 = Y_prob[0, 1].item()
    assert abs(p1 - saved_p1) < 1e-4
    A_soft = torch.softmax(torch.tensor(A), 0).numpy()
    facts.update(fold=fold, p_disease=p1, logits=[float(v) for v in logits[0].cpu()], n_bag=int(len(fco)),
                 attn_soft_max=float(A_soft.max()), attn_soft_min=float(A_soft.min()))

    # ------------------------------------------------------------------ 08 encoder input transform
    encoder, tf = get_encoder('uni_v1', target_img_size=224)
    encoder = encoder.cuda().eval()
    i_hi = int(np.argmax(A))
    x, y = fco[i_hi]
    p = slide.read_region((int(x), int(y)), 0, (ps, ps)).convert('RGB')
    t = tf(p)
    mean, std = MODEL2CONSTANTS['uni_v1']['mean'], MODEL2CONSTANTS['uni_v1']['std']
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.9), gridspec_kw={'width_ratios': [1, 1, 1, 1.25]})
    axes[0].imshow(np.asarray(p), interpolation='nearest'); axes[0].set_title(f'1. patch as read\n{ps} × {ps} px (#{i_hi}, highest attention)')
    rs = np.asarray(p.resize((224, 224), Image.BILINEAR))
    axes[1].imshow(rs); axes[1].set_title(f'2. Resize(224), bilinear\n224 × 224 (×{224 / ps:.2f})')
    vis = t.permute(1, 2, 0).numpy()
    axes[2].imshow(np.clip((vis - vis.min()) / (vis.max() - vis.min()), 0, 1))
    axes[2].set_title('3. ToTensor + Normalize\n(shown rescaled to 0–1)')
    for a in axes[:3]:
        off(a)
    for c, col in enumerate(['#d62728', '#2ca02c', '#1f77b4']):
        axes[3].hist(t[c].numpy().ravel(), bins=60, histtype='step', color=col, lw=1.3, label=f'{"RGB"[c]}: mean {mean[c]}, std {std[c]}')
    axes[3].axvline(0, color='#999', lw=0.8)
    axes[3].set_title('Normalized values per channel'); axes[3].set_xlabel('(value − ImageNet mean) / std')
    axes[3].legend(fontsize=7.5, frameon=False)
    fig.tight_layout()
    note(fig, 'CLAM transform for uni_v1 (utils/transform_utils.get_eval_transforms): Resize(224) → ToTensor (0–1) → Normalize with ImageNet '
              'mean/std. Most pixels are dark, so most normalized values are strongly negative.')
    save(fig, out, '08_encoder_input.png')

    # ------------------------------------------------------------------ 09 UNI features
    with torch.inference_mode():
        f_check = encoder(t[None].cuda()).float().cpu()[0]
    facts['feature_recompute_cos'] = float(torch.nn.functional.cosine_similarity(f_check, feats[i_hi].float(), 0))
    F = feats.float().numpy()
    order = np.lexsort((fco[:, 0], fco[:, 1]))  # grid row by grid row (y), left to right within a row
    row_breaks = np.where(np.diff(fco[order, 1]) != 0)[0] + 0.5
    Fz = (F - F.mean(0)) / (F.std(0) + 1e-6)
    U, S, Vt = np.linalg.svd(F - F.mean(0), full_matrices=False)
    pcs = (F - F.mean(0)) @ Vt[:3].T
    pc_rgb = (pcs - pcs.min(0)) / (pcs.max(0) - pcs.min(0))
    Fn = F / np.linalg.norm(F, axis=1, keepdims=True)
    sim = Fn[order] @ Fn[order].T
    fig = plt.figure(figsize=(14, 9))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.1], width_ratios=[1.6, 1])
    a = fig.add_subplot(gs[0, 0])
    im_ = a.imshow(Fz[order], aspect='auto', cmap='RdBu_r', vmin=-3, vmax=3, interpolation='nearest')
    a.set_title(f'Feature matrix: {F.shape[0]} patches × {F.shape[1]} dims (z-scored per dim)')
    a.set_xlabel('UNI feature dimension'); a.set_ylabel('Patch (grid row by row)')
    for b in row_breaks:
        a.axhline(b, color='black', lw=0.8)
    fig.colorbar(im_, ax=a, fraction=0.03)
    a = fig.add_subplot(gs[0, 1])
    im_ = a.imshow(sim, cmap='magma', vmin=sim.min(), vmax=1, interpolation='nearest')
    a.set_title('Cosine similarity between patches'); a.set_xlabel('Patch (grid row by row)'); a.set_ylabel('Patch')
    for b in row_breaks:
        a.axhline(b, color='white', lw=0.6); a.axvline(b, color='white', lw=0.6)
    fig.colorbar(im_, ax=a, fraction=0.046)
    a = fig.add_subplot(gs[1, :])
    a.imshow(crop, extent=ext)
    for (x, y), c in zip(fco, pc_rgb):
        a.add_patch(Rectangle((x, y), ps, ps, fc=c, alpha=0.75, ec='white', lw=0.4))
    a.set_xlim(cx0, cx1); a.set_ylim(cy1, cy0)
    a.set_title('First three principal components of the features as RGB: similar colour = similar features')
    a.set_xlabel('x (px)'); a.set_ylabel('y (px)')
    fig.tight_layout()
    ev = S ** 2 / (S ** 2).sum()
    note(fig, f'Features from {feat_dir}/pt_files/{sid}.pt (CLAM extract_features_fp.py, uni_v1). Mean vector length '
              f'{np.linalg.norm(F, axis=1).mean():.1f}. PC1–3 explain {ev[0]:.0%}, {ev[1]:.0%}, {ev[2]:.0%} of the variance. '
              f'Re-encoding patch #{i_hi} here gives cosine {facts["feature_recompute_cos"]:.4f} to the saved vector.')
    save(fig, out, '09_uni_features.png')
    facts.update(feat_norm=float(np.linalg.norm(F, axis=1).mean()), pc_var=[float(v) for v in ev[:3]],
                 sim_min=float(sim[~np.eye(len(sim), dtype=bool)].min()), sim_mean=float(sim[~np.eye(len(sim), dtype=bool)].mean()))

    # ------------------------------------------------------------------ 10 attention + prediction
    pct = np.array([score2percentile(s, A) for s in A])
    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 0.85], width_ratios=[2.2, 1])
    a = fig.add_subplot(gs[0, :])
    a.imshow(crop, extent=ext)
    cm = plt.get_cmap('jet')
    for (x, y), v in zip(fco, pct / 100):
        a.add_patch(Rectangle((x, y), ps, ps, fc=cm(v), alpha=0.55, ec='white', lw=0.4))
    a.set_xlim(cx0, cx1); a.set_ylim(cy1, cy0)
    a.set_title(f'Attention of the trained bag ({len(fco)} patches) as percentile within the slide (blue low, red high)')
    fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(0, 100), cmap='jet'), ax=a, fraction=0.015, label='percentile')
    a = fig.add_subplot(gs[1, 0])
    xs_o = fco[order, 0]
    a.bar(np.arange(len(order)), A_soft[order], color=cm(pct[order] / 100), width=0.85)
    a.set_xlabel('Patch (grid row by row; lines = new row, top row first)'); a.set_ylabel('softmax attention')
    for b in row_breaks:
        a.axvline(b, color='#999', lw=0.8, ls=':')
    a.set_title(f'Attention weights (sum = 1); uniform would be {1 / len(A):.3f}')
    a.axhline(1 / len(A), color='#777', lw=0.8, ls='--')
    a = fig.add_subplot(gs[1, 1]); a.axis('off')
    a.text(0, 0.95, '\n'.join([
        f'Run: {args.run}', f'Model: fold {fold} checkpoint (this slide was in', f'fold {fold}\'s test set)', '',
        f'logits  normal {facts["logits"][0]:+.2f}, disease {facts["logits"][1]:+.2f}',
        f'P(disease) = {p1:.4f}  → predicted normal', 'true label: normal (Control)', '',
        f'attention max {A_soft.max():.3f} (patch #{int(np.argmax(A_soft[order]))} from left)',
        f'attention min {A_soft.min():.4f}',
        f'top 5 patches hold {np.sort(A_soft)[-5:].sum():.0%} of the weight']), va='top', fontsize=9.5, family='monospace')
    fig.tight_layout()
    note(fig, 'CLAM-SB: h = W·x (1024→512), gated attention a = w·(tanh(V h) ⊙ σ(U h)), weights = softmax(a), slide vector = Σ weights·h, '
              'classifier on the slide vector. Percentiles are what the heatmaps colour.')
    save(fig, out, '10_attention_prediction.png')
    facts['top5_share'] = float(np.sort(A_soft)[-5:].sum())

    # ------------------------------------------------------------------ 10b what drives attention in this slide?
    from scipy.stats import spearmanr
    roi_share = np.array([coverage_fraction(buf, x, y, ps) for x, y in fco])
    bright = np.array([np.asarray(slide.read_region((int(x), int(y)), 0, (ps, ps)).convert('RGB')).mean() for x, y in fco])
    green = np.array([np.asarray(slide.read_region((int(x), int(y)), 0, (ps, ps)).convert('RGB'))[..., 1].mean() for x, y in fco])
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    for a, (v, lab) in zip(axes, [(roi_share, 'ROI share of the patch'), (bright, 'mean pixel value (all channels)'), (green, 'mean green value')]):
        r, pv = spearmanr(v, A)
        a.scatter(v, pct, c=pct, cmap='jet', vmin=0, vmax=100, s=40, edgecolors='black', linewidths=0.4)
        a.set_xlabel(lab); a.set_ylabel('attention percentile')
        a.set_title(f'Spearman r = {r:+.2f} (p = {pv:.1g}, n = {len(A)})')
        facts.setdefault('attn_spearman', {})[lab] = [round(float(r), 3), float(pv)]
    fig.tight_layout()
    note(fig, 'One dot per bag patch (128 px). Correlations within this one slide only; they describe what this model attends to here, not in general.')
    save(fig, out, '10b_attention_vs_content.png')
    facts['roi_share_top5_mean'] = float(roi_share[np.argsort(-A)[:5]].mean())
    facts['roi_share_bottom5_mean'] = float(roi_share[np.argsort(A)[:5]].mean())
    facts['roi_share_all_mean'] = float(roi_share.mean())

    # ------------------------------------------------------------------ fine heatmap scores (cached)
    cache = Path(args.out_dir) / f'fine_scores_overlap{int(args.overlap * 100)}.npz'
    step = int(round(ps * (1 - args.overlap)))
    if cache.exists():
        z = np.load(cache)
        fc, fs = z['coords'], z['pct']
    else:
        fc = generate_coords_roi_overlap(buf, ps, step, MIN_OVERLAP, True)
        raw_s = []
        for i in range(0, len(fc), 256):
            b = torch.stack([tf(slide.read_region(tuple(int(v) for v in c), 0, (ps, ps)).convert('RGB')) for c in fc[i:i + 256]]).cuda()
            with torch.inference_mode():
                raw_s.append(model(encoder(b), attention_only=True).view(-1).cpu().numpy())
        raw_s = np.concatenate(raw_s)
        fs = np.array([score2percentile(s, A) for s in raw_s])
        np.savez(cache, coords=fc, pct=fs, raw=raw_s)
    facts['fine_patches'] = int(len(fc)); facts['fine_step'] = step

    wsi = WholeSlideImage(img_path)
    wsi.initSegmentation(f'{PREP}/p{ps}_s{ps}_bufdown5/contours/{sid}.pkl')
    common = dict(wsi_object=wsi, vis_level=0, cmap='jet', alpha=0.4, patch_size=(ps, ps), blank_canvas=False,
                  use_holes=True, binarize=False, thresh=-1)
    hm95 = np.asarray(drawHeatmap(fs.copy(), fc, img_path, convert_to_percentiles=False, overlap=args.overlap, blur=True, **common))
    c50 = generate_coords_roi_overlap(buf, ps, ps // 2, MIN_OVERLAP, True)
    s50 = []
    for i in range(0, len(c50), 256):
        b = torch.stack([tf(slide.read_region(tuple(int(v) for v in c), 0, (ps, ps)).convert('RGB')) for c in c50[i:i + 256]]).cuda()
        with torch.inference_mode():
            s50.append(model(encoder(b), attention_only=True).view(-1).cpu().numpy())
    p50 = np.array([score2percentile(s, A) for s in np.concatenate(s50)])
    hm50 = np.asarray(drawHeatmap(p50.copy(), c50, img_path, convert_to_percentiles=False, overlap=0.5, **common))
    block = np.asarray(drawHeatmap(A.copy(), fco, img_path, convert_to_percentiles=True, **common))

    # ------------------------------------------------------------------ 11 heatmap stages
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    for a, (im_, t_) in zip(axes, [(block, f'Blockmap: the {len(fco)} bag patches, no overlap (what the model saw)'),
                                   (hm50, f'50% overlap: {len(c50)} patches, step {ps // 2} px'),
                                   (hm95, f'95% overlap + Gaussian blur: {len(fc)} patches, step {step} px (CLAM paper setting)')]):
        a.imshow(im_[cy0:cy1, cx0:cx1], extent=ext)
        a.set_title(t_); a.set_xlabel('x (px)'); a.set_ylabel('y (px)')
    fig.tight_layout()
    note(fig, 'All three drawn by CLAM\'s WholeSlideImage.visHeatmap (jet, alpha 0.4) inside the ROI contour. The overlapping versions score every '
              'extra patch with the same model and convert it to a percentile of the bag scores (CLAM use_ref_scores).')
    save(fig, out, '11_heatmap_stages.png')

    # ------------------------------------------------------------------ 12 zoom windows: heatmap vs original
    # attention along the band from the fine scores; pick windows at the max, the min and the median of a smoothed profile
    win = 420
    centres = fc[:, 0] + ps / 2
    xs_grid = np.arange(bx0 + win // 2, bx1 - win // 2, 20)
    prof = np.array([fs[np.abs(centres - xg) < win / 2].mean() if np.any(np.abs(centres - xg) < win / 2) else np.nan for xg in xs_grid])
    picks = {'high': xs_grid[np.nanargmax(prof)], 'low': xs_grid[np.nanargmin(prof)]}
    mid = np.nanmedian(prof)
    cand = [(abs(v - mid), xg) for v, xg in zip(prof, xs_grid) if abs(xg - picks['high']) > win and abs(xg - picks['low']) > win]
    picks['middle'] = min(cand)[1]
    windows = []
    for name in ['high', 'middle', 'low']:
        xc = int(picks[name])
        colm = buf[:, max(xc - win // 2, 0):xc + win // 2]
        yy = np.nonzero(colm.any(1))[0]
        yc_ = int((yy.min() + yy.max()) / 2)
        hgt = max(260, int(yy.max() - yy.min()) + 120)
        x0, x1 = max(0, xc - win // 2), min(W, xc + win // 2)
        y0, y1 = max(0, yc_ - hgt // 2), min(H, yc_ + hgt // 2)
        windows.append((name, x0, y0, x1, y1, float(prof[list(xs_grid).index(picks[name])])))
    facts['zoom_windows'] = [dict(name=n, x0=int(a), y0=int(b), x1=int(c), y1=int(d), mean_pct=round(m, 1)) for n, a, b, c, d, m in windows]

    fig = plt.figure(figsize=(14, 13))
    gs = fig.add_gridspec(4, 3, height_ratios=[0.75, 1, 1, 1])
    a = fig.add_subplot(gs[0, :])
    a.imshow(hm95[cy0:cy1, cx0:cx1], extent=ext)
    a2 = a.twinx()
    a2.plot(xs_grid, prof, color='white', lw=1.5)
    a2.set_ylim(0, 160); a2.set_yticks([0, 50, 100]); a2.set_ylabel('mean percentile\n(420 px window)', fontsize=8)
    for n, x0, y0, x1, y1, m in windows:
        a.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec='white', lw=1.5))
        a.text(x0 + 5, y0 - 8, n, color='white', fontsize=9, fontweight='bold')
    a.set_title('Where the zoom windows are (white boxes) and the attention profile along the band (white line)')
    for r, (n, x0, y0, x1, y1, m) in enumerate(windows, start=1):
        for c, (im_, t_) in enumerate([(rgb, 'original'), (hm95, 'heatmap 95% + blur'), (None, 'original + bag patch grid')]):
            ax = fig.add_subplot(gs[r, c])
            if im_ is None:
                ax.imshow(rgb[y0:y1, x0:x1], extent=[x0, x1, y1, y0])
                inside = fco[(fco[:, 0] + ps > x0) & (fco[:, 0] < x1) & (fco[:, 1] + ps > y0) & (fco[:, 1] < y1)]
                for (px_, py_) in inside:
                    v = pct[np.where((fco == (px_, py_)).all(1))[0][0]]
                    ax.add_patch(Rectangle((px_, py_), ps, ps, fill=False, ec=cm(v / 100), lw=2))
                    ax.text(max(px_, x0) + 4, max(py_, y0) + 14, f'{v:.0f}', color=cm(v / 100), fontsize=8, fontweight='bold')
                ax.plot(np.r_[tissue[0][:, 0, 0], tissue[0][0, 0, 0]], np.r_[tissue[0][:, 0, 1], tissue[0][0, 0, 1]], color=C_ROI, lw=0.8)
                ax.set_xlim(x0, x1); ax.set_ylim(y1, y0)
            else:
                ax.imshow(im_[y0:y1, x0:x1], extent=[x0, x1, y1, y0])
            ax.set_title(f'{n} attention (mean pct {m:.0f}): {t_}', fontsize=9.5)
            ax.set_xticks([]); ax.set_yticks([])
            if c == 0:
                scalebar(ax, 100)
    fig.tight_layout()
    note(fig, 'Windows are 420 px (≈ 270 µm) wide, centred where the smoothed attention profile is highest, lowest and closest to its median. '
              'Right column: bag patches (128 px) outlined in their attention colour with the percentile; yellow = ROI contour.')
    save(fig, out, '12_zoom_heatmap_vs_original.png')

    # ------------------------------------------------------------------ 13 top / bottom patches at native resolution
    k = 5
    hi_idx, lo_idx = np.argsort(-A)[:k], np.argsort(A)[:k]
    fig, axes = plt.subplots(2, k, figsize=(14, 7.6), gridspec_kw={'hspace': 0.35})
    for row, (idx, lab) in enumerate([(hi_idx, 'highest'), (lo_idx, 'lowest')]):
        for j, i in enumerate(idx):
            x, y = fco[i]
            p = np.asarray(slide.read_region((int(x), int(y)), 0, (ps, ps)).convert('RGB'))
            ax = axes[row, j]
            ax.imshow(p, interpolation='nearest')
            m = buf[max(y, 0):y + ps, max(x, 0):x + ps]
            ax.contour(m, levels=[0.5], colors=[C_ROI], linewidths=0.8)
            ax.set_title(f'{lab} #{j + 1}: ({x},{y})\nweight {A_soft[i]:.3f}, pct {pct[i]:.0f}', fontsize=8.5, loc='center')
            off(ax)
    fig.suptitle('The 5 bag patches with the highest (top) and lowest (bottom) attention, native 128 px, nearest-neighbour upscaling',
                 x=0.01, ha='left', fontsize=11, fontweight='semibold')
    fig.tight_layout()
    note(fig, 'Yellow line: buffered ROI boundary inside the patch. Each patch is ≈ 83 µm wide at the assumed scale.')
    save(fig, out, '13_top_bottom_patches.png')
    facts['top_patches'] = [dict(x=int(fco[i][0]), y=int(fco[i][1]), w=round(float(A_soft[i]), 4), roi=round(coverage_fraction(buf, *fco[i], ps), 2)) for i in hi_idx]
    facts['bottom_patches'] = [dict(x=int(fco[i][0]), y=int(fco[i][1]), w=round(float(A_soft[i]), 5), roi=round(coverage_fraction(buf, *fco[i], ps), 2)) for i in lo_idx]

    # ------------------------------------------------------------------ 14 ROI edge at pixel level
    n, x0, y0, x1, y1, _ = windows[0]
    xm = (x0 + x1) // 2
    ex0, ex1 = xm - 80, xm + 80
    colm = buf[:, ex0:ex1]
    yy = np.nonzero(colm.any(1))[0]
    ey0, ey1 = int(yy.min()) - 25, int(yy.max()) + 25
    fig, axes = plt.subplots(1, 3, figsize=(14, 5.6))
    sub = rgb[ey0:ey1, ex0:ex1]
    axes[0].imshow(sub, extent=[ex0, ex1, ey1, ey0], interpolation='nearest'); axes[0].set_title('Original pixels (no smoothing)')
    ov = sub.copy()
    mr, ma = raw[ey0:ey1, ex0:ex1] > 0, added[ey0:ey1, ex0:ex1]
    ov[mr] = (0.65 * ov[mr] + 0.35 * np.array([255, 212, 0])).astype(np.uint8); ov[ma] = np.array([255, 122, 0])
    axes[1].imshow(ov, extent=[ex0, ex1, ey1, ey0], interpolation='nearest'); axes[1].set_title('ROI (yellow) + 5 px buffer (orange)')
    axes[2].imshow(hm95[ey0:ey1, ex0:ex1], extent=[ex0, ex1, ey1, ey0], interpolation='nearest'); axes[2].set_title('Heatmap 95% + blur')
    for a in axes:
        a.plot(poly[:, 0], poly[:, 1], 'o', ms=3, color='white', mec='black', mew=0.4)
        a.set_xlim(ex0, ex1); a.set_ylim(ey1, ey0); a.set_xlabel('x (px)')
    axes[0].set_ylabel('y (px)')
    fig.tight_layout()
    note(fig, f'160 px wide strip (≈ {160 * UM_PER_PX:.0f} µm) through the "high" window; white dots = ROI vertices. Shows how closely the hand-drawn ROI follows '
              'the epidermis and where the coloured area ends.')
    save(fig, out, '14_roi_edge_pixels.png')

    # ------------------------------------------------------------------ 15 channels in the high and low windows
    fig, axes = plt.subplots(2, 4, figsize=(14, 6.6))
    for r, w_ in enumerate([windows[0], windows[2]]):
        n, x0, y0, x1, y1, m = w_
        sub = rgb[y0:y1, x0:x1]
        for c, (im_, t_, cmap) in enumerate([(sub, 'RGB', None), (sub[..., 0], 'red channel', 'gray'),
                                             (sub[..., 1], 'green channel', 'gray'), (sub[..., 2], 'blue channel', 'gray')]):
            ax = axes[r, c]
            ax.imshow(im_, cmap=cmap, vmin=0 if cmap else None, vmax=255 if cmap else None)
            ax.contour(buf[y0:y1, x0:x1], levels=[0.5], colors=[C_ROI], linewidths=0.7)
            ax.set_title(f'{n} attention: {t_}', fontsize=9.5)
            off(ax)
    fig.tight_layout()
    note(fig, 'Same windows as Figure 12 ("high" top, "low" bottom), split into the three RGB channels of the exported TIFF; yellow = ROI boundary. '
              'Channel display range 0–255 for all.')
    save(fig, out, '15_channels_high_low.png')

    # channel statistics inside the ROI part of each window
    for n, x0, y0, x1, y1, m in windows:
        mm = buf[y0:y1, x0:x1] > 0
        facts.setdefault('window_channel_means', {})[n] = [round(float(rgb[y0:y1, x0:x1][..., c][mm].mean()), 1) for c in range(3)]

    with open(Path(args.out_dir) / 'facts.json', 'w') as fh:
        json.dump(facts, fh, indent=1, default=float)
    print(json.dumps(facts, indent=1, default=float))


if __name__ == '__main__':
    main()
