"""Picture-first, one-figure-per-step version of the IENF -> CLAM walkthrough (default S1616-2_b).

Reuses the helpers and constants of heatmaps/ienf_walkthrough.py and its cached 95%-overlap scores
(run that script first). Figures:

  s0 overview (all steps on one page)   s5 extracted patches
  s1 original image                     s6 UNI features
  s2 epidermis ROI                      s7 CLAM attention + prediction
  s3 mask + 5 px buffer                 s8 heatmap vs. original
  s4 patch grid (224 and 128 px)

Usage (repo root, clam env):
  python heatmaps/ienf_walkthrough_simple.py \\
      --slide S1616-2_b \\
      --run ienf_2cls_uni_p128_CLAM_B8_s1 \\
      --cache results/ienf_preprocessing_walkthrough/fine_scores_overlap95.npz \\
      --out_dir results/ienf_preprocessing_walkthrough_simple
"""
import argparse
import json
import pickle
import sys
from argparse import Namespace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ienf_walkthrough as wt  # noqa: E402  (also sets up sys.path and the openslide stand-in)

import h5py  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from matplotlib.colors import Normalize  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Patch, Rectangle  # noqa: E402
from PIL import Image  # noqa: E402

from ienf_prep import slide_reader  # noqa: E402
from ienf_prep.patching import generate_coords_roi_overlap  # noqa: E402
from ienf_prep.roi import extend_mask_toward_dermis, mask_to_clam_contours, polygon_to_mask, read_roi_polygon  # noqa: E402
from utils.eval_utils import initiate_model  # noqa: E402
from vis_utils.heatmap_utils import drawHeatmap, score2percentile  # noqa: E402
from wsi_core.WholeSlideImage import WholeSlideImage  # noqa: E402

plt.rcParams.update({'font.size': 11, 'axes.titlesize': 13})
off, save, scalebar = wt.off, wt.save, wt.scalebar


def step_title(ax, n, text):
    ax.set_title(f'Step {n}  ·  {text}', loc='left', fontsize=13, fontweight='semibold')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--slide', default='S1616-2_b')
    ap.add_argument('--run', default='ienf_2cls_uni_p128_CLAM_B8_s1')
    ap.add_argument('--cache', default='results/ienf_preprocessing_walkthrough/fine_scores_overlap95.npz')
    ap.add_argument('--out_dir', default='results/ienf_preprocessing_walkthrough_simple')
    args = ap.parse_args()
    sid = args.slide
    out = Path(args.out_dir) / 'figures'
    out.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ inputs (same as the detailed walkthrough)
    img_path = f'{wt.DATA_ROOT}/Image/Original/{sid}.tif'
    roi_path = f'{wt.DATA_ROOT}/Epidermis_ROI/{sid}.roi'
    with Image.open(img_path) as im:
        size = im.size
        rgb = np.asarray(im.convert('RGB')).copy()
    W, H = size
    poly = read_roi_polygon(roi_path)
    raw = polygon_to_mask(poly, size)
    buf = extend_mask_toward_dermis(raw, wt.BUFFER, dermis_up=False)
    added = (buf > 0) & (raw == 0)
    tissue, _ = mask_to_clam_contours(buf)
    ys, xs = np.nonzero(buf)
    bx0, by0, bx1, by1 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
    cx0, cy0, cx1, cy1 = max(0, bx0 - 120), max(0, by0 - 120), min(W, bx1 + 120), min(H, by1 + 120)
    crop = rgb[cy0:cy1, cx0:cx1]
    ext = [cx0, cx1, cy1, cy0]
    cont = tissue[0][:, 0]
    cont = np.r_[cont, cont[:1]]
    slide = slide_reader.IENFSlide(img_path)
    facts = dict(slide=sid, W=W, H=H, n_vertices=len(poly), buffer_added=int(added.sum()))

    def show_crop(ax, img=None):
        ax.imshow(crop if img is None else img[cy0:cy1, cx0:cx1], extent=ext)
        ax.set_xlim(cx0, cx1); ax.set_ylim(cy1, cy0)
        off(ax)

    def tint(img, m, color, a):
        img[m] = ((1 - a) * img[m] + a * np.array(color)).astype(np.uint8)
        return img

    roi_img = tint(rgb.copy(), raw > 0, (255, 212, 0), 0.35)
    mask_img = tint(roi_img.copy(), added, (255, 122, 0), 1.0)

    # ------------------------------------------------------------------ grids, patches, model
    grids = {}
    for ps in [224, 128]:
        cand = [(x, y) for x in range(bx0, bx1 + 1, ps) for y in range(by0, by1 + 1, ps)]
        fr = np.array([wt.coverage_fraction(buf, x, y, ps) for x, y in cand])
        with h5py.File(f'{wt.PREP}/p{ps}_s{ps}_bufdown5/patches/{sid}.h5') as f:
            co = f['coords'][:]
        assert np.array_equal(np.sort(co, 0), np.sort(generate_coords_roi_overlap(buf, ps, ps, wt.MIN_OVERLAP, True), 0))
        grids[ps] = dict(cand=np.array(cand), frac=fr, kept=co[np.lexsort((co[:, 0], co[:, 1]))])
        facts[f'p{ps}'] = dict(candidates=len(cand), kept=int(len(co)))

    run_dir = wt.CLAM_ROOT / 'results' / args.run
    ps = int(args.run.split('_p')[1].split('_')[0])
    feat_dir = f'ienf_uni_p{ps}_s{ps}'
    for k in range(5):
        res = pickle.load(open(run_dir / f'split_{k}_results.pkl', 'rb'))
        if sid in res:
            fold, saved_p1 = k, float(res[sid]['prob'][0][1])
    model = initiate_model(Namespace(drop_out=0.25, n_classes=2, embed_dim=1024, model_size='small', model_type='clam_sb'),
                           str(run_dir / f's_{fold}_checkpoint.pt'))
    feats = torch.load(f'{wt.FEAT_ROOT}/{feat_dir}/pt_files/{sid}.pt')
    with h5py.File(f'{wt.FEAT_ROOT}/{feat_dir}/h5_files/{sid}.h5') as f:
        fco = f['coords'][:]
    with torch.inference_mode():
        _, Y_prob, _, _, _ = model(feats.cuda())
        A = model(feats.cuda(), attention_only=True).view(-1).cpu().numpy()
    p1 = Y_prob[0, 1].item()
    assert abs(p1 - saved_p1) < 1e-4
    pct = np.array([score2percentile(s, A) for s in A])
    order = np.lexsort((fco[:, 0], fco[:, 1]))
    facts.update(fold=fold, p_disease=p1, n_bag=int(len(fco)))

    z = np.load(args.cache)
    fc, fs = z['coords'], z['pct']
    wsi = WholeSlideImage(img_path)
    wsi.initSegmentation(f'{wt.PREP}/p{ps}_s{ps}_bufdown5/contours/{sid}.pkl')
    hm = np.asarray(drawHeatmap(fs.copy(), fc, img_path, wsi_object=wsi, vis_level=0, cmap='jet', alpha=0.4,
                                patch_size=(ps, ps), blank_canvas=False, use_holes=True, binarize=False, thresh=-1,
                                convert_to_percentiles=False, overlap=0.95, blur=True))
    facts.update(fine_patches=int(len(fc)))
    cm = plt.get_cmap('jet')

    def draw_grid(ax, g, ps, lw=1.4):
        for (x, y), f in zip(g['cand'], g['frac']):
            if f >= wt.MIN_OVERLAP:
                ax.add_patch(Rectangle((x, y), ps, ps, fill=False, ec=wt.C_KEEP, lw=lw))
            else:
                ax.add_patch(Rectangle((x, y), ps, ps, fill=False, ec='#bdbdbd', lw=0.7, ls=':'))

    def draw_attention(ax):
        for (x, y), v in zip(fco, pct / 100):
            ax.add_patch(Rectangle((x, y), ps, ps, fc=cm(v), alpha=0.6, ec='white', lw=0.4))

    # ------------------------------------------------------------------ s1 original
    fig, ax = plt.subplots(figsize=(14, 4.4))
    ax.imshow(rgb); off(ax)
    ax.add_patch(Rectangle((cx0, cy0), cx1 - cx0, cy1 - cy0, fill=False, ec='white', lw=1.2, ls='--'))
    ax.text(cx0 + 10, cy0 - 15, 'region shown in the next steps', color='white', fontsize=10)
    step_title(ax, 1, f'Original image: {sid}.tif, {W} × {H} px RGB')
    save(fig, out, 's1_original.png')

    # ------------------------------------------------------------------ s2 ROI
    fig, ax = plt.subplots(figsize=(14, 3.6))
    show_crop(ax, roi_img)
    ax.plot(poly[:, 0].tolist() + [poly[0, 0]], poly[:, 1].tolist() + [poly[0, 1]], color=wt.C_ROI, lw=1.4)
    ax.text(bx0 + 30, by0 - 30, 'skin surface', color='white', fontsize=10)
    ax.text(bx0 + 30, by1 + 70, 'dermis', color='white', fontsize=10)
    scalebar(ax, 200)
    step_title(ax, 2, f'Hand-drawn epidermis ROI (ImageJ .roi, {len(poly)} points)')
    save(fig, out, 's2_roi.png')

    # ------------------------------------------------------------------ s3 mask + buffer
    zx0, zx1 = 2300, 2500
    lower = np.array([np.nonzero(raw[:, x])[0].max() for x in range(zx0, zx1) if raw[:, x].any()])
    yc = int(np.median(lower))
    zy0, zy1 = yc - 45, yc + 30
    fig = plt.figure(figsize=(14, 6.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.25])
    ax = fig.add_subplot(gs[0, :])
    ax.imshow(buf[cy0:cy1, cx0:cx1], cmap='gray', extent=ext, interpolation='nearest'); off(ax)
    ax.add_patch(Rectangle((zx0, zy0), zx1 - zx0, zy1 - zy0, fill=False, ec=wt.C_BUF, lw=1.5))
    step_title(ax, 3, f'ROI → binary mask, then +{wt.BUFFER} px toward the dermis')
    for j, (img, t) in enumerate([(rgb, 'zoom: original'), (mask_img, f'zoom: ROI (yellow) + {wt.BUFFER} px buffer (orange)')]):
        a = fig.add_subplot(gs[1, j])
        a.imshow(img[zy0:zy1, zx0:zx1], interpolation='nearest'); off(a)
        a.set_title(t, loc='left', fontsize=11)
    fig.tight_layout()
    save(fig, out, 's3_mask_buffer.png')

    # ------------------------------------------------------------------ s4 grid
    fig, axes = plt.subplots(2, 1, figsize=(14, 7.2))
    for a, ps_ in zip(axes, [224, 128]):
        show_crop(a)
        draw_grid(a, grids[ps_], ps_)
        a.plot(cont[:, 0], cont[:, 1], color=wt.C_ROI, lw=0.8)
        f_ = facts[f'p{ps_}']
        a.set_title(f'{ps_} px: {f_["candidates"]} grid squares → {f_["kept"]} kept', loc='left', fontsize=11)
    step_title(axes[0], 4, 'Lay a grid over the ROI, keep squares with ≥ 5% ROI')
    axes[0].set_title(f'224 px: {facts["p224"]["candidates"]} grid squares → {facts["p224"]["kept"]} kept', loc='right', fontsize=11)
    axes[1].legend(handles=[Patch(fc='none', ec=wt.C_KEEP, label='kept patch'),
                            Patch(fc='none', ec='#bdbdbd', ls=':', label='dropped (< 5% ROI)'),
                            Patch(fc='none', ec=wt.C_ROI, label='ROI + buffer')],
                   loc='upper center', bbox_to_anchor=(0.5, -0.02), ncol=3, frameon=False)
    fig.tight_layout()
    save(fig, out, 's4_grid.png')

    # ------------------------------------------------------------------ s5 patches
    def mosaic(co, ps_, ncol, gap):
        nrow = int(np.ceil(len(co) / ncol))
        m = np.full((nrow * (ps_ + gap) - gap, ncol * (ps_ + gap) - gap, 3), 252, np.uint8)
        for i, (x, y) in enumerate(co):
            r, c = divmod(i, ncol)
            m[r * (ps_ + gap):r * (ps_ + gap) + ps_, c * (ps_ + gap):c * (ps_ + gap) + ps_] = \
                np.asarray(slide.read_region((int(x), int(y)), 0, (ps_, ps_)).convert('RGB'))
        return m

    m224, m128 = mosaic(grids[224]['kept'], 224, 11, 10), mosaic(grids[128]['kept'], 128, 13, 6)
    fig, axes = plt.subplots(2, 1, figsize=(14, 9.6), gridspec_kw={'height_ratios': [m224.shape[0] / m224.shape[1],
                                                                                    m128.shape[0] / m128.shape[1]]})
    for a, m, ps_ in [(axes[0], m224, 224), (axes[1], m128, 128)]:
        a.imshow(m); off(a)
        a.set_title(f'{ps_} px: {len(grids[ps_]["kept"])} patches (= one bag)', loc='left', fontsize=11)
    fig.suptitle('Step 5  ·  Cut out the kept patches (row by row, left to right)', x=0.01, ha='left',
                 fontsize=13, fontweight='semibold')
    fig.tight_layout()
    save(fig, out, 's5_patches.png')

    # ------------------------------------------------------------------ s6 features
    F = feats.float().numpy()
    Fz = (F - F.mean(0)) / (F.std(0) + 1e-6)
    ex = order[[0, len(order) // 3, 2 * len(order) // 3, len(order) - 1]]
    fig = plt.figure(figsize=(14, 5.2))
    gs = fig.add_gridspec(len(ex), 4, width_ratios=[0.4, 0.9, 2.2, 2.6], wspace=0.15, hspace=0.25)
    for r, i in enumerate(ex):
        a = fig.add_subplot(gs[r, 0])
        x, y = fco[i]
        a.imshow(np.asarray(slide.read_region((int(x), int(y)), 0, (ps, ps)).convert('RGB'))); off(a)
        if r == 0:
            a.set_title('example patches', loc='left', fontsize=11)
        a = fig.add_subplot(gs[r, 2])
        a.imshow(Fz[i][None, :], aspect='auto', cmap='RdBu_r', vmin=-3, vmax=3, interpolation='nearest'); off(a)
        if r == 0:
            a.set_title('1024 numbers per patch', loc='left', fontsize=11)
    a = fig.add_subplot(gs[:, 1]); a.axis('off')
    a.add_patch(FancyBboxPatch((0.1, 0.3), 0.8, 0.4, boxstyle='round,pad=0.02', fc='#e8eef7', ec='#4a6fa5', lw=1.5,
                               transform=a.transAxes))
    a.text(0.5, 0.5, 'UNI\nViT-L/16\n(frozen)', ha='center', va='center', fontsize=11, transform=a.transAxes)
    a.annotate('', xy=(0.1, 0.5), xytext=(-0.25, 0.5), xycoords='axes fraction', arrowprops=dict(arrowstyle='->', lw=1.5))
    a.annotate('', xy=(1.2, 0.5), xytext=(0.9, 0.5), xycoords='axes fraction', arrowprops=dict(arrowstyle='->', lw=1.5))
    a = fig.add_subplot(gs[:, 3])
    a.imshow(Fz[order], aspect='auto', cmap='RdBu_r', vmin=-3, vmax=3, interpolation='nearest')
    a.set_xlabel('feature dimension (1024)'); a.set_ylabel(f'patch ({len(F)})'); a.set_xticks([]); a.set_yticks([])
    a.set_title(f'whole slide: {len(F)} × 1024 matrix', loc='left', fontsize=11)
    fig.suptitle(f'Step 6  ·  Each {ps} px patch → resize to 224 → UNI → feature vector', x=0.01, y=0.99, ha='left',
                 fontsize=13, fontweight='semibold')
    save(fig, out, 's6_features.png')

    # ------------------------------------------------------------------ s7 attention + prediction
    fig, ax = plt.subplots(figsize=(14, 3.9))
    show_crop(ax)
    draw_attention(ax)
    ax.text(cx1 - 20, cy0 + 40, f'P(disease) = {p1:.4f} → predicted normal (true: normal)', ha='right', va='top',
            color='white', fontsize=11, bbox=dict(fc='black', alpha=0.6, ec='none'))
    fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(0, 100), cmap='jet'), ax=ax, fraction=0.015, pad=0.01,
                 label='attention percentile')
    step_title(ax, 7, f'CLAM weighs each of the {len(fco)} patches (attention) and classifies the slide')
    save(fig, out, 's7_attention.png')

    # ------------------------------------------------------------------ s8 heatmap vs original
    fig, axes = plt.subplots(2, 1, figsize=(14, 6.8))
    show_crop(axes[0])
    axes[0].plot(cont[:, 0], cont[:, 1], color=wt.C_ROI, lw=0.8)
    axes[0].set_title('original (yellow = ROI)', loc='left', fontsize=11)
    show_crop(axes[1], hm)
    axes[1].set_title(f'heatmap: {len(fc):,} overlapping patches (95% overlap) + blur, red = high attention',
                      loc='left', fontsize=11)
    step_title(axes[0], 8, 'Heatmap: attention painted back on the image')
    axes[0].set_title('original (yellow = ROI)', loc='right', fontsize=11)
    fig.tight_layout()
    save(fig, out, 's8_heatmap.png')

    # ------------------------------------------------------------------ s0 overview
    fig, axes = plt.subplots(6, 1, figsize=(12, 16.8))
    labels = ['1  original', '2  epidermis ROI', f'3  mask + {wt.BUFFER} px buffer', f'4  {ps} px patch grid',
              '7  attention per patch', '8  heatmap']
    show_crop(axes[0])
    show_crop(axes[1], roi_img)
    axes[1].plot(poly[:, 0].tolist() + [poly[0, 0]], poly[:, 1].tolist() + [poly[0, 1]], color=wt.C_ROI, lw=1)
    axes[2].imshow(buf[cy0:cy1, cx0:cx1], cmap='gray', extent=ext, interpolation='nearest'); off(axes[2])
    show_crop(axes[3]); draw_grid(axes[3], grids[ps], ps, lw=1)
    show_crop(axes[4]); draw_attention(axes[4])
    show_crop(axes[5], hm)
    for a, l in zip(axes, labels):
        a.set_title(l, loc='left', fontsize=12, fontweight='semibold')
    fig.suptitle(f'{sid}: from image to heatmap (steps 5–6, patches → UNI features, run between 4 and 7)',
                 x=0.01, ha='left', fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    save(fig, out, 's0_overview.png')

    with open(Path(args.out_dir) / 'facts.json', 'w') as fh:
        json.dump(facts, fh, indent=1, default=float)
    print(json.dumps(facts, indent=1, default=float))


if __name__ == '__main__':
    main()
