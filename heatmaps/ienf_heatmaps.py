"""Attention heatmaps for the IENF 2-class CLAM runs, using CLAM's own heatmap tools.

CLAM's create_heatmaps.py segments tissue with Otsu/HSV thresholds and opens slides with OpenSlide.
Neither fits IENF: the images are fluorescence, the TIFFs are plain (not pyramidal), and the bags come
from the epidermis ROI. This script runs the same steps as create_heatmaps.py with CLAM's own pieces:

  1. model:        utils.eval_utils.initiate_model with the checkpoint of the fold that TESTED the slide
  2. blockmap:     attention of the trained bag (saved UNI features + coords), CLAM's drawHeatmap
                   with convert_to_percentiles=True (= the "_blockmap" output of create_heatmaps.py)
  3. fine heatmap: patches with --overlap on the same ROI rule, UNI features via models.get_encoder,
                   attention converted to percentiles of the bag scores (score2percentile, as with
                   use_ref_scores: true), drawn with drawHeatmap(overlap=...)
  4. drawing:      WholeSlideImage.visHeatmap with the ROI contours loaded through initSegmentation
                   (segment=True), so only the buffered epidermis band is colored

Slides are chosen from the out-of-fold test predictions: the --n most confident true positives, true
negatives, false negatives and false positives (one slide per patient). Each slide's images are saved
separately; a combined overview figure is saved on its own.

Usage (repo root, clam env):
  UNI_CKPT_PATH=~/projects/UNI/assets/ckpts/uni/pytorch_model.bin \\
  python heatmaps/ienf_heatmaps.py \\
      --run ienf_2cls_uni_p128_CLAM_B8_s1 \\
      --variant p128_s128_bufdown5 --feature_dir ienf_uni_p128_s128 \\
      --n 2 --overlap 0.95 --blur \\
      --out_dir heatmaps/ienf/ienf_2cls_uni_p128_CLAM_B8_s1_overlap95_blur
"""
import argparse
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
sys.modules['openslide'] = slide_reader  # CLAM's WholeSlideImage opens slides through this stand-in

import h5py  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from matplotlib.colors import Normalize  # noqa: E402
from PIL import Image  # noqa: E402

from ienf_prep.patching import generate_coords_roi_overlap  # noqa: E402
from ienf_prep.roi import build_roi_contours, read_image_size  # noqa: E402
from models import get_encoder  # noqa: E402
from utils.eval_utils import initiate_model  # noqa: E402
from vis_utils.heatmap_utils import drawHeatmap, score2percentile  # noqa: E402
from wsi_core.WholeSlideImage import WholeSlideImage  # noqa: E402

DATA_ROOT = '/data/datasets/neuroimages/20250214'
PREP = '/var/tmp/shouteng_data/IENF_CLAM/clam_prep'
FEAT_ROOT = '/var/tmp/shouteng_data/CLAM_DATA'
DERMIS_UP = ['S1357-2_b']  # same list as create_patches_roi.py
GROUPS = [('TP', 'correct, disease'), ('TN', 'correct, normal'), ('FN', 'missed disease'), ('FP', 'false alarm')]
CMAP = 'jet'
ALPHA = 0.4  # CLAM config_template.yaml default


def test_predictions(run_dir, k):
    lab3 = pd.read_csv(CLAM_ROOT / 'dataset_csv' / 'ienf_3cls.csv').set_index('slide_id')
    rows = []
    for fold in range(k):
        res = pickle.load(open(os.path.join(run_dir, f'split_{fold}_results.pkl'), 'rb'))
        for sid, v in res.items():
            rows.append(dict(slide_id=sid, fold=fold, y=int(v['label']), p1=float(v['prob'][0][1]),
                             subtype=lab3.label[sid], case_id=lab3.case_id[sid]))
    d = pd.DataFrame(rows)
    d['pred'] = (d.p1 > 0.5).astype(int)
    d['group'] = np.select([(d.y == 1) & (d.pred == 1), (d.y == 0) & (d.pred == 0), (d.y == 1) & (d.pred == 0)],
                           ['TP', 'TN', 'FN'], 'FP')
    # confidence in the predicted class
    d['conf'] = np.where(d.pred == 1, d.p1, 1 - d.p1)
    return d


def pick(d, n):
    out = []
    for g, _ in GROUPS:
        sub = d[d.group == g].sort_values('conf', ascending=False).drop_duplicates('case_id')
        out.append(sub.head(n))
    return pd.concat(out, ignore_index=True)


def attention(model, feats):
    with torch.inference_mode():
        A = model(feats.cuda(), attention_only=True)  # 1 x N raw attention (CLAM-SB)
    return A.view(-1).cpu().numpy()


def fine_scores(model, encoder, tf, slide, mask, patch_size, step, ref_scores, batch_size=256):
    coords = generate_coords_roi_overlap(mask, patch_size, step, 0.05, True)
    scores = []
    for i in range(0, len(coords), batch_size):
        batch = torch.stack([tf(slide.read_region(tuple(int(v) for v in c), 0, (patch_size, patch_size)).convert('RGB'))
                             for c in coords[i:i + batch_size]]).cuda()
        with torch.inference_mode():
            scores.append(attention(model, encoder(batch)))
    raw = np.concatenate(scores)
    pct = np.array([score2percentile(s, ref_scores) for s in raw])  # as compute_from_patches(ref_scores=...)
    return coords, pct


def roi_box(mask, margin, size):
    ys, xs = np.nonzero(mask)
    x0, y0 = max(0, xs.min() - margin), max(0, ys.min() - margin)
    x1, y1 = min(size[0], xs.max() + margin), min(size[1], ys.max() + margin)
    return (x0, y0), (x1, y1)


def draw_overview(rec, out, args, patch_size, step):
    """Original | heatmap (ROI crop), one row per slide, grouped TP, TN, FN, FP; reads the saved slide images."""
    n = len(rec)
    fig, axes = plt.subplots(n, 2, figsize=(14, 2.1 * n + 0.8))
    for i, r in rec.iterrows():
        tag = f'{r.group}_{r.slide_id}'
        for j, name in enumerate(['orig_roi', 'heatmap_roi']):
            ax = axes[i, j]
            ax.imshow(Image.open(out / 'slides' / f'{tag}_{name}.jpg'))
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
        label = 'disease' if r.y == 1 else 'normal'
        desc = dict(GROUPS)[r.group]
        axes[i, 0].set_title(f'{r.group} ({desc}): {r.slide_id}, true {label} ({r.subtype}), '
                             f'P(disease) = {r.p1:.3f}, fold {r.fold}', loc='left', fontsize=9)
    axes[0, 0].set_title('Original (ROI crop)\n' + axes[0, 0].get_title(loc='left'), loc='left', fontsize=9)
    axes[0, 1].set_title('Attention heatmap (ROI crop)\n', loc='left', fontsize=9)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    cax = fig.add_axes([0.55, 0.045, 0.40, 0.010])
    fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(0, 100), cmap=CMAP), cax=cax, orientation='horizontal',
                 label='Attention percentile within the slide (blue = low, red = high)')
    fig.text(0.01, 0.012, f'{args.run}: CLAM-SB, UNI features, {patch_size}-px patches; '
             f'heatmap with {int(round(args.overlap * 100))}% patch overlap (step {step} px)' + (', Gaussian blur' if args.blur else '') + '. Out-of-fold test predictions.',
             fontsize=8, color='#52514e')
    fig.savefig(out / 'overview.png', dpi=130)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', default='ienf_2cls_uni_p128_CLAM_B8_s1')
    ap.add_argument('--variant', default='p128_s128_bufdown5')
    ap.add_argument('--feature_dir', default='ienf_uni_p128_s128')
    ap.add_argument('--k', type=int, default=5)
    ap.add_argument('--n', type=int, default=2, help='slides per group (TP, TN, FN, FP)')
    ap.add_argument('--overlap', type=float, default=0.5, help='patch overlap of the fine heatmap (CLAM paper: 0.95)')
    ap.add_argument('--blur', action='store_true', help="CLAM's visHeatmap Gaussian blur (kernel 2 * step + 1)")
    ap.add_argument('--batch_size', type=int, default=256, help='UNI batch size for the fine heatmap')
    ap.add_argument('--out_dir', default=None)
    ap.add_argument('--overview_only', action='store_true',
                    help='only redraw overview.png from selected_slides.csv and the saved slide images')
    args = ap.parse_args()

    run_dir = CLAM_ROOT / 'results' / args.run
    out = Path(args.out_dir or CLAM_ROOT / 'heatmaps' / 'ienf' / args.run)
    (out / 'slides').mkdir(parents=True, exist_ok=True)
    patch_size = int(args.variant.split('_')[0][1:])
    step = int(round(patch_size * (1 - args.overlap)))

    if args.overview_only:
        draw_overview(pd.read_csv(out / 'selected_slides.csv'), out, args, patch_size, step)
        print('wrote', out / 'overview.png')
        return

    preds = test_predictions(run_dir, args.k)
    chosen = pick(preds, args.n)
    print(chosen.to_string(index=False))

    model_args = Namespace(drop_out=0.25, n_classes=2, embed_dim=1024, model_size='small', model_type='clam_sb')
    encoder, tf = get_encoder('uni_v1', target_img_size=224)
    encoder = encoder.cuda().eval()

    records = []
    models = {}
    for _, r in chosen.iterrows():
        sid = r.slide_id
        if r.fold not in models:
            models[r.fold] = initiate_model(model_args, str(run_dir / f's_{r.fold}_checkpoint.pt'))
        model = models[r.fold]

        # trained bag: saved features + coords; check the prediction against the saved test result
        feats = torch.load(f'{FEAT_ROOT}/{args.feature_dir}/pt_files/{sid}.pt')
        with h5py.File(f'{FEAT_ROOT}/{args.feature_dir}/h5_files/{sid}.h5') as f:
            coords = f['coords'][:]
        with torch.inference_mode():
            _, Y_prob, _, _, _ = model(feats.cuda())
        p1 = Y_prob[0, 1].item()
        assert abs(p1 - r.p1) < 1e-4, f'{sid}: recomputed P(disease) {p1:.5f} != saved {r.p1:.5f}'
        A = attention(model, feats)

        slide_path = f'{DATA_ROOT}/Image/Original/{sid}.tif'
        wsi = WholeSlideImage(slide_path)
        wsi.initSegmentation(f'{PREP}/{args.variant}/contours/{sid}.pkl')
        size = read_image_size(slide_path)
        roi = build_roi_contours(f'{DATA_ROOT}/Epidermis_ROI/{sid}.roi', size, 5, sid in DERMIS_UP)
        top_left, bot_right = roi_box(roi['mask'], 2 * patch_size, size)

        fc, fs = fine_scores(model, encoder, tf, wsi.wsi, roi['mask'], patch_size, step, A, args.batch_size)
        # visHeatmap divides scores by 100 in place, so every call gets its own copy
        common = dict(wsi_object=wsi, vis_level=0, cmap=CMAP, alpha=ALPHA, patch_size=(patch_size, patch_size),
                      blank_canvas=False, use_holes=True, binarize=False, thresh=-1)
        fine = dict(convert_to_percentiles=False, overlap=args.overlap, blur=args.blur)
        images = {
            'blockmap': drawHeatmap(A.copy(), coords, slide_path, convert_to_percentiles=True, **common),
            'heatmap': drawHeatmap(fs.copy(), fc, slide_path, **fine, **common),
            'heatmap_roi': drawHeatmap(fs.copy(), fc, slide_path, **fine, top_left=top_left, bot_right=bot_right, **common),
            'orig_roi': wsi.wsi.read_region(top_left, 0, tuple(np.subtract(bot_right, top_left))).convert('RGB'),
        }
        tag = f'{r.group}_{sid}'
        for name, img in images.items():
            img.save(out / 'slides' / f'{tag}_{name}.jpg', quality=95)
        records.append(dict(r, p1_recomputed=round(p1, 6), n_patches=len(coords), n_fine_patches=len(fc),
                            files=f'slides/{tag}_*.jpg'))
        print(f'{tag}: P(disease) {p1:.4f}, {len(coords)} bag patches, {len(fc)} fine patches')

    rec = pd.DataFrame(records)
    rec.to_csv(out / 'selected_slides.csv', index=False)

    draw_overview(rec, out, args, patch_size, step)
    print('wrote', out)


if __name__ == '__main__':
    main()
