"""Attention heatmaps of the same slides at several patch sizes (IENF 2-class with iSFN, mask-only runs).

For each patch size, the model is the fold checkpoint that TESTED the slide in the mask-only run
ienf_ab_D_mask_p{ps}_CLAM_B{B}_s{seed} (see results/ienf_ablation_report.md). Two maps per slide and size:

  blockmap:  attention of the trained bag (saved UNI features + coords, no overlap), as percentiles within
             the slide, i.e. exactly the patches the model scored
  fine maps: one per --fine spec ("0.5", "0.95:blur"): patches with that overlap on the same ROI rule
             (buffer 20 px, >= 10% of the patch), UNI features of the mask-only image, attention as percentiles
             of the bag scores (score2percentile). The step is rounded to whole pixels (>= 1), and the overlap
             passed to visHeatmap is the actual one, 1 - step / ps: CLAM sizes the blur kernel as
             int(ps * (1 - overlap)) * 2 + 1, which with the nominal 0.95 is 1 x 1 (no blur) at 16 px.
             Fine scores are cached in slides/ (npz), so maps can be redrawn without UNI.

The model saw the mask-only images; the maps are drawn on the original image with the ROI + buffer contour,
using CLAM's drawHeatmap. Patient images: the output folder's slides/ and overview.png are git-ignored.

Usage (repo root, clam env):
  UNI_CKPT_PATH=~/projects/UNI/assets/ckpts/uni/pytorch_model.bin \\
  python heatmaps/ienf_heatmaps_patch_sizes.py \\
      --slides S1768-2_a S1445-2_a \\
      --sizes 32 16 --seed 1 \\
      --fine 0.5 0.95:blur --min_step 2 \\
      --out_dir heatmaps/ienf/patch_sizes_D_mask_s1
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
sys.path.insert(0, str(CLAM_ROOT / 'heatmaps'))
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

from ienf_heatmaps import attention, roi_box, CMAP, ALPHA  # noqa: E402
from ienf_prep.patching import generate_coords_roi_overlap  # noqa: E402
from ienf_prep.roi import build_roi_contours, read_image_size  # noqa: E402
from models import get_encoder  # noqa: E402
from utils.eval_utils import initiate_model  # noqa: E402
from vis_utils.heatmap_utils import drawHeatmap, score2percentile  # noqa: E402
from wsi_core.WholeSlideImage import WholeSlideImage  # noqa: E402

DATA_ROOT = '/data/datasets/neuroimages/20250214'
ORIG = f'{DATA_ROOT}/Image/Original'
MASKONLY = '/var/tmp/shouteng_data/IENF_CLAM/ablation/images_maskonly_bufdown20/images'
PREP = '/var/tmp/shouteng_data/IENF_CLAM/clam_prep_norm'
FEAT_ROOT = '/var/tmp/shouteng_data/CLAM_DATA'
BUFFER, MIN_OVERLAP = 20, 0.10
DERMIS_UP = ['S1357-2_b']  # same list as create_patches_roi.py
B_OF = {128: 8, 64: 16, 32: 32, 16: 64}


def test_result(run_dir, sid, k=5):
    for fold in range(k):
        res = pickle.load(open(run_dir / f'split_{fold}_results.pkl', 'rb'))
        if sid in res:
            return fold, int(res[sid]['label']), float(res[sid]['prob'][0][1])
    raise KeyError(sid)


def fine_scores(model, encoder, tf, slide, mask, ps, step, ref, batch_size=256):
    coords = generate_coords_roi_overlap(mask, ps, step, MIN_OVERLAP, True)
    raw = []
    for i in range(0, len(coords), batch_size):
        batch = torch.stack([tf(slide.read_region(tuple(int(v) for v in c), 0, (ps, ps)).convert('RGB'))
                             for c in coords[i:i + batch_size]]).cuda()
        with torch.inference_mode():
            raw.append(attention(model, encoder(batch)))
    raw = np.concatenate(raw)
    return coords, np.array([score2percentile(s, ref) for s in raw])


def parse_fine(spec):
    ov, _, opt = spec.partition(':')
    return float(ov), opt == 'blur'


def fine_tag(ov, blur):
    return f'fine{int(round(ov * 100))}' + ('_blur' if blur else '')


def draw_overview(rec, out, args):
    """Per slide: original crop, then blockmap and fine map for each size (rows); one column."""
    panels = []
    for sid, r in rec.groupby('slide_id', sort=False):
        r0 = r.iloc[0]
        panels.append((out / 'slides' / f'{sid}_orig_roi.jpg',
                       f'{sid}: true {r0.subtype} ({"disease" if r0.y else "normal"}), {r0.batch}, tested in fold {r0.fold}  |  original (ROI crop)'))
        for _, x in r.iterrows():
            panels.append((out / 'slides' / f'{sid}_p{x.ps}_blockmap_roi.jpg',
                           f'{x.ps} px, B = {B_OF[x.ps]}: bag attention, no overlap ({x.n_patches} patches), P(disease) = {x.p1:.3f}'))
            for spec in args.fine:
                ov, blur = parse_fine(spec)
                t = fine_tag(ov, blur)
                step = int(x[f'{t}_step'])
                panels.append((out / 'slides' / f'{sid}_p{x.ps}_{t}_roi.jpg',
                               f'{x.ps} px: {int(round(ov * 100))}% overlap (step {step} px = {100 * (1 - step / x.ps):.1f}%, '
                               f'{int(x[f"{t}_n"])} patches)' + (f', Gaussian blur {2 * step + 1} px' if blur else '')))
    imgs = [Image.open(p) for p, _ in panels]
    w = 14
    hs = [w * im.height / im.width for im in imgs]
    fig = plt.figure(figsize=(w, sum(hs) + 0.32 * len(imgs) + 1.4))
    y = 1.0
    H = fig.get_figheight()
    for im, (_, title), h in zip(imgs, panels, hs):
        y -= (h + 0.32) / H
        ax = fig.add_axes([0.01, y, 0.98, h / H])
        ax.imshow(im); ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(title, loc='left', fontsize=9)
    cax = fig.add_axes([0.55, 0.85 / H, 0.40, 0.12 / H])
    fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(0, 100), cmap=CMAP), cax=cax, orientation='horizontal',
                 label='Attention percentile within the slide (blue = low, red = high)')
    fig.text(0.01, 0.15 / H, f'Mask-only runs ienf_ab_D_mask_p*_s{args.seed}: CLAM-SB, UNI features; '
             'model input = original intensities masked outside ROI + 20 px; drawn on the original image.',
             fontsize=8, color='#52514e')
    fig.savefig(out / 'overview.png', dpi=130)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--slides', nargs='+', required=True)
    ap.add_argument('--sizes', nargs='+', type=int, default=[32, 16])
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--fine', nargs='+', default=['0.5'], help='fine maps as overlap[:blur], e.g. 0.5 0.95:blur')
    ap.add_argument('--min_step', type=int, default=1,
                    help='smallest step in px; 2 caps 16 px at 87.5%% overlap (1 px steps: ~0.7-0.85M UNI passes per slide)')
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--overview_only', action='store_true')
    args = ap.parse_args()
    out = Path(args.out_dir)
    (out / 'slides').mkdir(parents=True, exist_ok=True)
    if args.overview_only:
        draw_overview(pd.read_csv(out / 'selected_slides.csv'), out, args)
        return

    lab = pd.read_csv(CLAM_ROOT / 'dataset_csv' / 'ienf_isfn_4sub.csv').set_index('slide_id')
    model_args = Namespace(drop_out=0.25, n_classes=2, embed_dim=1024, model_size='small', model_type='clam_sb')
    encoder, tf = get_encoder('uni_v1', target_img_size=224)
    encoder = encoder.cuda().eval()

    records = []
    for sid in args.slides:
        size = read_image_size(f'{ORIG}/{sid}.tif')
        roi = build_roi_contours(f'{DATA_ROOT}/Epidermis_ROI/{sid}.roi', size, BUFFER, sid in DERMIS_UP)
        top_left, bot_right = roi_box(roi['mask'], 64, size)
        masked = WholeSlideImage(f'{MASKONLY}/{sid}.tif').wsi  # what the model saw
        for ps in args.sizes:
            run = f'ienf_ab_D_mask_p{ps}_CLAM_B{B_OF[ps]}_s{args.seed}'
            run_dir = CLAM_ROOT / 'results' / run
            fold, y, p_saved = test_result(run_dir, sid)
            model = initiate_model(model_args, str(run_dir / f's_{fold}_checkpoint.pt'))
            feat = f'{FEAT_ROOT}/ienf_ab_maskonly_buf20_uni_p{ps}_s{ps}'
            feats = torch.load(f'{feat}/pt_files/{sid}.pt')
            with h5py.File(f'{feat}/h5_files/{sid}.h5') as f:
                coords = f['coords'][:]
            with torch.inference_mode():
                _, Y_prob, _, _, _ = model(feats.cuda())
            p1 = Y_prob[0, 1].item()
            assert abs(p1 - p_saved) < 1e-4, f'{sid} {run}: recomputed {p1:.5f} != saved {p_saved:.5f}'
            A = attention(model, feats)

            wsi = WholeSlideImage(f'{ORIG}/{sid}.tif')
            wsi.initSegmentation(f'{PREP}/p{ps}_s{ps}_bufdown20/contours/{sid}.pkl')
            common = dict(wsi_object=wsi, vis_level=0, cmap=CMAP, alpha=ALPHA, patch_size=(ps, ps),
                          blank_canvas=False, use_holes=True, binarize=False, thresh=-1,
                          top_left=top_left, bot_right=bot_right)
            # visHeatmap divides scores by 100 in place, so every call gets its own copy
            drawHeatmap(A.copy(), coords, f'{ORIG}/{sid}.tif', convert_to_percentiles=True, **common) \
                .save(out / 'slides' / f'{sid}_p{ps}_blockmap_roi.jpg', quality=95)
            rec = dict(slide_id=sid, subtype=lab.label[sid], batch=lab.batch[sid], y=y, fold=fold,
                       ps=ps, run=run, p1=round(p1, 6), n_patches=len(coords))
            for spec in args.fine:
                ov, blur = parse_fine(spec)
                step = max(args.min_step, int(round(ps * (1 - ov))))
                cache = out / 'slides' / f'{sid}_p{ps}_step{step}_scores.npz'
                if cache.exists():
                    z = np.load(cache); fc, fs = z['coords'], z['pct']
                else:
                    fc, fs = fine_scores(model, encoder, tf, masked, roi['mask'], ps, step, A)
                    np.savez(cache, coords=fc, pct=fs)
                t = fine_tag(ov, blur)
                drawHeatmap(fs.copy(), fc, f'{ORIG}/{sid}.tif', convert_to_percentiles=False,
                            overlap=1 - step / ps, blur=blur, **common) \
                    .save(out / 'slides' / f'{sid}_p{ps}_{t}_roi.jpg', quality=95)
                rec[f'{t}_step'], rec[f'{t}_n'] = step, len(fc)
                print(f'{sid} {ps} px {t}: step {step}, {len(fc)} patches', flush=True)
            records.append(rec)
            print(f'{sid} {ps} px: fold {fold}, P(disease) {p1:.4f}, {len(coords)} bag patches', flush=True)
        wsi.wsi.read_region(top_left, 0, tuple(np.subtract(bot_right, top_left))).convert('RGB') \
            .save(out / 'slides' / f'{sid}_orig_roi.jpg', quality=95)
    recs = pd.DataFrame(records)
    recs.to_csv(out / 'selected_slides.csv', index=False)
    draw_overview(recs, out, args)
    print('wrote', out)


if __name__ == '__main__':
    main()
