"""PCA of CLAM's instance-level clustering for one IENF slide, with example patches per cluster.

CLAM has no plotting code for this, so the script reads it out of the trained model directly:

  h          512-d patch features after the first fc layer (model.attention_net), the input of both the
             attention head and the instance classifiers
  cluster    CLAM-SB keeps one instance classifier per class (512 -> 2). In training, for a slide of
             class c, classifier c learns top-B attention patches = positive, bottom-B = negative.
             Here classifier c (c = the slide's true label) is applied to every patch of the bag.
  PCA        of h over the bag; points coloured by cluster, ringed if they were top-B / bottom-B
             pseudo-labels (k = min(B, N // 2), as in models/model_clam.py)

Output in <out_dir>/<slide>/:
  instance_pca.png       left: PCA; right: one row per cluster, the 5 most confident patches
  patch_locations.png    the shown patches outlined on the original image
  patches.csv            every patch: index, x, y, cluster, P(positive), attention, PC1-2, pseudo-label
  patches/               every patch as PNG, named <cluster>_<rank>_x<x>_y<y>.png

Usage (repo root, clam env):
  python heatmaps/ienf_instance_pca.py \\
      --slide S1616-2_b \\
      --run ienf_2cls_uni_p128_CLAM_B8_s1 \\
      --out_dir heatmaps/ienf/instance_pca
"""
import argparse
import pickle
import sys
from argparse import Namespace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ienf_walkthrough as wt  # noqa: E402  (sets up sys.path and the openslide stand-in)

import h5py  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from PIL import Image  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.metrics import silhouette_score  # noqa: E402

from ienf_prep import slide_reader  # noqa: E402
from ienf_prep.roi import extend_mask_toward_dermis, polygon_to_mask, read_roi_polygon  # noqa: E402
from utils.eval_utils import initiate_model  # noqa: E402

CLASSES = ['normal', 'disease']
C_POS, C_NEG = '#d62728', '#1f77b4'
N_SHOW = 5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--slide', default='S1616-2_b')
    ap.add_argument('--run', default='ienf_2cls_uni_p128_CLAM_B8_s1')
    ap.add_argument('--out_dir', default='heatmaps/ienf/instance_pca')
    args = ap.parse_args()
    sid = args.slide
    out = Path(args.out_dir) / sid
    (out / 'patches').mkdir(parents=True, exist_ok=True)

    run_dir = wt.CLAM_ROOT / 'results' / args.run
    ps = int(args.run.split('_p')[1].split('_')[0])
    B = int(args.run.split('_B')[1].split('_')[0])
    feat_dir = f'ienf_uni_p{ps}_s{ps}'
    fold = None
    for k in range(5):
        res = pickle.load(open(run_dir / f'split_{k}_results.pkl', 'rb'))
        if sid in res:
            fold, label, saved_p1 = k, int(res[sid]['label']), float(res[sid]['prob'][0][1])
    assert fold is not None, f'{sid} is in no test split of {args.run}'
    model = initiate_model(Namespace(drop_out=0.25, n_classes=2, embed_dim=1024, model_size='small', model_type='clam_sb'),
                           str(run_dir / f's_{fold}_checkpoint.pt'))
    feats = torch.load(f'{wt.FEAT_ROOT}/{feat_dir}/pt_files/{sid}.pt').cuda()
    with h5py.File(f'{wt.FEAT_ROOT}/{feat_dir}/h5_files/{sid}.h5') as f:
        co = f['coords'][:]
    with torch.inference_mode():
        _, Y_prob, _, _, _ = model(feats)
        A, h = model.attention_net(feats)
        p_pos = torch.softmax(model.instance_classifiers[label](h), 1)[:, 1].cpu().numpy()
    p1 = Y_prob[0, 1].item()
    assert abs(p1 - saved_p1) < 1e-4
    A = A.view(-1).cpu().numpy()
    h = h.cpu().numpy()
    n = len(A)

    k = max(1, min(B, n // 2))
    pseudo = np.full(n, '', object)
    pseudo[np.argsort(-A)[:k]] = 'top-B'
    pseudo[np.argsort(A)[:k]] = 'bottom-B'
    cluster = np.where(p_pos >= 0.5, 'positive', 'negative')
    pca = PCA(2).fit(h)
    pc = pca.transform(h)
    sil = silhouette_score(h, cluster) if len(set(cluster)) == 2 else float('nan')

    img_path = f'{wt.DATA_ROOT}/Image/Original/{sid}.tif'
    slide = slide_reader.IENFSlide(img_path)
    with Image.open(img_path) as im:
        size = im.size
        rgb = np.asarray(im.convert('RGB')).copy()
    buf = extend_mask_toward_dermis(polygon_to_mask(read_roi_polygon(f'{wt.DATA_ROOT}/Epidermis_ROI/{sid}.roi'), size),
                                    wt.BUFFER, dermis_up=sid in {'S1357-2_b'})

    # rank inside each cluster by classifier confidence, save every patch
    conf = np.where(cluster == 'positive', p_pos, 1 - p_pos)
    rank = np.zeros(n, int)
    for c in ['positive', 'negative']:
        idx = np.where(cluster == c)[0]
        rank[idx[np.argsort(-conf[idx])]] = np.arange(1, len(idx) + 1)
    files = []
    for i, (x, y) in enumerate(co):
        name = f'{cluster[i]}_{rank[i]:02d}_x{x}_y{y}.png'
        slide.read_region((int(x), int(y)), 0, (ps, ps)).convert('RGB').save(out / 'patches' / name)
        files.append(f'patches/{name}')
    df = pd.DataFrame(dict(patch_index=np.arange(n), x=co[:, 0], y=co[:, 1], patch_size=ps, cluster=cluster,
                           rank_in_cluster=rank, p_positive=p_pos.round(4), attention_raw=A.round(4),
                           attention_softmax=(np.exp(A - A.max()) / np.exp(A - A.max()).sum()).round(5),
                           pseudo_label=pseudo, roi_share=[round(wt.coverage_fraction(buf, x, y, ps), 3) for x, y in co],
                           pc1=pc[:, 0].round(3), pc2=pc[:, 1].round(3), file=files))
    df.sort_values(['cluster', 'rank_in_cluster'], ascending=[False, True]).to_csv(out / 'patches.csv', index=False)

    rows = [(c, df[(df.cluster == c)].sort_values('rank_in_cluster').head(N_SHOW)) for c in ['positive', 'negative']]
    col = {'positive': C_POS, 'negative': C_NEG}
    tag = {'positive': 'P', 'negative': 'N'}
    cls = CLASSES[label]
    row_title = {'positive': f'positive: evidence for "{cls}"', 'negative': f'negative: not evidence for "{cls}"'}

    # ------------------------------------------------------------------ PCA + patch rows
    fig = plt.figure(figsize=(18.5, 6.8))
    gs = fig.add_gridspec(len(rows), 1 + N_SHOW, width_ratios=[N_SHOW * 0.95] + [1] * N_SHOW, wspace=0.14, hspace=0.35)
    ax = fig.add_subplot(gs[:, 0])
    for c in ['negative', 'positive']:
        m = cluster == c
        ax.scatter(pc[m, 0], pc[m, 1], s=55, c=col[c], alpha=0.8, edgecolors='white', linewidths=0.6,
                   label=f'{c} ({m.sum()})')
    for lab, mk in [('top-B', 'o'), ('bottom-B', 's')]:
        m = pseudo == lab
        ax.scatter(pc[m, 0], pc[m, 1], s=150, facecolors='none', edgecolors='black', marker=mk, linewidths=1.1)
    for c, sel in rows:
        for j, (_, r) in enumerate(sel.iterrows(), start=1):
            ax.annotate(f'{tag[c]}{j}', (r.pc1, r.pc2), xytext=(6, 5), textcoords='offset points', fontsize=9,
                        fontweight='bold', color=col[c])
    ev = pca.explained_variance_ratio_
    ax.set_xlabel(f'PC1 ({ev[0]:.0%} of variance)'); ax.set_ylabel(f'PC2 ({ev[1]:.0%})')
    handles, _ = ax.get_legend_handles_labels()
    handles += [Line2D([], [], marker='o', ls='', mfc='none', mec='black', ms=11, label=f'top-{k} attention (training positives)'),
                Line2D([], [], marker='s', ls='', mfc='none', mec='black', ms=11, label=f'bottom-{k} attention (training negatives)')]
    ax.legend(handles=handles, fontsize=8.5, loc='best', frameon=True, framealpha=0.85)
    ax.set_title(f'PCA of the {n} patch features h (512-d) · instance classifier "{cls}"', fontsize=11)
    for r_, (c, sel) in enumerate(rows):
        for j in range(N_SHOW):
            if j >= len(sel):
                if j == 0:
                    a = fig.add_subplot(gs[r_, 1:]); a.axis('off')
                    a.text(0, 1.08, row_title[c], transform=a.transAxes, fontsize=10.5, fontweight='semibold', color=col[c])
                    a.text(0, 0.5, 'no patches in this cluster', transform=a.transAxes, fontsize=10, color='#52514e')
                continue
            a = fig.add_subplot(gs[r_, 1 + j]); wt.off(a)
            r = sel.iloc[j]
            a.imshow(np.asarray(Image.open(out / r.file)), interpolation='nearest')
            m = buf[max(r.y, 0):r.y + ps, max(r.x, 0):r.x + ps]
            a.contour(m, levels=[0.5], colors=[wt.C_ROI], linewidths=0.8)
            for s in a.spines.values():
                s.set_visible(True); s.set_color(col[c]); s.set_linewidth(2.5)
            a.set_title(f'{tag[c]}{j + 1}  ({r.x},{r.y})\nP(pos) {r.p_positive:.2f} · ROI {r.roi_share:.0%}', fontsize=7.5, loc='center')
            if j == 0:
                a.text(0, 1.33, row_title[c], transform=a.transAxes, fontsize=10.5, fontweight='semibold', color=col[c])
    fig.suptitle(f'{sid}: CLAM instance-level clustering ({args.run}, fold {fold}; true {cls}, P(disease) = {p1:.4f})',
                 x=0.01, ha='left', fontsize=12.5, fontweight='semibold', y=1.0)
    fig.text(0.01, -0.02, f'Cluster = argmax of instance classifier "{cls}" on each patch. Right: the {N_SHOW} most confident patches per cluster, '
                          f'native {ps} px; yellow line = ROI boundary. Silhouette score of the two clusters in h: ' + (f'{sil:.2f}.' if sil == sil else 'n/a (only one cluster).'),
             fontsize=8.5, color='#52514e')
    fig.savefig(out / 'instance_pca.png')
    plt.close(fig)

    # ------------------------------------------------------------------ where the shown patches are
    ys, xs = np.nonzero(buf)
    cx0, cy0, cx1, cy1 = max(0, xs.min() - 120), max(0, ys.min() - 120), min(size[0], xs.max() + 120), min(size[1], ys.max() + 120)
    fig, ax = plt.subplots(figsize=(14, 14 * (cy1 - cy0) / (cx1 - cx0) + 0.6))
    ax.imshow(rgb[cy0:cy1, cx0:cx1], extent=[cx0, cx1, cy1, cy0]); wt.off(ax)
    for i, (x, y) in enumerate(co):
        ax.add_patch(Rectangle((x, y), ps, ps, fill=False, ec=col[cluster[i]], lw=0.9, alpha=0.7))
    for c, sel in rows:
        for j, (_, r) in enumerate(sel.iterrows(), start=1):
            ax.add_patch(Rectangle((r.x, r.y), ps, ps, fill=False, ec=col[c], lw=2.6))
            ax.text(r.x + 6, r.y + 24, f'{tag[c]}{j}', color='white', fontsize=10, fontweight='bold',
                    bbox=dict(fc=col[c], ec='none', pad=1.5))
    ax.set_title(f'{sid}: all {n} patches by cluster (red positive, blue negative); thick boxes = patches shown in instance_pca.png',
                 loc='left', fontsize=11)
    fig.savefig(out / 'patch_locations.png')
    plt.close(fig)

    print(f'{sid}: label {cls}, fold {fold}, n={n}, positive {int((cluster == "positive").sum())}, negative {int((cluster == "negative").sum())}, '
          f'PC1-2 {ev[0]:.2f}/{ev[1]:.2f}, silhouette {sil:.3f}')
    agree = {lab: f'{int(((pseudo == lab) & (cluster == want)).sum())}/{k}' for lab, want in [('top-B', 'positive'), ('bottom-B', 'negative')]}
    print('pseudo-labels matching the cluster:', agree)
    print('wrote', out)


if __name__ == '__main__':
    main()
