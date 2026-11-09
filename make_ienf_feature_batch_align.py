"""Feature-level batch alignment (ablation): align the UNI patch features of each acquisition batch to the
dataset-wide distribution, per feature dimension:

    x' = (x - mean_batch) / sd_batch * sd_all + mean_all

Means and SDs are computed over all patches of all slides in dataset_csv/ienf_isfn_4sub.csv (433 slides). This
is unsupervised (no labels) but transductive: test slides contribute to the statistics. Because iSFN exists only
in batch2, aligning the batches can also remove part of the disease signal; that is what the ablation measures.
Writes <out>/pt_files/*.pt and symlinks the h5 coordinate files of the source folder.

Usage:
  python make_ienf_feature_batch_align.py \\
      --src /var/tmp/shouteng_data/CLAM_DATA/ienf_ab_maskonly_buf20_uni_p128_s128 \\
      --out /var/tmp/shouteng_data/CLAM_DATA/ienf_ab_maskonly_featalign_uni_p128_s128
"""
import argparse
import json
import os

import pandas as pd
import torch

ap = argparse.ArgumentParser()
ap.add_argument('--src', required=True)
ap.add_argument('--out', required=True)
ap.add_argument('--csv', default='dataset_csv/ienf_isfn_4sub.csv')
a = ap.parse_args()

df = pd.read_csv(a.csv)
os.makedirs(f'{a.out}/pt_files', exist_ok=True)
os.makedirs(f'{a.out}/h5_files', exist_ok=True)


def moments(slides):
    n, s, ss = 0, 0, 0
    for sid in slides:
        x = torch.load(f'{a.src}/pt_files/{sid}.pt').double()
        n, s, ss = n + len(x), s + x.sum(0), ss + (x ** 2).sum(0)
    m = s / n
    return m, (ss / n - m ** 2).clamp_min(1e-12).sqrt(), n


m_all, sd_all, n_all = moments(df.slide_id)
stats = {}
for bt, g in df.groupby('batch'):
    m_b, sd_b, n_b = moments(g.slide_id)
    stats[bt] = (m_b, sd_b)
    print(f'{bt}: {len(g)} slides, {n_b} patches; mean |mean_b - mean_all| = {(m_b - m_all).abs().mean():.4f}, '
          f'median sd_b / sd_all = {(sd_b / sd_all).median():.3f}')
for sid, bt in zip(df.slide_id, df.batch):
    x = torch.load(f'{a.src}/pt_files/{sid}.pt').double()
    m_b, sd_b = stats[bt]
    torch.save(((x - m_b) / sd_b * sd_all + m_all).float(), f'{a.out}/pt_files/{sid}.pt')
    h5 = f'{a.out}/h5_files/{sid}.h5'
    if not os.path.lexists(h5):
        os.symlink(f'{a.src}/h5_files/{sid}.h5', h5)
with open(f'{a.out}/README.json', 'w') as f:
    json.dump(dict(src=a.src, csv=a.csv, n_slides=len(df), n_patches=n_all, method='per-batch per-dim standardize, '
                   'then rescale to the dataset mean / sd'), f, indent=1)
print('wrote', len(df), 'slides to', a.out)
