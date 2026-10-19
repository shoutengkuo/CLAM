"""Per-dimension feature statistics for z-score normalization (main.py --feat_norm zscore).

To avoid reading every feature file again for every fold, the per-slide sufficient statistics
(patch count, sum and sum of squares per dimension, in float64) are computed once per feature folder
and cached in feature_stats/<feature_dir>_slide_stats.npz (inside the repo, not in the data folder).
The mean / std for a fold are then computed exactly from the cached sums of that fold's TRAINING slides
only, so validation and test slides never influence the normalization.
"""
import os

import numpy as np
import torch

CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'feature_stats')


def slide_stats(data_root_dir, feature_dir):
    """Return dict slide_id -> (n, sum[D], sumsq[D]); computed once and cached."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache = os.path.join(CACHE_DIR, f'{feature_dir}_slide_stats.npz')
    pt_dir = os.path.join(data_root_dir, feature_dir, 'pt_files')
    if os.path.exists(cache):
        z = np.load(cache, allow_pickle=False)
        ids = z['slide_ids'].tolist()
        return {s: (int(z['n'][i]), z['sum'][i], z['sumsq'][i]) for i, s in enumerate(ids)}
    files = sorted(f for f in os.listdir(pt_dir) if f.endswith('.pt'))
    ids, n, sums, sumsqs = [], [], [], []
    for i, f in enumerate(files):
        x = torch.load(os.path.join(pt_dir, f), map_location='cpu').double()
        ids.append(f[:-3])
        n.append(x.shape[0])
        sums.append(x.sum(0).numpy())
        sumsqs.append((x * x).sum(0).numpy())
        if (i + 1) % 200 == 0:
            print(f'  feature stats: {i + 1}/{len(files)} slides')
    np.savez(cache, slide_ids=np.array(ids), n=np.array(n), sum=np.stack(sums), sumsq=np.stack(sumsqs))
    print(f'  feature stats cached in {cache}')
    return {s: (n[i], sums[i], sumsqs[i]) for i, s in enumerate(ids)}


def fold_mean_std(stats, slide_ids, eps=1e-6):
    """Exact per-dimension mean and std over all patches of the given slides."""
    missing = [s for s in slide_ids if s not in stats]
    assert not missing, f'no feature stats for {len(missing)} slides, e.g. {missing[:3]}'
    n = sum(stats[s][0] for s in slide_ids)
    tot = np.sum([stats[s][1] for s in slide_ids], axis=0)
    totsq = np.sum([stats[s][2] for s in slide_ids], axis=0)
    mean = tot / n
    var = np.maximum(totsq / n - mean ** 2, 0.0)
    std = np.maximum(np.sqrt(var), eps)
    return mean.astype(np.float32), std.astype(np.float32), int(n)
