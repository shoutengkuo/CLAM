"""Ablation splits: an iSFN split folder with the iSFN slides removed, so the earlier Control/DM/FAP setup
(task ienf_2cls, 387 slides) can be trained on the same folds.

    python create_splits_ienf_noisfn.py   # rebalanced folds (default)
    python create_splits_ienf_noisfn.py \\
        --src splits/ienf_isfn_kfold5_random \\
        --dst splits/ienf_kfold5_random_noisfn
"""
import argparse
import os
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument('--src', default='splits/ienf_isfn_kfold5_rebalanced')
ap.add_argument('--dst', default='splits/ienf_kfold5_rebalanced_noisfn')
a = ap.parse_args()
SRC, DST = a.src, a.dst
keep = set(pd.read_csv('dataset_csv/ienf_2cls.csv').slide_id)
os.makedirs(DST, exist_ok=True)
rows = []
for k in range(5):
    s = pd.read_csv(f'{SRC}/splits_{k}.csv', index_col=0)
    cols = {c: [x for x in s[c].dropna() if x in keep] for c in ['train', 'val', 'test']}
    n = max(len(v) for v in cols.values())
    out = pd.DataFrame({c: pd.Series(v + [None] * (n - len(v))) for c, v in cols.items()})
    out.to_csv(f'{DST}/splits_{k}.csv')
    rows.append(dict(fold=k, **{c: len(v) for c, v in cols.items()}))
    assert not (set(cols['train']) & set(cols['test'])) and not (set(cols['val']) & set(cols['test']))
summ = pd.DataFrame(rows)
summ.to_csv(f'{DST}/split_summary.csv', index=False)
tested = sum((pd.read_csv(f'{DST}/splits_{k}.csv').test.dropna().tolist() for k in range(5)), [])
assert sorted(tested) == sorted(keep), 'every Control/DM/FAP slide must be tested exactly once'
with open(f'{DST}/README.txt', 'w') as f:
    f.write(f'{SRC} without the iSFN slides (ablation, task ienf_2cls). Made by create_splits_ienf_noisfn.py.\n')
print(summ.to_string(index=False))
