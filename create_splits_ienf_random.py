"""Random patient-level 5-fold splits for IENF with iSFN (Control / DM / FAP / iSFN, 217 cases, 433 slides).

Unlike create_splits_ienf_rebalanced.py, no earlier test result is used: patients are assigned to test folds
by a stratified random K-fold over subtype x batch (iSFN exists only in batch2, so 7 strata). Stratification
uses only labels and the acquisition batch, never model predictions, so the folds are an untuned estimate.
Validation sets as in create_splits_kfold.py: val_frac of ALL patients, stratified by subtype, drawn from the
patients outside the test fold.

Output: splits/<out_dir>/splits_{i}.csv (train, val, test slide ids), split_summary.csv, patients.csv,
README.txt. Refuses to overwrite an existing folder.

Usage:
  python create_splits_ienf_random.py \\
      --seed 2026 \\
      --out_dir ienf_isfn_kfold5_random
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

p = argparse.ArgumentParser()
p.add_argument('--csv', default='dataset_csv/ienf_isfn_4sub.csv', help='case_id, slide_id, label (subtype), batch')
p.add_argument('--k', type=int, default=5)
p.add_argument('--seed', type=int, default=2026)
p.add_argument('--val_frac', type=float, default=0.1)
p.add_argument('--out_dir', required=True)
args = p.parse_args()
out = os.path.join('splits', args.out_dir)
if os.path.exists(out):
    sys.exit(f'ERROR: {out} already exists. Refusing to overwrite existing splits.')

df = pd.read_csv(args.csv)
slides_of = df.groupby('case_id').slide_id.apply(list).to_dict()
pat = df.groupby('case_id').agg(subtype=('label', 'first'), batch=('batch', 'first'), n_slides=('slide_id', 'size'))
assert (df.groupby('case_id').label.nunique() == 1).all() and (df.groupby('case_id').batch.nunique() == 1).all()
strata = pat.subtype + '_' + pat.batch

pat['fold'] = -1
skf = StratifiedKFold(n_splits=args.k, shuffle=True, random_state=args.seed)
for i, (_, te) in enumerate(skf.split(pat.index, strata)):
    pat.iloc[te, pat.columns.get_loc('fold')] = i

n_val = int(round(args.val_frac * len(pat)))
folds = []
for i in range(args.k):
    test = set(pat.index[pat.fold == i])
    rest = pat.index[pat.fold != i].to_numpy()
    tr, va = train_test_split(rest, test_size=n_val, stratify=pat.loc[rest, 'subtype'], random_state=args.seed * 1000 + i)
    folds.append(dict(train=set(tr), val=set(va), test=test))

all_cases = set(pat.index)
for i, f in enumerate(folds):
    assert not (f['train'] & f['val'] or f['train'] & f['test'] or f['val'] & f['test']), f'fold {i}: case overlap'
    assert f['train'] | f['val'] | f['test'] == all_cases, f'fold {i}: cases missing'
test_slides = [s for f in folds for c in f['test'] for s in slides_of[c]]
assert len(test_slides) == len(set(test_slides)) == len(df), 'test folds must cover every slide exactly once'

os.makedirs(out)
lab = df.set_index('slide_id')
rows = []
for i, f in enumerate(folds):
    cols = {name: sorted(s for c in f[name] for s in slides_of[c]) for name in ['train', 'val', 'test']}
    n = max(len(v) for v in cols.values())
    pd.DataFrame({k: pd.Series(v + [None] * (n - len(v))) for k, v in cols.items()}).to_csv(os.path.join(out, f'splits_{i}.csv'))
    for name, slides in cols.items():
        vc = lab.loc[slides, 'label'].value_counts()
        rows.append(dict(fold=i, set=name, cases=len(f[name]), slides=len(slides),
                         **{c: int(vc.get(c, 0)) for c in ['Control', 'DM', 'FAP', 'iSFN']},
                         batch2_slides=int((lab.loc[slides, 'batch'] == 'batch2').sum())))
summary = pd.DataFrame(rows)
summary.to_csv(os.path.join(out, 'split_summary.csv'), index=False)
pat.to_csv(os.path.join(out, 'patients.csv'))
with open(os.path.join(out, 'README.txt'), 'w') as fh:
    fh.write(f'Random patient-level {args.k}-fold splits for IENF Control / DM / FAP / iSFN ({len(pat)} cases, {len(df)} slides).\n'
             f'Test folds: StratifiedKFold over subtype x batch (shuffle, seed {args.seed}); no earlier results used.\n'
             f'val = {n_val} cases per fold, stratified by subtype, from outside the test fold.\n'
             f'Created by create_splits_ienf_random.py\n')
print(f'wrote {args.k} folds to {out}; n_val = {n_val} cases')
print(summary.to_string(index=False))
