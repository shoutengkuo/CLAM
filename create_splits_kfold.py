"""Patient-level, stratified k-fold splits for Camelyon+ (standard k-fold, not Monte Carlo).

Every case (patient) is in the test set of exactly one fold, so every slide is tested exactly once.
For fold i:
  test  = the i-th of k disjoint, stratified groups of cases          (1/k of all cases)
  val   = a stratified random sample of the remaining cases           (val_frac of ALL cases)
  train = all other cases
With k = 10 and val_frac = 0.1 this gives 80/10/10, with k = 5 it gives 70/10/20 (in cases; slide
fractions differ slightly because cases have different numbers of slides).

Stratification uses the case-level 4-class label (negative < itc < micro < macro; a case gets the
largest lesion among its slides), so the folds are balanced for the 2-class task as well.
All slides of a case always go to the same set.

Output: splits/<out_dir>/splits_{i}.csv in the same format as create_splits_seq.py (columns train, val,
test with slide ids), plus split_summary.csv with per-fold class counts. The script refuses to write
into an existing folder, so existing splits can never be overwritten.

Usage:
  python create_splits_kfold.py --k 10 --seed 1 --out_dir camelyon_plus_kfold10_seed1
  python create_splits_kfold.py --k 5  --seed 1 --out_dir camelyon_plus_kfold5_seed1
Train with: python main.py ... --k <k> --split_dir <out_dir>
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

p = argparse.ArgumentParser()
p.add_argument('--k', type=int, required=True)
p.add_argument('--seed', type=int, default=1)
p.add_argument('--val_frac', type=float, default=0.1, help='validation size as a fraction of ALL cases')
p.add_argument('--csv', default='dataset_csv/camelyon_plus_4cls.csv', help='case_id, slide_id, label (used for stratification)')
p.add_argument('--order', nargs='+', default=['negative', 'itc', 'micro', 'macro'], help='label order; a case gets the max')
p.add_argument('--out_dir', required=True, help='folder name under splits/ (must not exist)')
args = p.parse_args()

out = os.path.join('splits', args.out_dir)
if os.path.exists(out):
    sys.exit(f'ERROR: {out} already exists. Refusing to overwrite existing splits.')

df = pd.read_csv(args.csv)
rank = {c: i for i, c in enumerate(args.order)}
assert set(df.label) <= set(rank), f'unknown labels {set(df.label) - set(rank)}'
df['rank'] = df.label.map(rank)
cases = df.groupby('case_id')['rank'].max().reset_index()          # case label = largest lesion
case_ids, case_y = cases.case_id.values, cases['rank'].values
n_val = int(round(args.val_frac * len(cases)))
slides_of = df.groupby('case_id').slide_id.apply(list).to_dict()

skf = StratifiedKFold(n_splits=args.k, shuffle=True, random_state=args.seed)
folds = []
for i, (rest_idx, test_idx) in enumerate(skf.split(case_ids, case_y)):
    tr_idx, val_idx = train_test_split(rest_idx, test_size=n_val, stratify=case_y[rest_idx],
                                       random_state=args.seed * 1000 + i)
    folds.append({name: set(case_ids[idx]) for name, idx in [('train', tr_idx), ('val', val_idx), ('test', test_idx)]})

# ------------------------------------------------------------------ checks (fail loudly)
all_cases = set(case_ids)
test_count = pd.Series(0, index=case_ids)
for i, f in enumerate(folds):
    assert not (f['train'] & f['val'] or f['train'] & f['test'] or f['val'] & f['test']), f'fold {i}: case overlap'
    assert f['train'] | f['val'] | f['test'] == all_cases, f'fold {i}: cases missing'
    test_count[list(f['test'])] += 1
assert (test_count == 1).all(), 'every case must be in exactly one test fold'
test_slides = [s for f in folds for c in f['test'] for s in slides_of[c]]
assert len(test_slides) == len(set(test_slides)) == len(df), 'test folds must cover every slide exactly once'

# ------------------------------------------------------------------ write
os.makedirs(out)
label_of = df.set_index('slide_id').label
rows = []
for i, f in enumerate(folds):
    cols = {name: sorted(s for c in f[name] for s in slides_of[c]) for name in ['train', 'val', 'test']}
    n = max(len(v) for v in cols.values())
    pd.DataFrame({k: pd.Series(v + [None] * (n - len(v))) for k, v in cols.items()}).to_csv(
        os.path.join(out, f'splits_{i}.csv'))
    for name, slides in cols.items():
        vc = label_of[slides].value_counts()
        rows.append(dict(fold=i, set=name, cases=len(f[name]), slides=len(slides),
                         **{c: int(vc.get(c, 0)) for c in args.order}))
summary = pd.DataFrame(rows)
summary.to_csv(os.path.join(out, 'split_summary.csv'), index=False)
with open(os.path.join(out, 'README.txt'), 'w') as fh:
    fh.write(f'Patient-level stratified {args.k}-fold splits (standard k-fold, each case tested once).\n'
             f'seed={args.seed} val_frac={args.val_frac} (of all cases, n_val={n_val}) csv={args.csv}\n'
             f'stratification: case label = max over slides in order {args.order}\n'
             f'created by create_splits_kfold.py\n')

pd.set_option('display.width', 200)
print(f'wrote {args.k} folds to {out}  ({len(cases)} cases, {len(df)} slides, n_val = {n_val} cases)')
print(summary.pivot_table(index='fold', columns='set', values=['cases', 'slides'] + args.order).astype(int).to_string())
