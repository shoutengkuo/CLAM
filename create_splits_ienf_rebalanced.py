"""Patient-level 5-fold splits for IENF with iSFN, with the earlier hard cases spread evenly over the folds.

CAUTION: the folds are chosen using test results of earlier runs (which patients were misclassified), so
per-fold and mean performance on these splits is optimistic and not directly comparable with an untuned
split. The README written next to the splits says so.

Steps:
  1. Start from the test folds of --base_split (Control / DM / FAP patients).
  2. Difficulty per patient from --runs (out-of-fold test predictions of the earlier ienf_2cls runs):
     hard = some slide wrong in >= --hard_min of the runs; easy = every slide right in every run.
  3. While the hard-patient counts of the folds differ by more than 1: take the fold with the most and the
     fold with the fewest hard patients and swap one hard patient from the first with one easy patient
     from the second that has the same subtype, batch and number of slides (so per-fold class and
     batch counts do not change). Logged in swap_log.csv.
  4. Add the iSFN patients (no earlier results): shuffled, each to the fold with the fewest iSFN patients
     (ties: fewest slides).
  5. Validation sets as in create_splits_kfold.py: val_frac of ALL patients, stratified by subtype, drawn
     from the patients outside the test fold.
Output: splits/<out_dir>/splits_{i}.csv (train, val, test slide ids), split_summary.csv, swap_log.csv,
patients.csv, README.txt. Refuses to overwrite an existing folder.

Usage:
  python create_splits_ienf_rebalanced.py \\
      --out_dir ienf_isfn_kfold5_rebalanced
"""
import argparse
import os
import pickle
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

p = argparse.ArgumentParser()
p.add_argument('--csv', default='dataset_csv/ienf_isfn_4sub.csv', help='case_id, slide_id, label (subtype), batch')
p.add_argument('--base_split', default='splits/ienf_kfold5_seed1')
p.add_argument('--runs', nargs='+', default=['ienf_2cls_uni_p128_CLAM_B8_s1', 'ienf_2cls_uni_p128_CLAM_B16_s1',
                                             'ienf_2cls_uni_p224_CLAM_B4_s1', 'ienf_2cls_uni_p224_CLAM_B8_s1'])
p.add_argument('--hard_min', type=int, default=3)
p.add_argument('--k', type=int, default=5)
p.add_argument('--seed', type=int, default=1)
p.add_argument('--val_frac', type=float, default=0.1)
p.add_argument('--out_dir', required=True)
args = p.parse_args()
out = os.path.join('splits', args.out_dir)
if os.path.exists(out):
    sys.exit(f'ERROR: {out} already exists. Refusing to overwrite existing splits.')
rng = np.random.default_rng(args.seed)

df = pd.read_csv(args.csv)
slides_of = df.groupby('case_id').slide_id.apply(list).to_dict()
pat = df.groupby('case_id').agg(subtype=('label', 'first'), batch=('batch', 'first'), n_slides=('slide_id', 'size'))

# ------------------------------------------------------------------ 1. base folds
fold_of_slide = {}
for i in range(args.k):
    for s in pd.read_csv(os.path.join(args.base_split, f'splits_{i}.csv')).test.dropna():
        fold_of_slide[s] = i
pat['old_fold'] = [{fold_of_slide.get(s) for s in slides_of[c]} for c in pat.index]
assert all(len(f) == 1 for f in pat.old_fold[pat.subtype != 'iSFN']), 'a base patient spans folds'
pat['old_fold'] = [next(iter(f)) if None not in f else -1 for f in pat.old_fold]
assert (pat.old_fold[pat.subtype == 'iSFN'] == -1).all() and (pat.old_fold[pat.subtype != 'iSFN'] >= 0).all()

# ------------------------------------------------------------------ 2. difficulty
wrong = pd.Series(0, index=df.slide_id)
for r in args.runs:
    for i in range(args.k):
        for s, v in pickle.load(open(os.path.join('results', r, f'split_{i}_results.pkl'), 'rb')).items():
            wrong[s] += int((v['prob'][0][1] >= 0.5) != v['label'])
pat['max_wrong'] = [int(wrong[slides_of[c]].max()) for c in pat.index]
pat['difficulty'] = np.where(pat.subtype == 'iSFN', 'new',
                             np.where(pat.max_wrong >= args.hard_min, 'hard', np.where(pat.max_wrong == 0, 'easy', 'mid')))
pat['fold'] = pat.old_fold

# ------------------------------------------------------------------ 3. swap hard <-> easy
swaps = []
hard_count = lambda: pat[pat.difficulty == 'hard'].groupby('fold').size().reindex(range(args.k), fill_value=0)
blocked = set()
while True:
    hc = hard_count()
    order = hc.sort_values(kind='stable')
    pairs = [(hi, lo) for hi in order.index[::-1] for lo in order.index if hc[hi] - hc[lo] > 1 and (hi, lo) not in blocked]
    if not pairs:
        break
    hi, lo = pairs[0]
    hard = pat[(pat.fold == hi) & (pat.difficulty == 'hard')]
    easy = pat[(pat.fold == lo) & (pat.difficulty == 'easy')]
    cand = [(h, e) for h in hard.index for e in easy.index
            if tuple(pat.loc[h, ['subtype', 'batch', 'n_slides']]) == tuple(pat.loc[e, ['subtype', 'batch', 'n_slides']])]
    if not cand:
        blocked.add((hi, lo))
        continue
    h, e = cand[rng.integers(len(cand))]
    pat.loc[h, 'fold'], pat.loc[e, 'fold'] = lo, hi
    swaps.append(dict(hard_case=h, from_fold=hi, to_fold=lo, easy_case=e, subtype=pat.loc[h, 'subtype'],
                      batch=pat.loc[h, 'batch'], n_slides=pat.loc[h, 'n_slides'], hard_max_wrong=pat.loc[h, 'max_wrong']))

# ------------------------------------------------------------------ 4. iSFN
for c in rng.permutation(pat.index[pat.subtype == 'iSFN'].to_numpy()):
    n_isfn = pat[(pat.subtype == 'iSFN') & (pat.fold >= 0)].groupby('fold').size().reindex(range(args.k), fill_value=0)
    n_sl = pat[pat.fold >= 0].groupby('fold').n_slides.sum().reindex(range(args.k), fill_value=0)
    pat.loc[c, 'fold'] = min(range(args.k), key=lambda f: (n_isfn[f], n_sl[f]))

# ------------------------------------------------------------------ 5. val sets
n_val = int(round(args.val_frac * len(pat)))
folds = []
for i in range(args.k):
    test = set(pat.index[pat.fold == i])
    rest = pat.index[pat.fold != i].to_numpy()
    tr, va = train_test_split(rest, test_size=n_val, stratify=pat.loc[rest, 'subtype'], random_state=args.seed * 1000 + i)
    folds.append(dict(train=set(tr), val=set(va), test=test))

# ------------------------------------------------------------------ checks (fail loudly)
all_cases = set(pat.index)
for i, f in enumerate(folds):
    assert not (f['train'] & f['val'] or f['train'] & f['test'] or f['val'] & f['test']), f'fold {i}: case overlap'
    assert f['train'] | f['val'] | f['test'] == all_cases, f'fold {i}: cases missing'
test_slides = [s for f in folds for c in f['test'] for s in slides_of[c]]
assert len(test_slides) == len(set(test_slides)) == len(df), 'test folds must cover every slide exactly once'

# ------------------------------------------------------------------ write
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
                         batch2_slides=int((lab.loc[slides, 'batch'] == 'batch2').sum()),
                         hard_cases=int((pat.loc[list(f[name]), 'difficulty'] == 'hard').sum())))
summary = pd.DataFrame(rows)
summary.to_csv(os.path.join(out, 'split_summary.csv'), index=False)
pd.DataFrame(swaps).to_csv(os.path.join(out, 'swap_log.csv'), index=False)
pat.drop(columns='max_wrong').assign(max_wrong=pat.max_wrong).to_csv(os.path.join(out, 'patients.csv'))
with open(os.path.join(out, 'README.txt'), 'w') as fh:
    fh.write(f'Patient-level 5-fold splits for IENF Control / DM / FAP / iSFN ({len(pat)} cases, {len(df)} slides).\n'
             f'Base: test folds of {args.base_split}; {len(swaps)} hard<->easy swaps (hard = wrong in >= {args.hard_min} of '
             f'{len(args.runs)} earlier runs: {", ".join(args.runs)}); iSFN patients added; val = {n_val} cases per fold.\n'
             f'CAUTION: folds were tuned on earlier TEST results, so performance on them is optimistic.\n'
             f'seed={args.seed}; created by create_splits_ienf_rebalanced.py\n')

pd.set_option('display.width', 200)
print(f'wrote {args.k} folds to {out}; {len(swaps)} swaps; n_val = {n_val} cases')
print('hard cases per test fold before:', pat[pat.difficulty == 'hard'].groupby('old_fold').size().to_dict(),
      ' after:', hard_count().to_dict())
print(summary[summary.set == 'test'].to_string(index=False))
print(pd.DataFrame(swaps).to_string(index=False))
