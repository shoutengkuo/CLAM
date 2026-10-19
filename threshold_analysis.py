"""Decision-threshold analysis for finished CLAM runs (binary task, class 1 = positive).

For every run and fold:
  * test predictions come from results/<run>/split_{k}_results.pkl (written by main.py);
  * validation predictions come from eval_results/EVAL_<run>_val/fold_{k}.csv
    (written by eval.py --split val --save_exp_code <run>_val).

A threshold is chosen on the fold's VALIDATION predictions with each rule below and then applied
unchanged to the fold's TEST predictions. The test set is never used to choose a threshold, so the
test metrics remain an honest estimate:

  default   t = 0.5 (argmax; what all reports so far use)
  youden    maximises sensitivity + specificity - 1 on val (ties: median of the tied thresholds)
  f1        maximises F1 of the positive class on val (ties: median)
  sens90    the highest threshold that still gives sensitivity >= 0.90 on val

For reference it also writes the mean test metrics over a grid of fixed thresholds (descriptive
curve) and the "oracle" threshold that is best on the test set itself. The oracle is an upper
bound only and must not be reported as a performance estimate.

Outputs (results/threshold_analysis/):
  per_fold.csv   one row per run x fold x rule
  summary.md     markdown tables: mean ± SD per rule, paired Wilcoxon vs. default
  curves.csv     mean test metrics per fixed threshold, per run

Usage:
  python threshold_analysis.py <run_dir_name> [<run_dir_name> ...]
"""
import os
import pickle
import sys

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

RESULTS = 'results'
EVAL = 'eval_results'
OUT = os.path.join(RESULTS, 'threshold_analysis')
GRID = np.round(np.arange(0.01, 1.0, 0.01), 2)
RULES = ['default', 'youden', 'f1', 'sens90']
METRICS = ['acc', 'bacc', 'sens', 'spec', 'prec', 'f1']


def metrics_at(y, p, t):
    pred = (p >= t).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    sens = tp / (tp + fn) if tp + fn else 0.0
    spec = tn / (tn + fp) if tn + fp else 0.0
    prec = tp / (tp + fp) if tp + fp else 0.0
    f1 = 2 * prec * sens / (prec + sens) if prec + sens else 0.0
    return dict(acc=(tp + tn) / len(y), bacc=(sens + spec) / 2, sens=sens, spec=spec, prec=prec, f1=f1)


def pick(y, p, rule):
    if rule == 'default':
        return 0.5
    rows = [metrics_at(y, p, t) for t in GRID]
    if rule in ('youden', 'f1'):
        score = np.array([r['sens'] + r['spec'] - 1 if rule == 'youden' else r['f1'] for r in rows])
        best = GRID[np.isclose(score, score.max())]
        return float(np.median(best))
    if rule == 'sens90':
        ok = [t for t, r in zip(GRID, rows) if r['sens'] >= 0.90]
        return float(max(ok)) if ok else float(GRID[0])
    raise ValueError(rule)


def load(run, k):
    res = pickle.load(open(os.path.join(RESULTS, run, f'split_{k}_results.pkl'), 'rb'))
    yt = np.array([int(r['label']) for r in res.values()])
    pt = np.array([float(r['prob'][0][1]) for r in res.values()])
    val_csv = os.path.join(EVAL, f'EVAL_{run}_val', f'fold_{k}.csv')
    if not os.path.exists(val_csv):
        raise FileNotFoundError(f'{val_csv} missing: run eval.py --split val --save_exp_code {run}_val first')
    v = pd.read_csv(val_csv)
    return v.Y.values.astype(int), v.p_1.values, yt, pt


def main(runs):
    os.makedirs(OUT, exist_ok=True)
    rows, curves = [], []
    for run in runs:
        nfold = len(pd.read_csv(os.path.join(RESULTS, run, 'summary.csv')))
        fold_curves = []
        for k in range(nfold):
            yv, pv, yt, pt = load(run, k)
            for rule in RULES:
                t = pick(yv, pv, rule)
                rows.append(dict(run=run, fold=k, rule=rule, threshold=t, **metrics_at(yt, pt, t)))
            fold_curves.append(pd.DataFrame([dict(threshold=t, **metrics_at(yt, pt, t)) for t in GRID]))
            # oracle: best balanced accuracy on the test set itself (upper bound only)
            tb = pick(yt, pt, 'youden')
            rows.append(dict(run=run, fold=k, rule='oracle_test', threshold=tb, **metrics_at(yt, pt, tb)))
        c = pd.concat(fold_curves).groupby('threshold').mean().reset_index()
        c.insert(0, 'run', run)
        curves.append(c)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, 'per_fold.csv'), index=False)
    pd.concat(curves).to_csv(os.path.join(OUT, 'curves.csv'), index=False)

    lines = ['# Threshold analysis', '',
             'Thresholds chosen on each fold\'s validation set and applied to its test set. '
             '`oracle_test` is chosen on the test set itself and is an upper bound only.', '']
    for run in runs:
        d = df[df.run == run]
        lines += [f'## {run}', '', '| Rule | Threshold (mean, range) | ' + ' | '.join(METRICS) + ' |',
                  '|---|---|' + '---|' * len(METRICS)]
        for rule in RULES + ['oracle_test']:
            g = d[d.rule == rule]
            cells = [f'{g[m].mean():.4f} ± {g[m].std(ddof=1):.4f}' for m in METRICS]
            lines.append(f'| {rule} | {g.threshold.mean():.2f} ({g.threshold.min():.2f}–{g.threshold.max():.2f}) | ' + ' | '.join(cells) + ' |')
        lines += ['', 'Paired difference vs. default (rule − default), folds better, Wilcoxon p:', '',
                  '| Rule | ' + ' | '.join(METRICS) + ' |', '|---|' + '---|' * len(METRICS)]
        base = d[d.rule == 'default'].sort_values('fold')
        for rule in ['youden', 'f1', 'sens90']:
            g = d[d.rule == rule].sort_values('fold')
            cells = []
            for m in METRICS:
                diff = g[m].values - base[m].values
                try:
                    p = wilcoxon(g[m].values, base[m].values).pvalue
                except ValueError:
                    p = float('nan')
                cells.append(f'{diff.mean():+.4f}, {(diff > 0).sum()}/{len(diff)}, p={p:.3g}')
            lines.append(f'| {rule} | ' + ' | '.join(cells) + ' |')
        lines.append('')
    open(os.path.join(OUT, 'summary.md'), 'w').write('\n'.join(lines))
    print('\n'.join(lines))
    print(f'\nwrote {OUT}/per_fold.csv, curves.csv, summary.md')


if __name__ == '__main__':
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(sys.argv[1:])
