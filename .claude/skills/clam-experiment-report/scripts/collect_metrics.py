"""Collect every number a CLAM experiment report needs, straight from the saved run outputs.

For each run directory (results/<exp_code>_s<seed>/) it reads summary.csv, split_{k}_results.pkl,
experiment_*.txt and the TensorBoard logs {k}/events.out.tfevents.*, and prints markdown-ready
tables: summary metrics, per-fold results, paper-style metrics, pooled confusion matrix,
sensitivity by subtype (optional), training time, training-curve statistics. With two or more
runs it also prints paired per-fold comparisons (Wilcoxon signed-rank) and prediction agreement.

Usage (from the repo root, inside the clam conda env):
  python .claude/skills/clam-experiment-report/scripts/collect_metrics.py \
      results/camelyon_plus_2cls_uni_CLAM_B32_s1 results/camelyon_plus_2cls_conch_CLAM_B32_s1 \
      --names UNI CONCH --subtypes "/mnt/nas/Camelyon+/slide-lables/Camelyon+(4-classes).xlsx" \
      --csv-dir /tmp/metrics

Assumes a binary task (class 1 = positive). Positive-class predictions use argmax (= p1 > 0.5).
"""
import argparse
import ast
import datetime
import glob
import os
import pickle
import struct
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, roc_auc_score
from tensorboardX.proto import event_pb2

METRICS = ['auc', 'acc', 'bacc', 'sens', 'spec', 'prec', 'f1_tumor', 'f1_macro', 'f1_weighted', 'ap_tumor', 'map_macro']
LABEL = {'auc': 'AUC', 'acc': 'Accuracy', 'bacc': 'Balanced accuracy', 'sens': 'Sensitivity', 'spec': 'Specificity',
         'prec': 'Precision (pos.)', 'f1_tumor': 'F1 (pos.)', 'f1_macro': 'F1 (macro)', 'f1_weighted': 'F1 (weighted)',
         'ap_tumor': 'AP (pos.)', 'map_macro': 'mAP (macro)', 'val_auc': 'Val AUC', 'val_acc': 'Val Acc'}


def read_events(path):
    out = []
    with open(path, 'rb') as f:
        while True:
            h = f.read(8)
            if len(h) < 8:
                break
            n = struct.unpack('Q', h)[0]
            f.read(4)
            data = f.read(n)
            f.read(4)
            e = event_pb2.Event()
            e.ParseFromString(data)
            out.append(e)
    return out


def fold_logs(fold_dir):
    """Return scalars + wall-time span of the event file that holds data, and how many files exist.
    An extra empty file usually means a crashed earlier attempt; mention it in the report."""
    files = sorted(glob.glob(os.path.join(fold_dir, 'events.out.tfevents.*')))
    for path in files:
        ev = read_events(path)
        sc = {}
        for e in ev:
            for v in e.summary.value:
                sc.setdefault(v.tag, {})[e.step] = v.simple_value
        if 'val/loss' in sc:
            arr = {t: np.array([sc[t][s] for s in sorted(sc[t])]) for t in sc}
            return arr, min(e.wall_time for e in ev), max(e.wall_time for e in ev), len(files)
    raise FileNotFoundError(f'no event file with val/loss in {fold_dir}')


def metrics(y, prob):
    p1 = prob[:, 1]
    pred = np.argmax(prob, 1)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    sens, spec = tp / (tp + fn), tn / (tn + fp)
    ap1, ap0 = average_precision_score(y, p1), average_precision_score(1 - y, 1 - p1)
    return dict(auc=roc_auc_score(y, p1), acc=(tp + tn) / len(y), bacc=(sens + spec) / 2, sens=sens, spec=spec,
                prec=tp / (tp + fp) if tp + fp else 0.0, f1_tumor=f1_score(y, pred),
                f1_macro=f1_score(y, pred, average='macro'), f1_weighted=f1_score(y, pred, average='weighted'),
                ap_tumor=ap1, map_macro=(ap0 + ap1) / 2, n=len(y), n_neg=int((y == 0).sum()), n_pos=int((y == 1).sum()),
                tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp))


def load_run(run_dir, subtype):
    summary = pd.read_csv(os.path.join(run_dir, 'summary.csv'))
    settings = {}
    for path in glob.glob(os.path.join(run_dir, 'experiment_*.txt')):
        settings = ast.literal_eval(open(path).read())
    folds, preds, curves = [], [], []
    for k in sorted(summary.folds.astype(int)):
        res = pickle.load(open(os.path.join(run_dir, f'split_{k}_results.pkl'), 'rb'))
        slides = list(res)
        y = np.array([int(res[s]['label']) for s in slides])
        prob = np.vstack([res[s]['prob'] for s in slides])
        m = metrics(y, prob)
        row = summary[summary.folds == k].iloc[0]
        # sanity check: recomputed metrics must match what CLAM wrote
        assert np.isclose(m['auc'], row.test_auc) and np.isclose(m['acc'], row.test_acc), f'fold {k} mismatch'
        sc, t0, t1, nfiles = fold_logs(os.path.join(run_dir, str(k)))
        vl = sc['val/loss']
        m.update(fold=k, val_auc=row.val_auc, val_acc=row.val_acc, epochs=len(vl), best_epoch=int(np.argmin(vl)),
                 start=t0, end=t1, n_event_files=nfiles,
                 val_loss_first=vl[0], val_loss_best=vl.min(), val_loss_last=vl[-1],
                 train_loss_first=sc['train/loss'][0], train_loss_last=sc['train/loss'][-1])
        folds.append(m)
        curves.append(sc)
        preds.append(pd.DataFrame(dict(fold=k, slide=slides, y=y, p=prob[:, 1], pred=np.argmax(prob, 1),
                                       sub=[subtype.get(s, '') for s in slides] if subtype else '')))
    return dict(dir=run_dir, settings=settings, folds=pd.DataFrame(folds), preds=pd.concat(preds, ignore_index=True))


f4 = lambda x: f'{x:.4f}'
ms = lambda s: f'{s.mean():.4f} ± {s.std(ddof=1):.4f}'


def report_run(name, run):
    df, preds, st = run['folds'], run['preds'], run['settings']
    print(f'\n\n# ===== {name}  ({run["dir"]})')
    print('\nSettings:', st)
    print('\n## Summary (test, mean ± SD over folds, ddof=1)\n')
    print('| Metric | Mean ± SD | Min | Max |\n|---|---|---|---|')
    for m in ['auc', 'acc', 'bacc', 'sens', 'spec', 'prec', 'f1_tumor']:
        print(f'| {LABEL[m]} | {ms(df[m])} | {f4(df[m].min())} | {f4(df[m].max())} |')
    print('\nAlternative definitions:', ', '.join(f'{LABEL[m]} {ms(df[m])}' for m in ['f1_macro', 'f1_weighted', 'ap_tumor', 'map_macro']))

    print('\n## Per-fold results\n')
    print('| Fold | Epochs run | Best val-loss epoch | Val AUC | Val Acc | **Test AUC** | **Test Acc** | Sens. | Spec. | F1 |')
    print('|---|---|---|---|---|---|---|---|---|---|')
    cols = ['val_auc', 'val_acc', 'auc', 'acc', 'sens', 'spec', 'f1_tumor']
    for _, r in df.iterrows():
        print(f'| {int(r.fold)} | {int(r.epochs)} | {int(r.best_epoch)} | ' + ' | '.join(f4(r[c]) for c in cols) + ' |')
    print('| **Mean** | | | ' + ' | '.join(f4(df[c].mean()) for c in cols) + ' |')
    print('| **SD** | | | ' + ' | '.join(f4(df[c].std(ddof=1)) for c in cols) + ' |')

    print('\n## Paper-style metrics per fold\n')
    print('| Fold | AUC | bACC | F1 (macro) | mAP (macro) |\n|---|---|---|---|---|')
    for _, r in df.iterrows():
        print(f'| {int(r.fold)} | {f4(r.auc)} | {f4(r.bacc)} | {f4(r.f1_macro)} | {f4(r.map_macro)} |')

    print('\n## Test-set composition per fold\n')
    has_sub = preds['sub'].astype(str).str.len().gt(0).any()
    for k, g in preds.groupby('fold'):
        extra = ''
        if has_sub:
            extra = '  subtypes: ' + ', '.join(f'{s}={c}' for s, c in g['sub'].value_counts().items())
        print(f'fold {k}: n={len(g)} neg={(g.y == 0).sum()} pos={(g.y == 1).sum()}{extra}')
    print(f'pooled predictions {len(preds)}, unique slides {preds.slide.nunique()}')

    cm = confusion_matrix(preds.y, preds.pred, labels=[0, 1])
    print('\n## Pooled confusion matrix\n')
    print(f'TN {cm[0, 0]}  FP {cm[0, 1]}  FN {cm[1, 0]}  TP {cm[1, 1]}   pooled AUC {roc_auc_score(preds.y, preds.p):.4f}')

    if has_sub:
        print('\n## Detection rate by subtype (pooled; per-fold mean ± SD)\n')
        for sub, g in preds.groupby('sub'):
            hit = (g.pred == 1).sum()
            pf = g.groupby('fold').pred.mean()
            print(f'{sub}: {hit}/{len(g)} = {hit / len(g):.3f}  per-fold {pf.mean():.3f} ± {pf.std(ddof=1):.3f}')

    print('\n## Training time (TensorBoard wall time)\n')
    print('| Fold | Start | End | Duration | Epochs run | Time / epoch |\n|---|---|---|---|---|---|')
    T = lambda t: datetime.datetime.fromtimestamp(t).strftime('%H:%M:%S')
    for _, r in df.iterrows():
        print(f'| {int(r.fold)} | {T(r.start)} | {T(r.end)} | {(r.end - r.start) / 60:.1f} min | {int(r.epochs)} | {(r.end - r.start) / r.epochs:.1f} s |')
    tot = (df.end.max() - df.start.min()) / 60
    print(f'| **Total** | {T(df.start.min())} | {T(df.end.max())} | **{tot:.1f} min ({int(tot // 60)} h {int(round(tot % 60))} min)** '
          f'| {df.epochs.sum()} | **{(df.end - df.start).sum() / df.epochs.sum():.1f} s (mean)** |')
    print(f'run from {datetime.datetime.fromtimestamp(df.start.min())} to {datetime.datetime.fromtimestamp(df.end.max())}')
    extra_files = df[df.n_event_files > 1].fold.tolist()
    if extra_files:
        print(f'NOTE: folds {extra_files} have more than one event file (probably a crashed earlier attempt)')

    print('\n## Training dynamics\n')
    print(f'best val-loss epoch: {df.best_epoch.tolist()}  mean {df.best_epoch.mean():.1f} ± {df.best_epoch.std(ddof=1):.1f}')
    print(f'epochs run: {df.epochs.tolist()}  total {df.epochs.sum()}')
    for c in ['val_loss_first', 'val_loss_best', 'val_loss_last', 'train_loss_first', 'train_loss_last']:
        print(f'{c}: mean {df[c].mean():.3f}  range {df[c].min():.3f}–{df[c].max():.3f}')
    print(f'corr(val AUC, test AUC) across folds: r = {np.corrcoef(df.val_auc, df.auc)[0, 1]:.2f}')


def compare(names, runs):
    print('\n\n# ===== Paired comparisons (same splits => same test set per fold)')
    print('Wilcoxon signed-rank, two-sided, exact. With 10 folds the smallest possible p is 1/512 ≈ 0.00195.')
    for a, b in combinations(range(len(names)), 2):
        A, B = runs[a]['folds'], runs[b]['folds']
        if not np.array_equal(A.fold.values, B.fold.values):
            print(f'\n{names[a]} vs {names[b]}: different folds, skipped')
            continue
        same = all(set(runs[a]['preds'][runs[a]['preds'].fold == k].slide) == set(runs[b]['preds'][runs[b]['preds'].fold == k].slide) for k in A.fold)
        print(f'\n## {names[a]} − {names[b]}' + ('' if same else '   WARNING: test sets differ, pairing invalid'))
        print('| Metric | Difference (mean ± SD) | Folds first better | p |\n|---|---|---|---|')
        for m in ['auc', 'acc', 'bacc', 'sens', 'spec', 'prec', 'f1_tumor', 'f1_macro', 'map_macro', 'ap_tumor', 'val_auc']:
            d = A[m].values - B[m].values
            try:
                p = wilcoxon(A[m], B[m]).pvalue
            except ValueError:
                p = float('nan')
            ties = int((d == 0).sum())
            print(f'| {LABEL[m]} | {d.mean():+.4f} ± {d.std(ddof=1):.4f} | {(d > 0).sum()} / {len(d)}' + (f' ({ties} ties)' if ties else '') + f' | {p:.3g} |')
        m = runs[a]['preds'].merge(runs[b]['preds'], on=['fold', 'slide'], suffixes=('_a', '_b'))
        print(f'prediction agreement {(m.pred_a == m.pred_b).mean():.3f} ({(m.pred_a == m.pred_b).sum()}/{len(m)}), '
              f'prob corr {np.corrcoef(m.p_a, m.p_b)[0, 1]:.3f}, flips a0->b1 {((m.pred_a == 0) & (m.pred_b == 1)).sum()}, a1->b0 {((m.pred_a == 1) & (m.pred_b == 0)).sum()}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('runs', nargs='+', help='run directories, e.g. results/<exp_code>_s1')
    ap.add_argument('--names', nargs='+', help='display names, same order as runs')
    ap.add_argument('--subtypes', help='xlsx/csv with columns slide,label for a per-subtype breakdown (optional)')
    ap.add_argument('--csv-dir', help='also write per-fold metrics and per-slide predictions as CSV here')
    args = ap.parse_args()
    names = args.names or [os.path.basename(r.rstrip('/')) for r in args.runs]
    subtype = None
    if args.subtypes:
        t = pd.read_excel(args.subtypes) if args.subtypes.endswith(('.xlsx', '.xls')) else pd.read_csv(args.subtypes)
        subtype = t.set_index(t.columns[0])[t.columns[1]].astype(str).to_dict()
    runs = [load_run(r, subtype) for r in args.runs]
    for n, r in zip(names, runs):
        report_run(n, r)
        if args.csv_dir:
            os.makedirs(args.csv_dir, exist_ok=True)
            r['folds'].to_csv(os.path.join(args.csv_dir, f'{n}_folds.csv'), index=False)
            r['preds'].to_csv(os.path.join(args.csv_dir, f'{n}_preds.csv'), index=False)
    if len(runs) > 1:
        compare(names, runs)


if __name__ == '__main__':
    main()
