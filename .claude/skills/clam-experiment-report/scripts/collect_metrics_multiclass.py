"""Collect the numbers for a multi-class CLAM experiment report from the saved run outputs.

For each run directory it reads summary.csv, split_{k}_results.pkl, experiment_*.txt and the
TensorBoard logs, and writes per-fold CSVs plus a printed summary:

  * multi-class metrics per fold: macro one-vs-rest AUC (checked against summary.csv), accuracy,
    balanced accuracy (mean per-class recall), macro / weighted F1, Cohen's kappa (unweighted and
    quadratic-weighted, for ordered classes);
  * per-class one-vs-rest AUC, recall, precision and F1 per fold;
  * the pooled confusion matrix;
  * a binary collapse (class 0 = negative vs. all other classes = positive):
      P(positive) = 1 - p_0, predicted positive when argmax != 0 (the model's own 4-class decision).
    This makes the run comparable fold by fold with binary runs on the same splits;
  * training time and dynamics from TensorBoard.

Usage (repo root, clam env):
  python .claude/skills/clam-experiment-report/scripts/collect_metrics_multiclass.py \
      results/<run_A> results/<run_B> --names A B --classes negative itc micro macro --csv-dir <dir>
"""
import argparse
import glob
import os
import pickle
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import (cohen_kappa_score, confusion_matrix, f1_score, precision_score, recall_score,
                             roc_auc_score)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from collect_metrics import fold_logs  # noqa: E402  (same TensorBoard reader as the binary script)


def load_run(run_dir, classes):
    C = len(classes)
    summary = pd.read_csv(os.path.join(run_dir, 'summary.csv'))
    folds, preds = [], []
    for k in sorted(summary.folds.astype(int)):
        res = pickle.load(open(os.path.join(run_dir, f'split_{k}_results.pkl'), 'rb'))
        slides = list(res)
        y = np.array([int(res[s]['label']) for s in slides])
        P = np.vstack([res[s]['prob'] for s in slides])
        pred = P.argmax(1)
        row = summary[summary.folds == k].iloc[0]
        # CLAM's multi-class AUC: mean over classes of one-vs-rest AUC (classes absent from the fold are skipped)
        per_auc = [roc_auc_score(y == c, P[:, c]) if 0 < (y == c).sum() < len(y) else np.nan for c in range(C)]
        auc = np.nanmean(per_auc)
        assert np.isclose(auc, row.test_auc), f'fold {k}: AUC {auc} != summary {row.test_auc}'
        assert np.isclose((pred == y).mean(), row.test_acc), f'fold {k}: accuracy mismatch'
        rec = recall_score(y, pred, labels=range(C), average=None, zero_division=0)
        prec = precision_score(y, pred, labels=range(C), average=None, zero_division=0)
        f1 = f1_score(y, pred, labels=range(C), average=None, zero_division=0)
        # binary collapse: negative (class 0) vs. any other class
        yb, pb, predb = (y != 0).astype(int), 1 - P[:, 0], (pred != 0).astype(int)
        tn, fp, fn, tp = confusion_matrix(yb, predb, labels=[0, 1]).ravel()
        m = dict(fold=k, auc=auc, acc=(pred == y).mean(), bacc=rec.mean(),
                 f1_macro=f1_score(y, pred, labels=range(C), average='macro', zero_division=0),
                 f1_weighted=f1_score(y, pred, labels=range(C), average='weighted', zero_division=0),
                 kappa=cohen_kappa_score(y, pred), kappa_qw=cohen_kappa_score(y, pred, weights='quadratic'),
                 val_auc=row.val_auc, val_acc=row.val_acc,
                 bin_auc=roc_auc_score(yb, pb), bin_acc=(predb == yb).mean(), bin_sens=tp / (tp + fn),
                 bin_spec=tn / (tn + fp), bin_bacc=(tp / (tp + fn) + tn / (tn + fp)) / 2,
                 bin_f1=f1_score(yb, predb))
        for c, name in enumerate(classes):
            # 'cls_' prefix: a class may be called e.g. 'macro', which must not overwrite f1_macro
            m.update({f'cls_auc_{name}': per_auc[c], f'cls_recall_{name}': rec[c], f'cls_prec_{name}': prec[c],
                      f'cls_f1_{name}': f1[c], f'cls_n_{name}': int((y == c).sum())})
        sc, t0, t1, nfiles = fold_logs(os.path.join(run_dir, str(k)))
        vl = sc['val/loss']
        m.update(epochs=len(vl), best_epoch=int(np.argmin(vl)), start=t0, end=t1, n_event_files=nfiles,
                 val_loss_first=vl[0], val_loss_best=vl.min(), val_loss_last=vl[-1],
                 train_loss_first=sc['train/loss'][0], train_loss_last=sc['train/loss'][-1])
        folds.append(m)
        d = pd.DataFrame(dict(fold=k, slide=slides, y=y, pred=pred))
        for c, name in enumerate(classes):
            d[f'p_{name}'] = P[:, c]
        preds.append(d)
    return pd.DataFrame(folds), pd.concat(preds, ignore_index=True)


def ms(s):
    return f'{s.mean():.4f} ± {s.std(ddof=1):.4f}'


def report(name, df, preds, classes):
    print(f'\n\n# ===== {name}')
    print('\n## Multi-class (mean ± SD over folds)')
    for m in ['auc', 'acc', 'bacc', 'f1_macro', 'f1_weighted', 'kappa', 'kappa_qw', 'val_auc']:
        print(f'{m:12s} {ms(df[m])}   (min {df[m].min():.4f}, max {df[m].max():.4f})')
    print('\n## Per class (mean ± SD over folds)')
    print('| Class | n per test fold | AUC (OvR) | Recall | Precision | F1 |\n|---|---|---|---|---|---|')
    for c in classes:
        print(f'| {c} | {df[f"cls_n_{c}"].min()}–{df[f"cls_n_{c}"].max()} | {ms(df[f"cls_auc_{c}"])} | {ms(df[f"cls_recall_{c}"])} | '
              f'{ms(df[f"cls_prec_{c}"])} | {ms(df[f"cls_f1_{c}"])} |')
    cm = confusion_matrix(preds.y, preds.pred, labels=range(len(classes)))
    print('\n## Pooled confusion matrix (rows = true, columns = predicted)')
    print('| | ' + ' | '.join(classes) + ' |\n|---|' + '---|' * len(classes))
    for i, c in enumerate(classes):
        print(f'| {c} ({cm[i].sum()}) | ' + ' | '.join(str(v) for v in cm[i]) + ' |')
    print('\n## Binary collapse (negative vs. tumor)')
    for m in ['bin_auc', 'bin_acc', 'bin_bacc', 'bin_sens', 'bin_spec', 'bin_f1']:
        print(f'{m:10s} {ms(df[m])}')
    print('\n## Training')
    print(f'best val-loss epoch {df.best_epoch.tolist()} mean {df.best_epoch.mean():.1f}; epochs {df.epochs.tolist()} total {df.epochs.sum()}')
    for c in ['val_loss_first', 'val_loss_best', 'val_loss_last', 'train_loss_first', 'train_loss_last']:
        print(f'{c}: mean {df[c].mean():.3f} range {df[c].min():.3f}–{df[c].max():.3f}')
    tot = (df.end.max() - df.start.min()) / 60
    print(f'time: total {tot:.1f} min ({int(tot // 60)} h {int(round(tot % 60))} min), {(df.end - df.start).sum() / df.epochs.sum():.1f} s/epoch')
    print(f'corr(val AUC, test AUC) r = {np.corrcoef(df.val_auc, df.auc)[0, 1]:.2f}')
    if (df.n_event_files > 1).any():
        print('NOTE: extra event files in folds', df[df.n_event_files > 1].fold.tolist())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('runs', nargs='+')
    ap.add_argument('--names', nargs='+')
    ap.add_argument('--classes', nargs='+', required=True, help='class names in label-index order')
    ap.add_argument('--csv-dir')
    a = ap.parse_args()
    names = a.names or [os.path.basename(r.rstrip('/')) for r in a.runs]
    for n, r in zip(names, a.runs):
        df, preds = load_run(r, a.classes)
        report(n, df, preds, a.classes)
        if a.csv_dir:
            os.makedirs(a.csv_dir, exist_ok=True)
            df.to_csv(os.path.join(a.csv_dir, f'{n}_folds.csv'), index=False)
            preds.to_csv(os.path.join(a.csv_dir, f'{n}_preds.csv'), index=False)


if __name__ == '__main__':
    main()
