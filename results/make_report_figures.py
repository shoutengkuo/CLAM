"""Generate the figures used in the Camelyon+ experiment reports.

Reads split_{k}_results.pkl, summary.csv and the TensorBoard logs of each run, plus the
Camelyon+ 4-class label file, and writes PNG figures:

  results/<run>/figures/*.png   figures for a single run (reports 1-3)
  results/figures/*.png         comparison figures (reports 1-5)

Usage (from the repo root):
  python results/make_report_figures.py              # all figures
  python results/make_report_figures.py --only ienf  # only the IENF report figures
  python results/make_report_figures.py --only ienf_isfn  # only the IENF + iSFN report figures
  python results/make_report_figures.py --only ienf_ablation  # only the IENF ablation figures
  python results/make_report_figures.py --only ienf_norm_rf  # only the IENF normalization / random-fold figures
"""
import glob
import os
import pickle
import struct
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from PIL import Image
from scipy import stats
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_auc_score, roc_curve
from tensorboardX.proto import event_pb2

RESULTS = os.path.dirname(os.path.abspath(__file__))
LABELS_4CLS = '/mnt/nas/Camelyon+/slide-lables/Camelyon+(4-classes).xlsx'
K = 10

RUNS = {
    'r50_b8': dict(dir='camelyon_plus_2cls_resnet50_CLAM_s1', name='ResNet-50, B = 8', color='#86b6ef'),
    'r50_b32': dict(dir='camelyon_plus_2cls_resnet50_CLAM_B32_s1', name='ResNet-50', color='#2a78d6'),
    'uni_b32': dict(dir='camelyon_plus_2cls_uni_CLAM_B32_s1', name='UNI', color='#eb6834'),
    'conch_b32': dict(dir='camelyon_plus_2cls_conch_CLAM_B32_s1', name='CONCH', color='#1baf7a'),
    # model_size = big: a darker step of the same encoder hue
    'r50_big': dict(dir='camelyon_plus_2cls_resnet50_CLAM_B32_big_s1', name='ResNet-50, big', color='#184f95'),
    'uni_big': dict(dir='camelyon_plus_2cls_uni_CLAM_B32_big_s1', name='UNI, big', color='#a8441b'),
    'conch_big': dict(dir='camelyon_plus_2cls_conch_CLAM_B32_big_s1', name='CONCH, big', color='#0f7a54'),
}
SIZE_PAIRS = [('r50_b32', 'r50_big', 'ResNet-50'), ('uni_b32', 'uni_big', 'UNI'), ('conch_b32', 'conch_big', 'CONCH')]
# CLAM-MB (model_type clam_mb, small): the darkest step of each encoder hue (the middle step is model_size big)
RUNS.update({
    'r50_mb': dict(dir='camelyon_plus_2cls_resnet50_CLAM_MB_B32_s1', name='ResNet-50, MB', color='#0d366b'),
    'uni_mb': dict(dir='camelyon_plus_2cls_uni_CLAM_MB_B32_s1', name='UNI, MB', color='#6e2c10'),
    'conch_mb': dict(dir='camelyon_plus_2cls_conch_CLAM_MB_B32_s1', name='CONCH, MB', color='#0a4f37'),
})
MB_PAIRS = [('r50_b32', 'r50_mb', 'ResNet-50'), ('uni_b32', 'uni_mb', 'UNI'), ('conch_b32', 'conch_mb', 'CONCH')]
ENCODERS = ['r50_b32', 'uni_b32', 'conch_b32']

# Values reported in the original CLAM paper (CLAM-SB, ResNet-50), used in report 4.
CLAM_PAPER = {'auc': (0.9532, 0.0292), 'bacc': (0.9042, 0.0365), 'f1_macro': (0.8878, 0.0471), 'map_macro': (0.9464, 0.0302)}

# Chart chrome (light surface).
SURFACE = '#fcfcfb'
INK = '#0b0b0b'
INK2 = '#52514e'
MUTED = '#898781'
GRID = '#e1e0d9'
AXIS = '#c3c2b7'
REF_GRAY = '#898781'
BLUE_RAMP = ['#f3f8fe', '#cde2fb', '#9ec5f4', '#6da7ec', '#3987e5', '#256abf', '#184f95', '#0d366b']

plt.rcParams.update({
    'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
    'font.family': 'DejaVu Sans', 'font.size': 10,
    'text.color': INK, 'axes.labelcolor': INK2, 'axes.titlecolor': INK,
    'axes.titlesize': 11, 'axes.titleweight': 'semibold', 'axes.titlelocation': 'left', 'axes.titlepad': 10,
    'axes.edgecolor': AXIS, 'axes.linewidth': 0.8,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.8, 'grid.linestyle': '-',
    'axes.axisbelow': True,
    'xtick.color': MUTED, 'ytick.color': MUTED, 'xtick.labelcolor': INK2, 'ytick.labelcolor': INK2,
    'xtick.major.size': 0, 'ytick.major.size': 0,
    'legend.frameon': False, 'legend.fontsize': 9, 'legend.labelcolor': INK2,
    'lines.linewidth': 2, 'lines.solid_capstyle': 'round', 'lines.solid_joinstyle': 'round',
    'savefig.dpi': 160, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.15,
})

METRIC_LABELS = {
    'auc': 'AUC', 'acc': 'Accuracy', 'bacc': 'Balanced acc.', 'sens': 'Sensitivity', 'spec': 'Specificity',
    'prec': 'Precision', 'f1_tumor': 'F1 (tumor)', 'f1_macro': 'F1 (macro)', 'map_macro': 'mAP (macro)',
}
SUBTYPES = [('macro', 'Macro'), ('micro', 'Micro'), ('itc', 'ITC')]


# ----------------------------------------------------------------------------- data
def read_events(path):
    events = []
    with open(path, 'rb') as f:
        while True:
            header = f.read(8)
            if len(header) < 8:
                break
            n = struct.unpack('Q', header)[0]
            f.read(4)
            data = f.read(n)
            f.read(4)
            e = event_pb2.Event()
            e.ParseFromString(data)
            events.append(e)
    return events


def read_scalars(fold_dir):
    """Scalars of the event file that has content (fold 0 of the B = 8 run has an extra empty one)."""
    for path in sorted(glob.glob(os.path.join(fold_dir, 'events.out.tfevents.*'))):
        scalars = {}
        for e in read_events(path):
            for v in e.summary.value:
                scalars.setdefault(v.tag, {})[e.step] = v.simple_value
        if 'val/loss' in scalars:
            out = {}
            for tag in ['train/loss', 'val/loss', 'val/auc']:
                steps = sorted(scalars[tag])
                out[tag] = np.array([scalars[tag][s] for s in steps])
            return out
    raise FileNotFoundError(fold_dir)


def fold_metrics(y, p1):
    pred = (p1 > 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    sens, spec = tp / (tp + fn), tn / (tn + fp)
    prec = tp / (tp + fp) if tp + fp else 0.0
    from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
    return dict(auc=roc_auc_score(y, p1), acc=(tp + tn) / len(y), bacc=(sens + spec) / 2, sens=sens, spec=spec,
                prec=prec, f1_tumor=f1_score(y, pred), f1_macro=f1_score(y, pred, average='macro'),
                map_macro=(average_precision_score(y, p1) + average_precision_score(1 - y, 1 - p1)) / 2)


def load_run(key, subtype):
    d = os.path.join(RESULTS, RUNS[key]['dir'])
    summary = pd.read_csv(os.path.join(d, 'summary.csv'))
    folds, preds, curves = [], [], []
    for k in range(K):
        res = pickle.load(open(os.path.join(d, f'split_{k}_results.pkl'), 'rb'))
        slides = list(res)
        y = np.array([res[s]['label'] for s in slides])
        prob = np.vstack([res[s]['prob'] for s in slides])
        p1 = prob[:, 1]
        # argmax == (p1 > 0.5) except for exact ties, which do not occur here
        assert np.array_equal(np.argmax(prob, 1), (p1 > 0.5).astype(int))
        m = fold_metrics(y, p1)
        row = summary[summary.folds == k].iloc[0]
        assert np.isclose(m['auc'], row.test_auc)
        sc = read_scalars(os.path.join(d, str(k)))
        m.update(fold=k, val_auc=row.val_auc, best_epoch=int(np.argmin(sc['val/loss'])), epochs=len(sc['val/loss']))
        folds.append(m)
        curves.append(sc)
        preds.append(pd.DataFrame(dict(fold=k, slide=slides, y=y, p=p1, pred=(p1 > 0.5).astype(int),
                                       sub=[subtype[s] for s in slides])))
    return dict(folds=pd.DataFrame(folds), preds=pd.concat(preds, ignore_index=True), curves=curves)


# ----------------------------------------------------------------------------- helpers
def save(fig, path):
    """tight_layout first, then the figure-level legend and footnote below the plots."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.tight_layout()
    handles = getattr(fig, '_legend_handles', None)
    y = 0.0
    if handles:
        fig.legend(handles=handles, loc='upper left', bbox_to_anchor=(0.01, y), ncol=len(handles),
                   handlelength=1.6, columnspacing=1.8, frameon=False)
        y -= 0.09 * (4.0 / fig.get_figheight())
    note = getattr(fig, '_note', None)
    if note:
        fig.text(0.01, y - 0.02, note, color=MUTED, fontsize=8, va='top')
    fig.savefig(path)
    plt.close(fig)
    print('wrote', os.path.relpath(path, RESULTS))


def legend_below(ax, handles):
    ax.figure._legend_handles = handles


def note(fig, text):
    fig._note = text


def mean_roc(preds):
    grid = np.linspace(0, 1, 201)
    tprs = []
    for _, g in preds.groupby('fold'):
        fpr, tpr, _ = roc_curve(g.y, g.p)
        t = np.interp(grid, fpr, tpr)
        t[0] = 0.0
        tprs.append(t)
    tprs = np.array(tprs)
    return grid, tprs.mean(0), tprs.std(0, ddof=1)


def mean_pr(preds):
    grid = np.linspace(0, 1, 201)
    precs = []
    for _, g in preds.groupby('fold'):
        prec, rec, _ = precision_recall_curve(g.y, g.p)
        # interpolated precision: best precision at recall >= r
        prec_i = np.maximum.accumulate(prec)  # recall is decreasing along the arrays
        order = np.argsort(rec)
        precs.append(np.interp(grid, rec[order], prec_i[order]))
    precs = np.array(precs)
    return grid, precs.mean(0), precs.std(0, ddof=1)


def curve_matrix(run, tag, n=52):
    """Per-fold curve truncated to the first n epochs (every fold ran at least 52)."""
    return np.array([c[tag][:n] for c in run['curves']])


def subtype_rates(preds):
    out = {}
    for sub, _ in SUBTYPES:
        g = preds[preds['sub'] == sub]
        per_fold = g.groupby('fold').pred.mean()
        out[sub] = dict(pooled=(g.pred == 1).mean(), n=len(g), hit=int((g.pred == 1).sum()), per_fold=per_fold.values)
    return out


def text_on(color_hex):
    r, g, b = [int(color_hex[i:i + 2], 16) / 255 for i in (1, 3, 5)]
    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return INK if lum > 0.55 else '#ffffff'


# ----------------------------------------------------------------------------- single-run figures
def fig_training_curves(run, key, path):
    color = RUNS[key]['color']
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for ax, tag, title in zip(axes, ['train/loss', 'val/loss', 'val/auc'], ['Training loss', 'Validation loss', 'Validation AUC']):
        for k, c in enumerate(run['curves']):
            ax.plot(np.arange(len(c[tag])), c[tag], color=color, lw=1, alpha=0.35)
        m = curve_matrix(run, tag)
        ax.plot(np.arange(m.shape[1]), m.mean(0), color=color, lw=2.2)
        if tag == 'val/loss':
            for c in run['curves']:
                b = int(np.argmin(c[tag]))
                ax.plot(b, c[tag][b], 'o', ms=6, color=color, mec=SURFACE, mew=1.5, zorder=5)
        ax.axvline(50, color=AXIS, lw=1, zorder=1)
        ax.set_title(title)
        ax.set_xlabel('Epoch')
        ax.margins(x=0.01)
    axes[0].set_ylim(bottom=0)
    axes[1].set_ylim(bottom=0)
    axes[2].text(50.8, axes[2].get_ylim()[0], ' min. stop epoch', color=MUTED, fontsize=8, va='bottom')
    handles = [Line2D([], [], color=color, lw=1, alpha=0.5, label='Each fold'),
               Line2D([], [], color=color, lw=2.2, label='Mean over folds (first 52 epochs)'),
               Line2D([], [], color=color, marker='o', lw=0, ms=6, mec=SURFACE, label='Lowest val loss (checkpoint used for testing)')]
    legend_below(axes[0], handles)
    save(fig, path)


def fig_roc_pr(run, key, path):
    color = RUNS[key]['color']
    preds = run['preds']
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))
    ax = axes[0]
    for _, g in preds.groupby('fold'):
        fpr, tpr, _ = roc_curve(g.y, g.p)
        ax.plot(fpr, tpr, color=color, lw=1, alpha=0.3)
    grid, m, s = mean_roc(preds)
    ax.fill_between(grid, np.clip(m - s, 0, 1), np.clip(m + s, 0, 1), color=color, alpha=0.12, lw=0)
    auc = run['folds'].auc
    ax.plot(grid, m, color=color, lw=2.2, label=f'Mean ROC (AUC {auc.mean():.3f} ± {auc.std(ddof=1):.3f})')
    ax.plot([0, 1], [0, 1], color=AXIS, lw=1)
    ax.set(xlim=(0, 1), ylim=(0, 1.01), xlabel='False positive rate (1 − specificity)', ylabel='True positive rate (sensitivity)')
    ax.set_title('ROC curve, 10 test folds')
    ax.set_aspect('equal')
    ax.legend(loc='lower right')
    ax = axes[1]
    for _, g in preds.groupby('fold'):
        prec, rec, _ = precision_recall_curve(g.y, g.p)
        ax.plot(rec, prec, color=color, lw=1, alpha=0.3)
    grid, m, s = mean_pr(preds)
    ax.fill_between(grid, np.clip(m - s, 0, 1), np.clip(m + s, 0, 1), color=color, alpha=0.12, lw=0)
    ax.plot(grid, m, color=color, lw=2.2, label='Mean precision-recall curve')
    prevalence = preds.y.mean()
    ax.axhline(prevalence, color=AXIS, lw=1)
    ax.text(0.02, prevalence + 0.015, f'chance level (tumor prevalence {prevalence:.2f})', color=MUTED, fontsize=8)
    ax.set(xlim=(0, 1), ylim=(0, 1.01), xlabel='Recall (sensitivity)', ylabel='Precision')
    ax.set_title('Precision-recall curve (tumor class)')
    ax.set_aspect('equal')
    ax.legend(loc='lower left')
    note(fig, 'Thin lines: individual folds. Band: ±1 SD across folds.')
    save(fig, path)


def draw_confusion(ax, preds, title):
    cm = confusion_matrix(preds.y, preds.pred, labels=[0, 1])
    row_pct = cm / cm.sum(1, keepdims=True)
    cmap = LinearSegmentedColormap.from_list('blue', BLUE_RAMP)
    ax.imshow(row_pct, cmap=cmap, vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            fill = cmap(row_pct[i, j])
            hexc = '#%02x%02x%02x' % tuple(int(255 * c) for c in fill[:3])
            tc = text_on(hexc)
            ax.text(j, i - 0.08, f'{cm[i, j]}', ha='center', va='center', fontsize=14, fontweight='semibold', color=tc)
            ax.text(j, i + 0.2, f'{row_pct[i, j]:.1%} of row', ha='center', va='center', fontsize=8.5, color=tc)
    ax.set_xticks([0, 1], ['Negative', 'Tumor'])
    ax.set_yticks([0, 1], ['Negative', 'Tumor'])
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(title)


def fig_confusion_subtype(run, key, path):
    color = RUNS[key]['color']
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4), gridspec_kw=dict(width_ratios=[1, 1.25]))
    draw_confusion(axes[0], run['preds'], 'Pooled confusion matrix\n(1284 test predictions)')
    ax = axes[1]
    rates = subtype_rates(run['preds'])
    x = np.arange(len(SUBTYPES))
    vals = [rates[s]['pooled'] for s, _ in SUBTYPES]
    ax.bar(x, vals, width=0.32, color=color)
    rng = np.random.default_rng(0)
    for i, (s, _) in enumerate(SUBTYPES):
        pf = rates[s]['per_fold']
        ax.plot(i + rng.uniform(-0.12, 0.12, len(pf)), pf, 'o', ms=4.5, color=INK2, alpha=0.55, mec=SURFACE, mew=0.8, zorder=4)
        ax.text(i, min(vals[i] + 0.04, 1.06), f'{vals[i]:.0%}\n({rates[s]["hit"]}/{rates[s]["n"]})', ha='center', va='bottom', fontsize=9, color=INK)
    ax.set_xticks(x, [f'{lab}-metastasis' if s != 'itc' else 'ITC' for s, lab in SUBTYPES])
    ax.set_ylim(0, 1.2)
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.set_yticklabels([f'{v:.0%}' for v in np.linspace(0, 1, 6)])
    ax.set_ylabel('Detected as tumor')
    ax.set_title('Sensitivity by metastasis size')
    ax.grid(axis='x', visible=False)
    legend_below(ax, [Line2D([], [], color=color, lw=8, label='Pooled over folds'),
                      Line2D([], [], color=INK2, marker='o', lw=0, ms=4.5, alpha=0.6, label='Single fold')])
    save(fig, path)


def fig_per_fold_metrics(run, key, path):
    color = RUNS[key]['color']
    metrics = ['auc', 'acc', 'bacc', 'sens', 'spec', 'f1_tumor']
    df = run['folds']
    fig, ax = plt.subplots(figsize=(9, 3.9))
    rng = np.random.default_rng(1)
    for i, m in enumerate(metrics):
        v = df[m].values
        ax.plot(i + rng.uniform(-0.15, 0.15, len(v)), v, 'o', ms=6, color=color, alpha=0.6, mec=SURFACE, mew=1, zorder=3)
        mu, sd = v.mean(), v.std(ddof=1)
        ax.errorbar(i + 0.32, mu, yerr=sd, fmt='o', ms=7, color=INK, ecolor=INK, elinewidth=1.5, capsize=0, zorder=4)
        ax.text(i + 0.42, mu, f'{mu:.3f}', va='center', fontsize=8.5, color=INK)
    ax.set_xticks(range(len(metrics)), [METRIC_LABELS[m] for m in metrics])
    ax.set_xlim(-0.5, len(metrics) - 0.2)
    ax.set_ylabel('Test set value')
    ax.grid(axis='x', visible=False)
    ax.set_title('Test metrics per fold')
    legend_below(ax, [Line2D([], [], color=color, marker='o', lw=0, ms=6, label='Single fold'),
                      Line2D([], [], color=INK, marker='o', lw=1.5, ms=7, label='Mean ± SD')])
    save(fig, path)


def fig_val_vs_test(run, key, path):
    color = RUNS[key]['color']
    df = run['folds']
    fig, ax = plt.subplots(figsize=(5, 4.6))
    lo = min(df.val_auc.min(), df.auc.min()) - 0.02
    hi = min(1.0, max(df.val_auc.max(), df.auc.max()) + 0.015)
    ax.plot([lo, hi], [lo, hi], color=AXIS, lw=1)
    ax.text(hi - 0.002, hi - 0.006, 'val = test', color=MUTED, fontsize=8, ha='right', va='top', rotation=45, rotation_mode='anchor')
    ax.plot(df.val_auc, df.auc, 'o', ms=8, color=color, mec=SURFACE, mew=2, zorder=3)
    for _, r in df.iterrows():
        ax.annotate(str(int(r.fold)), (r.val_auc, r.auc), xytext=(6, 2), textcoords='offset points', fontsize=8, color=INK2)
    r = np.corrcoef(df.val_auc, df.auc)[0, 1]
    ax.set(xlim=(lo, hi), ylim=(lo, hi), xlabel='Validation AUC', ylabel='Test AUC')
    ax.set_aspect('equal')
    ax.set_title(f'Validation vs. test AUC per fold (r = {r:.2f})'.replace('= -', '= −'))
    save(fig, path)


# ----------------------------------------------------------------------------- comparison figures
def grouped_bars(ax, keys, runs, metrics, values=None, width=None):
    n = len(keys)
    width = width or min(0.8 / n, 0.24)
    x = np.arange(len(metrics))
    for i, key in enumerate(keys):
        if values is not None:
            mu = np.array([values[key][m][0] for m in metrics])
            sd = np.array([values[key][m][1] for m in metrics])
        else:
            df = runs[key]['folds']
            mu = np.array([df[m].mean() for m in metrics])
            sd = np.array([df[m].std(ddof=1) for m in metrics])
        pos = x + (i - (n - 1) / 2) * (width + 0.02)
        color = REF_GRAY if key == 'paper' else RUNS[key]['color']
        ax.bar(pos, mu, width=width, color=color, zorder=2)
        ax.errorbar(pos, mu, yerr=sd, fmt='none', ecolor=INK2, elinewidth=1.2, capsize=0, zorder=3)
    ax.set_xticks(x, [METRIC_LABELS[m] for m in metrics])
    ax.grid(axis='x', visible=False)


def fig_encoder_metrics(runs, path):
    metrics = ['auc', 'acc', 'bacc', 'sens', 'spec', 'prec', 'f1_tumor', 'f1_macro', 'map_macro']
    fig, ax = plt.subplots(figsize=(12, 4.2))
    grouped_bars(ax, ENCODERS, runs, metrics)
    ax.set_ylim(0.6, 1.0)
    ax.set_ylabel('Test set value (mean ± SD)')
    ax.set_title('Test performance by encoder (10 folds, B = 32)')
    legend_below(ax, [Line2D([], [], color=RUNS[k]['color'], lw=8, label=RUNS[k]['name']) for k in ENCODERS])
    note(fig, 'The y-axis starts at 0.6 to make the differences visible.')
    save(fig, path)


def fig_encoder_per_fold_auc(runs, path):
    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(K)
    for i, key in enumerate(ENCODERS):
        v = runs[key]['folds'].auc.values
        ax.plot(x + (i - 1) * 0.18, v, 'o', ms=8, color=RUNS[key]['color'], mec=SURFACE, mew=2, zorder=3)
    for k in range(K):
        vals = [runs[key]['folds'].auc.values[k] for key in ENCODERS]
        ax.plot([k - 0.18, k, k + 0.18], vals, color=AXIS, lw=1.2, zorder=1)
    ax.set_xticks(x, [f'Fold {k}' for k in range(K)])
    ax.set_ylabel('Test AUC')
    ax.grid(axis='x', visible=False)
    ax.set_title('Test AUC per fold (same test set for all encoders within a fold)')
    legend_below(ax, [Line2D([], [], color=RUNS[k]['color'], marker='o', lw=0, ms=8, label=RUNS[k]['name']) for k in ENCODERS])
    save(fig, path)


def fig_encoder_roc_pr(runs, path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6))
    for key in ENCODERS:
        c = RUNS[key]['color']
        preds = runs[key]['preds']
        grid, m, s = mean_roc(preds)
        axes[0].fill_between(grid, np.clip(m - s, 0, 1), np.clip(m + s, 0, 1), color=c, alpha=0.10, lw=0)
        auc = runs[key]['folds'].auc
        axes[0].plot(grid, m, color=c, label=f'{RUNS[key]["name"]} (AUC {auc.mean():.3f})')
        grid, m, s = mean_pr(preds)
        axes[1].fill_between(grid, np.clip(m - s, 0, 1), np.clip(m + s, 0, 1), color=c, alpha=0.10, lw=0)
        mp = runs[key]['folds'].map_macro
        axes[1].plot(grid, m, color=c, label=RUNS[key]['name'])
    axes[0].plot([0, 1], [0, 1], color=AXIS, lw=1)
    axes[0].set(xlim=(0, 1), ylim=(0, 1.01), xlabel='False positive rate', ylabel='True positive rate')
    axes[0].set_title('Mean ROC curve')
    axes[0].legend(loc='lower right')
    prevalence = runs['r50_b32']['preds'].y.mean()
    axes[1].axhline(prevalence, color=AXIS, lw=1)
    axes[1].set(xlim=(0, 1), ylim=(0, 1.01), xlabel='Recall', ylabel='Precision')
    axes[1].set_title('Mean precision-recall curve (tumor class)')
    axes[1].legend(loc='lower left')
    for ax in axes:
        ax.set_aspect('equal')
    note(fig, 'Curves are averaged over the 10 test folds. Band: ±1 SD across folds.')
    save(fig, path)


def fig_subtype_compare(runs, keys, path, title):
    fig, ax = plt.subplots(figsize=(8, 4))
    n = len(keys)
    width = min(0.8 / n, 0.24)
    x = np.arange(len(SUBTYPES))
    for i, key in enumerate(keys):
        rates = subtype_rates(runs[key]['preds'])
        vals = [rates[s]['pooled'] for s, _ in SUBTYPES]
        pos = x + (i - (n - 1) / 2) * (width + 0.02)
        ax.bar(pos, vals, width=width, color=RUNS[key]['color'], zorder=2)
        for p, v, (s, _) in zip(pos, vals, SUBTYPES):
            ax.text(p, v + 0.015, f'{v:.0%}', ha='center', va='bottom', fontsize=8.5, color=INK)
    totals = subtype_rates(runs[keys[0]]['preds'])
    ax.set_xticks(x, [f'{lab}-metastasis\n(n = {totals[s]["n"]})' if s != 'itc' else f'ITC\n(n = {totals[s]["n"]})' for s, lab in SUBTYPES])
    ax.set_ylim(0, 1.1)
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.set_yticklabels([f'{v:.0%}' for v in np.linspace(0, 1, 6)])
    ax.set_ylabel('Detected as tumor (pooled)')
    ax.grid(axis='x', visible=False)
    ax.set_title(title)
    legend_below(ax, [Line2D([], [], color=RUNS[k]['color'], lw=8, label=RUNS[k]['name']) for k in keys])
    save(fig, path)


def fig_encoder_confusion(runs, path):
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for ax, key in zip(axes, ENCODERS):
        draw_confusion(ax, runs[key]['preds'], RUNS[key]['name'])
    note(fig, 'Pooled over the 10 test folds (1284 predictions). Cell color = share of the true-class row.')
    save(fig, path)


def fig_training_compare(runs, keys, path, title):
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for ax, tag, t in zip(axes, ['train/loss', 'val/loss', 'val/auc'], ['Training loss', 'Validation loss', 'Validation AUC']):
        for key in keys:
            m = curve_matrix(runs[key], tag)
            mu, sd = m.mean(0), m.std(0, ddof=1)
            ep = np.arange(len(mu))
            c = RUNS[key]['color']
            ax.fill_between(ep, mu - sd, mu + sd, color=c, alpha=0.12, lw=0)
            ax.plot(ep, mu, color=c)
            if tag == 'val/loss':
                b = int(np.argmin(mu))
                ax.plot(b, mu[b], 'o', ms=7, color=c, mec=SURFACE, mew=2, zorder=5)
        ax.set_title(t)
        ax.set_xlabel('Epoch')
        ax.margins(x=0.01)
    axes[0].set_ylim(bottom=0)
    axes[1].set_ylim(bottom=0)
    legend_below(axes[0], [Line2D([], [], color=RUNS[k]['color'], label=RUNS[k]['name']) for k in keys]
                 + [Line2D([], [], color=INK2, marker='o', lw=0, ms=7, mec=SURFACE, label='Lowest mean val loss')])
    fig.suptitle(title, x=0.01, ha='left', fontsize=11, fontweight='semibold', color=INK)
    note(fig, 'Mean ± SD over 10 folds, first 52 epochs (all folds ran at least 52).')
    save(fig, path)


def fig_paired_diff(runs, path):
    metrics = ['auc', 'acc', 'bacc', 'sens', 'spec', 'f1_tumor', 'f1_macro', 'map_macro']
    pairs = [('uni_b32', 'r50_b32'), ('conch_b32', 'r50_b32'), ('uni_b32', 'conch_b32')]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True, sharex=True)
    tcrit = stats.t.ppf(0.975, K - 1)
    for ax, (a, b) in zip(axes, pairs):
        for i, m in enumerate(metrics):
            d = runs[a]['folds'][m].values - runs[b]['folds'][m].values
            mu, ci = d.mean(), tcrit * d.std(ddof=1) / np.sqrt(K)
            p = stats.wilcoxon(runs[a]['folds'][m], runs[b]['folds'][m]).pvalue
            sig = p < 0.05
            ax.plot([mu - ci, mu + ci], [i, i], color=INK2, lw=1.5)
            ax.plot(mu, i, 'o', ms=7, color=INK if sig else SURFACE, mec=INK, mew=1.5, zorder=3)
        ax.axvline(0, color=AXIS, lw=1.2, zorder=1)
        ax.set_title(f'{RUNS[a]["name"]} − {RUNS[b]["name"]}')
        ax.set_xlabel('Paired difference')
        ax.grid(axis='y', visible=False)
    axes[0].set_yticks(range(len(metrics)), [METRIC_LABELS[m] for m in metrics])
    axes[0].invert_yaxis()
    legend_below(axes[0], [Line2D([], [], color=INK, marker='o', lw=1.5, ms=7, label='Wilcoxon p < 0.05'),
                           Line2D([], [], color=INK2, marker='o', lw=1.5, ms=7, mfc=SURFACE, mec=INK, label='Not significant')])
    note(fig, 'Mean per-fold difference with 95% confidence interval (t distribution, 10 folds).')
    save(fig, path)


def fig_val_vs_test_compare(runs, path):
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    lo, hi = 0.84, 1.005
    ax.plot([lo, hi], [lo, hi], color=AXIS, lw=1)
    ax.text(hi - 0.003, hi - 0.008, 'val = test', color=MUTED, fontsize=8, ha='right', va='top', rotation=45, rotation_mode='anchor')
    for key in ENCODERS:
        df = runs[key]['folds']
        r = np.corrcoef(df.val_auc, df.auc)[0, 1]
        ax.plot(df.val_auc, df.auc, 'o', ms=8, color=RUNS[key]['color'], mec=SURFACE, mew=2, zorder=3)
    ax.set(xlim=(lo, hi), ylim=(lo, hi), xlabel='Validation AUC', ylabel='Test AUC')
    ax.set_aspect('equal')
    ax.set_title('Validation vs. test AUC (one dot per fold)')
    legend_below(ax, [Line2D([], [], color=RUNS[k]['color'], marker='o', lw=0, ms=8,
                             label=f"{RUNS[k]['name']} (r = {np.corrcoef(runs[k]['folds'].val_auc, runs[k]['folds'].auc)[0, 1]:.2f})".replace('= -', '= −'))
                      for k in ENCODERS])
    save(fig, path)


def fig_cost(runs, path):
    # time per epoch and total time from the TensorBoard wall times are in the reports;
    # here they are recomputed from the event files.
    rows = {}
    for key in ENCODERS:
        d = os.path.join(RESULTS, RUNS[key]['dir'])
        t0, t1, epochs, per = [], [], 0, []
        for k in range(K):
            for path_ev in sorted(glob.glob(os.path.join(d, str(k), 'events.out.tfevents.*'))):
                ev = read_events(path_ev)
                if any(len(e.summary.value) for e in ev):
                    a, b = min(e.wall_time for e in ev), max(e.wall_time for e in ev)
                    n = runs[key]['folds'].epochs[k]
                    t0.append(a); t1.append(b); epochs += n
        rows[key] = dict(per_epoch=(np.array(t1) - np.array(t0)).sum() / epochs, total=(max(t1) - min(t0)) / 60)
    storage = {'r50_b32': 42, 'uni_b32': 42, 'conch_b32': 21}
    fig, axes = plt.subplots(1, 3, figsize=(12, 2.4))
    names = [RUNS[k]['name'] for k in ENCODERS]
    colors = [RUNS[k]['color'] for k in ENCODERS]
    panels = [([rows[k]['per_epoch'] for k in ENCODERS], 'Time per epoch', '{:.1f} s'),
              ([rows[k]['total'] for k in ENCODERS], 'Total training time, 10 folds', None),
              ([storage[k] for k in ENCODERS], 'Feature storage (1350 slides)', '{:.0f} GB')]
    for ax, (vals, title, fmt) in zip(axes, panels):
        y = np.arange(len(vals))
        ax.barh(y, vals, height=0.42, color=colors)
        for yi, v in zip(y, vals):
            label = f'{int(v // 60)} h {int(round(v % 60))} min' if fmt is None else fmt.format(v)
            ax.text(v + max(vals) * 0.02, yi, label, va='center', fontsize=9, color=INK)
        ax.set_yticks(y, names)
        ax.invert_yaxis()
        ax.set_xlim(0, max(vals) * 1.35)
        ax.set_xticks([])
        ax.grid(False)
        ax.spines['bottom'].set_visible(False)
        ax.set_title(title)
    save(fig, path)


def fig_b8_b32_paper(runs, path):
    metrics = ['auc', 'bacc', 'f1_macro', 'map_macro']
    values = {'paper': CLAM_PAPER}
    for key in ['r50_b8', 'r50_b32']:
        df = runs[key]['folds']
        values[key] = {m: (df[m].mean(), df[m].std(ddof=1)) for m in metrics}
    fig, ax = plt.subplots(figsize=(9, 4))
    keys = ['r50_b8', 'r50_b32', 'paper']
    grouped_bars(ax, keys, runs, metrics, values=values)
    ax.set_ylim(0.7, 1.0)
    ax.set_ylabel('Test set value (mean ± SD)')
    ax.set_title('Our ResNet-50 runs (Camelyon+) vs. the original CLAM paper')
    legend_below(ax, [Line2D([], [], color=RUNS['r50_b8']['color'], lw=8, label='Ours, B = 8'),
                      Line2D([], [], color=RUNS['r50_b32']['color'], lw=8, label='Ours, B = 32'),
                      Line2D([], [], color=REF_GRAY, lw=8, label='CLAM paper (original CAMELYON16/17 data)')])
    note(fig, 'F1 = macro F1, mAP = macro mAP. The y-axis starts at 0.7.')
    save(fig, path)


def fig_b8_b32_per_fold(runs, path):
    a, b = runs['r50_b8']['folds'].auc.values, runs['r50_b32']['folds'].auc.values
    fig, ax = plt.subplots(figsize=(9, 3.8))
    x = np.arange(K)
    for k in range(K):
        ax.plot([k - 0.12, k + 0.12], [a[k], b[k]], color=AXIS, lw=1.2, zorder=1)
    ax.plot(x - 0.12, a, 'o', ms=8, color=RUNS['r50_b8']['color'], mec=SURFACE, mew=2, zorder=3)
    ax.plot(x + 0.12, b, 'o', ms=8, color=RUNS['r50_b32']['color'], mec=SURFACE, mew=2, zorder=3)
    ax.axhline(CLAM_PAPER['auc'][0], color=REF_GRAY, lw=1.5)
    ax.text(K - 0.5, CLAM_PAPER['auc'][0] + 0.003, 'CLAM paper mean AUC 0.953', color=INK2, fontsize=8.5, ha='right', va='bottom')
    ax.set_xticks(x, [f'Fold {k}' for k in range(K)])
    ax.set_ylabel('Test AUC')
    ax.set_ylim(0.83, 0.97)
    ax.grid(axis='x', visible=False)
    ax.set_title('Test AUC per fold: B = 8 vs. B = 32 (same test sets)')
    legend_below(ax, [Line2D([], [], color=RUNS['r50_b8']['color'], marker='o', lw=0, ms=8, label='B = 8'),
                      Line2D([], [], color=RUNS['r50_b32']['color'], marker='o', lw=0, ms=8, label='B = 32')])
    save(fig, path)


def fig_b8_b32_roc(runs, path):
    fig, ax = plt.subplots(figsize=(4.8, 4.6))
    for key in ['r50_b8', 'r50_b32']:
        c = RUNS[key]['color']
        grid, m, s = mean_roc(runs[key]['preds'])
        ax.fill_between(grid, np.clip(m - s, 0, 1), np.clip(m + s, 0, 1), color=c, alpha=0.12, lw=0)
        auc = runs[key]['folds'].auc
        ax.plot(grid, m, color=c, label=f'{RUNS[key]["name"]} (AUC {auc.mean():.3f})')
    ax.plot([0, 1], [0, 1], color=AXIS, lw=1)
    ax.set(xlim=(0, 1), ylim=(0, 1.01), xlabel='False positive rate', ylabel='True positive rate')
    ax.set_aspect('equal')
    ax.set_title('Mean ROC curve, ResNet-50')
    ax.legend(loc='lower right')
    save(fig, path)


# ----------------------------------------------------------------------------- 5-fold vs. 10-fold MCCV
FIRST5 = [0, 1, 2, 3, 4]  # a 5-fold run with the same seed reproduces exactly these splits


def fig_5v10_subsets(runs, path):
    """Distribution of the 5-fold mean AUC over all C(10, 5) = 252 ways to pick 5 of the 10 folds."""
    from itertools import combinations
    subsets = list(combinations(range(K), 5))
    fig, axes = plt.subplots(3, 1, figsize=(9, 5.6), sharex=True)
    bins = np.arange(0.875, 0.980, 0.0025)
    for ax, key in zip(axes, ENCODERS):
        auc = runs[key]['folds'].auc.values
        means = np.array([auc[list(c)].mean() for c in subsets])
        ax.hist(means, bins=bins, color=RUNS[key]['color'], alpha=0.35, rwidth=0.85)
        ax.axvline(auc.mean(), color=INK, lw=1.5)
        ax.plot(auc[FIRST5].mean(), ax.get_ylim()[1] * 0.55, 'v', ms=9, color=RUNS[key]['color'], mec=INK, mew=1, zorder=4)
        ax.text(0.01, 0.82, RUNS[key]['name'], transform=ax.transAxes, fontsize=10, fontweight='semibold', color=INK)
        ax.set_yticks([])
        ax.grid(axis='y', visible=False)
        ax.spines['left'].set_visible(False)
    axes[-1].set_xlabel('Mean test AUC of 5 folds')
    axes[0].set_title('Where a 5-fold estimate can land: all 252 choices of 5 out of the 10 folds')
    legend_below(axes[-1], [Line2D([], [], color=MUTED, lw=8, alpha=0.5, label='5-fold mean, one per choice of folds'),
                            Line2D([], [], color=INK, lw=1.5, label='10-fold mean'),
                            Line2D([], [], color=MUTED, marker='v', lw=0, ms=9, mec=INK, label='Folds 0-4 (= 5-fold run with seed 1)')])
    save(fig, path)


def fig_5v10_estimates(runs, path):
    """Mean with 95% t confidence interval, 10 folds vs. folds 0-4, for the main metrics."""
    metrics = ['auc', 'bacc', 'sens', 'spec']
    fig, axes = plt.subplots(1, len(metrics), figsize=(13, 3.6))
    for ax, m in zip(axes, metrics):
        for i, key in enumerate(ENCODERS):
            c = RUNS[key]['color']
            for j, (folds, filled) in enumerate([(list(range(K)), True), (FIRST5, False)]):
                v = runs[key]['folds'][m].values[folds]
                mu = v.mean()
                ci = stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
                y = i + (j - 0.5) * 0.32
                ax.plot([mu - ci, mu + ci], [y, y], color=c, lw=2)
                ax.plot(mu, y, 'o', ms=7, color=c if filled else SURFACE, mec=c, mew=2, zorder=3)
        ax.set_yticks(range(len(ENCODERS)), [RUNS[k]['name'] for k in ENCODERS] if ax is axes[0] else [''] * len(ENCODERS))
        ax.invert_yaxis()
        ax.set_title(METRIC_LABELS[m])
        ax.grid(axis='y', visible=False)
    legend_below(axes[0], [Line2D([], [], color=INK2, marker='o', lw=2, ms=7, label='10 folds'),
                           Line2D([], [], color=INK2, marker='o', lw=2, ms=7, mfc=SURFACE, label='5 folds (folds 0-4)')])
    note(fig, 'Mean with 95% confidence interval (t distribution; 9 and 4 degrees of freedom). Upper marker of each pair: 10 folds.')
    save(fig, path)


# ----------------------------------------------------------------------------- model_size big vs. small
def _size_handles(small, big):
    return [Line2D([], [], color=RUNS[small]['color'], lw=8, label='small (attention dim 256)'),
            Line2D([], [], color=RUNS[big]['color'], lw=8, label='big (attention dim 384)')]


def fig_size_metrics(runs, path, pairs=SIZE_PAIRS, labels=('small', 'big')):
    metrics = ['auc', 'bacc', 'sens', 'spec', 'f1_tumor', 'map_macro']
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.9), sharey=True)
    x = np.arange(len(metrics))
    for ax, (small, big, title) in zip(axes, pairs):
        for j, key in enumerate([small, big]):
            df = runs[key]['folds']
            mu = np.array([df[m].mean() for m in metrics])
            sd = np.array([df[m].std(ddof=1) for m in metrics])
            pos = x + (j - 0.5) * 0.36
            ax.bar(pos, mu, width=0.34, color=RUNS[key]['color'], zorder=2)
            ax.errorbar(pos, mu, yerr=sd, fmt='none', ecolor=INK2, elinewidth=1.2, capsize=0, zorder=3)
        ax.set_xticks(x, [METRIC_LABELS[m].replace('Balanced acc.', 'bACC') for m in metrics], rotation=30, ha='right')
        ax.set_title(title)
        ax.grid(axis='x', visible=False)
    axes[0].set_ylim(0.6, 1.0)
    axes[0].set_ylabel('Test set value (mean ± SD)')
    legend_below(axes[0], [Line2D([], [], color=MUTED, lw=8, alpha=0.6, label=f'{labels[0]} (left bar of each pair, lighter)'),
                           Line2D([], [], color=INK2, lw=8, label=f'{labels[1]} (right bar, darker)')])
    note(fig, f'The y-axis starts at 0.6. Each encoder keeps its color; {labels[1]} is the darker shade.')
    save(fig, path)


def fig_size_per_fold(runs, path, pairs=SIZE_PAIRS, labels=('small', 'big')):
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    x = np.arange(K)
    for ax, (small, big, title) in zip(axes, pairs):
        a, b = runs[small]['folds'].auc.values, runs[big]['folds'].auc.values
        for k in range(K):
            ax.plot([k - 0.12, k + 0.12], [a[k], b[k]], color=AXIS, lw=1.2, zorder=1)
        ax.plot(x - 0.12, a, 'o', ms=8, color=RUNS[small]['color'], mec=SURFACE, mew=2, zorder=3)
        ax.plot(x + 0.12, b, 'o', ms=8, color=RUNS[big]['color'], mec=SURFACE, mew=2, zorder=3)
        ax.text(0.005, 0.9, f'{title}   {labels[0]} {a.mean():.4f}  vs.  {labels[1]} {b.mean():.4f}', transform=ax.transAxes,
                fontsize=9.5, fontweight='semibold', color=INK, va='top')
        ax.set_ylabel('Test AUC')
        ax.grid(axis='x', visible=False)
        lo, hi = min(a.min(), b.min()), max(a.max(), b.max())
        ax.set_ylim(lo - 0.01, hi + 0.025)
    axes[-1].set_xticks(x, [f'Fold {k}' for k in range(K)])
    axes[0].set_title(f'Test AUC per fold: {labels[0]} (left dot) vs. {labels[1]} (right dot), same test set within a fold')
    legend_below(axes[-1], [Line2D([], [], color=MUTED, marker='o', lw=0, ms=8, alpha=0.6, label=f'{labels[0]} (lighter)'),
                            Line2D([], [], color=INK2, marker='o', lw=0, ms=8, label=f'{labels[1]} (darker)')])
    save(fig, path)


def fig_size_paired_diff(runs, path, pairs=SIZE_PAIRS, labels=('small', 'big')):
    metrics = ['auc', 'acc', 'bacc', 'sens', 'spec', 'f1_tumor', 'f1_macro', 'map_macro']
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True, sharex=True)
    tcrit = stats.t.ppf(0.975, K - 1)
    for ax, (small, big, title) in zip(axes, pairs):
        for i, m in enumerate(metrics):
            d = runs[big]['folds'][m].values - runs[small]['folds'][m].values
            mu, ci = d.mean(), tcrit * d.std(ddof=1) / np.sqrt(K)
            sig = stats.wilcoxon(runs[big]['folds'][m], runs[small]['folds'][m]).pvalue < 0.05
            ax.plot([mu - ci, mu + ci], [i, i], color=INK2, lw=1.5)
            ax.plot(mu, i, 'o', ms=7, color=INK if sig else SURFACE, mec=INK, mew=1.5, zorder=3)
        ax.axvline(0, color=AXIS, lw=1.2, zorder=1)
        ax.set_title(f'{title}: {labels[1]} − {labels[0]}')
        ax.set_xlabel('Paired difference')
        ax.grid(axis='y', visible=False)
    axes[0].set_yticks(range(len(metrics)), [METRIC_LABELS[m] for m in metrics])
    axes[0].invert_yaxis()
    legend_below(axes[0], [Line2D([], [], color=INK, marker='o', lw=1.5, ms=7, label='Wilcoxon p < 0.05 (uncorrected)'),
                           Line2D([], [], color=INK2, marker='o', lw=1.5, ms=7, mfc=SURFACE, mec=INK, label='Not significant')])
    note(fig, f'Mean per-fold difference ({labels[1]} − {labels[0]}) with 95% confidence interval (t distribution, 10 folds). Negative = {labels[1]} is worse.')
    save(fig, path)


def fig_size_training(runs, path, pairs=SIZE_PAIRS, labels=('small', 'big')):
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)
    for ax, (small, big, title) in zip(axes, pairs):
        for key in [small, big]:
            m = curve_matrix(runs[key], 'val/loss')
            mu, sd = m.mean(0), m.std(0, ddof=1)
            ep = np.arange(len(mu))
            c = RUNS[key]['color']
            ax.fill_between(ep, mu - sd, mu + sd, color=c, alpha=0.12, lw=0)
            ax.plot(ep, mu, color=c)
            b = int(np.argmin(mu))
            ax.plot(b, mu[b], 'o', ms=7, color=c, mec=SURFACE, mew=2, zorder=5)
        ax.set_title(f'{title}: validation loss')
        ax.set_xlabel('Epoch')
        ax.margins(x=0.01)
    axes[0].set_ylim(bottom=0)
    legend_below(axes[0], [Line2D([], [], color=MUTED, alpha=0.6, label=f'{labels[0]} (lighter)'),
                           Line2D([], [], color=INK2, label=f'{labels[1]} (darker)'),
                           Line2D([], [], color=INK2, marker='o', lw=0, ms=7, mec=SURFACE, label='Lowest mean val loss')])
    note(fig, 'Mean ± SD over 10 folds, first 52 epochs.')
    save(fig, path)


def fig_size_subtype(runs, path, pairs=SIZE_PAIRS, labels=('small', 'big')):
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)
    x = np.arange(len(SUBTYPES))
    for ax, (small, big, title) in zip(axes, pairs):
        for j, key in enumerate([small, big]):
            rates = subtype_rates(runs[key]['preds'])
            vals = [rates[s]['pooled'] for s, _ in SUBTYPES]
            pos = x + (j - 0.5) * 0.36
            ax.bar(pos, vals, width=0.34, color=RUNS[key]['color'], zorder=2)
            for p, v in zip(pos, vals):
                ax.text(p, v + 0.015, f'{v:.0%}', ha='center', va='bottom', fontsize=8, color=INK)
        ax.set_xticks(x, [lab if s != 'itc' else 'ITC' for s, lab in SUBTYPES])
        ax.set_title(title)
        ax.grid(axis='x', visible=False)
    axes[0].set_ylim(0, 1.12)
    axes[0].set_yticks(np.linspace(0, 1, 6))
    axes[0].set_yticklabels([f'{v:.0%}' for v in np.linspace(0, 1, 6)])
    axes[0].set_ylabel('Detected as tumor (pooled)')
    legend_below(axes[0], [Line2D([], [], color=MUTED, lw=8, alpha=0.6, label=f'{labels[0]} (left, lighter)'),
                           Line2D([], [], color=INK2, lw=8, label=f'{labels[1]} (right, darker)')])
    note(fig, 'Pooled over the 10 test folds: macro n = 235, micro n = 186, ITC n = 46.')
    save(fig, path)


# ----------------------------------------------------------------------------- decision-threshold analysis
THR_DIR = os.path.join(RESULTS, 'threshold_analysis')
THR_RUNS = ['r50_b32', 'uni_b32', 'conch_b32', 'r50_big', 'uni_big', 'conch_big', 'r50_mb', 'uni_mb', 'conch_mb']


def fig_thr_curves(path):
    c = pd.read_csv(os.path.join(THR_DIR, 'curves.csv'))
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharex=True)
    for ax, (m, title) in zip(axes, [('bacc', 'Balanced accuracy'), ('sens', 'Sensitivity'), ('spec', 'Specificity')]):
        for key in ENCODERS:
            g = c[c.run == RUNS[key]['dir']]
            ax.plot(g.threshold, g[m], color=RUNS[key]['color'])
            ax.plot(0.5, g.set_index('threshold').loc[0.5, m], 'o', ms=7, color=RUNS[key]['color'], mec=SURFACE, mew=2, zorder=4)
        ax.axvline(0.5, color=AXIS, lw=1, zorder=1)
        ax.set_title(title)
        ax.set_xlabel('Decision threshold on P(tumor)')
        ax.set_xlim(0, 1)
    axes[0].set_ylim(0.5, 1.0)
    legend_below(axes[0], [Line2D([], [], color=RUNS[k]['color'], label=f'{RUNS[k]["name"]} (CLAM-SB, small)') for k in ENCODERS]
                 + [Line2D([], [], color=INK2, marker='o', lw=0, ms=7, mec=SURFACE, label='t = 0.5')])
    note(fig, 'Mean over 10 test folds for each fixed threshold. Descriptive only: choosing a threshold from these curves would use the test set.')
    save(fig, path)


def fig_thr_rule_diff(path):
    d = pd.read_csv(os.path.join(THR_DIR, 'per_fold.csv'))
    rules = [('youden', 'Youden (val)'), ('f1', 'max F1 (val)'), ('sens90', 'sens ≥ 0.90 (val)'), ('oracle_test', 'oracle (test, upper bound)')]
    fig, axes = plt.subplots(1, len(rules), figsize=(14, 4.6), sharey=True, sharex=True)
    tcrit = stats.t.ppf(0.975, K - 1)
    for ax, (rule, title) in zip(axes, rules):
        for i, key in enumerate(THR_RUNS):
            g = d[d.run == RUNS[key]['dir']]
            a = g[g.rule == rule].sort_values('fold').bacc.values
            b = g[g.rule == 'default'].sort_values('fold').bacc.values
            diff = a - b
            mu, ci = diff.mean(), tcrit * diff.std(ddof=1) / np.sqrt(K)
            sig = stats.wilcoxon(a, b).pvalue < 0.05 if np.any(diff != 0) else False
            c = RUNS[key]['color']
            ax.plot([mu - ci, mu + ci], [i, i], color=c, lw=1.8)
            ax.plot(mu, i, 'o', ms=7, color=c if sig else SURFACE, mec=c, mew=1.8, zorder=3)
        ax.axvline(0, color=AXIS, lw=1.2, zorder=1)
        ax.set_title(title)
        ax.set_xlabel('bACC change vs. t = 0.5')
        ax.grid(axis='y', visible=False)
    axes[0].set_yticks(range(len(THR_RUNS)))
    axes[0].set_yticklabels(['ResNet-50, SB small', 'UNI, SB small', 'CONCH, SB small', 'ResNet-50, SB big', 'UNI, SB big', 'CONCH, SB big',
                             'ResNet-50, MB', 'UNI, MB', 'CONCH, MB'])
    axes[0].invert_yaxis()
    legend_below(axes[0], [Line2D([], [], color=INK2, marker='o', lw=1.8, ms=7, label='Wilcoxon p < 0.05 (uncorrected)'),
                           Line2D([], [], color=INK2, marker='o', lw=1.8, ms=7, mfc=SURFACE, label='Not significant')])
    note(fig, 'Mean per-fold change in test balanced accuracy with 95% confidence interval (t distribution, 10 folds). The oracle is chosen on the test set and is not a valid estimate.')
    save(fig, path)


def fig_thr_val_vs_oracle(path):
    d = pd.read_csv(os.path.join(THR_DIR, 'per_fold.csv'))
    fig, ax = plt.subplots(figsize=(5.8, 5.4))
    ax.plot([0, 1], [0, 1], color=AXIS, lw=1)
    handles = []
    for key in ENCODERS:
        g = d[d.run == RUNS[key]['dir']]
        v = g[g.rule == 'youden'].sort_values('fold').threshold.values
        o = g[g.rule == 'oracle_test'].sort_values('fold').threshold.values
        r = np.corrcoef(v, o)[0, 1]
        ax.plot(v, o, 'o', ms=8, color=RUNS[key]['color'], mec=SURFACE, mew=2, zorder=3)
        handles.append(Line2D([], [], color=RUNS[key]['color'], marker='o', lw=0, ms=8, label=f'{RUNS[key]["name"]} (r = {r:.2f})'.replace('= -', '= −')))
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel='Threshold chosen on validation (Youden)', ylabel='Best threshold on test (oracle)')
    ax.set_aspect('equal')
    ax.set_title('Validation-chosen vs. test-optimal threshold')
    legend_below(ax, handles)
    note(fig, 'One dot per fold, CLAM-SB small. If validation picked the right threshold, dots would lie on the diagonal.')
    save(fig, path)


def fig_thr_sens90(path):
    d = pd.read_csv(os.path.join(THR_DIR, 'per_fold.csv'))
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    for key in ENCODERS:
        g = d[d.run == RUNS[key]['dir']]
        p0 = g[g.rule == 'default'][['spec', 'sens']].mean().values
        p1 = g[g.rule == 'sens90'][['spec', 'sens']].mean().values
        c = RUNS[key]['color']
        ax.annotate('', xy=(p1[0], p1[1]), xytext=(p0[0], p0[1]), arrowprops=dict(arrowstyle='-|>', color=c, lw=1.8))
        ax.plot(*p0, 'o', ms=9, color=c, mec=SURFACE, mew=2, zorder=4)
        ax.plot(*p1, 'o', ms=9, color=SURFACE, mec=c, mew=2, zorder=4)
        offset = {'r50_b32': (8, -4), 'uni_b32': (8, 6), 'conch_b32': (-12, 6)}[key]
        ax.annotate(RUNS[key]['name'], p0, xytext=offset, textcoords='offset points', fontsize=9, color=INK2,
                    ha='right' if offset[0] < 0 else 'left')
    ax.axhline(0.90, color=AXIS, lw=1)
    ax.text(0.655, 0.902, 'target sensitivity 0.90', color=MUTED, fontsize=8, va='bottom')
    ax.set(xlim=(0.65, 1.0), ylim=(0.65, 1.0), xlabel='Test specificity (mean over folds)', ylabel='Test sensitivity (mean over folds)')
    ax.set_aspect('equal')
    ax.set_title('Effect of the "sensitivity ≥ 0.90 on validation" rule')
    legend_below(ax, [Line2D([], [], color=INK2, marker='o', lw=0, ms=9, label='t = 0.5'),
                      Line2D([], [], color=INK2, marker='o', lw=0, ms=9, mfc=SURFACE, label='sens90 threshold')])
    note(fig, 'CLAM-SB small. The arrow shows the move from t = 0.5 to the threshold chosen on validation.')
    save(fig, path)


# ----------------------------------------------------------------------------- 4-class task (negative / itc / micro / macro)
C4_CLASSES = ['negative', 'itc', 'micro', 'macro']
C4_LABELS = ['Negative', 'ITC', 'Micro', 'Macro']
C4_RUNS = {  # key: (run dir, encoder key for the colour, model_size)
    'r50_4s': ('camelyon_plus_4cls_resnet50_CLAM_MB_B32_s1', 'r50_b32', 'small'),
    'uni_4s': ('camelyon_plus_4cls_uni_CLAM_MB_B32_s1', 'uni_b32', 'small'),
    'conch_4s': ('camelyon_plus_4cls_conch_CLAM_MB_B32_s1', 'conch_b32', 'small'),
    'r50_4b': ('camelyon_plus_4cls_resnet50_CLAM_MB_B32_big_s1', 'r50_big', 'big'),
    'uni_4b': ('camelyon_plus_4cls_uni_CLAM_MB_B32_big_s1', 'uni_big', 'big'),
    'conch_4b': ('camelyon_plus_4cls_conch_CLAM_MB_B32_big_s1', 'conch_big', 'big'),
}
C4_PAIRS = [('r50_4s', 'r50_4b', 'ResNet-50'), ('uni_4s', 'uni_4b', 'UNI'), ('conch_4s', 'conch_4b', 'CONCH')]
C4_ENC = ['r50_4s', 'uni_4s', 'conch_4s']


def load_run_4cls(run_dir):
    """Per-fold multi-class metrics and per-slide predictions (same definitions as collect_metrics_multiclass.py)."""
    from sklearn.metrics import cohen_kappa_score, f1_score, recall_score, roc_auc_score
    d = os.path.join(RESULTS, run_dir)
    summary = pd.read_csv(os.path.join(d, 'summary.csv'))
    folds, preds = [], []
    for k in range(K):
        res = pickle.load(open(os.path.join(d, f'split_{k}_results.pkl'), 'rb'))
        y = np.array([int(r['label']) for r in res.values()])
        P = np.vstack([r['prob'] for r in res.values()])
        pred = P.argmax(1)
        per_auc = [roc_auc_score(y == c, P[:, c]) for c in range(4)]
        assert np.isclose(np.mean(per_auc), summary.test_auc[k])
        rec = recall_score(y, pred, labels=range(4), average=None, zero_division=0)
        yb, predb = (y != 0), (pred != 0)
        m = dict(auc=np.mean(per_auc), acc=(pred == y).mean(), bacc=rec.mean(),
                 f1_macro=f1_score(y, pred, labels=range(4), average='macro', zero_division=0),
                 kappa_qw=cohen_kappa_score(y, pred, weights='quadratic'),
                 bin_auc=roc_auc_score(yb, 1 - P[:, 0]), sens=(predb & yb).sum() / yb.sum(),
                 spec=(~predb & ~yb).sum() / (~yb).sum())
        for c, name in enumerate(C4_CLASSES):
            m[f'recall_{name}'] = rec[c]
            m[f'clsauc_{name}'] = per_auc[c]
        folds.append(m)
        preds.append(pd.DataFrame(dict(fold=k, y=y, pred=pred)))
    return dict(folds=pd.DataFrame(folds), preds=pd.concat(preds, ignore_index=True))


def fig_c4_confusion(c4, path):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))
    cmap = LinearSegmentedColormap.from_list('blue', BLUE_RAMP)
    for ax, key in zip(axes, C4_ENC):
        p = c4[key]['preds']
        cm = confusion_matrix(p.y, p.pred, labels=range(4))
        row = cm / cm.sum(1, keepdims=True)
        ax.imshow(row, cmap=cmap, vmin=0, vmax=1)
        for i in range(4):
            for j in range(4):
                fill = cmap(row[i, j])
                tc = text_on('#%02x%02x%02x' % tuple(int(255 * c) for c in fill[:3]))
                ax.text(j, i - 0.12, f'{cm[i, j]}', ha='center', va='center', fontsize=11, fontweight='semibold', color=tc)
                ax.text(j, i + 0.22, f'{row[i, j]:.0%}', ha='center', va='center', fontsize=8, color=tc)
        ax.set_xticks(range(4), C4_LABELS)
        ax.set_yticks(range(4), [f'{l} ({n})' for l, n in zip(C4_LABELS, cm.sum(1))])
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.grid(False)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_title(f'{RUNS[C4_RUNS[key][1]]["name"]}, CLAM-MB small')
    note(fig, 'Pooled over the 10 test folds (1284 predictions). Cell colour and percentage: share of the true-class row.')
    save(fig, path)


def fig_c4_per_class(c4, path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    x = np.arange(4)
    for ax, (prefix, title) in zip(axes, [('recall_', 'Per-class recall (sensitivity)'), ('clsauc_', 'Per-class AUC (one-vs-rest)')]):
        for i, key in enumerate(C4_ENC):
            df = c4[key]['folds']
            mu = np.array([df[prefix + c].mean() for c in C4_CLASSES])
            sd = np.array([df[prefix + c].std(ddof=1) for c in C4_CLASSES])
            pos = x + (i - 1) * 0.26
            col = RUNS[C4_RUNS[key][1]]['color']
            ax.bar(pos, mu, width=0.24, color=col, zorder=2)
            ax.errorbar(pos, mu, yerr=sd, fmt='none', ecolor=INK2, elinewidth=1.1, capsize=0, zorder=3)
        ax.set_xticks(x, C4_LABELS)
        ax.set_ylim(0, 1.05)
        ax.set_title(title)
        ax.grid(axis='x', visible=False)
    legend_below(axes[0], [Line2D([], [], color=RUNS[C4_RUNS[k][1]]['color'], lw=8, label=RUNS[C4_RUNS[k][1]]['name']) for k in C4_ENC])
    note(fig, 'CLAM-MB small, mean ± SD over 10 test folds. ITC has only 2–9 test slides per fold, hence the large SD.')
    save(fig, path)


def fig_c4_vs_2cls(runs, c4, path):
    """Tumor vs. negative performance: 2-class CLAM-MB small vs. 4-class CLAM-MB collapsed (small, big)."""
    metrics = [('bin_auc', 'auc', 'AUC (tumor vs. negative)'), ('sens', 'sens', 'Sensitivity'), ('spec', 'spec', 'Specificity')]
    two = {'r50_4s': 'r50_mb', 'uni_4s': 'uni_mb', 'conch_4s': 'conch_mb'}
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.9), sharey=True)
    x = np.arange(3)
    for ax, (m4, m2, title) in zip(axes, metrics):
        for i, (small, big, name) in enumerate(C4_PAIRS):
            vals = [(runs[two[small]]['folds'][m2], '#c3c2b7'),
                    (c4[small]['folds'][m4], RUNS[C4_RUNS[small][1]]['color']),
                    (c4[big]['folds'][m4], RUNS[C4_RUNS[big][1]]['color'])]
            for j, (v, col) in enumerate(vals):
                pos = i + (j - 1) * 0.26
                ax.bar(pos, v.mean(), width=0.24, color=col, zorder=2)
                ax.errorbar(pos, v.mean(), yerr=v.std(ddof=1), fmt='none', ecolor=INK2, elinewidth=1.1, capsize=0, zorder=3)
        ax.set_xticks(x, [p[2] for p in C4_PAIRS])
        ax.set_title(title)
        ax.grid(axis='x', visible=False)
    axes[0].set_ylim(0.6, 1.0)
    legend_below(axes[0], [Line2D([], [], color='#c3c2b7', lw=8, label='2-class CLAM-MB small'),
                           Line2D([], [], color=MUTED, lw=8, label='4-class CLAM-MB small, collapsed (encoder colour)'),
                           Line2D([], [], color=INK2, lw=8, label='4-class CLAM-MB big, collapsed (darker)')])
    note(fig, '4-class predictions collapsed to tumor vs. negative: tumor = argmax is ITC, micro or macro; score = 1 − P(negative). Same splits for all runs. y-axis starts at 0.6.')
    save(fig, path)


def fig_c4_size_diff(c4, path):
    metrics = [('auc', 'Macro AUC'), ('acc', 'Accuracy'), ('bacc', 'Balanced acc.'), ('f1_macro', 'Macro F1'),
               ('kappa_qw', 'Quadratic kappa'), ('recall_negative', 'Recall negative'), ('recall_itc', 'Recall ITC'),
               ('recall_micro', 'Recall micro'), ('recall_macro', 'Recall macro')]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4), sharey=True, sharex=True)
    tcrit = stats.t.ppf(0.975, K - 1)
    for ax, (small, big, title) in zip(axes, C4_PAIRS):
        for i, (m, _) in enumerate(metrics):
            a, b = c4[big]['folds'][m].values, c4[small]['folds'][m].values
            d = a - b
            mu, ci = d.mean(), tcrit * d.std(ddof=1) / np.sqrt(K)
            sig = stats.wilcoxon(a, b).pvalue < 0.05 if np.any(d != 0) else False
            ax.plot([mu - ci, mu + ci], [i, i], color=INK2, lw=1.5)
            ax.plot(mu, i, 'o', ms=7, color=INK if sig else SURFACE, mec=INK, mew=1.5, zorder=3)
        ax.axvline(0, color=AXIS, lw=1.2, zorder=1)
        ax.set_title(f'{title}: big − small')
        ax.set_xlabel('Paired difference')
        ax.grid(axis='y', visible=False)
    axes[0].set_yticks(range(len(metrics)), [l for _, l in metrics])
    axes[0].invert_yaxis()
    legend_below(axes[0], [Line2D([], [], color=INK, marker='o', lw=1.5, ms=7, label='Wilcoxon p < 0.05 (uncorrected)'),
                           Line2D([], [], color=INK2, marker='o', lw=1.5, ms=7, mfc=SURFACE, mec=INK, label='Not significant')])
    note(fig, 'Mean per-fold difference (big − small) with 95% confidence interval (t distribution, 10 folds), 4-class CLAM-MB.')
    save(fig, path)


# ----------------------------------------------------------------------------- standard k-fold vs. Monte Carlo CV (ResNet-50)
KF = {  # key: (run dir, label, colour)
    'mccv10': ('camelyon_plus_2cls_resnet50_CLAM_B32_s1', 'Monte Carlo 10-fold (80/10/10)', '#2a78d6'),
    'kfold10': ('camelyon_plus_2cls_resnet50_CLAM_B32_kfold10_s1', 'k-fold 10 (80/10/10)', '#4a3aa7'),
    'kfold5': ('camelyon_plus_2cls_resnet50_CLAM_B32_kfold5_s1', 'k-fold 5 (70/10/20)', '#e87ba4'),
}


def load_kf(labels):
    out = {}
    for key, (d, _, _) in KF.items():
        RUNS['_tmp'] = dict(dir=d, name=key, color='#000000')
        n = len(pd.read_csv(os.path.join(RESULTS, d, 'summary.csv')))
        global K
        k_saved, K = K, n
        out[key] = load_run('_tmp', labels)
        K = k_saved
    del RUNS['_tmp']
    return out


def _case_bootstrap(preds, case_of, B=2000, seed=0):
    rng = np.random.default_rng(seed)
    case = preds.slide.map(case_of).values
    cases = np.unique(case)
    groups = {c: np.where(case == c)[0] for c in cases}
    y, p = preds.y.values, preds.p.values
    aucs = []
    for _ in range(B):
        sel = np.concatenate([groups[c] for c in rng.choice(cases, len(cases), replace=True)])
        aucs.append(roc_auc_score(y[sel], p[sel]))
    return np.percentile(aucs, [2.5, 97.5])


def fig_kf_per_fold(kf, path):
    metrics = ['auc', 'bacc', 'sens', 'spec']
    fig, axes = plt.subplots(1, 4, figsize=(14, 4), sharey=False)
    rng = np.random.default_rng(3)
    for ax, m in zip(axes, metrics):
        for i, key in enumerate(KF):
            v = kf[key]['folds'][m].values
            col = KF[key][2]
            ax.plot(i + rng.uniform(-0.12, 0.12, len(v)), v, 'o', ms=6, color=col, alpha=0.65, mec=SURFACE, mew=1, zorder=3)
            mu = v.mean()
            ci = stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
            ax.errorbar(i + 0.3, mu, yerr=ci, fmt='o', ms=7, color=INK, ecolor=INK, elinewidth=1.5, capsize=0, zorder=4)
        ax.set_xticks(range(3), ['MCCV 10', 'k-fold 10', 'k-fold 5'])
        ax.set_xlim(-0.5, 2.6)
        ax.set_title(METRIC_LABELS[m])
        ax.grid(axis='x', visible=False)
    legend_below(axes[0], [Line2D([], [], color=KF[k][2], marker='o', lw=0, ms=6, label=KF[k][1] + ', one dot per fold') for k in KF]
                 + [Line2D([], [], color=INK, marker='o', lw=1.5, ms=7, label='Mean with 95% CI (t)')])
    note(fig, 'ResNet-50, CLAM-SB small, B = 32. k-fold 5 has twice as many test slides per fold (about 270 vs. 135), so its per-fold values vary less.')
    save(fig, path)


def fig_kf_roc(kf, case_of, path):
    fig, ax = plt.subplots(figsize=(5.4, 5.2))
    handles = []
    for key in ['kfold10', 'kfold5']:
        p = kf[key]['preds']
        fpr, tpr, _ = roc_curve(p.y, p.p)
        auc = roc_auc_score(p.y, p.p)
        lo, hi = _case_bootstrap(p, case_of)
        ax.plot(fpr, tpr, color=KF[key][2])
        handles.append(Line2D([], [], color=KF[key][2], label=f'{KF[key][1]}: AUC {auc:.3f} [{lo:.3f}, {hi:.3f}]'))
    ax.plot([0, 1], [0, 1], color=AXIS, lw=1)
    ax.set(xlim=(0, 1), ylim=(0, 1.01), xlabel='False positive rate', ylabel='True positive rate')
    ax.set_aspect('equal')
    ax.set_title('Pooled out-of-fold ROC (each slide once)')
    legend_below(ax, handles)
    note(fig, 'All 1350 slides, one out-of-fold prediction each. Brackets: 95% CI, bootstrap over patients (2000 resamples).')
    save(fig, path)


# ----------------------------------------------------------------------------- input feature normalization (--feat_norm)
FN_GROUPS = [  # (encoder title, base key, {norm: run dir})
    ('ResNet-50', 'r50_b32', {'l2': 'camelyon_plus_2cls_resnet50_CLAM_B32_l2_s1', 'zscore': 'camelyon_plus_2cls_resnet50_CLAM_B32_zscore_s1'}),
    ('UNI', 'uni_b32', {'l2': 'camelyon_plus_2cls_uni_CLAM_B32_l2_s1', 'zscore': 'camelyon_plus_2cls_uni_CLAM_B32_zscore_s1'}),
    ('CONCH', 'conch_b32', {'zscore': 'camelyon_plus_2cls_conch_CLAM_B32_zscore_s1'}),
]
FN_COLORS = {'l2': '#c3c2b7', 'zscore': '#52514e'}
FN_LABELS = {'none': 'none (raw features)', 'l2': 'L2 (unit-length vectors)', 'zscore': 'z-score (per dimension, train-fold stats)'}
# mean L2 norm of the input vectors as seen by CLAM (raw: all patches; z-score: fold-0 stats on 20 test slides)
FN_INPUT_NORM = {('ResNet-50', 'none'): 2.26, ('ResNet-50', 'l2'): 1.0, ('ResNet-50', 'zscore'): 31.2,
                 ('UNI', 'none'): 38.7, ('UNI', 'l2'): 1.0, ('UNI', 'zscore'): 32.1,
                 ('CONCH', 'none'): 1.0, ('CONCH', 'zscore'): 21.8}


def load_fn(runs, labels):
    fn = {}
    for title, base, variants in FN_GROUPS:
        fn[(title, 'none')] = runs[base]
        for norm, d in variants.items():
            RUNS['_tmp'] = dict(dir=d, name=f'{title} {norm}', color='#000000')
            fn[(title, norm)] = load_run('_tmp', labels)
    RUNS.pop('_tmp', None)
    return fn


def fig_fn_metrics(fn, path):
    metrics = ['auc', 'bacc', 'sens', 'spec']
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.9), sharey=True)
    x = np.arange(len(metrics))
    for ax, (title, base, variants) in zip(axes, FN_GROUPS):
        norms = ['none'] + list(variants)
        w = 0.8 / 3
        for j, norm in enumerate(norms):
            df = fn[(title, norm)]['folds']
            mu = np.array([df[m].mean() for m in metrics])
            sd = np.array([df[m].std(ddof=1) for m in metrics])
            pos = x + (j - (len(norms) - 1) / 2) * (w + 0.02)
            col = RUNS[base]['color'] if norm == 'none' else FN_COLORS[norm]
            ax.bar(pos, mu, width=w, color=col, zorder=2)
            ax.errorbar(pos, mu, yerr=sd, fmt='none', ecolor=INK2, elinewidth=1.1, capsize=0, zorder=3)
        ax.set_xticks(x, [METRIC_LABELS[m] for m in metrics])
        ax.set_title(title)
        ax.grid(axis='x', visible=False)
    axes[0].set_ylim(0.6, 1.0)
    axes[0].set_ylabel('Test set value (mean ± SD)')
    legend_below(axes[0], [Line2D([], [], color=MUTED, lw=8, label=FN_LABELS['none'] + ', encoder colour'),
                           Line2D([], [], color=FN_COLORS['l2'], lw=8, label=FN_LABELS['l2']),
                           Line2D([], [], color=FN_COLORS['zscore'], lw=8, label=FN_LABELS['zscore'])])
    note(fig, 'CLAM-SB small, B = 32, same Monte Carlo splits. No L2 run for CONCH: its features already have unit length. The y-axis starts at 0.6.')
    save(fig, path)


def fig_fn_paired_diff(fn, path):
    metrics = ['auc', 'acc', 'bacc', 'sens', 'spec', 'f1_tumor', 'map_macro']
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True, sharex=True)
    tcrit = stats.t.ppf(0.975, K - 1)
    for ax, (title, base, variants) in zip(axes, FN_GROUPS):
        for i, m in enumerate(metrics):
            for norm, off, marker in [('l2', -0.15, 'o'), ('zscore', 0.15, 's')]:
                if norm not in variants:
                    continue
                a, b = fn[(title, norm)]['folds'][m].values, fn[(title, 'none')]['folds'][m].values
                d = a - b
                mu, ci = d.mean(), tcrit * d.std(ddof=1) / np.sqrt(K)
                sig = stats.wilcoxon(a, b).pvalue < 0.05 if np.any(d != 0) else False
                col = FN_COLORS[norm] if norm == 'zscore' else '#898781'
                ax.plot([mu - ci, mu + ci], [i + off, i + off], color=col, lw=1.6)
                ax.plot(mu, i + off, marker, ms=7, color=col if sig else SURFACE, mec=col, mew=1.6, zorder=3)
        ax.axvline(0, color=AXIS, lw=1.2, zorder=1)
        ax.set_title(f'{title}: normalized − raw')
        ax.set_xlabel('Paired difference')
        ax.grid(axis='y', visible=False)
    axes[0].set_yticks(range(len(metrics)), [METRIC_LABELS[m] for m in metrics])
    axes[0].invert_yaxis()
    legend_below(axes[0], [Line2D([], [], color='#898781', marker='o', lw=1.6, ms=7, label='L2 − none'),
                           Line2D([], [], color=FN_COLORS['zscore'], marker='s', lw=1.6, ms=7, label='z-score − none'),
                           Line2D([], [], color=INK2, marker='o', lw=0, ms=7, label='filled: Wilcoxon p < 0.05 (uncorrected)')])
    note(fig, 'Mean per-fold difference with 95% confidence interval (t distribution, 10 folds), shared x-axis.')
    save(fig, path)


def fig_fn_curves(fn, path):
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.6), sharex=True)
    for col_i, (title, base, variants) in enumerate(FN_GROUPS):
        for row, (tag, ylab) in enumerate([('train/loss', 'Training loss'), ('val/loss', 'Validation loss')]):
            ax = axes[row, col_i]
            for norm in ['none'] + list(variants):
                m = curve_matrix(fn[(title, norm)], tag)
                mu = m.mean(0)
                col = RUNS[base]['color'] if norm == 'none' else FN_COLORS[norm]
                ax.plot(np.arange(len(mu)), mu, color=col)
                if tag == 'val/loss':
                    b = int(np.argmin(mu))
                    ax.plot(b, mu[b], 'o', ms=7, color=col, mec=SURFACE, mew=2, zorder=5)
            ax.set_ylim(bottom=0)
            if row == 0:
                ax.set_title(title)
            if col_i == 0:
                ax.set_ylabel(ylab)
            if row == 1:
                ax.set_xlabel('Epoch')
    legend_below(axes[1, 0], [Line2D([], [], color=MUTED, label='none (encoder colour)'),
                              Line2D([], [], color=FN_COLORS['l2'], label='L2'),
                              Line2D([], [], color=FN_COLORS['zscore'], label='z-score'),
                              Line2D([], [], color=INK2, marker='o', lw=0, ms=7, mec=SURFACE, label='lowest mean val loss')])
    note(fig, 'Mean over 10 folds, first 52 epochs.')
    save(fig, path)


def fig_fn_scale(fn, path):
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    markers = {'none': 'o', 'l2': 'D', 'zscore': 's'}
    for title, base, variants in FN_GROUPS:
        pts = []
        for norm in ['none'] + list(variants):
            n = FN_INPUT_NORM[(title, norm)]
            be = fn[(title, norm)]['folds'].best_epoch.mean()
            pts.append((n, be))
            ax.plot(n, be, markers[norm], ms=9, color=RUNS[base]['color'], mec=SURFACE, mew=1.5, zorder=3)
            if norm == 'zscore':
                off = {'ResNet-50': (-30, 40), 'UNI': (-45, -28), 'CONCH': (-75, 22)}[title]
                ax.annotate(f'{title} z-score', (n, be), xytext=off, textcoords='offset points', fontsize=8, color=INK2,
                            arrowprops=dict(arrowstyle='-', color=AXIS, lw=0.8))
            else:
                ax.annotate(norm, (n, be), xytext=(7, 3), textcoords='offset points', fontsize=8, color=INK2)
        p = sorted(pts)
        ax.plot([q[0] for q in p], [q[1] for q in p], color=RUNS[base]['color'], lw=1, alpha=0.5, zorder=2)
    ax.set_xscale('log')
    ax.set_ylim(-2, 40)
    ax.set_xlabel('Mean length of the input vectors (log scale)')
    ax.set_ylabel('Best val-loss epoch (mean over folds)')
    ax.set_title('Input scale vs. how fast the model fits')
    legend_below(ax, [Line2D([], [], color=RUNS[b]['color'], marker='o', lw=0, ms=9, label=t) for t, b, _ in FN_GROUPS])
    note(fig, 'Marker shape: circle = raw, diamond = L2, square = z-score. Lines connect the runs of one encoder.')
    save(fig, path)


# ----------------------------------------------------------------------------- lower learning rate (UNI, CONCH)
LR_GROUPS = [  # (encoder title, base key at lr 2e-4, {lr: run dir})
    ('UNI', 'uni_b32', {'5e-5': 'camelyon_plus_2cls_uni_CLAM_B32_lr5e-5_s1', '2e-5': 'camelyon_plus_2cls_uni_CLAM_B32_lr2e-5_s1'}),
    ('CONCH', 'conch_b32', {'5e-5': 'camelyon_plus_2cls_conch_CLAM_B32_lr5e-5_s1', '2e-5': 'camelyon_plus_2cls_conch_CLAM_B32_lr2e-5_s1'}),
]
LR_ORDER = ['2e-4', '5e-5', '2e-5']
LR_COLORS = {'5e-5': '#c3c2b7', '2e-5': '#52514e'}  # lighter = 4x lower, darker = 10x lower; 2e-4 keeps the encoder colour


def load_lr(runs, labels):
    lr = {}
    for title, base, variants in LR_GROUPS:
        lr[(title, '2e-4')] = runs[base]
        for rate, d in variants.items():
            RUNS['_tmp'] = dict(dir=d, name=f'{title} {rate}', color='#000000')
            lr[(title, rate)] = load_run('_tmp', labels)
    RUNS.pop('_tmp', None)
    return lr


def _lr_color(base, rate):
    return RUNS[base]['color'] if rate == '2e-4' else LR_COLORS[rate]


def fig_lr_metrics(lr, path):
    metrics = ['auc', 'bacc', 'sens', 'spec']
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.9), sharey=True)
    x = np.arange(len(metrics))
    w = 0.8 / 3
    for ax, (title, base, _) in zip(axes, LR_GROUPS):
        for j, rate in enumerate(LR_ORDER):
            df = lr[(title, rate)]['folds']
            mu = np.array([df[m].mean() for m in metrics])
            sd = np.array([df[m].std(ddof=1) for m in metrics])
            pos = x + (j - 1) * (w + 0.02)
            ax.bar(pos, mu, width=w, color=_lr_color(base, rate), zorder=2)
            ax.errorbar(pos, mu, yerr=sd, fmt='none', ecolor=INK2, elinewidth=1.1, capsize=0, zorder=3)
        ax.set_xticks(x, [METRIC_LABELS[m] for m in metrics])
        ax.set_title(title)
        ax.grid(axis='x', visible=False)
    axes[0].set_ylim(0.6, 1.0)
    axes[0].set_ylabel('Test set value (mean ± SD)')
    legend_below(axes[0], [Line2D([], [], color=MUTED, lw=8, label='lr 2e-4 (earlier runs), encoder colour'),
                           Line2D([], [], color=LR_COLORS['5e-5'], lw=8, label='lr 5e-5'),
                           Line2D([], [], color=LR_COLORS['2e-5'], lw=8, label='lr 2e-5')])
    note(fig, 'CLAM-SB small, B = 32, same Monte Carlo splits. The y-axis starts at 0.6.')
    save(fig, path)


def fig_lr_paired_diff(lr, path):
    metrics = ['auc', 'acc', 'bacc', 'sens', 'spec', 'f1_tumor', 'map_macro']
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True, sharex=True)
    tcrit = stats.t.ppf(0.975, K - 1)
    for ax, (title, base, _) in zip(axes, LR_GROUPS):
        for i, m in enumerate(metrics):
            for rate, off, marker in [('5e-5', -0.15, 'o'), ('2e-5', 0.15, 's')]:
                a, b = lr[(title, rate)]['folds'][m].values, lr[(title, '2e-4')]['folds'][m].values
                d = a - b
                mu, ci = d.mean(), tcrit * d.std(ddof=1) / np.sqrt(K)
                sig = stats.wilcoxon(a, b).pvalue < 0.05 if np.any(d != 0) else False
                col = '#898781' if rate == '5e-5' else LR_COLORS['2e-5']
                ax.plot([mu - ci, mu + ci], [i + off, i + off], color=col, lw=1.6)
                ax.plot(mu, i + off, marker, ms=7, color=col if sig else SURFACE, mec=col, mew=1.6, zorder=3)
        ax.axvline(0, color=AXIS, lw=1.2, zorder=1)
        ax.set_title(f'{title}: lower lr − 2e-4')
        ax.set_xlabel('Paired difference')
        ax.grid(axis='y', visible=False)
    axes[0].set_yticks(range(len(metrics)), [METRIC_LABELS[m] for m in metrics])
    axes[0].invert_yaxis()
    legend_below(axes[0], [Line2D([], [], color='#898781', marker='o', lw=1.6, ms=7, label='5e-5 − 2e-4'),
                           Line2D([], [], color=LR_COLORS['2e-5'], marker='s', lw=1.6, ms=7, label='2e-5 − 2e-4'),
                           Line2D([], [], color=INK2, marker='o', lw=0, ms=7, label='filled: Wilcoxon p < 0.05 (uncorrected)')])
    note(fig, 'Mean per-fold difference with 95% confidence interval (t distribution, 10 folds), shared x-axis.')
    save(fig, path)


def fig_lr_curves(lr, path):
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.6), sharex=True)
    for col_i, (title, base, _) in enumerate(LR_GROUPS):
        for row, (tag, ylab) in enumerate([('train/loss', 'Training loss'), ('val/loss', 'Validation loss')]):
            ax = axes[row, col_i]
            for rate in LR_ORDER:
                mu = curve_matrix(lr[(title, rate)], tag).mean(0)
                col = _lr_color(base, rate)
                ax.plot(np.arange(len(mu)), mu, color=col)
                if tag == 'val/loss':
                    b = int(np.argmin(mu))
                    ax.plot(b, mu[b], 'o', ms=7, color=col, mec=SURFACE, mew=2, zorder=5)
            ax.set_ylim(bottom=0)
            if row == 0:
                ax.set_title(title)
            if col_i == 0:
                ax.set_ylabel(ylab)
            if row == 1:
                ax.set_xlabel('Epoch')
    legend_below(axes[1, 0], [Line2D([], [], color=MUTED, label='2e-4 (encoder colour)'),
                              Line2D([], [], color=LR_COLORS['5e-5'], label='5e-5'),
                              Line2D([], [], color=LR_COLORS['2e-5'], label='2e-5'),
                              Line2D([], [], color=INK2, marker='o', lw=0, ms=7, mec=SURFACE, label='lowest mean val loss')])
    note(fig, 'Mean over 10 folds, first 52 epochs (all folds ran at least 52; CONCH at 2e-5 ran up to 82 in some folds).')
    save(fig, path)


def fig_lr_best_epoch(lr, path):
    """Per fold: best val-loss epoch (left) and the val loss reached there (right), by learning rate."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, (field, ylab, title) in zip(axes, [('best_epoch', 'Best val-loss epoch (0-indexed)', 'When the best checkpoint is reached'),
                                               ('best_val_loss', 'Lowest validation loss', 'How good the best checkpoint is (val loss)')]):
        for g, (enc, base, _) in enumerate(LR_GROUPS):
            for j, rate in enumerate(LR_ORDER):
                run = lr[(enc, rate)]
                if field == 'best_epoch':
                    v = run['folds'].best_epoch.values.astype(float)
                else:
                    v = np.array([c['val/loss'].min() for c in run['curves']])
                pos = g * 3.6 + j
                xs = pos + np.linspace(-0.22, 0.22, len(v))  # spread the 10 folds so no dot hides another
                ax.plot(xs, v, 'o', ms=5.5, color=_lr_color(base, rate), mec=SURFACE, mew=0.8, alpha=0.9, zorder=3)
                ax.plot([pos - 0.3, pos + 0.3], [v.mean()] * 2, color=INK, lw=2, zorder=4)
        ax.set_xticks([g * 3.6 + j for g in range(2) for j in range(3)], [r for _ in LR_GROUPS for r in LR_ORDER])
        for g, (enc, base, _) in enumerate(LR_GROUPS):
            ax.text(g * 3.6 + 1, -0.16, enc, transform=ax.get_xaxis_transform(), ha='center', va='top', color=INK, fontsize=10)
        ax.set_ylim(bottom=-2 if field == 'best_epoch' else 0)  # room for the dots at epoch 0
        ax.set_ylabel(ylab)
        ax.set_title(title)
        ax.grid(axis='x', visible=False)
    legend_below(axes[0], [Line2D([], [], color=MUTED, marker='o', lw=0, ms=6, label='one dot per fold'),
                           Line2D([], [], color=INK, lw=2, label='mean over folds')])
    note(fig, 'x-axis: learning rate. Early stopping saves the checkpoint with the lowest validation loss, so these are the tested models.')
    save(fig, path)


# ----------------------------------------------------------------------------- IENF 2-class (Control vs. DM + FAP), 5-fold
IENF_RUNS = {  # key: (run dir, label, colour). Hue = patch size (yellow 224, violet 128); darker = larger B
    'ienf_224_b4': ('ienf_2cls_uni_p224_CLAM_B4_s1', '224 px, B = 4', '#f2c35b'),
    'ienf_224_b8': ('ienf_2cls_uni_p224_CLAM_B8_s1', '224 px, B = 8', '#c98700'),
    'ienf_128_b8': ('ienf_2cls_uni_p128_CLAM_B8_s1', '128 px, B = 8', '#9d93d9'),
    'ienf_128_b16': ('ienf_2cls_uni_p128_CLAM_B16_s1', '128 px, B = 16', '#4a3aa7'),
}
IENF_K = 5
IENF_GROUPS = [('Control', 'Control\n(specificity)'), ('DM', 'DM\n(sensitivity)'), ('FAP', 'FAP\n(sensitivity)')]


def load_ienf():
    global K
    sub = pd.read_csv(os.path.join(os.path.dirname(RESULTS), 'dataset_csv', 'ienf_3cls.csv')).set_index('slide_id').label.to_dict()
    out, k_saved = {}, K
    K = IENF_K
    for key, (d, name, color) in IENF_RUNS.items():
        RUNS[key] = dict(dir=d, name=name, color=color)
        out[key] = load_run(key, sub)
    K = k_saved
    return out


def _ienf_handles(marker=None):
    kw = dict(marker='o', lw=0, ms=8) if marker else dict(lw=8)
    return [Line2D([], [], color=RUNS[k]['color'], label=RUNS[k]['name'], **kw) for k in IENF_RUNS]


def fig_ienf_metrics(runs, path):
    metrics = ['auc', 'acc', 'bacc', 'sens', 'spec', 'f1_tumor', 'map_macro']
    fig, ax = plt.subplots(figsize=(12, 4.2))
    grouped_bars(ax, list(IENF_RUNS), runs, metrics)
    ax.set_xticks(np.arange(len(metrics)), [METRIC_LABELS[m].replace('tumor', 'disease') for m in metrics])
    ax.set_ylim(0.5, 1.0)
    ax.set_ylabel('Test set value (mean ± SD)')
    ax.set_title('IENF normal vs. disease: test performance (5 folds)')
    legend_below(ax, _ienf_handles())
    note(fig, 'CLAM-SB small, UNI features, patient-level 5-fold CV. Positive class = disease (DM + FAP). The y-axis starts at 0.5.')
    save(fig, path)


def fig_ienf_per_fold(runs, path):
    keys = list(IENF_RUNS)
    fig, ax = plt.subplots(figsize=(9, 4))
    offs = np.linspace(-0.27, 0.27, len(keys))
    for k in range(IENF_K):
        ax.plot(k + offs, [runs[key]['folds'].auc.values[k] for key in keys], color=AXIS, lw=1.2, zorder=1)
    for off, key in zip(offs, keys):
        ax.plot(np.arange(IENF_K) + off, runs[key]['folds'].auc.values, 'o', ms=8, color=RUNS[key]['color'],
                mec=SURFACE, mew=1.5, zorder=3)
    ax.set_xticks(np.arange(IENF_K), [f'Fold {k}' for k in range(IENF_K)])
    ax.set_ylabel('Test AUC')
    ax.grid(axis='x', visible=False)
    ax.set_title('Test AUC per fold (same test patients for all runs within a fold)')
    legend_below(ax, _ienf_handles(marker=True))
    save(fig, path)


def fig_ienf_roc_pr(runs, path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6))
    for key in IENF_RUNS:
        c, preds = RUNS[key]['color'], runs[key]['preds']
        grid, m, s = mean_roc(preds)
        axes[0].fill_between(grid, np.clip(m - s, 0, 1), np.clip(m + s, 0, 1), color=c, alpha=0.08, lw=0)
        axes[0].plot(grid, m, color=c, label=f'{RUNS[key]["name"]} (AUC {runs[key]["folds"].auc.mean():.3f})')
        grid, m, s = mean_pr(preds)
        axes[1].plot(grid, m, color=c, label=RUNS[key]['name'])
    axes[0].plot([0, 1], [0, 1], color=AXIS, lw=1)
    axes[0].set(xlim=(0, 1), ylim=(0, 1.01), xlabel='False positive rate', ylabel='True positive rate')
    axes[0].set_title('Mean ROC curve')
    axes[0].legend(loc='lower right')
    axes[1].axhline(runs['ienf_224_b4']['preds'].y.mean(), color=AXIS, lw=1)
    axes[1].set(xlim=(0, 1), ylim=(0, 1.01), xlabel='Recall', ylabel='Precision')
    axes[1].set_title('Mean precision-recall curve (disease class)')
    axes[1].legend(loc='lower left')
    for ax in axes:
        ax.set_aspect('equal')
    note(fig, 'Averaged over the 5 test folds; ROC band = ±1 SD across folds. Gray line in the PR panel: share of disease slides (0.56).')
    save(fig, path)


def fig_ienf_groups(runs, path):
    keys = list(IENF_RUNS)
    fig, ax = plt.subplots(figsize=(9, 4.2))
    width = 0.8 / len(keys)
    rng = np.random.default_rng(0)
    for i, key in enumerate(keys):
        p = runs[key]['preds']
        for g, (grp, _) in enumerate(IENF_GROUPS):
            sel = p[p['sub'] == grp]
            correct = (sel.pred == (0 if grp == 'Control' else 1))
            pos = g + (i - (len(keys) - 1) / 2) * (width + 0.01)
            ax.bar(pos, correct.mean(), width=width, color=RUNS[key]['color'], zorder=2)
            per_fold = correct.groupby(sel.fold).mean().values
            ax.plot(pos + rng.uniform(-0.04, 0.04, len(per_fold)), per_fold, 'o', ms=3.5, color=INK2, alpha=0.7, zorder=3)
    ax.set_xticks(range(len(IENF_GROUPS)), [lab for _, lab in IENF_GROUPS])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel('Share classified correctly')
    ax.set_title('Correct classification by diagnosis (pooled over 5 test folds)')
    ax.grid(axis='x', visible=False)
    legend_below(ax, _ienf_handles())
    note(fig, 'Bars: pooled share (Control n = 172, DM n = 99, FAP n = 116 slides). Dots: individual folds.')
    save(fig, path)


def fig_ienf_training(runs, path):
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for ax, tag, t in zip(axes, ['train/loss', 'val/loss', 'val/auc'], ['Training loss', 'Validation loss', 'Validation AUC']):
        for key in IENF_RUNS:
            m = curve_matrix(runs[key], tag)
            mu, c = m.mean(0), RUNS[key]['color']
            ax.plot(np.arange(len(mu)), mu, color=c, lw=1.6)
            if tag == 'val/loss':
                b = int(np.argmin(mu))
                ax.plot(b, mu[b], 'o', ms=7, color=c, mec=SURFACE, mew=2, zorder=5)
        ax.set_title(t)
        ax.set_xlabel('Epoch')
        ax.margins(x=0.01)
    axes[0].set_ylim(bottom=0)
    axes[1].set_ylim(bottom=0)
    legend_below(axes[0], [Line2D([], [], color=RUNS[k]['color'], label=RUNS[k]['name']) for k in IENF_RUNS]
                 + [Line2D([], [], color=INK2, marker='o', lw=0, ms=7, mec=SURFACE, label='Lowest mean val loss')])
    note(fig, 'Mean over 5 folds, first 52 epochs (every fold ran exactly 52).')
    save(fig, path)


IENF_HEATMAPS = os.path.join(os.path.dirname(RESULTS), 'heatmaps', 'ienf')


def fig_ienf_heatmap_overlap(path, slides=('FP_S425-2_b', 'TN_S1616-2_b')):
    """Same slide, same model: heatmap with 50% overlap vs. 95% overlap + blur (heatmaps/ienf_heatmaps.py output)."""
    variants = [('ienf_2cls_uni_p128_CLAM_B8_s1', '50% overlap (step 64 px), no blur'),
                ('ienf_2cls_uni_p128_CLAM_B8_s1_overlap95_blur', '95% overlap (step 6 px) + Gaussian blur')]
    files = [[os.path.join(IENF_HEATMAPS, d, 'slides', f'{s}_heatmap_roi.jpg') for d, _ in variants] for s in slides]
    if not all(os.path.isfile(f) for row in files for f in row):
        print('skipped', os.path.basename(path), '(heatmaps not found)')
        return
    fig, axes = plt.subplots(len(slides) * 2, 1, figsize=(12, 2.6 * len(slides) * 2))
    for i, s in enumerate(slides):
        for j, (_, label) in enumerate(variants):
            ax = axes[2 * i + j]
            ax.imshow(Image.open(files[i][j]))
            ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
            for sp in ax.spines.values():
                sp.set_visible(False)
            ax.set_title(f'{s.replace("_", " ", 1)}: {label}', fontsize=10)
    note(fig, 'CLAM-SB, 128 px, B = 8. Colors: attention percentile within the slide (jet; blue = low, red = high).')
    save(fig, path)


def ienf_figures(shared):
    fig_ienf_heatmap_overlap(os.path.join(shared, 'ienf_heatmap_overlap.png'))
    runs = load_ienf()
    fig_ienf_metrics(runs, os.path.join(shared, 'ienf_metrics.png'))
    fig_ienf_per_fold(runs, os.path.join(shared, 'ienf_per_fold_auc.png'))
    fig_ienf_roc_pr(runs, os.path.join(shared, 'ienf_roc_pr.png'))
    fig_ienf_groups(runs, os.path.join(shared, 'ienf_by_diagnosis.png'))
    fig_ienf_training(runs, os.path.join(shared, 'ienf_training_curves.png'))


# ----------------------------------------------------------------------------- IENF 2-class with iSFN (Control vs. DM + FAP + iSFN)
# New preprocessing (intensity normalization, mask outside ROI + 20 px, >= 10% rule), rebalanced folds.
# Hue = patch size (128 keeps the violet of the earlier 128 px runs); the earlier run is a gray reference.
ISFN_RUNS = {
    'isfn_128': ('ienf_isfn_2cls_uni_p128_CLAM_B8_s1', '128 px, B = 8', '#4a3aa7'),
    'isfn_64': ('ienf_isfn_2cls_uni_p64_CLAM_B16_s1', '64 px, B = 16', '#e87ba4'),
    'isfn_32': ('ienf_isfn_2cls_uni_p32_CLAM_B32_s1', '32 px, B = 32', '#008300'),
    'isfn_16': ('ienf_isfn_2cls_uni_p16_CLAM_B64_s1', '16 px, B = 64', '#e34948'),
}
ISFN_OLD = ('isfn_old', 'ienf_2cls_uni_p128_CLAM_B8_s1', 'earlier 128 px, B = 8 (old pipeline, no iSFN)', REF_GRAY)
ISFN_GROUPS = [('Control', 'batch1'), ('Control', 'batch2'), ('DM', 'batch1'), ('DM', 'batch2'),
               ('FAP', 'batch1'), ('FAP', 'batch2'), ('iSFN', 'batch2')]


def load_isfn():
    global K
    meta = pd.read_csv(os.path.join(os.path.dirname(RESULTS), 'dataset_csv', 'ienf_isfn_4sub.csv')).set_index('slide_id')
    pat = pd.read_csv(os.path.join(os.path.dirname(RESULTS), 'splits', 'ienf_isfn_kfold5_rebalanced', 'patients.csv')).set_index('case_id')
    new_fold = meta.case_id.map(pat.fold)
    out, k_saved = {}, K
    K = IENF_K
    keys = [(k, *v) for k, v in ISFN_RUNS.items()] + [ISFN_OLD]
    for key, d, name, color in keys:
        RUNS[key] = dict(dir=d, name=name, color=color)
        r = load_run(key, meta.label.to_dict())
        r['preds']['batch'] = r['preds'].slide.map(meta.batch)
        r['preds']['new_fold'] = r['preds'].slide.map(new_fold)
        out[key] = r
    K = k_saved
    return out


def _isfn_handles(marker=None, old=True):
    kw = dict(marker='o', lw=0, ms=8) if marker else dict(lw=8)
    keys = list(ISFN_RUNS) + ([ISFN_OLD[0]] if old else [])
    return [Line2D([], [], color=RUNS[k]['color'], label=RUNS[k]['name'], **kw) for k in keys]


def fig_isfn_metrics(runs, path):
    metrics = ['auc', 'acc', 'bacc', 'sens', 'spec', 'f1_tumor', 'map_macro']
    fig, ax = plt.subplots(figsize=(12.5, 4.2))
    grouped_bars(ax, list(ISFN_RUNS) + [ISFN_OLD[0]], runs, metrics, width=0.15)
    ax.set_xticks(np.arange(len(metrics)), [METRIC_LABELS[m].replace('tumor', 'disease') for m in metrics])
    ax.set_ylim(0.5, 1.0)
    ax.set_ylabel('Test set value (mean ± SD)')
    ax.set_title('IENF normal vs. disease with iSFN: test performance (5 folds)')
    legend_below(ax, _isfn_handles())
    note(fig, 'New runs: 433 slides, rebalanced folds. Gray: the earlier run (387 slides without iSFN, earlier folds), '
              'shown as context, not a paired comparison. The y-axis starts at 0.5.')
    save(fig, path)


def fig_isfn_per_fold(runs, path):
    keys = list(ISFN_RUNS)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2), sharey=True)
    offs = np.linspace(-0.27, 0.27, len(keys) + 1)
    for ax, subset in zip(axes, ['all', 'no_isfn']):
        vals = {}
        for key in keys:
            p = runs[key]['preds']
            if subset == 'no_isfn':
                p = p[p['sub'] != 'iSFN']
            vals[key] = [roc_auc_score(g.y, g.p) for _, g in p.groupby('fold')]
        if subset == 'no_isfn':  # earlier predictions of the same slides, grouped by the NEW test folds
            p = runs[ISFN_OLD[0]]['preds']
            vals[ISFN_OLD[0]] = [roc_auc_score(g.y, g.p) for _, g in p.groupby('new_fold')]
        ks = list(vals)
        for k in range(IENF_K):
            ax.plot(k + offs[:len(ks)], [vals[key][k] for key in ks], color=AXIS, lw=1.2, zorder=1)
        for off, key in zip(offs, ks):
            ax.plot(np.arange(IENF_K) + off, vals[key], 'o', ms=8, color=RUNS[key]['color'], mec=SURFACE, mew=1.5, zorder=3)
        ax.set_xticks(np.arange(IENF_K), [f'Fold {k}' for k in range(IENF_K)])
        ax.grid(axis='x', visible=False)
    axes[0].set_ylabel('Test AUC')
    axes[0].set_title('All test slides (with iSFN)')
    axes[1].set_title('Control / DM / FAP test slides only')
    legend_below(axes[0], _isfn_handles(marker=True))
    note(fig, 'Right panel, gray: out-of-fold predictions of the earlier run for the same slides, grouped by the new test folds '
              '(those predictions were used to build the folds).')
    save(fig, path)


def fig_isfn_roc_pr(runs, path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6))
    for key in ISFN_RUNS:
        c, preds = RUNS[key]['color'], runs[key]['preds']
        grid, m, s = mean_roc(preds)
        axes[0].fill_between(grid, np.clip(m - s, 0, 1), np.clip(m + s, 0, 1), color=c, alpha=0.07, lw=0)
        axes[0].plot(grid, m, color=c, label=f'{RUNS[key]["name"]} (AUC {runs[key]["folds"].auc.mean():.3f})')
        grid, m, s = mean_pr(preds)
        axes[1].plot(grid, m, color=c, label=RUNS[key]['name'])
    axes[0].plot([0, 1], [0, 1], color=AXIS, lw=1)
    axes[0].set(xlim=(0, 1), ylim=(0, 1.01), xlabel='False positive rate', ylabel='True positive rate')
    axes[0].set_title('Mean ROC curve')
    axes[0].legend(loc='lower right')
    prior = runs['isfn_128']['preds'].y.mean()
    axes[1].axhline(prior, color=AXIS, lw=1)
    axes[1].set(xlim=(0, 1), ylim=(0, 1.01), xlabel='Recall', ylabel='Precision')
    axes[1].set_title('Mean precision-recall curve (disease class)')
    axes[1].legend(loc='lower left')
    for ax in axes:
        ax.set_aspect('equal')
    note(fig, f'Averaged over the 5 test folds; ROC band = ±1 SD across folds. Gray line in the PR panel: share of disease slides ({prior:.2f}).')
    save(fig, path)


def fig_isfn_groups(runs, path):
    keys = list(ISFN_RUNS) + [ISFN_OLD[0]]
    fig, ax = plt.subplots(figsize=(13, 4.4))
    width = 0.8 / len(keys)
    labels = []
    for g, (sub, batch) in enumerate(ISFN_GROUPS):
        n = None
        for i, key in enumerate(keys):
            p = runs[key]['preds']
            sel = p[(p['sub'] == sub) & (p.batch == batch)]
            if not len(sel):
                continue
            n = len(sel)
            correct = (sel.pred == (0 if sub == 'Control' else 1)).mean()
            ax.bar(g + (i - (len(keys) - 1) / 2) * (width + 0.01), correct, width=width, color=RUNS[key]['color'], zorder=2)
        kind = 'specificity' if sub == 'Control' else 'sensitivity'
        labels.append(f'{sub}, {batch}\n({kind}, n = {n})')
    ax.set_xticks(range(len(ISFN_GROUPS)), labels, fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel('Share classified correctly')
    ax.set_title('Correct classification by diagnosis and image batch (pooled over 5 test folds)')
    ax.grid(axis='x', visible=False)
    legend_below(ax, _isfn_handles())
    note(fig, 'batch1 and batch2 = the two image deliveries (first and second batch). All iSFN slides are in batch2, so the gray reference has no iSFN bar.')
    save(fig, path)


def fig_isfn_training(runs, path):
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for ax, tag, t in zip(axes, ['train/loss', 'val/loss', 'val/auc'], ['Training loss', 'Validation loss', 'Validation AUC']):
        for key in ISFN_RUNS:
            m = curve_matrix(runs[key], tag)
            mu, c = m.mean(0), RUNS[key]['color']
            ax.plot(np.arange(len(mu)), mu, color=c, lw=1.6)
            if tag == 'val/loss':
                b = int(np.argmin(mu))
                ax.plot(b, mu[b], 'o', ms=7, color=c, mec=SURFACE, mew=2, zorder=5)
        ax.set_title(t)
        ax.set_xlabel('Epoch')
        ax.margins(x=0.01)
    axes[0].set_ylim(bottom=0)
    axes[1].set_ylim(bottom=0)
    legend_below(axes[0], [Line2D([], [], color=RUNS[k]['color'], label=RUNS[k]['name']) for k in ISFN_RUNS]
                 + [Line2D([], [], color=INK2, marker='o', lw=0, ms=7, mec=SURFACE, label='Lowest mean val loss')])
    note(fig, 'Mean over 5 folds, first 52 epochs (every fold ran at least 52; a few ran 54–65).')
    save(fig, path)


def isfn_figures(shared):
    runs = load_isfn()
    fig_isfn_metrics(runs, os.path.join(shared, 'ienf_isfn_metrics.png'))
    fig_isfn_per_fold(runs, os.path.join(shared, 'ienf_isfn_per_fold_auc.png'))
    fig_isfn_roc_pr(runs, os.path.join(shared, 'ienf_isfn_roc_pr.png'))
    fig_isfn_groups(runs, os.path.join(shared, 'ienf_isfn_by_subtype_batch.png'))
    fig_isfn_training(runs, os.path.join(shared, 'ienf_isfn_training_curves.png'))


# ----------------------------------------------------------------------------- IENF ablation ladder (128 px, B = 8)
AB_RUNGS = {  # rung: (exp_code without seed, short label)
    'A': ('ienf_ab_A_oldprep_noisfn_p128_CLAM_B8', 'A  earlier prep,\nno iSFN'),
    'B': ('ienf_ab_B_oldprep_p128_CLAM_B8', 'B  + iSFN'),
    'C': ('ienf_ab_C_buf20_p128_CLAM_B8', 'C  + buffer 20 px,\n≥ 10% ROI'),
    'D': ('ienf_ab_D_mask_p128_CLAM_B8', 'D  + mask'),
    'E': ('ienf_isfn_2cls_uni_p128_CLAM_B8', 'E  + normalization\n(new pipeline)'),
}
AB_SEEDS = [1, 2, 3]
AB_MEAN = '#256abf'


def load_ablation():
    """Per-slide test predictions of every rung and seed, with subtype, batch and the normalization QC flag."""
    root = os.path.dirname(RESULTS)
    meta = pd.read_csv(os.path.join(root, 'dataset_csv', 'ienf_isfn_4sub.csv')).set_index('slide_id')
    qc = pd.read_csv('/var/tmp/shouteng_data/IENF_CLAM/images_norm_bufdown20/normalize_qc.csv').set_index('slide_id')
    rows = []
    for rung, (exp, _) in AB_RUNGS.items():
        for seed in AB_SEEDS:
            for k in range(5):
                with open(os.path.join(RESULTS, f'{exp}_s{seed}', f'split_{k}_results.pkl'), 'rb') as f:
                    for s, v in pickle.load(f).items():
                        rows.append(dict(rung=rung, seed=seed, fold=k, slide=s, y=int(v['label']), p=float(v['prob'][0][1])))
    p = pd.DataFrame(rows).join(meta[['label', 'batch']], on='slide').join(qc[['within_applied']], on='slide')
    p['ok'] = (p.p >= 0.5).astype(int) == p.y
    return p


def _ab_fold_metrics(p):
    """Fold-level metrics on the 387 Control/DM/FAP slides (shared by all rungs), per rung, seed and fold."""
    rows = []
    for (rung, seed, k), g in p[p.label != 'iSFN'].groupby(['rung', 'seed', 'fold']):
        c = g[g.label == 'Control']
        sens, spec = g[g.y == 1].ok.mean(), c.ok.mean()
        rows.append(dict(rung=rung, seed=seed, fold=k, auc=roc_auc_score(g.y, g.p), sens=sens, spec=spec, bacc=(sens + spec) / 2,
                         spec_b1=c[c.batch == 'batch1'].ok.mean(), spec_b2=c[c.batch == 'batch2'].ok.mean()))
    return pd.DataFrame(rows)


AB_SHORT = {'A': 'A\nearlier prep,\nno iSFN', 'B': 'B\n+ iSFN', 'C': 'C\n+ buffer 20,\n≥ 10% ROI', 'D': 'D\n+ mask', 'E': 'E\n+ normal-\nization'}


def _ab_ladder_panel(ax, per_seed, title, ylim=None, short=False):
    """Seed dots just left of each rung, the seed mean on the rung with its value to the right."""
    x = np.arange(len(AB_RUNGS))
    for i, rung in enumerate(AB_RUNGS):
        v = per_seed[per_seed.rung == rung].value.values
        ax.scatter(np.full(len(v), i - 0.16), v, s=16, color=MUTED, zorder=3)
    means = per_seed.groupby('rung').value.mean().reindex(list(AB_RUNGS))
    ax.plot(x, means.values, color=AB_MEAN, marker='o', ms=6, lw=1.6, zorder=4)
    for i, m in enumerate(means.values):
        if np.isfinite(m):
            ax.annotate(f'{m:.3f}', (i, m), textcoords='offset points', xytext=(7, -3), ha='left', va='top',
                        fontsize=8, color=AB_MEAN)
    ax.set_xticks(x, [AB_SHORT[r] if short else AB_RUNGS[r][1] for r in AB_RUNGS], fontsize=8)
    ax.set_xlim(-0.45, len(AB_RUNGS) - 0.4)
    ax.set_title(title)
    ax.grid(axis='x', visible=False)
    if ylim:
        ax.set_ylim(*ylim)


def _ab_handles():
    return [Line2D([], [], color=AB_MEAN, marker='o', label='mean of 3 seeds'),
            Line2D([], [], color=MUTED, marker='o', ls='', label='one seed (mean over 5 folds)')]


def fig_ablation_ladder(fm, path):
    seed_means = fm.groupby(['rung', 'seed'])[['auc', 'spec', 'sens', 'bacc']].mean().reset_index()
    fig, axes = plt.subplots(2, 2, figsize=(13, 7.6))
    for ax, (m, title) in zip(axes.flat, [('auc', 'AUC'), ('spec', 'Specificity (Control)'),
                                          ('sens', 'Sensitivity (DM + FAP)'), ('bacc', 'Balanced accuracy')]):
        _ab_ladder_panel(ax, seed_means[['rung', m]].rename(columns={m: 'value'}), title)
        lo, hi = seed_means[m].min(), seed_means[m].max()
        ax.set_ylim(lo - 0.03, hi + 0.035)
    legend_below(axes[1, 0], _ab_handles())
    note(fig, 'All rungs: 128 px, B = 8, rebalanced folds, Control/DM/FAP test slides only (387; iSFN excluded so that A is comparable). '
              'Each rung adds one change to the one before.')
    save(fig, path)


def fig_ablation_batches(p, fm, path):
    seed_means = fm.groupby(['rung', 'seed'])[['spec_b1', 'spec_b2']].mean().reset_index()
    isfn = p[p.label == 'iSFN'].groupby(['rung', 'seed']).ok.mean().reset_index().rename(columns={'ok': 'value'})
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    _ab_ladder_panel(axes[0], seed_means[['rung', 'spec_b1']].rename(columns={'spec_b1': 'value'}), 'Specificity, Control batch1 (116 slides)', (0.55, 0.95), short=True)
    _ab_ladder_panel(axes[1], seed_means[['rung', 'spec_b2']].rename(columns={'spec_b2': 'value'}), 'Specificity, Control batch2 (56 slides)', (0.55, 0.95), short=True)
    full = pd.concat([pd.DataFrame(dict(rung=['A'], seed=[0], value=[np.nan])), isfn])
    _ab_ladder_panel(axes[2], full, 'Sensitivity, iSFN (46 slides; not in A)', (0.55, 1.0), short=True)
    legend_below(axes[0], _ab_handles())
    note(fig, 'Batch specificities are means of fold-level values (fold sizes are small: about 11 batch2 Controls per fold).')
    save(fig, path)


def fig_ablation_steps(fm, path):
    """Per-fold change from the previous rung (mean over the 3 seeds)."""
    f = fm.groupby(['rung', 'fold'])[['auc', 'spec', 'sens']].mean()
    order = list(AB_RUNGS)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharey=False)
    for ax, (m, title) in zip(axes, [('auc', 'Δ AUC'), ('spec', 'Δ specificity'), ('sens', 'Δ sensitivity')]):
        for i, (a, b) in enumerate(zip(order, order[1:])):
            d = (f.loc[b, m] - f.loc[a, m]).values
            ax.scatter(np.full(len(d), i) + np.linspace(-0.12, 0.12, len(d)), d, s=22, color=MUTED, zorder=3)
            ax.plot([i - 0.22, i + 0.22], [d.mean()] * 2, color=AB_MEAN, lw=2.2, zorder=4)
        ax.axhline(0, color=AXIS, lw=1, zorder=1)
        ax.set_xticks(range(len(order) - 1), [f'{a} → {b}' for a, b in zip(order, order[1:])])
        ax.set_title(title)
        ax.grid(axis='x', visible=False)
    legend_below(axes[0], [Line2D([], [], color=AB_MEAN, lw=2.2, label='mean over folds'),
                           Line2D([], [], color=MUTED, marker='o', ls='', label='one test fold (mean of 3 seeds)')])
    note(fig, 'A → B: + iSFN;  B → C: buffer 20 px, ≥ 10% ROI;  C → D: mask outside ROI + 20 px;  D → E: intensity normalization.')
    save(fig, path)


def fig_ablation_norm_controls(p, path):
    """Seed-averaged P(disease) of every Control slide without (D) and with (E) normalization."""
    w = p[(p.label == 'Control') & p.rung.isin(['D', 'E'])].groupby(['slide', 'rung']).p.mean().unstack()
    info = p[p.label == 'Control'].drop_duplicates('slide').set_index('slide')[['batch', 'within_applied']]
    w = w.join(info)
    fig, ax = plt.subplots(figsize=(6.4, 6.0))
    styles = {(False, 'batch1'): ('#9ec5f4', 'o'), (False, 'batch2'): ('#9ec5f4', 's'),
              (True, 'batch1'): ('#e34948', 'o'), (True, 'batch2'): ('#e34948', 's')}
    for (wa, b), (col, mk) in styles.items():
        s = w[(w.within_applied == wa) & (w.batch == b)]
        ax.scatter(s.D, s.E, s=26 if wa else 16, color=col, marker=mk, edgecolor=INK if wa else 'none', lw=0.5, zorder=3 if wa else 2)
    ax.plot([0, 1], [0, 1], color=AXIS, lw=1, zorder=1)
    ax.axhline(0.5, color=AXIS, lw=0.8, ls='--'); ax.axvline(0.5, color=AXIS, lw=0.8, ls='--')
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel('P(disease), D: mask only')
    ax.set_ylabel('P(disease), E: mask + normalization')
    ax.set_title('Control slides: effect of normalization')
    legend_below(ax, [Line2D([], [], color='#9ec5f4', marker='o', ls='', label='no within-image correction'),
                      Line2D([], [], color='#e34948', marker='o', ls='', mec=INK, label='within-image correction applied'),
                      Line2D([], [], color=MUTED, marker='s', ls='', label='batch2 (circles: batch1)')])
    note(fig, 'Each dot: one Control slide, P averaged over 3 seeds. Above the diagonal = looks more diseased after normalization;\n'
              'upper-left quadrant = correct without normalization, false positive with it.')
    save(fig, path)


AB_SIZES = {  # patch size: {rung: exp_code without seed}
    128: {'D': 'ienf_ab_D_mask_p128_CLAM_B8', 'E': 'ienf_isfn_2cls_uni_p128_CLAM_B8'},
    32: {'D': 'ienf_ab_D_mask_p32_CLAM_B32', 'E': 'ienf_isfn_2cls_uni_p32_CLAM_B32'},
    16: {'D': 'ienf_ab_D_mask_p16_CLAM_B64', 'E': 'ienf_isfn_2cls_uni_p16_CLAM_B64'},
}
AB_DE_COLORS = {'D': '#008300', 'E': '#4a3aa7'}
AB_DE_LABELS = {'D': 'D: mask only', 'E': 'E: mask + normalization'}


def fig_ablation_patch_sizes(path):
    """D vs. E at 128 / 32 / 16 px on all 433 slides: seed dots and seed means of the 5-fold means."""
    meta = pd.read_csv(os.path.join(os.path.dirname(RESULTS), 'dataset_csv', 'ienf_isfn_4sub.csv')).set_index('slide_id')
    rows = []
    for ps, rungs in AB_SIZES.items():
        for rung, exp in rungs.items():
            for seed in AB_SEEDS:
                folds, isfn_ok = [], []
                for k in range(5):
                    with open(os.path.join(RESULTS, f'{exp}_s{seed}', f'split_{k}_results.pkl'), 'rb') as f:
                        d = pd.DataFrame([dict(slide=s, y=int(v['label']), p=float(v['prob'][0][1])) for s, v in pickle.load(f).items()])
                    d = d.join(meta[['label']], on='slide')
                    d['ok'] = (d.p >= 0.5).astype(int) == d.y
                    folds.append(dict(auc=roc_auc_score(d.y, d.p), spec=d[d.label == 'Control'].ok.mean(),
                                      sens=d[d.y == 1].ok.mean()))
                    isfn_ok += d[d.label == 'iSFN'].ok.tolist()
                # iSFN: pooled share of the 46 slides (fold sizes 8-10 are too small for fold means)
                rows.append(dict(ps=ps, rung=rung, seed=seed, **pd.DataFrame(folds).mean(), isfn=np.mean(isfn_ok)))
    r = pd.DataFrame(rows)
    sizes = list(AB_SIZES)
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.3))
    for ax, (m, title) in zip(axes, [('auc', 'AUC'), ('spec', 'Specificity (Control)'),
                                     ('sens', 'Sensitivity (DM + FAP + iSFN)'), ('isfn', 'Sensitivity, iSFN (pooled, 46 slides)')]):
        for j, rung in enumerate(['D', 'E']):
            off = (j - 0.5) * 0.22
            g = r[r.rung == rung]
            for i, ps in enumerate(sizes):
                v = g[g.ps == ps][m].values
                ax.scatter(np.full(len(v), i + off), v, s=14, color=AB_DE_COLORS[rung], alpha=0.45, zorder=3, lw=0)
            means = [g[g.ps == ps][m].mean() for ps in sizes]
            ax.plot(np.arange(len(sizes)) + off, means, color=AB_DE_COLORS[rung], marker='o', ms=6, lw=1.6, zorder=4)
            for i, mv in enumerate(means):
                ax.annotate(f'{mv:.3f}', (i + off, mv), textcoords='offset points', xytext=(0, 7 if rung == 'D' else -14),
                            ha='center', fontsize=7.5, color=AB_DE_COLORS[rung])
        ax.set_xticks(range(len(sizes)), [f'{ps} px' for ps in sizes])
        ax.set_xlim(-0.5, len(sizes) - 0.5)
        lo, hi = r[m].min(), r[m].max()
        ax.set_ylim(lo - 0.03, hi + 0.03)
        ax.set_title(title)
        ax.grid(axis='x', visible=False)
    legend_below(axes[0], [Line2D([], [], color=AB_DE_COLORS[k], marker='o', label=AB_DE_LABELS[k] + ' (mean of 3 seeds)') for k in 'DE']
                 + [Line2D([], [], color=MUTED, marker='o', ls='', alpha=0.6, label='one seed (5-fold mean)')])
    note(fig, 'All 433 test slides, rebalanced folds, buffer 20 px, ≥ 10% ROI + buffer. B = 8 / 32 / 64 for 128 / 32 / 16 px.')
    save(fig, path)


def ablation_figures(shared):
    fig_ablation_patch_sizes(os.path.join(shared, 'ienf_ablation_patch_sizes.png'))
    p = load_ablation()
    fm = _ab_fold_metrics(p)
    fig_ablation_ladder(fm, os.path.join(shared, 'ienf_ablation_ladder.png'))
    fig_ablation_batches(p, fm, os.path.join(shared, 'ienf_ablation_batches.png'))
    fig_ablation_steps(fm, os.path.join(shared, 'ienf_ablation_steps.png'))
    fig_ablation_norm_controls(p, os.path.join(shared, 'ienf_ablation_norm_controls.png'))


# ----------------------------------------------------------------------------- IENF normalization variants, random folds
RF_LADDER = {'A': 'ienf_rf_A_oldprep_noisfn_p128_CLAM_B8', 'B': 'ienf_rf_B_oldprep_p128_CLAM_B8',
             'C': 'ienf_rf_C_buf20_p128_CLAM_B8', 'D': 'ienf_rf_D_mask_p128_CLAM_B8', 'E': 'ienf_rf_E_full_p128_CLAM_B8'}
RF_VARIANTS = {  # variant: (folder name part, short label)
    'D': ('mask', 'D\nnone'), 'E': ('full', 'E\nwithin +\nbetween'), 'W': ('within', 'W\nwithin\nonly'),
    'G': ('between', 'G\nbetween\nonly'), 'Gk': ('betweenkeepG', 'Gk\nbetween,\nG kept'),
    'Bt': ('batchgain', 'Bt\nbatch\ngain'), 'F': ('featalign', 'F\nfeature\nalign'),
}
RF_SIZES = {128: (8, '#256abf'), 32: (32, '#eda100'), 16: (64, '#e34948')}  # patch size: (B, color)


def _rf_exp(v, ps):
    return f'ienf_rf_{v}_{RF_VARIANTS[v][0]}_p{ps}_CLAM_B{RF_SIZES[ps][0]}'


def _rf_seed_metrics(exp, meta, isfn=True):
    """Per seed: 5-fold means of AUC / sens / spec (all test slides, or Control/DM/FAP when isfn=False),
    plus the pooled iSFN sensitivity. None if a seed is missing."""
    rows = []
    for seed in AB_SEEDS:
        d = os.path.join(RESULTS, f'{exp}_s{seed}')
        if not os.path.exists(os.path.join(d, 'summary.csv')):
            return None
        folds, isfn_ok = [], []
        for k in range(5):
            with open(os.path.join(d, f'split_{k}_results.pkl'), 'rb') as f:
                g = pd.DataFrame([dict(slide=s, y=int(v['label']), p=float(v['prob'][0][1])) for s, v in pickle.load(f).items()])
            g = g.join(meta[['label']], on='slide')
            g['ok'] = (g.p >= 0.5).astype(int) == g.y
            isfn_ok += g[g.label == 'iSFN'].ok.tolist()
            if not isfn:
                g = g[g.label != 'iSFN']
            folds.append(dict(fold=k, auc=roc_auc_score(g.y, g.p), spec=g[g.label == 'Control'].ok.mean(),
                              sens=g[g.y == 1].ok.mean()))
        f = pd.DataFrame(folds)
        rows.append(dict(seed=seed, **f.drop(columns='fold').mean(), isfn=np.mean(isfn_ok) if isfn_ok else np.nan,
                         folds=f.set_index('fold')))
    return rows


def fig_rf_ladder(meta, path):
    """Ladder A-E on the random folds next to the same ladder on the rebalanced folds (387 Control/DM/FAP slides)."""
    reb = dict(AB_RUNGS)
    series = [('random folds', RF_LADDER, AB_MEAN, 0.1), ('rebalanced folds (earlier report)', {r: v[0] for r, v in reb.items()}, MUTED, -0.1)]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    for name, runs, col, off in series:
        per = {r: _rf_seed_metrics(exp, meta, isfn=False) for r, exp in runs.items()}
        for ax, m in zip(axes, ['auc', 'spec', 'sens']):
            means = []
            for i, r in enumerate(runs):
                v = [s[m] for s in per[r]]
                ax.scatter(np.full(len(v), i + off), v, s=12, color=col, alpha=0.45, lw=0, zorder=3)
                means.append(np.mean(v))
            ax.plot(np.arange(len(runs)) + off, means, color=col, marker='o', ms=5, lw=1.4, zorder=4)
            if col == AB_MEAN:
                for i, mv in enumerate(means):
                    ax.annotate(f'{mv:.3f}', (i + off, mv), textcoords='offset points', xytext=(6, 4), ha='left', fontsize=7.5, color=INK2)
    for ax, title in zip(axes, ['AUC', 'Specificity (Control)', 'Sensitivity (DM + FAP)']):
        ax.set_xticks(range(5), [AB_SHORT[r] for r in 'ABCDE'], fontsize=8)
        ax.set_xlim(-0.45, 4.6)
        ax.set_title(title)
        ax.grid(axis='x', visible=False)
    legend_below(axes[0], [Line2D([], [], color=AB_MEAN, marker='o', label='random folds, mean of 3 seeds'),
                           Line2D([], [], color=MUTED, marker='o', label='rebalanced folds, mean of 3 seeds'),
                           Line2D([], [], color=MUTED, marker='o', ls='', alpha=0.5, label='one seed (5-fold mean)')])
    note(fig, '128 px, B = 8, Control/DM/FAP test slides only (387). The two fold schemes have different test sets, so their levels are not paired.')
    save(fig, path)


def fig_rf_variants(meta, path):
    """Normalization variants at each patch size where they were run: seed dots and the 3-seed mean, all 433 slides."""
    sizes = [ps for ps in (128, 32) if _rf_seed_metrics(_rf_exp('D', ps), meta) is not None]
    data = {}
    for ps in sizes:
        for v in RF_VARIANTS:
            r = _rf_seed_metrics(_rf_exp(v, ps), meta)
            if r is not None:
                data[(v, ps)] = r
    vs = list(RF_VARIANTS)
    fig, axes = plt.subplots(2, 2, figsize=(14, 8.2))
    offs = {ps: (j - (len(sizes) - 1) / 2) * 0.24 for j, ps in enumerate(sizes)}
    for ax, (m, title) in zip(axes.flat, [('auc', 'AUC'), ('spec', 'Specificity (Control)'),
                                          ('sens', 'Sensitivity (DM + FAP + iSFN)'), ('isfn', 'Sensitivity, iSFN (pooled, 46 slides)')]):
        for ps in sizes:
            col = RF_SIZES[ps][1]
            ref = np.mean([s[m] for s in data[('D', ps)]])
            ax.axhline(ref, color=col, lw=0.8, ls='--', alpha=0.7, zorder=1)
            xs, means = [], []
            for i, v in enumerate(vs):
                if (v, ps) not in data:
                    continue
                vals = [s[m] for s in data[(v, ps)]]
                ax.scatter(np.full(len(vals), i + offs[ps]), vals, s=14, color=col, alpha=0.45, lw=0, zorder=3)
                xs.append(i + offs[ps]); means.append(np.mean(vals))
                ax.scatter([i + offs[ps]], [np.mean(vals)], s=34, color=col, zorder=4, edgecolor='white', lw=0.6)
                left = offs[ps] < 0 or len(sizes) == 1
                ax.annotate(f'{np.mean(vals):.3f}', (i + offs[ps], np.mean(vals)), textcoords='offset points',
                            xytext=(-6 if left else 6, 0), ha='right' if left else 'left', va='center', fontsize=6.8, color=INK2,
                            bbox=dict(fc='#fcfcfb', ec='none', pad=0.6), zorder=5)
        ax.set_xticks(range(len(vs)), [RF_VARIANTS[v][1] for v in vs], fontsize=8)
        ax.set_xlim(-0.5, len(vs) - 0.5)
        allv = [s[m] for r in data.values() for s in r]
        ax.set_ylim(min(allv) - 0.035, max(allv) + 0.02)
        ax.set_title(title)
        ax.grid(axis='x', visible=False)
    legend_below(axes[1, 0], [Line2D([], [], color=RF_SIZES[ps][1], marker='o', ls='', label=f'{ps} px, mean of 3 seeds') for ps in sizes]
                 + [Line2D([], [], color=MUTED, marker='o', ls='', alpha=0.5, label='one seed (5-fold mean)'),
                    Line2D([], [], color=MUTED, ls='--', label='D (no correction) at that size')])
    note(fig, 'Random folds, all 433 test slides. All variants use the same patch coordinates and the mask outside ROI + 20 px; they differ only in '
              'pixel values (E, W, G, Gk, Bt)\nor in the features (F). The y-axes do not start at 0.')
    save(fig, path)


def fig_rf_paired(meta, path):
    """Per-fold difference to D at the same patch size (seed-averaged), AUC and specificity, shared y-axis per metric."""
    sizes = [ps for ps in (128, 32) if _rf_seed_metrics(_rf_exp('D', ps), meta) is not None]
    vs = [v for v in RF_VARIANTS if v != 'D']
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    offs = {ps: (j - (len(sizes) - 1) / 2) * 0.3 for j, ps in enumerate(sizes)}
    for ps in sizes:
        ref = pd.concat([s['folds'] for s in _rf_seed_metrics(_rf_exp('D', ps), meta)]).groupby(level=0).mean()
        for i, v in enumerate(vs):
            r = _rf_seed_metrics(_rf_exp(v, ps), meta)
            if r is None:
                continue
            f = pd.concat([s['folds'] for s in r]).groupby(level=0).mean()
            for ax, m in zip(axes, ['auc', 'spec', 'sens']):
                d = (f[m] - ref[m]).values
                x = i + offs[ps] + np.linspace(-0.09, 0.09, len(d))
                ax.scatter(x, d, s=14, color=RF_SIZES[ps][1], alpha=0.6, lw=0, zorder=3)
                ax.plot([i + offs[ps] - 0.12, i + offs[ps] + 0.12], [d.mean()] * 2, color=RF_SIZES[ps][1], lw=2.2, zorder=4)
    for ax, title in zip(axes, ['Δ AUC vs. D', 'Δ specificity vs. D', 'Δ sensitivity vs. D']):
        ax.axhline(0, color=AXIS, lw=1, zorder=1)
        ax.set_xticks(range(len(vs)), [RF_VARIANTS[v][1] for v in vs], fontsize=8)
        ax.set_title(title)
        ax.grid(axis='x', visible=False)
    legend_below(axes[0], [Line2D([], [], color=RF_SIZES[ps][1], lw=2.2, label=f'{ps} px, mean over folds') for ps in sizes]
                 + [Line2D([], [], color=MUTED, marker='o', ls='', alpha=0.6, label='one test fold (mean of 3 seeds)')])
    note(fig, 'Random folds, all 433 test slides. Each variant is compared with D (mask, no intensity correction) at the same patch size.')
    save(fig, path)


def fig_rf_patch_sizes(meta, path):
    """D vs. E at 128 / 32 / 16 px on the random folds, all 433 slides."""
    sizes = [128, 32, 16]
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.3))
    allr = {(v, ps): _rf_seed_metrics(_rf_exp(v, ps), meta) for v in 'DE' for ps in sizes}
    for ax, (m, title) in zip(axes, [('auc', 'AUC'), ('spec', 'Specificity (Control)'),
                                     ('sens', 'Sensitivity (DM + FAP + iSFN)'), ('isfn', 'Sensitivity, iSFN (pooled, 46 slides)')]):
        for j, v in enumerate('DE'):
            off = (j - 0.5) * 0.22
            means = []
            for i, ps in enumerate(sizes):
                vals = [s[m] for s in allr[(v, ps)]]
                ax.scatter(np.full(len(vals), i + off), vals, s=14, color=AB_DE_COLORS[v], alpha=0.45, zorder=3, lw=0)
                means.append(np.mean(vals))
            ax.plot(np.arange(len(sizes)) + off, means, color=AB_DE_COLORS[v], marker='o', ms=6, lw=1.6, zorder=4)
            for i, mv in enumerate(means):
                ax.annotate(f'{mv:.3f}', (i + off, mv), textcoords='offset points', xytext=(-7 if v == 'D' else 7, 0),
                            ha='right' if v == 'D' else 'left', va='center', fontsize=7.5, color=INK2,
                            bbox=dict(fc='#fcfcfb', ec='none', pad=0.6), zorder=5)
        allv = [s[m] for r in allr.values() for s in r]
        ax.set_ylim(min(allv) - 0.03, max(allv) + 0.03)
        ax.set_xticks(range(len(sizes)), [f'{ps} px' for ps in sizes])
        ax.set_xlim(-0.6, len(sizes) - 0.4)
        ax.set_title(title)
        ax.grid(axis='x', visible=False)
    legend_below(axes[0], [Line2D([], [], color=AB_DE_COLORS[k], marker='o', label=AB_DE_LABELS[k] + ' (mean of 3 seeds)') for k in 'DE']
                 + [Line2D([], [], color=MUTED, marker='o', ls='', alpha=0.6, label='one seed (5-fold mean)')])
    note(fig, 'Random folds, all 433 test slides, buffer 20 px, ≥ 10% ROI + buffer. B = 8 / 32 / 64 for 128 / 32 / 16 px.')
    save(fig, path)


def norm_rf_figures(shared):
    meta = pd.read_csv(os.path.join(os.path.dirname(RESULTS), 'dataset_csv', 'ienf_isfn_4sub.csv')).set_index('slide_id')
    fig_rf_ladder(meta, os.path.join(shared, 'ienf_norm_rf_ladder.png'))
    fig_rf_variants(meta, os.path.join(shared, 'ienf_norm_rf_variants.png'))
    fig_rf_paired(meta, os.path.join(shared, 'ienf_norm_rf_paired.png'))
    fig_rf_patch_sizes(meta, os.path.join(shared, 'ienf_norm_rf_patch_sizes.png'))


# ----------------------------------------------------------------------------- main
def main():
    if '--only' in sys.argv and sys.argv[sys.argv.index('--only') + 1] == 'ienf_norm_rf':
        norm_rf_figures(os.path.join(RESULTS, 'figures'))
        return
    if '--only' in sys.argv and sys.argv[sys.argv.index('--only') + 1] == 'ienf':
        ienf_figures(os.path.join(RESULTS, 'figures'))
        return
    if '--only' in sys.argv and sys.argv[sys.argv.index('--only') + 1] == 'ienf_isfn':
        isfn_figures(os.path.join(RESULTS, 'figures'))
        return
    if '--only' in sys.argv and sys.argv[sys.argv.index('--only') + 1] == 'ienf_ablation':
        ablation_figures(os.path.join(RESULTS, 'figures'))
        return
    labels = pd.read_excel(LABELS_4CLS).set_index('slide').label.to_dict()
    runs = {key: load_run(key, labels) for key in RUNS}
    RUNS['r50_b8']['name'] = 'B = 8'  # only shown next to B = 32
    for key in ENCODERS:
        out = os.path.join(RESULTS, RUNS[key]['dir'], 'figures')
        fig_training_curves(runs[key], key, os.path.join(out, 'training_curves.png'))
        fig_roc_pr(runs[key], key, os.path.join(out, 'roc_pr.png'))
        fig_confusion_subtype(runs[key], key, os.path.join(out, 'confusion_subtype.png'))
        fig_per_fold_metrics(runs[key], key, os.path.join(out, 'per_fold_metrics.png'))
        fig_val_vs_test(runs[key], key, os.path.join(out, 'val_vs_test_auc.png'))
    shared = os.path.join(RESULTS, 'figures')
    fig_encoder_metrics(runs, os.path.join(shared, 'encoder_metrics.png'))
    fig_encoder_per_fold_auc(runs, os.path.join(shared, 'encoder_per_fold_auc.png'))
    fig_encoder_roc_pr(runs, os.path.join(shared, 'encoder_roc_pr.png'))
    fig_subtype_compare(runs, ENCODERS, os.path.join(shared, 'encoder_subtype_sensitivity.png'), 'Sensitivity by metastasis size')
    fig_encoder_confusion(runs, os.path.join(shared, 'encoder_confusion.png'))
    fig_training_compare(runs, ENCODERS, os.path.join(shared, 'encoder_training_curves.png'), 'Training curves by encoder')
    fig_paired_diff(runs, os.path.join(shared, 'encoder_paired_differences.png'))
    fig_val_vs_test_compare(runs, os.path.join(shared, 'encoder_val_vs_test_auc.png'))
    fig_cost(runs, os.path.join(shared, 'encoder_cost.png'))
    RUNS['r50_b32']['name'] = 'B = 32'
    fig_b8_b32_paper(runs, os.path.join(shared, 'r50_b8_b32_paper.png'))
    fig_b8_b32_per_fold(runs, os.path.join(shared, 'r50_b8_b32_per_fold_auc.png'))
    fig_b8_b32_roc(runs, os.path.join(shared, 'r50_b8_b32_roc.png'))
    fig_subtype_compare(runs, ['r50_b8', 'r50_b32'], os.path.join(shared, 'r50_b8_b32_subtype.png'), 'Sensitivity by metastasis size, ResNet-50')
    fig_training_compare(runs, ['r50_b8', 'r50_b32'], os.path.join(shared, 'r50_b8_b32_training_curves.png'), 'Training curves, ResNet-50: B = 8 vs. B = 32')
    RUNS['r50_b32']['name'] = 'ResNet-50'
    fig_5v10_subsets(runs, os.path.join(shared, 'mccv_5v10_subset_auc.png'))
    fig_5v10_estimates(runs, os.path.join(shared, 'mccv_5v10_estimates.png'))
    fig_size_metrics(runs, os.path.join(shared, 'size_metrics.png'))
    fig_size_per_fold(runs, os.path.join(shared, 'size_per_fold_auc.png'))
    fig_size_paired_diff(runs, os.path.join(shared, 'size_paired_differences.png'))
    fig_size_training(runs, os.path.join(shared, 'size_val_loss.png'))
    fig_size_subtype(runs, os.path.join(shared, 'size_subtype.png'))
    mb = dict(pairs=MB_PAIRS, labels=('SB', 'MB'))
    fig_size_metrics(runs, os.path.join(shared, 'mb_metrics.png'), **mb)
    fig_size_per_fold(runs, os.path.join(shared, 'mb_per_fold_auc.png'), **mb)
    fig_size_paired_diff(runs, os.path.join(shared, 'mb_paired_differences.png'), **mb)
    fig_size_training(runs, os.path.join(shared, 'mb_val_loss.png'), **mb)
    fig_size_subtype(runs, os.path.join(shared, 'mb_subtype.png'), **mb)
    fig_thr_curves(os.path.join(shared, 'thr_curves.png'))
    fig_thr_rule_diff(os.path.join(shared, 'thr_rule_diff.png'))
    fig_thr_val_vs_oracle(os.path.join(shared, 'thr_val_vs_oracle.png'))
    fig_thr_sens90(os.path.join(shared, 'thr_sens90.png'))
    RUNS['r50_b32']['name'], RUNS['uni_b32']['name'], RUNS['conch_b32']['name'] = 'ResNet-50', 'UNI', 'CONCH'
    c4 = {key: load_run_4cls(v[0]) for key, v in C4_RUNS.items()}
    fig_c4_confusion(c4, os.path.join(shared, 'c4_confusion.png'))
    fig_c4_per_class(c4, os.path.join(shared, 'c4_per_class.png'))
    fig_c4_vs_2cls(runs, c4, os.path.join(shared, 'c4_vs_2cls.png'))
    fig_c4_size_diff(c4, os.path.join(shared, 'c4_size_diff.png'))
    kf = load_kf(labels)
    case_of = pd.read_csv(os.path.join(os.path.dirname(RESULTS), 'dataset_csv', 'camelyon_plus_2cls.csv')).set_index('slide_id').case_id.to_dict()
    fig_kf_per_fold(kf, os.path.join(shared, 'kf_per_fold.png'))
    fig_kf_roc(kf, case_of, os.path.join(shared, 'kf_oof_roc.png'))
    fn = load_fn(runs, labels)
    fig_fn_metrics(fn, os.path.join(shared, 'fn_metrics.png'))
    fig_fn_paired_diff(fn, os.path.join(shared, 'fn_paired_differences.png'))
    fig_fn_curves(fn, os.path.join(shared, 'fn_loss_curves.png'))
    fig_fn_scale(fn, os.path.join(shared, 'fn_scale_vs_best_epoch.png'))
    lr = load_lr(runs, labels)
    fig_lr_metrics(lr, os.path.join(shared, 'lr_metrics.png'))
    fig_lr_paired_diff(lr, os.path.join(shared, 'lr_paired_differences.png'))
    fig_lr_curves(lr, os.path.join(shared, 'lr_loss_curves.png'))
    fig_lr_best_epoch(lr, os.path.join(shared, 'lr_best_epoch.png'))
    ienf_figures(shared)
    isfn_figures(shared)
    ablation_figures(shared)


if __name__ == '__main__':
    main()
