---
name: clam-experiment-report
description: Write experiment reports for CLAM / MIL training runs in this repo (results/<exp_code>_s<seed>/), in the established house style - per-run reports, encoder or hyperparameter comparison reports, and comparisons against published numbers, with figures. Use this whenever the user asks to summarize, document, report on, write up, compare or visualize a finished training run or several runs under results/, even if they just say "整理實驗結果", "寫報告", "看這次訓練做了什麼", "跟上次比較", or "跟論文比"; also use it when adding a new run to an existing set of reports.
---

# CLAM experiment reports

This skill captures how experiment reports are written in this repo. The reference examples are the
existing reports, which are the best guide to tone and level of detail. Read at least one before
writing:

- Single run: `results/camelyon_plus_2cls_uni_CLAM_B32_s1/experiment_report.md`
- Comparison with a published result and with another setting of the same run:
  `results/camelyon_plus_2cls_resnet50_CLAM_B32_s1/paper_comparison_report.md`
- Several runs side by side: `results/encoder_comparison_B32_report.md`

Section templates for the three report types are in `references/report-templates.md`. Figure
specifications are in `references/figures.md`.

## Why these reports look the way they do

The reports are read later by the user and by others (advisor, lab mates). People use them to decide
what to run next and to defend the numbers. That leads to the main rules:

1. **Every number comes from the saved outputs, never from memory or a log you skimmed.** Run
   `scripts/collect_metrics.py` (below), which recomputes all metrics from `split_{k}_results.pkl`
   and checks them against `summary.csv`. Copy its tables into the report rather than retyping
   numbers.
2. **Every claim about a count or a range is checked against the data before it goes in.** For
   example, "8 of 10 folds", "best epoch 17–50" or "UNI better in all folds" must each be checked.
   Earlier drafts contained miscounts ("4 folds" that were really 6) and an incorrect statistics
   claim. Re-read each observation against the script output before finishing.
3. **Comparisons are only meaningful if the runs differ in one controlled way.** State exactly what
   is the same and what differs, including changes forced by the setup, such as `--embed_dim 512`
   for CONCH. Use paired tests whenever runs share splits.
4. **Separate what is measured from what is inferred.** Use phrases like "suggests", "was not tested"
   or "possible causes that have not been checked" for anything not verified. Never invent reference
   numbers. If a paper's value for some setting is not in hand, say so instead of filling it in.
5. **Attribute sources correctly.** In this project, Camelyon+ is only the **dataset** (slides,
   labels, pre-extracted features on NAS `/mnt/nas/Camelyon+/`). The reference numbers, and the
   choice of B = 32, come from the **original CLAM paper (Lu et al., 2021)**. That paper used the
   original CAMELYON16/17 data and its own features, so any comparison with it is a reference point,
   not like-for-like. Never write "Camelyon+ paper". If the user names a different source for future
   reference numbers, follow the user.

## Workflow

### 1. Gather facts about the run(s)

For each run directory `results/<exp_code>_s<seed>/`:

- `experiment_*.txt` holds the settings dict: hyperparameters, `split_dir`, and `embed_dim` /
  `feature_dir` for newer runs.
- Reconstruct the training command from the settings and `main.py` arguments. Check whether
  `--early_stopping`, `--weighted_sample` and `--log_data` were used, and which `--data_root_dir`
  and `--feature_dir`.
- Data facts: the dataset CSV in `dataset_csv/`, the feature folder (count the files, check one
  tensor's shape), the split files in `splits/<split_dir>/`, and whether patient stratification is
  on (`create_splits_seq.py`).
- `git status` / `git diff` show the code changes the run depends on, for the "Code changes" section.

### 2. Compute all numbers with the bundled script

Run it inside the clam environment:

```bash
/var/tmp/shouteng_conda/envs/clam/bin/python .claude/skills/clam-experiment-report/scripts/collect_metrics.py \
    results/<run_A> results/<run_B> --names A B \
    --subtypes "/mnt/nas/Camelyon+/slide-lables/Camelyon+(4-classes).xlsx" --csv-dir <scratchpad>/metrics
```

It prints markdown tables for:
- the summary, per-fold results and paper-style metrics (AUC / bACC / macro F1 / macro mAP);
- test-set composition, the pooled confusion matrix and detection rate by subtype;
- training time, training dynamics, and the correlation between validation and test AUC;
- with two or more runs, paired differences with Wilcoxon p-values and prediction agreement.

Leave out `--subtypes` for datasets that have no subtype labels. `--csv-dir` writes per-fold and
per-slide CSVs for any extra analysis.

The script asserts that recomputed test AUC and accuracy match `summary.csv`. If that fails, stop and
find out why before writing anything.

For **multi-class tasks** (e.g. `camelyon_plus_4cls`), use `scripts/collect_metrics_multiclass.py
<runs> --names ... --classes negative itc micro macro --csv-dir <dir>` instead. It reports:
- macro one-vs-rest AUC (checked against `summary.csv`), balanced accuracy, macro and weighted F1,
  Cohen's kappa (plain and quadratic-weighted for ordered classes);
- per-class AUC, recall, precision and F1, and the pooled confusion matrix;
- a **binary collapse** (class 0 vs. the rest), so a multi-class run can be compared fold by fold with
  a binary run on the same splits. When the binary operating points differ, also compare sensitivity
  at matched specificity. A shifted threshold can look like a sensitivity gain.

### 3. Make the figures

Follow `references/figures.md`. In short: extend `RUNS` (and, if needed, the figure lists) in
`results/make_report_figures.py`, run it, then open every PNG and look at it. That pass has caught
squashed panels, a wrong precision-recall interpolation, hidden overlapping dots and a legend covering
data. Fix and re-render until nothing collides.

### 4. Write the report(s)

Use the matching template in `references/report-templates.md`. Placement:

- Per-run report: `results/<run>/experiment_report.md`, with figures in `results/<run>/figures/`.
- Report comparing a run with a paper or with a variant of itself: `results/<run>/<topic>_report.md`.
- Report across several runs: `results/<topic>_report.md`, with shared figures in `results/figures/`.
- When a new run joins an existing comparison set, also update the comparison sections of the
  existing reports, and say that you did.

Reports are written in **English**, in plain, direct sentences. Talk to the user in their language
(Traditional Chinese so far) and give a short summary there.

### 5. Final check before handing over

- Re-read every observation and interpretation bullet against the script output: counts, ranges,
  "all folds", fold indices.
- Check that every image link resolves:

  ```bash
  grep -o '!\[[^]]*\]([^)]*)' report.md
  ```

  Then test each path from the report's folder.
- Check that anchor links (`[Section 5](#5-...)`) match the headings.
- Tell the user where the files are, the main findings, anything you inferred rather than verified,
  and any decision you made on their behalf (for example, which baseline you compared against).

## Conventions that apply to every report type

**Metrics.**
- The positive class is the tumor / positive class (label 1).
- Threshold-based metrics use argmax, which is a 0.5 threshold for two classes.
- Report mean ± SD over folds, with sample SD (ddof = 1) and 4 decimals.
- bACC = (sensitivity + specificity) / 2. F1 and mAP in paper comparisons are macro-averaged unless
  the paper says otherwise. Also list the alternative definitions (positive-class F1, weighted F1,
  positive-class AP) so a reader can match whichever definition the paper used.
- "Best val-loss epoch" is 0-indexed. It is the checkpoint that early stopping saved and that was
  tested.

**Monte Carlo CV caveats** (mention them where relevant):
- Test sets of different folds overlap, so pooled counts include some slides more than once. Give
  both numbers, e.g. 1284 predictions covering 853 unique slides.
- With patient-level splits the 80/10/10 ratio holds for cases, not exactly for slides.
- Pooled AUC is usually lower than the mean per-fold AUC because calibration differs between fold
  models. Say this when you report the pooled value.

**Statistics.**
- When runs share splits, use paired per-fold comparisons: the mean difference ± SD, how many folds
  the first run wins (with ties), and the two-sided exact Wilcoxon signed-rank p-value.
- With 10 folds the smallest possible p is 1/512 ≈ 0.002, which happens when one run wins all 10
  folds. This value passes Bonferroni over up to about 25 tests; p ≈ 0.006–0.008 does not. Do not
  claim robustness to multiple-comparison correction without checking.
- Call a difference "significant" only if p < 0.05. Otherwise call it a trend, and give the win count.
- Note the single-seed limitation: fold variance is captured, but seed variance is not.

**Training-time numbers** come from TensorBoard wall times: from the first to the last event of each
fold, which includes validation and the final test. An extra empty event file in a fold means an
earlier crashed attempt. Mention it, and exclude it from the times.

**Things worth looking for in the observations section** (from past runs, as examples):
- Where sensitivity is lost, broken down by subtype (ITC / micro / macro).
- Whether validation AUC predicts test AUC (it usually does not, because validation sets are small).
- Whether early stopping fires at the minimum epoch, which shows how fast the model converges or
  overfits.
- Whether shared hyperparameters may disadvantage one setting. Say this, but do not claim it as fact.
