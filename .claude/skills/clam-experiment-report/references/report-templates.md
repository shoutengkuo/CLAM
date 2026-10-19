# Report templates

Three report types. Section order and headings are fixed so reports stay easy to compare. Section
contents adapt to the run. Angle brackets are placeholders. The example reports named in SKILL.md
show the full level of detail.

Contents:
1. Single-run report
2. Comparison with a published result and/or a variant of the same run
3. Multi-run comparison report
4. Figure and caption format

---

## 1. Single-run report — `results/<run>/experiment_report.md`

```markdown
# CLAM on <dataset> (<task>, <encoder> features, B = <B>): Experiment Report

**Experiment code:** `<exp_code>_s<seed>`
**Run date:** <YYYY-MM-DD>, <HH:MM>–<HH:MM> (<total> for all <k> folds; see [Section 6](#6-training-time))
**Hardware:** <GPU>

## 1. Summary
<2–3 sentences: model, task, features, CV protocol, what is special about this run and which
earlier run it is otherwise identical to.>

| Metric (test set, <k> folds) | Mean ± SD | Min | Max |   <- AUC and Accuracy in bold; then bACC,
                                                                Sensitivity, Specificity, Precision, F1
<One line: SD definition (ddof = 1), positive class, argmax threshold.>

## 2. Data
Table: dataset (and its source), labels file, slide counts per class, subtype counts if available,
case count and how cases are formed, feature archive (with its NAS path) and tensor shape, local
feature location (number of files, size), dataset CSV.
Bullets: features used as provided or re-extracted; file-name check against the dataset CSV; patch
count consistency with other encoders if relevant; extra label files used only for analysis.

## 3. Data Splits: <k>-fold Monte Carlo Cross-Validation
The command that created the splits, then bullets on sampling, overlap of test sets (predictions vs.
unique slides), patient stratification and the slide vs. case ratio. Then a per-fold table: train /
val / test slides, test negative, test positive (with a subtype breakdown if available).
Say explicitly if the splits are shared with other runs ("identical files").

## 4. Training Configuration
The exact training command in a bash block (wrapped with \ for readability), then a settings table:
model + layer widths, embed dim, dropout, optimizer, losses + B, loss weighting, class-imbalance
handling, max epochs, early-stopping rule, seed.
End with one sentence: "Compared with <baseline run>, only <flags> changed." Explain any forced
change (e.g. embed_dim -> the first layer goes from Linear(1024, 512) to Linear(512, 512)).

## 5. Per-Fold Test Results
Per-fold table (Fold | Epochs run | Best val-loss epoch | Val AUC | Val Acc | **Test AUC** | **Test Acc**
| Sens. | Spec. | F1) with Mean and SD rows. Note that the epochs come from TensorBoard and are 0-indexed.
[Figure: ROC + PR] [Figure: per-fold metric strip]
### Pooled confusion matrix (all <k> test folds, <n> slide predictions)
2×2 table + the overlap caveat + pooled AUC vs. mean per-fold AUC.
### Sensitivity by <subtype> (only if subtype labels exist)
Table: detected / total (pooled), pooled rate, per-fold mean ± SD, and the false-positive rate on negatives.
[Figure: confusion matrix + subtype bars]

## 6. Training Time
Per-fold table: Start | End | Duration | Epochs run | Time / epoch, plus a Total row. The date line.
Bullets: time per epoch, what drives fold duration (early-stopping epoch), and any excluded crashed
attempts.

## 7. Observations
[Figure: training curves] then 3–5 bold-lead bullets, each a finding backed by numbers:
- where the errors are (e.g. by subtype)
- fold variance and val-vs-test AUC (with r)
- convergence / overfitting (best epoch range, train/val loss at the end)
- caveats about hyperparameters that were not tuned for this setting
[Figure: val vs. test AUC]

## 8. Comparison with <other runs> (only when comparable runs exist)
What is identical and what differs; links to the other runs' reports. State which run "<name>" refers to
(e.g. "our ResNet-50 run, not the number reported in the original CLAM paper").
### Final performance (test set, mean ± SD over <k> folds)
One table, one column per run, this run's column in bold. Do not repeat the other runs' per-fold numbers.
[Figure: grouped metric bars] [Figure: per-fold AUC by run]
### Paired difference of <this run> from each other run (same splits in every fold)
Table: Metric | A − B | Folds A better | p | ... (Wilcoxon note: exact test, smallest possible p).
### Sensitivity by <subtype> (pooled over all test folds)
[Figure: subtype bars by run]
### Interpretation
Bold-lead bullets: this run vs. each other run (significant or trend), where the gain comes from, and
cost (time, storage).

## 9. Code Changes to the CLAM Repository
Table File | Change. Include changes made after this run, and say so if the run did not use them.

## 10. Output Files
The list of files in the run folder (summary.csv, checkpoints, pkl, splits, TensorBoard logs,
settings txt, figures/, any other reports), then the eval.py command to re-evaluate the checkpoints,
with the flags that are needed (--feature_dir, --embed_dim).
```

---

## 2. Comparison with a published result / variant — `results/<run>/<topic>_report.md`

Model: `results/camelyon_plus_2cls_resnet50_CLAM_B32_s1/paper_comparison_report.md`.

```markdown
# <Model> with <encoder> on <dataset>: Comparison with <source> and Effect of <variable>

**Runs compared:** table Label | Experiment code | <variable> | Run date, plus a row for the published result.
<Paragraph: what "the paper" means (full citation), what was taken from the dataset provider vs.
from the paper, and a bold "The comparison is not like-for-like" paragraph if the data or features differ.>

## 1. Summary            <- 3 bullets: gap to the paper, effect of the variable (with p), most likely explanation
## 2. Setup              <- side-by-side settings table; command; ### Metric definitions (assumed if the paper is silent)
## 3. <Run> vs. <Paper>  <- table (paper row labeled with its data); alternative-definition table; per-fold paper metrics; Interpretation
## 4. <Variant A> vs. <Variant B>
   ### Test performance  <- every metric: A | B | paired difference | folds better | p; per-fold AUC row table
   ### Agreement of predictions   <- agreement %, prob correlation, 2×2 flip table
   ### Pooled confusion matrices / ### Sensitivity by subtype / ### Training behavior and time
   **Interpretation.**
## 5. <A> vs. <B> vs. <Paper>   <- one table + gaps + "gap closed by <variable>" (paired improvement / gap)
   ### What could explain the remaining gap   <- ordered from most to least likely, each marked unchecked; a concrete next step
## 6. Files
```

---

## 3. Multi-run comparison report — `results/<topic>_report.md`

Model: `results/encoder_comparison_B32_report.md`.

```markdown
# <Model> on <dataset>: <A> vs. <B> vs. <C> (<shared setting>)
**Runs compared:** table Name | Experiment code | Run date | Report path. State that these are our own runs
and why the shared setting was chosen.

## 1. Summary                  <- compact metric table + 4–5 bold-lead conclusion bullets
## 2. Setup                    <- ### What is the same in all runs (table), ### What differs (table), ### Metrics (definitions + the paired-test note)
## 3. Overall Test Performance <- every metric; bold the best mean; last column "Significant differences (p < 0.05)"
## 4. Per-Fold Test Results    <- per-fold AUC table with a "best in fold" column; sens/spec per fold
## 5. Paired Statistical Comparison  <- one cell per pair: diff ± SD, wins, p; multiple-comparison note (see SKILL.md)
## 6. Error Analysis           <- pooled confusion matrices, sensitivity by subtype (pooled + per fold), agreement / overlap of errors between runs
## 7. Training Dynamics and Model Selection  <- best epoch, epochs run, loss first/best/last, train loss last, val AUC, val-test r
## 8. Computational Cost       <- feature dim, storage, parameters of the first layer, time per epoch, total time
## 9. Context: <published reference>   <- short; say what is and is not comparable
## 10. Conclusions             <- numbered conclusions; ### Limitations; ### Possible next steps
## Figures                     <- where the figures live and the command that regenerates them
```

---

## 4. Figure and caption format

Insert figures right after the table or paragraph they support, using paths relative to the report:

```markdown
![Short alt text](figures/roc_pr.png)

*Figure 3. One or two sentences: what is plotted, how to read it (thin lines = folds, band = ±1 SD,
y-axis starts at 0.6, ...), and the one thing the reader should notice.*
```

Number figures from 1 within each report. Per-run figures use `figures/...`. Shared figures used from
a per-run report use `../figures/...`.
