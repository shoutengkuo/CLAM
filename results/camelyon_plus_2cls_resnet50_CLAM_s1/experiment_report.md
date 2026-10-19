# CLAM on Camelyon+ (2-class, ResNet-50 features): Experiment Report

**Experiment code:** `camelyon_plus_2cls_resnet50_CLAM_s1`
**Run date:** 2026-09-24, 15:37–18:20 (2 h 42 min for all 10 folds; see [Section 6](#6-training-time))
**Hardware:** 1 × NVIDIA GeForce RTX 3090 (24 GB)

## 1. Summary

CLAM-SB was trained on the Camelyon+ 2-class task (negative vs. tumor) with pre-extracted ResNet-50 features, using 10-fold Monte Carlo cross-validation (80 / 10 / 10 train / val / test).

| Metric (test set, 10 folds) | Mean ± SD | Min | Max |
|---|---|---|---|
| **AUC** | **0.9007 ± 0.0287** | 0.8506 | 0.9377 |
| **Accuracy** | **0.8732 ± 0.0274** | 0.8205 | 0.9220 |
| Balanced accuracy | 0.8442 ± 0.0306 | 0.7916 | 0.8964 |
| Sensitivity (tumor recall) | 0.7367 ± 0.0561 | 0.6500 | 0.8043 |
| Specificity (negative recall) | 0.9517 ± 0.0308 | 0.8831 | 0.9889 |
| Precision (tumor) | 0.8989 ± 0.0617 | 0.7568 | 0.9762 |
| F1 (tumor) | 0.8078 ± 0.0428 | 0.7273 | 0.8817 |

SD is the sample standard deviation (ddof = 1) across the 10 folds. The positive class is `tumor`. Threshold-based metrics use argmax, which for two classes is a 0.5 probability threshold.

## 2. Data

| Item | Value |
|---|---|
| Dataset | Camelyon+ (cleaned CAMELYON16 + CAMELYON17), downloaded from ScienceDB |
| Labels | `slide-lables/Camelyon+(2-classes).xlsx` |
| Slides | 1350 (negative 871, tumor 479) |
| Cases | 586. CAMELYON17 slides are grouped by patient (`patient_XXX`); each CAMELYON16 slide is its own case |
| Features | Provided `ResNet-50.zip`: one `[N_patches, 1024]` float32 tensor per slide |
| Feature location | `/var/tmp/shouteng_data/CLAM_DATA/camelyon_plus_resnet50/pt_files/` |
| Dataset CSV | `dataset_csv/camelyon_plus_2cls.csv` (`case_id, slide_id, label`) |

The features were used as provided and were not re-extracted with CLAM's `extract_features_fp.py`. The accompanying `h5py-files` hold only patch coordinates (patch size 256, patch level 1) and were not used for training.

## 3. Data Splits: 10-fold Monte Carlo Cross-Validation

The splits were generated with CLAM's default splitting script:

```bash
python create_splits_seq.py --task camelyon_plus_2cls --seed 1 --k 10
```

- Each fold independently samples 10% validation and 10% test cases per class. The remaining 80% of cases are used for training.
- Test sets of different folds may overlap, which is expected for Monte Carlo CV.
- Sampling is done at the patient level (`patient_strat=True`), so no patient appears in more than one of train / val / test within a fold. The patient label is `max` over its slides, meaning a patient is tumor if any of its slides is tumor.
- Because CAMELYON17 patients contribute up to 5 slides, the 80/10/10 ratio holds for cases, not exactly for slides.
- Split files: `splits/camelyon_plus_2cls_100/splits_{0..9}.csv`.

| Fold | Train slides | Val slides | Test slides | Test negative | Test tumor |
|---|---|---|---|---|---|
| 0 | 1091 | 136 | 123 | 83 | 40 |
| 1 | 1064 | 150 | 136 | 89 | 47 |
| 2 | 1085 | 138 | 127 | 77 | 50 |
| 3 | 1080 | 139 | 131 | 86 | 45 |
| 4 | 1087 | 122 | 141 | 90 | 51 |
| 5 | 1065 | 151 | 134 | 88 | 46 |
| 6 | 1093 | 142 | 115 | 70 | 45 |
| 7 | 1095 | 122 | 133 | 86 | 47 |
| 8 | 1114 | 119 | 117 | 77 | 40 |
| 9 | 1098 | 125 | 127 | 71 | 56 |

## 4. Training Configuration

```bash
CUDA_VISIBLE_DEVICES=0 python main.py --drop_out 0.25 --early_stopping --lr 2e-4 --k 10 \
  --exp_code camelyon_plus_2cls_resnet50_CLAM --weighted_sample --bag_loss ce --inst_loss svm \
  --task camelyon_plus_2cls --model_type clam_sb --log_data \
  --data_root_dir /var/tmp/shouteng_data/CLAM_DATA --embed_dim 1024
```

| Setting | Value |
|---|---|
| Model | CLAM-SB (single attention branch), `model_size = small` |
| Input embedding dim | 1024 |
| Dropout | 0.25 |
| Optimizer | Adam, lr = 2e-4, weight decay = 1e-5 |
| Bag-level loss | Cross-entropy |
| Instance-level loss | Smooth top-1 SVM, B = 8 |
| Loss weighting | 0.7 × bag loss + 0.3 × instance loss |
| Class imbalance | `--weighted_sample`: sampling weight per slide = N / N_class, i.e. inversely proportional to class size |
| Max epochs | 200 |
| Early stopping | Monitors validation loss, patience 20, not before epoch 50. The checkpoint with the lowest validation loss is used for testing |
| Seed | 1 |

## 5. Per-Fold Test Results

| Fold | Epochs run | Best val-loss epoch | Val AUC | Val Acc | **Test AUC** | **Test Acc** | Sens. | Spec. | F1 |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 70 | 49 | 0.9272 | 0.9265 | 0.9033 | 0.8699 | 0.6500 | 0.9759 | 0.7647 |
| 1 | 52 | 13 | 0.8491 | 0.8467 | 0.9288 | 0.8897 | 0.7660 | 0.9551 | 0.8276 |
| 2 | 52 | 26 | 0.8809 | 0.8551 | 0.9143 | 0.8819 | 0.8000 | 0.9351 | 0.8421 |
| 3 | 52 | 30 | 0.9370 | 0.9209 | 0.9106 | 0.8779 | 0.7556 | 0.9419 | 0.8095 |
| 4 | 52 | 25 | 0.9567 | 0.9344 | 0.9377 | 0.9220 | 0.8039 | 0.9889 | 0.8817 |
| 5 | 52 | 14 | 0.9096 | 0.8675 | 0.8990 | 0.8881 | 0.8043 | 0.9318 | 0.8315 |
| 6 | 52 | 17 | 0.9495 | 0.9296 | 0.8654 | 0.8522 | 0.6889 | 0.9571 | 0.7848 |
| 7 | 52 | 31 | 0.9202 | 0.9098 | 0.9233 | 0.8797 | 0.7021 | 0.9767 | 0.8049 |
| 8 | 70 | 49 | 0.9296 | 0.9160 | 0.8506 | 0.8205 | 0.7000 | 0.8831 | 0.7273 |
| 9 | 52 | 22 | 0.9722 | 0.9360 | 0.8740 | 0.8504 | 0.6964 | 0.9718 | 0.8041 |
| **Mean** | | | 0.9232 | 0.9042 | **0.9007** | **0.8732** | 0.7367 | 0.9517 | 0.8078 |
| **SD** | | | 0.0364 | 0.0343 | 0.0287 | 0.0274 | 0.0561 | 0.0308 | 0.0428 |

"Epochs run" and "Best val-loss epoch" (0-indexed) are read from the TensorBoard logs.

### Pooled confusion matrix (all 10 test folds, 1284 slide predictions)

Test sets overlap across folds, so some slides are counted more than once.

| | Predicted negative | Predicted tumor |
|---|---|---|
| **True negative** | 778 | 39 |
| **True tumor** | 122 | 345 |

The AUC computed on the pooled predictions is 0.8883. It is lower than the mean per-fold AUC because the score calibration differs between fold models.

## 6. Training Time

Folds were run sequentially on a single RTX 3090. The times below are taken from the TensorBoard event timestamps. Each time runs from the moment a fold's log writer was created to the fold's last logged event, so it includes training, per-epoch validation and the final test evaluation.

| Fold | Start | End | Duration | Epochs run | Time / epoch |
|---|---|---|---|---|---|
| 0 | 15:37:34 | 15:57:43 | 20.2 min | 70 | 17.3 s |
| 1 | 15:57:43 | 16:12:33 | 14.8 min | 52 | 17.1 s |
| 2 | 16:12:33 | 16:27:42 | 15.1 min | 52 | 17.5 s |
| 3 | 16:27:42 | 16:43:07 | 15.4 min | 52 | 17.8 s |
| 4 | 16:43:07 | 16:58:13 | 15.1 min | 52 | 17.4 s |
| 5 | 16:58:13 | 17:13:27 | 15.2 min | 52 | 17.6 s |
| 6 | 17:13:27 | 17:28:44 | 15.3 min | 52 | 17.6 s |
| 7 | 17:28:44 | 17:43:57 | 15.2 min | 52 | 17.6 s |
| 8 | 17:43:57 | 18:04:18 | 20.4 min | 70 | 17.4 s |
| 9 | 18:04:18 | 18:19:40 | 15.4 min | 52 | 17.7 s |
| **Total** | 15:37:34 | 18:19:40 | **162.1 min (2 h 42 min)** | 556 | **17.5 s (mean)** |

All times are on 2026-09-24.

- One epoch (a training pass over about 1080 slides plus validation on about 135 slides) takes about 17.5 s.
- Fold duration is determined almost entirely by when early stopping triggers: 52 epochs take about 15 min, and 70 epochs take about 20 min.
- If no fold had stopped early, the maximum of 200 epochs would take about 58 min per fold, or about 9.7 h for 10 folds.
- Not included in these times: an earlier attempt that crashed on the `np.Inf` error at 15:35:00. It is the empty log in fold 0.

## 7. Observations

- **Low sensitivity is the main weakness.** Specificity is high (0.95), but about 26% of tumor slides are missed at the default 0.5 threshold, even with class-balanced sampling. A likely cause is small metastases (micro-metastases and isolated tumor cells), which occupy very few patches. The 4-class labels (`macro` / `micro` / `itc`) can be used to confirm this.
- **Fold variance is moderate.** Test AUC ranges from 0.851 to 0.938. Validation AUC and test AUC are only weakly related across folds. For example, fold 9 has the highest val AUC (0.972) but one of the lowest test AUCs (0.874). Validation sets are small (about 120–150 slides, 40–56 tumor), so model selection is noisy.
- **Early stopping triggers at the minimum allowed epoch in most folds.** 8 of 10 folds stopped at epoch 52, with the best validation loss reached between epochs 13 and 31. This suggests the model converges quickly, and the fixed 50-epoch minimum mostly adds training time.

## 8. Comparison with the Original CLAM Paper

The reference numbers below are the CLAM-SB / ResNet-50 results reported in the original CLAM paper (Lu et al., 2021). The Camelyon+ project provided only the dataset used in this run, not the reference numbers. The CLAM paper predates Camelyon+, so its results come from different data (the original CAMELYON16/17 slides and labels) and from features extracted by the CLAM authors. The comparison is therefore a reference point, not a like-for-like replication.

| Method | Features | Protocol | AUC | bACC | F1 | mAP |
|---|---|---|---|---|---|---|
| CLAM-SB (CLAM paper) | ResNet-50 | 10-fold MCCV, 80/10/10 | 0.9532 ± 0.0292 | 0.9042 ± 0.0365 | 0.8878 ± 0.0471 | 0.9464 ± 0.0302 |
| CLAM-SB (this run) | ResNet-50 | 10-fold MCCV, 80/10/10 | 0.9007 ± 0.0287 | 0.8442 ± 0.0306 | 0.8563 ± 0.0308 | 0.8933 ± 0.0377 |
| Difference (this run − paper) | | | −0.0525 | −0.0600 | −0.0315 | −0.0531 |

For this run, metrics are computed per fold from `split_{k}_results.pkl` and reported as mean ± SD over the 10 folds. The paper does not state its metric definitions in the numbers provided, so the following definitions were assumed:

- **bACC**: mean of per-class recall, i.e. (sensitivity + specificity) / 2, at the argmax threshold.
- **F1**: macro-averaged F1 over the two classes. If the paper instead reports F1 of the tumor class only, the corresponding value here is 0.8078 ± 0.0428. If it reports the class-frequency-weighted F1, the value is 0.8700 ± 0.0280.
- **mAP**: average precision computed for each class (one-vs-rest) and then averaged over the two classes. If the paper instead reports AP of the tumor class only, the corresponding value here is 0.8711 ± 0.0595.

Per-fold values:

| Fold | AUC | bACC | F1 (macro) | mAP (macro) |
|---|---|---|---|---|
| 0 | 0.9033 | 0.8130 | 0.8374 | 0.9040 |
| 1 | 0.9288 | 0.8605 | 0.8733 | 0.9235 |
| 2 | 0.9143 | 0.8675 | 0.8739 | 0.9137 |
| 3 | 0.9106 | 0.8487 | 0.8598 | 0.8882 |
| 4 | 0.9377 | 0.8964 | 0.9118 | 0.9357 |
| 5 | 0.8990 | 0.8681 | 0.8738 | 0.9040 |
| 6 | 0.8654 | 0.8230 | 0.8361 | 0.8500 |
| 7 | 0.9233 | 0.8394 | 0.8590 | 0.9282 |
| 8 | 0.8506 | 0.7916 | 0.7968 | 0.8183 |
| 9 | 0.8740 | 0.8341 | 0.8416 | 0.8674 |

**Interpretation.** This run is lower than the paper on all four metrics. The AUC, bACC and mAP gaps (0.05–0.06) are larger than 1 SD of either result (about 0.03), so they are unlikely to be explained by split randomness alone. The F1 gap (0.03) is about 1 SD. The paper's split seeds are not known, so the per-fold test sets are not identical to the paper's. Possible sources of the gap that have not yet been checked:

- **Dataset.** The CLAM paper used the original CAMELYON16/17 data. This run uses Camelyon+, which has a different slide set and cleaned labels. A different test population alone can shift all metrics.
- **Features.** This run used the `ResNet-50.zip` features provided with Camelyon+. The CLAM paper extracted its own features with CLAM's pipeline, which may use different ResNet-50 settings, such as the truncated ResNet-50 with ImageNet normalization or a different magnification or patch level.
- **Hyperparameters.** This run used CLAM's README defaults (lr 2e-4, dropout 0.25, `clam_sb`, `small`). The paper's settings may differ.
- **Metric definitions.** These are the assumptions listed above.
- **Model selection.** This run used early stopping on validation loss.

## 9. Code Changes to the CLAM Repository

| File | Change |
|---|---|
| `main.py`, `eval.py` | Added task `camelyon_plus_2cls` (labels `negative: 0`, `tumor: 1`, data dir `camelyon_plus_resnet50`) |
| `create_splits_seq.py` | Added task `camelyon_plus_2cls` (`patient_strat=True`) |
| `utils/core_utils.py` | Replaced `np.Inf` with `np.inf` for NumPy 2.x compatibility |
| `dataset_csv/camelyon_plus_2cls.csv` | New dataset CSV built from the Camelyon+ 2-class label file |
| `create_splits_camelyon_plus.py` | New script for the official CAMELYON16/17 train/test splits (`splits/camelyon_plus_2cls_official/`). Not used in this experiment |

## 10. Output Files

All files are under `results/camelyon_plus_2cls_resnet50_CLAM_s1/`:

- `summary.csv`: per-fold test / val AUC and accuracy
- `s_{k}_checkpoint.pt`: model checkpoint with the lowest validation loss for fold k
- `split_{k}_results.pkl`: per-slide test predictions (`slide_id`, class probabilities, label)
- `splits_{k}.csv`: copy of the split used for fold k
- `{k}/events.out.tfevents.*`: TensorBoard logs. Fold 0 has an extra empty log from an initial run that crashed on the `np.Inf` error
- `experiment_camelyon_plus_2cls_resnet50_CLAM.txt`: full settings dictionary

To re-evaluate the saved checkpoints:

```bash
CUDA_VISIBLE_DEVICES=0 python eval.py --k 10 --models_exp_code camelyon_plus_2cls_resnet50_CLAM_s1 \
  --save_exp_code camelyon_plus_2cls_resnet50_CLAM_s1_cv --task camelyon_plus_2cls --model_type clam_sb \
  --results_dir results --data_root_dir /var/tmp/shouteng_data/CLAM_DATA --embed_dim 1024
```
