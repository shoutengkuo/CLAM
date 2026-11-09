Random patient-level 5-fold splits for IENF Control / DM / FAP / iSFN (217 cases, 433 slides).
Test folds: StratifiedKFold over subtype x batch (shuffle, seed 2026); no earlier results used.
val = 22 cases per fold, stratified by subtype, from outside the test fold.
Created by create_splits_ienf_random.py
