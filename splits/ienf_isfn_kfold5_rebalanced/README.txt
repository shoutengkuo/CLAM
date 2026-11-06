Patient-level 5-fold splits for IENF Control / DM / FAP / iSFN (217 cases, 433 slides).
Base: test folds of splits/ienf_kfold5_seed1; 5 hard<->easy swaps (hard = wrong in >= 3 of 4 earlier runs: ienf_2cls_uni_p128_CLAM_B8_s1, ienf_2cls_uni_p128_CLAM_B16_s1, ienf_2cls_uni_p224_CLAM_B4_s1, ienf_2cls_uni_p224_CLAM_B8_s1); iSFN patients added; val = 22 cases per fold.
CAUTION: folds were tuned on earlier TEST results, so performance on them is optimistic.
seed=1; created by create_splits_ienf_rebalanced.py
