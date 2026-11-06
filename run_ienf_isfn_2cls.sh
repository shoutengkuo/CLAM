#!/usr/bin/env bash
# IENF 2-class with iSFN (Control vs. DM + FAP + iSFN), new preprocessing: intensity-normalized images,
# pixels outside ROI + 20 px buffer set to 0, patches 128/64/32/16 px kept at >= 10% ROI
# (features from run_ienf_isfn_features.sh). CLAM-SB small, B = 8/16/32/64 roughly tracking the patch
# count; k = min(B, N // 2) per bag (models/model_clam.py). Other settings as in run_ienf_2cls.sh.
# Splits: splits/ienf_isfn_kfold5_rebalanced (patient-level; hard cases spread using earlier TEST
# results, so performance on these folds is optimistic -- see the README there).
#
# Usage (repo root), e.g. inside tmux:
#   bash run_ienf_isfn_2cls.sh 2>&1 | tee logs/run_ienf_isfn_2cls.log
# Each experiment waits until its feature folder has all 439 pt files, so this can start while
# run_ienf_isfn_features.sh is still running.

cd "$(dirname "$0")"
mkdir -p logs

REST_SECONDS=120
GPU=0
PYTHON=${PYTHON:-/var/tmp/shouteng_conda/envs/clam/bin/python}
DATA_ROOT=/var/tmp/shouteng_data/CLAM_DATA
N_SLIDES=439

COMMON=(
  --task ienf_isfn_2cls --split_dir ienf_isfn_kfold5_rebalanced --k 5
  --model_type clam_sb --model_size small
  --drop_out 0.25 --lr 2e-4 --reg 1e-5 --max_epochs 200 --early_stopping
  --weighted_sample --bag_loss ce --inst_loss svm
  --data_root_dir "$DATA_ROOT" --embed_dim 1024 --seed 1 --log_data
)

# patch | B
EXPERIMENTS=(
  "128 8"
  "64 16"
  "32 32"
  "16 64"
)

n=${#EXPERIMENTS[@]}
for i in "${!EXPERIMENTS[@]}"; do
  read -r patch B <<< "${EXPERIMENTS[$i]}"
  feat=ienf_norm_uni_p${patch}_s${patch}
  exp_code=ienf_isfn_2cls_uni_p${patch}_CLAM_B${B}
  until [ "$(ls "$DATA_ROOT/$feat/pt_files" 2>/dev/null | wc -l)" -ge "$N_SLIDES" ]; do sleep 300; done
  sleep 60   # let the extractor finish writing the last file
  echo "[$(date '+%F %T')] start ${exp_code} ($((i + 1))/${n})"
  CUDA_VISIBLE_DEVICES=$GPU "$PYTHON" main.py "${COMMON[@]}" \
    --feature_dir "$feat" --B "$B" --exp_code "$exp_code" > "logs/${exp_code}.log" 2>&1
  rc=$?
  echo "[$(date '+%F %T')] finished ${exp_code} with exit code ${rc} (log: logs/${exp_code}.log)"
  if [ "$i" -lt $((n - 1)) ]; then
    echo "[$(date '+%F %T')] resting ${REST_SECONDS} s"
    sleep "$REST_SECONDS"
  fi
done
echo "[$(date '+%F %T')] all experiments done"
