#!/usr/bin/env bash
# IENF 2-class (Control vs. DM + FAP), UNI features from ROI patching at 224 and 128 px (no overlap).
# CLAM-SB small; B = 4, 8 for 224 px and B = 8, 16 for 128 px. Instance sampling uses k = min(B, N // 2)
# per bag (models/model_clam.py), so small bags never label a patch both positive and negative.
# Everything else as in the Camelyon+ runs (CLAM README settings).
# Patient-level stratified 5-fold CV (standard k-fold): splits/ienf_kfold5_seed1 (70/10/20 in cases).
#
# Usage (repo root), e.g. inside tmux:
#   bash run_ienf_2cls.sh 2>&1 | tee logs/run_ienf_2cls.log

cd "$(dirname "$0")"
mkdir -p logs

REST_SECONDS=120
GPU=0
PYTHON=${PYTHON:-/var/tmp/shouteng_conda/envs/clam/bin/python}
DATA_ROOT=/var/tmp/shouteng_data/CLAM_DATA

COMMON=(
  --task ienf_2cls --split_dir ienf_kfold5_seed1 --k 5
  --model_type clam_sb --model_size small
  --drop_out 0.25 --lr 2e-4 --reg 1e-5 --max_epochs 200 --early_stopping
  --weighted_sample --bag_loss ce --inst_loss svm
  --data_root_dir "$DATA_ROOT" --embed_dim 1024 --seed 1 --log_data
)

# patch | feature_dir | B
EXPERIMENTS=(
  "224 ienf_uni_p224_s224 4"
  "224 ienf_uni_p224_s224 8"
  "128 ienf_uni_p128_s128 8"
  "128 ienf_uni_p128_s128 16"
)

n=${#EXPERIMENTS[@]}
for i in "${!EXPERIMENTS[@]}"; do
  read -r patch feat B <<< "${EXPERIMENTS[$i]}"
  exp_code=ienf_2cls_uni_p${patch}_CLAM_B${B}
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
