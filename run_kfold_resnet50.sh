#!/usr/bin/env bash
# Standard (non-Monte-Carlo) k-fold CV with patient-level, stratified splits from create_splits_kfold.py:
#   10-fold: 80 / 10 / 10   (splits/camelyon_plus_kfold10_seed1)
#    5-fold: 70 / 10 / 20   (splits/camelyon_plus_kfold5_seed1)
# ResNet-50, CLAM-SB, model_size small, B = 32, 2-class task; all other settings as in the existing
# camelyon_plus_2cls_resnet50_CLAM_B32 (Monte Carlo) run. Runs execute one after another with a pause.
#
# Usage (repo root, clam env active), e.g. inside tmux:
#   bash run_kfold_resnet50.sh 2>&1 | tee logs/run_kfold_resnet50.log

cd "$(dirname "$0")"
mkdir -p logs

REST_SECONDS=600
GPU=0
DATA_ROOT=/var/tmp/shouteng_data/CLAM_DATA

COMMON="--drop_out 0.25 --early_stopping --lr 2e-4 --weighted_sample --bag_loss ce --inst_loss svm \
--task camelyon_plus_2cls --model_type clam_sb --model_size small --log_data --data_root_dir $DATA_ROOT --B 32 \
--feature_dir camelyon_plus_resnet50 --embed_dim 1024"

# k | split folder
EXPERIMENTS=(
  "10 camelyon_plus_kfold10_seed1"
  "5 camelyon_plus_kfold5_seed1"
)

n=${#EXPERIMENTS[@]}
for i in "${!EXPERIMENTS[@]}"; do
  read -r k split <<< "${EXPERIMENTS[$i]}"
  exp_code=camelyon_plus_2cls_resnet50_CLAM_B32_kfold${k}
  echo "[$(date '+%F %T')] start ${exp_code} ($((i + 1))/${n})"
  CUDA_VISIBLE_DEVICES=$GPU python main.py $COMMON --k "$k" --split_dir "$split" --exp_code "$exp_code" \
    > "logs/${exp_code}.log" 2>&1
  echo "[$(date '+%F %T')] finished ${exp_code} with exit code $? (log: logs/${exp_code}.log)"
  if [ "$i" -lt $((n - 1)) ]; then
    echo "[$(date '+%F %T')] resting ${REST_SECONDS} s"
    sleep "$REST_SECONDS"
  fi
done
echo "[$(date '+%F %T')] all experiments done"
