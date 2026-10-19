#!/usr/bin/env bash
# Input feature normalization experiment (main.py --feat_norm), CLAM-SB small, B = 32, 2-class task,
# same Monte Carlo splits (splits/camelyon_plus_2cls_100), seed and hyperparameters as the existing B = 32 runs,
# so every run can be compared fold by fold with its un-normalized counterpart.
#   l2     : every patch vector scaled to unit length (CONCH features already have unit length -> no CONCH l2 run)
#   zscore : per-dimension (x - mean) / std, statistics from the training slides of each fold only
# Runs execute one after another with a cool-down pause.
#
# Usage (repo root, clam env active), e.g. inside tmux:
#   bash run_feat_norm.sh 2>&1 | tee logs/run_feat_norm.log

cd "$(dirname "$0")"
mkdir -p logs

REST_SECONDS=600
GPU=0
DATA_ROOT=/var/tmp/shouteng_data/CLAM_DATA

COMMON="--drop_out 0.25 --early_stopping --lr 2e-4 --k 10 --weighted_sample --bag_loss ce --inst_loss svm \
--task camelyon_plus_2cls --model_type clam_sb --model_size small --log_data --data_root_dir $DATA_ROOT --B 32"

# name | feature_dir | embed_dim | feat_norm
EXPERIMENTS=(
  "uni camelyon_plus_uni 1024 l2"
  "uni camelyon_plus_uni 1024 zscore"
  "resnet50 camelyon_plus_resnet50 1024 l2"
  "resnet50 camelyon_plus_resnet50 1024 zscore"
  "conch camelyon_plus_conch 512 zscore"
)

n=${#EXPERIMENTS[@]}
for i in "${!EXPERIMENTS[@]}"; do
  read -r name feat dim norm <<< "${EXPERIMENTS[$i]}"
  exp_code=camelyon_plus_2cls_${name}_CLAM_B32_${norm}
  echo "[$(date '+%F %T')] start ${exp_code} ($((i + 1))/${n})"
  CUDA_VISIBLE_DEVICES=$GPU python main.py $COMMON --feat_norm "$norm" --exp_code "$exp_code" \
    --feature_dir "$feat" --embed_dim "$dim" > "logs/${exp_code}.log" 2>&1
  echo "[$(date '+%F %T')] finished ${exp_code} with exit code $? (log: logs/${exp_code}.log)"
  if [ "$i" -lt $((n - 1)) ]; then
    echo "[$(date '+%F %T')] resting ${REST_SECONDS} s"
    sleep "$REST_SECONDS"
  fi
done
echo "[$(date '+%F %T')] all experiments done"
