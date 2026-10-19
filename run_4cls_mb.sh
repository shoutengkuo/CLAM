#!/usr/bin/env bash
# 4-class Camelyon+ (negative / itc / micro / macro) with CLAM-MB, B = 32,
# for ResNet-50, UNI and CONCH, each with model_size small and big (6 runs).
# Splits: the existing 2-class splits (splits/camelyon_plus_2cls_100), used read-only, so every run has exactly the
# same train / val / test slides as the 2-class runs. All other settings match the 2-class B = 32 runs.
# The runs execute one after another, with a cool-down pause between them.
#
# Usage (from the repo root, with the clam env active), e.g. inside tmux:
#   bash run_4cls_mb.sh 2>&1 | tee logs/run_4cls_mb.log

cd "$(dirname "$0")"
mkdir -p logs

REST_SECONDS=600   # pause between experiments (10 min)
GPU=0
DATA_ROOT=/var/tmp/shouteng_data/CLAM_DATA

COMMON="--drop_out 0.25 --early_stopping --lr 2e-4 --k 10 --weighted_sample --bag_loss ce --inst_loss svm \
--task camelyon_plus_4cls --model_type clam_mb --log_data --data_root_dir $DATA_ROOT --B 32 \
--split_dir camelyon_plus_2cls_100"

# name | feature_dir | embed_dim | model_size
EXPERIMENTS=(
  "resnet50 camelyon_plus_resnet50 1024 small"
  "uni camelyon_plus_uni 1024 small"
  "conch camelyon_plus_conch 512 small"
  "resnet50 camelyon_plus_resnet50 1024 big"
  "uni camelyon_plus_uni 1024 big"
  "conch camelyon_plus_conch 512 big"
)

n=${#EXPERIMENTS[@]}
for i in "${!EXPERIMENTS[@]}"; do
  read -r name feat dim size <<< "${EXPERIMENTS[$i]}"
  exp_code=camelyon_plus_4cls_${name}_CLAM_MB_B32
  [ "$size" = "big" ] && exp_code=${exp_code}_big
  echo "[$(date '+%F %T')] start ${exp_code} ($((i + 1))/${n})"
  CUDA_VISIBLE_DEVICES=$GPU python main.py $COMMON --model_size "$size" --exp_code "$exp_code" \
    --feature_dir "$feat" --embed_dim "$dim" > "logs/${exp_code}.log" 2>&1
  echo "[$(date '+%F %T')] finished ${exp_code} with exit code $? (log: logs/${exp_code}.log)"
  if [ "$i" -lt $((n - 1)) ]; then
    echo "[$(date '+%F %T')] resting ${REST_SECONDS} s"
    sleep "$REST_SECONDS"
  fi
done
echo "[$(date '+%F %T')] all experiments done"
