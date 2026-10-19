#!/usr/bin/env bash
# Lower learning rate for the foundation-model encoders (UNI, CONCH), which reach their best validation loss
# within the first few epochs at lr = 2e-4 and then overfit. Two lower rates give a dose-response:
#   5e-5 (4x lower) and 2e-5 (10x lower) than the 2e-4 used in all earlier runs.
# Everything else is identical to the existing CLAM-SB small B = 32 runs (same Monte Carlo splits, seed,
# weight decay 1e-5, early stopping with min. 50 epochs / patience 20, max. 200 epochs), so every run can be
# compared fold by fold with its lr = 2e-4 counterpart.
#
# Usage (repo root, clam env active), e.g. inside tmux:
#   bash run_lower_lr.sh 2>&1 | tee logs/run_lower_lr.log

cd "$(dirname "$0")"
mkdir -p logs

REST_SECONDS=600
GPU=0
DATA_ROOT=/var/tmp/shouteng_data/CLAM_DATA

COMMON="--drop_out 0.25 --early_stopping --k 10 --weighted_sample --bag_loss ce --inst_loss svm \
--task camelyon_plus_2cls --model_type clam_sb --model_size small --log_data --data_root_dir $DATA_ROOT --B 32"

# name | feature_dir | embed_dim | lr
EXPERIMENTS=(
  "uni camelyon_plus_uni 1024 5e-5"
  "uni camelyon_plus_uni 1024 2e-5"
  "conch camelyon_plus_conch 512 5e-5"
  "conch camelyon_plus_conch 512 2e-5"
)

n=${#EXPERIMENTS[@]}
for i in "${!EXPERIMENTS[@]}"; do
  read -r name feat dim lr <<< "${EXPERIMENTS[$i]}"
  exp_code=camelyon_plus_2cls_${name}_CLAM_B32_lr${lr}
  echo "[$(date '+%F %T')] start ${exp_code} ($((i + 1))/${n})"
  CUDA_VISIBLE_DEVICES=$GPU python main.py $COMMON --lr "$lr" --exp_code "$exp_code" \
    --feature_dir "$feat" --embed_dim "$dim" > "logs/${exp_code}.log" 2>&1
  echo "[$(date '+%F %T')] finished ${exp_code} with exit code $? (log: logs/${exp_code}.log)"
  if [ "$i" -lt $((n - 1)) ]; then
    echo "[$(date '+%F %T')] resting ${REST_SECONDS} s"
    sleep "$REST_SECONDS"
  fi
done
echo "[$(date '+%F %T')] all experiments done"
