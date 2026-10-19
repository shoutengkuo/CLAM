#!/usr/bin/env bash
# CLAM-SB with model_size=big (attention hidden dim 384 instead of 256) for ResNet-50, UNI and CONCH, B = 32.
# Everything else is identical to the existing model_size=small B = 32 runs (same splits, seed and hyperparameters).
# The runs execute one after another, with a cool-down pause between them.
#
# Usage (from the repo root, with the clam env active):
#   nohup bash run_model_size_big.sh > logs/run_model_size_big.log 2>&1 &

cd "$(dirname "$0")"
mkdir -p logs

REST_SECONDS=600   # pause between experiments (10 min)
GPU=0
DATA_ROOT=/var/tmp/shouteng_data/CLAM_DATA

COMMON="--drop_out 0.25 --early_stopping --lr 2e-4 --k 10 --weighted_sample --bag_loss ce --inst_loss svm \
--task camelyon_plus_2cls --model_type clam_sb --model_size big --log_data --data_root_dir $DATA_ROOT --B 32"

# name | feature_dir | embed_dim
EXPERIMENTS=(
  "resnet50 camelyon_plus_resnet50 1024"
  "uni camelyon_plus_uni 1024"
  "conch camelyon_plus_conch 512"
)

n=${#EXPERIMENTS[@]}
for i in "${!EXPERIMENTS[@]}"; do
  read -r name feat dim <<< "${EXPERIMENTS[$i]}"
  exp_code=camelyon_plus_2cls_${name}_CLAM_B32_big
  echo "[$(date '+%F %T')] start ${exp_code} ($((i + 1))/${n})"
  CUDA_VISIBLE_DEVICES=$GPU python main.py $COMMON --exp_code "$exp_code" --feature_dir "$feat" --embed_dim "$dim" \
    > "logs/${exp_code}.log" 2>&1
  status=$?
  echo "[$(date '+%F %T')] finished ${exp_code} with exit code ${status} (log: logs/${exp_code}.log)"
  if [ "$i" -lt $((n - 1)) ]; then
    echo "[$(date '+%F %T')] resting ${REST_SECONDS} s"
    sleep "$REST_SECONDS"
  fi
done
echo "[$(date '+%F %T')] all experiments done"
