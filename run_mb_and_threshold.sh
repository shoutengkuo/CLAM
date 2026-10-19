#!/usr/bin/env bash
# Experiment 1: CLAM-MB (model_type clam_mb), model_size small, B = 32, for ResNet-50, UNI and CONCH.
#               Everything else identical to the existing CLAM-SB small B = 32 runs (same splits, seed, hyperparameters).
# Experiment 2: decision-threshold analysis. Validation-set predictions are regenerated from the saved
#               checkpoints with eval.py (read-only use of results/ and splits/), thresholds are chosen on
#               validation and applied to test by threshold_analysis.py.
# Steps run one after another, with a cool-down pause after each training run.
#
# Usage (from the repo root, with the clam env active), e.g. inside tmux:
#   bash run_mb_and_threshold.sh 2>&1 | tee logs/run_mb_and_threshold.log

cd "$(dirname "$0")"
mkdir -p logs

REST_SECONDS=600   # pause after each training run (10 min)
GPU=0
DATA_ROOT=/var/tmp/shouteng_data/CLAM_DATA
log() { echo "[$(date '+%F %T')] $*"; }

# ---------------------------------------------------------------- experiment 1: CLAM-MB training
COMMON="--drop_out 0.25 --early_stopping --lr 2e-4 --k 10 --weighted_sample --bag_loss ce --inst_loss svm \
--task camelyon_plus_2cls --model_type clam_mb --model_size small --log_data --data_root_dir $DATA_ROOT --B 32"

# name | feature_dir | embed_dim
ENCODERS=(
  "resnet50 camelyon_plus_resnet50 1024"
  "uni camelyon_plus_uni 1024"
  "conch camelyon_plus_conch 512"
)

for i in "${!ENCODERS[@]}"; do
  read -r name feat dim <<< "${ENCODERS[$i]}"
  exp_code=camelyon_plus_2cls_${name}_CLAM_MB_B32
  log "experiment 1: start ${exp_code} ($((i + 1))/${#ENCODERS[@]})"
  CUDA_VISIBLE_DEVICES=$GPU python main.py $COMMON --exp_code "$exp_code" --feature_dir "$feat" --embed_dim "$dim" \
    > "logs/${exp_code}.log" 2>&1
  log "experiment 1: finished ${exp_code} with exit code $? (log: logs/${exp_code}.log)"
  log "resting ${REST_SECONDS} s"
  sleep "$REST_SECONDS"
done

# ---------------------------------------------------------------- experiment 2: threshold analysis
# run dir | model_type | model_size | feature_dir | embed_dim
RUNS=(
  "camelyon_plus_2cls_resnet50_CLAM_B32_s1 clam_sb small camelyon_plus_resnet50 1024"
  "camelyon_plus_2cls_uni_CLAM_B32_s1 clam_sb small camelyon_plus_uni 1024"
  "camelyon_plus_2cls_conch_CLAM_B32_s1 clam_sb small camelyon_plus_conch 512"
  "camelyon_plus_2cls_resnet50_CLAM_B32_big_s1 clam_sb big camelyon_plus_resnet50 1024"
  "camelyon_plus_2cls_uni_CLAM_B32_big_s1 clam_sb big camelyon_plus_uni 1024"
  "camelyon_plus_2cls_conch_CLAM_B32_big_s1 clam_sb big camelyon_plus_conch 512"
  "camelyon_plus_2cls_resnet50_CLAM_MB_B32_s1 clam_mb small camelyon_plus_resnet50 1024"
  "camelyon_plus_2cls_uni_CLAM_MB_B32_s1 clam_mb small camelyon_plus_uni 1024"
  "camelyon_plus_2cls_conch_CLAM_MB_B32_s1 clam_mb small camelyon_plus_conch 512"
)

done_runs=()
for entry in "${RUNS[@]}"; do
  read -r run mtype msize feat dim <<< "$entry"
  if [ ! -f "results/${run}/summary.csv" ]; then
    log "experiment 2: skip ${run} (no finished training run found)"
    continue
  fi
  log "experiment 2: validation predictions for ${run}"
  CUDA_VISIBLE_DEVICES=$GPU python eval.py --k 10 --models_exp_code "$run" --save_exp_code "${run}_val" \
    --task camelyon_plus_2cls --model_type "$mtype" --model_size "$msize" --split val --results_dir results \
    --data_root_dir "$DATA_ROOT" --feature_dir "$feat" --embed_dim "$dim" > "logs/eval_${run}_val.log" 2>&1
  status=$?
  log "experiment 2: eval finished with exit code ${status}"
  [ "$status" -eq 0 ] && done_runs+=("$run")
done

log "experiment 2: threshold analysis on ${#done_runs[@]} runs"
python threshold_analysis.py "${done_runs[@]}" > logs/threshold_analysis.log 2>&1
log "experiment 2: finished with exit code $? (results: results/threshold_analysis/, log: logs/threshold_analysis.log)"
log "all done"
