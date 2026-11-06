#!/usr/bin/env bash
# IENF ablation follow-up: rung D (mask only, no intensity normalization) vs. rung E (mask + normalization)
# at 32 px (B = 32) and 16 px (B = 64), seeds 1-3, on the rebalanced folds. See results/ienf_ablation_report.md.
# The patch coordinates depend only on the ROI files, so D reuses E's patch files
# (clam_prep_norm/p{32,16}_s*_bufdown20) and reads the mask-only images written by run_ienf_ablation.sh.
#
# Two parts, which can run at the same time on one GPU (e.g. in two tmux windows):
#   PART=E bash run_ienf_ablation_small.sh 2>&1 | tee logs/run_ienf_ablation_small_E.log   # E seeds 2-3 (seed 1 exists)
#   UNI_CKPT_PATH=~/projects/UNI/assets/ckpts/uni/pytorch_model.bin \
#   PART=D bash run_ienf_ablation_small.sh 2>&1 | tee logs/run_ienf_ablation_small_D.log   # D features + seeds 1-3

cd "$(dirname "$0")"
mkdir -p logs
GPU=0
REST_SECONDS=60
PYTHON=${PYTHON:-/var/tmp/shouteng_conda/envs/clam/bin/python}
DATA_ROOT=/var/tmp/shouteng_data/CLAM_DATA
PREP=/var/tmp/shouteng_data/IENF_CLAM/clam_prep_norm
MASKONLY=/var/tmp/shouteng_data/IENF_CLAM/ablation/images_maskonly_bufdown20/images
IENF_PREP=$HOME/projects/IENF_CLAM_Preprocessing
PART=${PART:?set PART=D or PART=E}
log() { echo "[$(date '+%F %T')] $*"; }

COMMON=(
  --task ienf_isfn_2cls --split_dir ienf_isfn_kfold5_rebalanced --k 5
  --model_type clam_sb --model_size small
  --drop_out 0.25 --lr 2e-4 --reg 1e-5 --max_epochs 200 --early_stopping
  --weighted_sample --bag_loss ce --inst_loss svm
  --data_root_dir "$DATA_ROOT" --embed_dim 1024 --log_data
)

train() {  # train <feature_dir> <B> <exp_code> <seed>
  if [ -f "results/${3}_s${4}/summary.csv" ]; then log "skip ${3} s${4}"; return; fi
  log "start ${3} s${4}"
  CUDA_VISIBLE_DEVICES=$GPU "$PYTHON" main.py "${COMMON[@]}" \
    --feature_dir "$1" --B "$2" --seed "$4" --exp_code "$3" > "logs/${3}_s${4}.log" 2>&1
  log "finished ${3} s${4} with exit code $?"
  sleep "$REST_SECONDS"
}

for spec in "32 32" "16 64"; do
  read -r ps B <<< "$spec"
  if [ "$PART" = E ]; then
    for seed in 2 3; do
      train "ienf_norm_uni_p${ps}_s${ps}" "$B" "ienf_isfn_2cls_uni_p${ps}_CLAM_B${B}" "$seed"
    done
  else
    : "${UNI_CKPT_PATH:?set UNI_CKPT_PATH to the UNI v1 pytorch_model.bin}"
    export UNI_CKPT_PATH
    feat=ienf_ab_maskonly_buf20_uni_p${ps}_s${ps}
    variant=$PREP/p${ps}_s${ps}_bufdown20
    if [ "$(ls "$DATA_ROOT/$feat/pt_files" 2>/dev/null | wc -l)" -lt 439 ]; then
      log "extract ${variant} on mask-only images -> ${feat}"
      CUDA_VISIBLE_DEVICES=$GPU "$PYTHON" "$IENF_PREP/scripts/extract_features_ienf.py" --clam_root "$PWD" \
        --data_h5_dir "$variant" --data_slide_dir "$MASKONLY" --slide_ext .tif \
        --csv_path "$variant/slides_with_patches.csv" --feat_dir "$DATA_ROOT/$feat" \
        --model_name uni_v1 --batch_size 256 > "logs/${feat}.log" 2>&1
      log "extract ${feat} exit code $? ($(ls "$DATA_ROOT/$feat/pt_files" | wc -l) pt files)"
    fi
    for seed in 1 2 3; do
      train "$feat" "$B" "ienf_ab_D_mask_p${ps}_CLAM_B${B}" "$seed"
    done
  fi
done
log "part ${PART} done"
