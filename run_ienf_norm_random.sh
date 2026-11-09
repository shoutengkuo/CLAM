#!/usr/bin/env bash
# IENF with iSFN on RANDOM folds (splits/ienf_isfn_kfold5_random; no earlier test results used):
# (1) the earlier ablation ladder, (2) normalization variants that separate within-image and between-image
# normalization and try other strategies, (3) mask-only vs. full normalization at 32 / 16 px.
# All runs: CLAM-SB small, UNI v1 features, seeds 1-3. Variants D-F share the patch coordinates
# (clam_prep_norm/p{ps}_s{ps}_bufdown20, computed from the ROI files) and differ only in pixel values / features.
#
#   A   earlier preprocessing (original images, buffer 5 px, >= 5% ROI), Control/DM/FAP only
#   B   A + iSFN slides
#   C   buffer 20 px, >= 10% ROI, original images, no mask
#   D   C + mask outside ROI + 20 px                      (= "none": no intensity correction)
#   E   D + within + between normalization               (= "full", the earlier new pipeline)
#   W   D + within-image only (tile profile + tile gains, own level kept)
#   G   D + between-image only (dark offset + ROI median -> target, no seam correction)
#   Gk  G on red and blue only, green (nerve-fibre channel) untouched
#   Bt  D + one gain per batch and channel (batch median -> dataset median)
#   F   D features aligned per batch in feature space (per-dim mean / sd)
#
# Parts (can run at the same time on one GPU, e.g. in two tmux windows):
#   UNI_CKPT_PATH=~/projects/UNI/assets/ckpts/uni/pytorch_model.bin \
#   PART=P128 bash run_ienf_norm_random.sh 2>&1 | tee logs/run_ienf_norm_random_P128.log
#
#   PART=SMALL bash run_ienf_norm_random.sh 2>&1 | tee logs/run_ienf_norm_random_SMALL.log
#
#   UNI_CKPT_PATH=~/projects/UNI/assets/ckpts/uni/pytorch_model.bin \
#   PART=P32 VARIANTS="W Bt" bash run_ienf_norm_random.sh 2>&1 | tee logs/run_ienf_norm_random_P32.log

cd "$(dirname "$0")"
mkdir -p logs
GPU=0
REST_SECONDS=60
PYTHON=${PYTHON:-/var/tmp/shouteng_conda/envs/clam/bin/python}
DATA_ROOT=/var/tmp/shouteng_data/CLAM_DATA
SOURCE=/data/datasets/neuroimages/20250214
AB=/var/tmp/shouteng_data/IENF_CLAM/ablation
PREP=/var/tmp/shouteng_data/IENF_CLAM/clam_prep_norm
IENF_PREP=$HOME/projects/IENF_CLAM_Preprocessing
SPLIT=ienf_isfn_kfold5_random
SEEDS=${SEEDS:-1 2 3}
PART=${PART:?set PART=P128, SMALL or P32}
log() { echo "[$(date '+%F %T')] $*"; }
B_OF() { case $1 in 128) echo 8;; 64) echo 16;; 32) echo 32;; 16) echo 64;; esac; }

# variant -> image mode (pixel variants) and feature folder name per patch size
declare -A MODE=([W]=within [G]=between [Gk]=between_keepG [Bt]=batch)
declare -A NAME=([D]=mask [E]=full [W]=within [G]=between [Gk]=betweenkeepG [Bt]=batchgain [F]=featalign)
feat_of() {  # feat_of <variant> <ps>
  case $1 in
    D) echo "ienf_ab_maskonly_buf20_uni_p$2_s$2";;
    E) echo "ienf_norm_uni_p$2_s$2";;
    F) echo "ienf_ab_maskonly_featalign_uni_p$2_s$2";;
    *) echo "ienf_ab_${NAME[$1]}_buf20_uni_p$2_s$2";;
  esac
}

COMMON=(
  --k 5
  --model_type clam_sb --model_size small
  --drop_out 0.25 --lr 2e-4 --reg 1e-5 --max_epochs 200 --early_stopping
  --weighted_sample --bag_loss ce --inst_loss svm
  --data_root_dir "$DATA_ROOT" --embed_dim 1024 --log_data
)
train() {  # train <task> <split_dir> <feature_dir> <B> <exp_code> <seed>
  if [ -f "results/${5}_s${6}/summary.csv" ]; then log "skip ${5} s${6}"; return; fi
  log "start ${5} s${6}"
  CUDA_VISIBLE_DEVICES=$GPU "$PYTHON" main.py "${COMMON[@]}" \
    --task "$1" --split_dir "$2" --feature_dir "$3" --B "$4" \
    --seed "$6" --exp_code "$5" > "logs/${5}_s${6}.log" 2>&1
  log "finished ${5} s${6} with exit code $?"
  sleep "$REST_SECONDS"
}
images() {  # images <variant>: write the variant's masked images (CPU)
  local mode=${MODE[$1]} dir=$AB/images_${MODE[$1]}_bufdown20
  if [ "$(ls "$dir/images" 2>/dev/null | wc -l)" -ge 439 ]; then log "images $mode exist"; return; fi
  local fit=(); [ "$mode" != within ] && fit=(--fit)
  log "normalize images: mode $mode -> $dir"
  "$PYTHON" "$IENF_PREP/scripts/normalize_images.py" \
    --source "$SOURCE" \
    --batches "$SOURCE/labels/canonical_original_images.csv" \
    --out_dir "$dir" \
    --buffer_down 20 --mode "$mode" "${fit[@]}" --workers 14 > "logs/ienf_norm_images_${mode}.log" 2>&1
  log "normalize $mode exit code $? ($(ls "$dir/images" | wc -l) images)"
}
extract() {  # extract <variant> <ps>
  local feat; feat=$(feat_of "$1" "$2")
  local variant=$PREP/p$2_s$2_bufdown20
  if [ "$1" = F ]; then
    [ -f "$DATA_ROOT/$feat/README.json" ] && { log "features $feat exist"; return; }
    log "feature-level batch alignment -> $feat"
    "$PYTHON" make_ienf_feature_batch_align.py \
      --src "$DATA_ROOT/$(feat_of D "$2")" \
      --out "$DATA_ROOT/$feat" > "logs/${feat}.log" 2>&1
    log "align $feat exit code $?"; return
  fi
  if [ "$(ls "$DATA_ROOT/$feat/pt_files" 2>/dev/null | wc -l)" -ge 439 ]; then log "features $feat exist"; return; fi
  log "extract $variant on ${MODE[$1]} images -> $feat"
  CUDA_VISIBLE_DEVICES=$GPU "$PYTHON" "$IENF_PREP/scripts/extract_features_ienf.py" --clam_root "$PWD" \
    --data_h5_dir "$variant" --data_slide_dir "$AB/images_${MODE[$1]}_bufdown20/images" --slide_ext .tif \
    --csv_path "$variant/slides_with_patches.csv" --feat_dir "$DATA_ROOT/$feat" \
    --model_name uni_v1 --batch_size 256 > "logs/${feat}.log" 2>&1
  log "extract $feat exit code $? ($(ls "$DATA_ROOT/$feat/pt_files" | wc -l) pt files)"
}
need_uni() { : "${UNI_CKPT_PATH:?set UNI_CKPT_PATH to the UNI v1 pytorch_model.bin}"; export UNI_CKPT_PATH; }

if [ "$PART" = P128 ]; then
  need_uni
  for v in W G Gk Bt; do images "$v"; done
  for v in W G Gk Bt F; do extract "$v" 128; done
  for seed in $SEEDS; do
    train ienf_2cls ienf_kfold5_random_noisfn ienf_uni_p128_s128 8 ienf_rf_A_oldprep_noisfn_p128_CLAM_B8 "$seed"
    train ienf_isfn_2cls $SPLIT ienf_ab_oldprep_all_uni_p128_s128 8 ienf_rf_B_oldprep_p128_CLAM_B8 "$seed"
    train ienf_isfn_2cls $SPLIT ienf_ab_orig_buf20_uni_p128_s128 8 ienf_rf_C_buf20_p128_CLAM_B8 "$seed"
    for v in D E W G Gk Bt F; do
      train ienf_isfn_2cls $SPLIT "$(feat_of $v 128)" 8 "ienf_rf_${v}_${NAME[$v]}_p128_CLAM_B8" "$seed"
    done
  done
elif [ "$PART" = SMALL ]; then
  for seed in $SEEDS; do
    for ps in 32 16; do
      for v in D E; do
        train ienf_isfn_2cls $SPLIT "$(feat_of $v $ps)" "$(B_OF $ps)" "ienf_rf_${v}_${NAME[$v]}_p${ps}_CLAM_B$(B_OF $ps)" "$seed"
      done
    done
  done
elif [ "$PART" = P32 ]; then
  need_uni
  for v in ${VARIANTS:?set VARIANTS, e.g. \"W Bt\"}; do
    [ -n "${MODE[$v]}" ] && images "$v"
    extract "$v" 32
    for seed in $SEEDS; do
      train ienf_isfn_2cls $SPLIT "$(feat_of $v 32)" 32 "ienf_rf_${v}_${NAME[$v]}_p32_CLAM_B32" "$seed"
    done
  done
fi
log "part ${PART} done"
