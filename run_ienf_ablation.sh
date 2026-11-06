#!/usr/bin/env bash
# IENF ablation ladder: which change between the earlier 2-class pipeline (ienf_2cls_uni_p128_CLAM_B8)
# and the new one (ienf_isfn_2cls_uni_p128_CLAM_B8) costs specificity? All rungs: 128 px, CLAM-SB small,
# B = 8, the rebalanced folds, seeds 1-3. Each rung adds one change to the previous one:
#   A  earlier preprocessing (original images, buffer 5 px, >= 5% ROI), Control/DM/FAP only  -> fold effect
#   B  A + iSFN slides (patched the earlier way)                                              -> iSFN effect
#   C  B with buffer 20 px and >= 10% ROI (original images, no mask)                         -> patching effect
#   D  C + pixels outside ROI + 20 px set to 0 (original intensities)                         -> mask effect
#   E  D + intensity normalization = the new pipeline (seed 1 already exists)                -> normalization effect
# C, D and E use identical patch coordinates (computed from the ROI files), so they differ only in pixel values.
#
# Usage (repo root), e.g. inside tmux:
#   UNI_CKPT_PATH=~/projects/UNI/assets/ckpts/uni/pytorch_model.bin \
#   bash run_ienf_ablation.sh 2>&1 | tee logs/run_ienf_ablation.log

cd "$(dirname "$0")"
mkdir -p logs
: "${UNI_CKPT_PATH:?set UNI_CKPT_PATH to the UNI v1 pytorch_model.bin}"
export UNI_CKPT_PATH

GPU=0
REST_SECONDS=60
PYTHON=${PYTHON:-/var/tmp/shouteng_conda/envs/clam/bin/python}
DATA_ROOT=/var/tmp/shouteng_data/CLAM_DATA
SOURCE=/data/datasets/neuroimages/20250214
ORIG=$SOURCE/Image/Original
AB=/var/tmp/shouteng_data/IENF_CLAM/ablation
IENF_PREP=$HOME/projects/IENF_CLAM_Preprocessing
SEEDS=${SEEDS:-1 2 3}
log() { echo "[$(date '+%F %T')] $*"; }

extract() {  # extract <variant_dir> <slide_dir> <feat_name>
  if [ "$(ls "$DATA_ROOT/$3/pt_files" 2>/dev/null | wc -l)" -ge "$(($(wc -l < "$1/slides_with_patches.csv") - 1))" ]; then
    log "features $3 already complete, skipping"; return
  fi
  log "extract $1 -> $3"
  CUDA_VISIBLE_DEVICES=$GPU "$PYTHON" "$IENF_PREP/scripts/extract_features_ienf.py" --clam_root "$PWD" \
    --data_h5_dir "$1" --data_slide_dir "$2" --slide_ext .tif \
    --csv_path "$1/slides_with_patches.csv" --feat_dir "$DATA_ROOT/$3" \
    --model_name uni_v1 --batch_size 256 > "logs/$3.log" 2>&1
  log "extract $3 exit code $? ($(ls "$DATA_ROOT/$3/pt_files" | wc -l) pt files)"
}

# ---------- preprocessing (CPU) ----------
mkdir -p "$AB"
# B: the 46 iSFN slides, patched the earlier way (buffer 5 px, >= 5% ROI)
"$PYTHON" -c "import pandas as pd; d = pd.read_csv('dataset_csv/ienf_isfn_4sub.csv'); \
d[d.label == 'iSFN'][['slide_id']].to_csv('$AB/isfn_slides.csv', index=False)"
[ -f "$AB/prep_isfn_old/p128_s128_bufdown5/slides_with_patches.csv" ] || \
  "$PYTHON" "$IENF_PREP/scripts/create_patches_roi.py" \
    --source "$SOURCE" \
    --csv "$AB/isfn_slides.csv" \
    --save_dir "$AB/prep_isfn_old" \
    --clam_root "$PWD" \
    --patch_size 128 --overlap 0 \
    --buffer_down 5 --min_roi_overlap 0.05 > logs/ienf_ab_patch_isfn_old.log 2>&1
# C: original images, buffer 20 px, >= 10% ROI
[ -f "$AB/prep_orig/p128_s128_bufdown20/slides_with_patches.csv" ] || \
  "$PYTHON" "$IENF_PREP/scripts/create_patches_roi.py" \
    --source "$SOURCE" \
    --save_dir "$AB/prep_orig" \
    --clam_root "$PWD" \
    --patch_size 128 --overlap 0 \
    --buffer_down 20 --min_roi_overlap 0.10 > logs/ienf_ab_patch_orig.log 2>&1
# D: original intensities, masked outside ROI + 20 px; the patch coordinates are the same as in C
[ "$(ls "$AB/images_maskonly_bufdown20/images" 2>/dev/null | wc -l)" -ge 439 ] || \
  "$PYTHON" "$IENF_PREP/scripts/normalize_images.py" \
    --source "$SOURCE" \
    --batches "$SOURCE/labels/canonical_original_images.csv" \
    --out_dir "$AB/images_maskonly_bufdown20" \
    --buffer_down 20 --no_norm --workers 14 > logs/ienf_ab_maskonly.log 2>&1
log "preprocessing done"

# ---------- features (GPU) ----------
extract "$AB/prep_isfn_old/p128_s128_bufdown5" "$ORIG" ienf_ab_isfn_oldprep_uni_p128_s128
extract "$AB/prep_orig/p128_s128_bufdown20" "$ORIG" ienf_ab_orig_buf20_uni_p128_s128
extract "$AB/prep_orig/p128_s128_bufdown20" "$AB/images_maskonly_bufdown20/images" ienf_ab_maskonly_buf20_uni_p128_s128
# B's feature folder = the earlier 387 Control/DM/FAP features + the 46 iSFN ones (symlinks)
COMB=$DATA_ROOT/ienf_ab_oldprep_all_uni_p128_s128
mkdir -p "$COMB/pt_files"
ln -sf "$DATA_ROOT"/ienf_uni_p128_s128/pt_files/*.pt "$DATA_ROOT"/ienf_ab_isfn_oldprep_uni_p128_s128/pt_files/*.pt "$COMB/pt_files/"
log "combined folder: $(ls "$COMB/pt_files" | wc -l) pt files"

# ---------- training ----------
COMMON=(
  --k 5
  --model_type clam_sb --model_size small --B 8
  --drop_out 0.25 --lr 2e-4 --reg 1e-5 --max_epochs 200 --early_stopping
  --weighted_sample --bag_loss ce --inst_loss svm
  --data_root_dir "$DATA_ROOT" --embed_dim 1024 --log_data
)
# rung | task | split_dir | feature_dir | exp_code
RUNGS=(
  "A ienf_2cls ienf_kfold5_rebalanced_noisfn ienf_uni_p128_s128 ienf_ab_A_oldprep_noisfn_p128_CLAM_B8"
  "B ienf_isfn_2cls ienf_isfn_kfold5_rebalanced ienf_ab_oldprep_all_uni_p128_s128 ienf_ab_B_oldprep_p128_CLAM_B8"
  "C ienf_isfn_2cls ienf_isfn_kfold5_rebalanced ienf_ab_orig_buf20_uni_p128_s128 ienf_ab_C_buf20_p128_CLAM_B8"
  "D ienf_isfn_2cls ienf_isfn_kfold5_rebalanced ienf_ab_maskonly_buf20_uni_p128_s128 ienf_ab_D_mask_p128_CLAM_B8"
  "E ienf_isfn_2cls ienf_isfn_kfold5_rebalanced ienf_norm_uni_p128_s128 ienf_isfn_2cls_uni_p128_CLAM_B8"
)
for seed in $SEEDS; do
  for r in "${RUNGS[@]}"; do
    read -r rung task split feat exp_code <<< "$r"
    if [ -f "results/${exp_code}_s${seed}/summary.csv" ]; then
      log "skip ${rung} ${exp_code} s${seed} (summary.csv exists)"; continue
    fi
    log "start ${rung} ${exp_code} s${seed}"
    CUDA_VISIBLE_DEVICES=$GPU "$PYTHON" main.py "${COMMON[@]}" \
      --task "$task" --split_dir "$split" --feature_dir "$feat" \
      --seed "$seed" --exp_code "$exp_code" > "logs/${exp_code}_s${seed}.log" 2>&1
    log "finished ${rung} ${exp_code} s${seed} with exit code $?"
    sleep "$REST_SECONDS"
  done
done
log "all ablation runs done"
