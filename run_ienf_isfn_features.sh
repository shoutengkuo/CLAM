#!/usr/bin/env bash
# UNI feature extraction for IENF with iSFN, new preprocessing: intensity-normalized images with
# everything outside ROI + 20 px buffer set to 0 (IENF_CLAM_Preprocessing/scripts/normalize_images.py),
# patches 128/64/32/16 px, step = patch size, kept when ROI + buffer covers >= 10% of the patch
# (scripts/create_patches_roi.py). Every patch is resized to 224 px by CLAM's UNI transform.
# Features land in $DATA_ROOT/ienf_norm_uni_p{ps}_s{ps}/{pt_files,h5_files}, ready for main.py --feature_dir.
#
# Usage (repo root), e.g. inside tmux:
#   UNI_CKPT_PATH=~/projects/UNI/assets/ckpts/uni/pytorch_model.bin \
#   bash run_ienf_isfn_features.sh 2>&1 | tee logs/run_ienf_isfn_features.log

cd "$(dirname "$0")"
mkdir -p logs
: "${UNI_CKPT_PATH:?set UNI_CKPT_PATH to the UNI v1 pytorch_model.bin}"
export UNI_CKPT_PATH

GPU=0
PYTHON=${PYTHON:-/var/tmp/shouteng_conda/envs/clam/bin/python}
PREP=/var/tmp/shouteng_data/IENF_CLAM/clam_prep_norm
SLIDES=/var/tmp/shouteng_data/IENF_CLAM/images_norm_bufdown20/images
DATA_ROOT=${DATA_ROOT:-/var/tmp/shouteng_data/CLAM_DATA}
IENF_PREP=$HOME/projects/IENF_CLAM_Preprocessing

for ps in ${SIZES:-128 64 32 16}; do
  variant=p${ps}_s${ps}_bufdown20
  feat=ienf_norm_uni_p${ps}_s${ps}
  echo "[$(date '+%F %T')] start ${variant} -> ${DATA_ROOT}/${feat}"
  CUDA_VISIBLE_DEVICES=$GPU "$PYTHON" "$IENF_PREP/scripts/extract_features_ienf.py" --clam_root "$PWD" \
    --data_h5_dir "$PREP/$variant" --data_slide_dir "$SLIDES" --slide_ext .tif \
    --csv_path "${CSV:-$PREP/$variant/slides_with_patches.csv}" --feat_dir "$DATA_ROOT/$feat" \
    --model_name uni_v1 --batch_size 256 > "logs/ienf_norm_uni_${variant}.log" 2>&1
  rc=$?
  echo "[$(date '+%F %T')] finished ${variant} with exit code ${rc} ($(ls "$DATA_ROOT/$feat/pt_files" 2>/dev/null | wc -l) pt files)"
done
