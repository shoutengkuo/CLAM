#!/usr/bin/env bash
# UNI feature extraction for the IENF ROI-patching variants (patch 224 and 128, no overlap).
# Patches were made with ~/projects/IENF_CLAM_Preprocessing/scripts/create_patches_roi.py
# (log: logs/ienf_patching_224_128.log). Every patch is resized to 224 px by CLAM's UNI transform.
# Features land in $DATA_ROOT/<feature_dir>/{pt_files,h5_files}, ready for main.py --feature_dir.
#
# Usage (repo root, clam env active), e.g. inside tmux:
#   UNI_CKPT_PATH=~/projects/UNI/assets/ckpts/uni/pytorch_model.bin bash run_ienf_uni_features.sh 2>&1 | tee logs/run_ienf_uni_features.log

cd "$(dirname "$0")"
mkdir -p logs
: "${UNI_CKPT_PATH:?set UNI_CKPT_PATH to the UNI v1 pytorch_model.bin}"
export UNI_CKPT_PATH

GPU=0
PYTHON=${PYTHON:-/var/tmp/shouteng_conda/envs/clam/bin/python}
PREP=/var/tmp/shouteng_data/IENF_CLAM/clam_prep
DATA_ROOT=/var/tmp/shouteng_data/CLAM_DATA
SLIDES=/data/datasets/neuroimages/20250214/Image/Original
IENF_PREP=$HOME/projects/IENF_CLAM_Preprocessing

for variant in p224_s224_bufdown5 p128_s128_bufdown5; do
  feat=ienf_uni_${variant%_bufdown5}
  echo "[$(date '+%F %T')] start ${variant} -> ${DATA_ROOT}/${feat}"
  CUDA_VISIBLE_DEVICES=$GPU "$PYTHON" "$IENF_PREP/scripts/extract_features_ienf.py" --clam_root "$PWD" \
    --data_h5_dir "$PREP/$variant" --data_slide_dir "$SLIDES" --slide_ext .tif \
    --csv_path "$PREP/$variant/slides_with_patches.csv" --feat_dir "$DATA_ROOT/$feat" \
    --model_name uni_v1 --batch_size 64 > "logs/ienf_uni_${variant}.log" 2>&1
  rc=$?
  echo "[$(date '+%F %T')] finished ${variant} with exit code ${rc} ($(ls "$DATA_ROOT/$feat/pt_files" 2>/dev/null | wc -l) pt files)"
done
