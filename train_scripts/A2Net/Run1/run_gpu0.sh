#!/bin/bash
# A2Net Run1 — clean baseline: LEVIR -> SYSU -> WHU -> CDD (GPU 0)
set -euo pipefail

GPU_ID="${1:-0}"
PROJ="/home/yqwang/project/LS-Rep_BCD_PLUG"
DATA_BASE="/data/CD"
SAVE_BASE="$PROJ/saved_models/A2Net/Run1"

cd "$PROJ"

run_one() {
  local DS="$1"; local DS_FOLDER="$2"
  local SAVE_DIR="$SAVE_BASE/$DS_FOLDER"; local LOG_FILE="$SAVE_DIR/train_log.txt"
  if grep -q 'Test Results (Best Model)' "$LOG_FILE" 2>/dev/null; then
    echo "[skip] $DS already complete"; return 0
  fi
  mkdir -p "$SAVE_DIR"
  python models/scripts/train.py \
    --dataset_name "$DS" --data_root "$DATA_BASE/$DS_FOLDER" \
    --model_type L0 --batch_size 32 --max_steps 40000 --lr 5e-4 \
    --gpu_id "$GPU_ID" --seed 2333 \
    --save_dir "$SAVE_DIR"
}

run_one LEVIR LEVIR-CD-256
run_one SYSU SYSU-CD-256
run_one WHU WHU-CD-256
run_one CDD CDD-CD-256
