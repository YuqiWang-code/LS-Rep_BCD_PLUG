#!/bin/bash
# GRAFT-PLUG Run1 — Phase A 机制消融矩阵（SYSU + CDD，seed 2333），物理 GPU 1
# R0 / R1-T / R1-D / R1-U / R1-F / R1-L
set -euo pipefail

GPU_ID="${1:-1}"
PROJ="/home/yqwang/project/LS-Rep_BCD_PLUG"
DATA_BASE="/data/CD"
SAVE_BASE="/storage/yqwang/LS-Rep_BCD/saved_models/GRAFT-PLUG/Run1"

cd "$PROJ"

run_one() {
  local EXP="$1"; local DS="$2"; local DS_FOLDER="$3"; shift 3
  local SAVE_DIR="$SAVE_BASE/$EXP/$DS_FOLDER"; local LOG_FILE="$SAVE_DIR/train_log.txt"
  if grep -q 'Test Results (Best Model)' "$LOG_FILE" 2>/dev/null; then
    echo "[skip] $EXP/$DS already complete"; return 0
  fi
  mkdir -p "$SAVE_DIR"
  python models/scripts/train.py \
    --experiment "$EXP" --dataset_name "$DS" --data_root "$DATA_BASE/$DS_FOLDER" \
    --model_type L0 --batch_size 64 --max_steps 40000 --lr 5e-4 \
    --gpu_id "$GPU_ID" --seed 2333 \
    --save_dir "$SAVE_DIR" "$@"
}

# Phase A — SYSU-CD-256
run_one R0   SYSU SYSU-CD-256
run_one R1-T SYSU SYSU-CD-256 --graft_ablation T
run_one R1-D SYSU SYSU-CD-256 --graft_ablation D
run_one R1-U SYSU SYSU-CD-256 --graft_ablation U
run_one R1-F SYSU SYSU-CD-256 --graft_ablation F
run_one R1-L SYSU SYSU-CD-256 --graft_ablation L

# Phase A — CDD-CD-256
run_one R0   CDD CDD-CD-256
run_one R1-T CDD CDD-CD-256 --graft_ablation T
run_one R1-D CDD CDD-CD-256 --graft_ablation D
run_one R1-U CDD CDD-CD-256 --graft_ablation U
run_one R1-F CDD CDD-CD-256 --graft_ablation F
run_one R1-L CDD CDD-CD-256 --graft_ablation L
