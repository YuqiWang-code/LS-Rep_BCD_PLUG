#!/bin/bash
# PLUG Run1 — GPU 0 queue (matrix template)
# 每个实验一行 run_one <EXP> <DS> <DS_FOLDER> [extra args...]
set -euo pipefail

GPU_ID="${1:-0}"
PROJ="/home/yqwang/project/LS-Rep_BCD_PLUG"
DATA_BASE="/data/CD"
SAVE_BASE="$PROJ/saved_models/PLUG/Run1"

cd "$PROJ"

run_one() {
  local EXP="$1"; local DS="$2"; local DS_FOLDER="$3"; shift 3
  local SAVE_DIR="$SAVE_BASE/$EXP/$DS_FOLDER"; local LOG_FILE="$SAVE_DIR/train_log.txt"
  if grep -q 'Test Results (Best Model)' "$LOG_FILE" 2>/dev/null; then
    echo "[skip] $EXP/$DS already complete"; return 0
  fi
  mkdir -p "$SAVE_DIR"
  # 外挂模块实验在此追加 --plug_* 参数（模块实现后启用）
  python models/scripts/train.py \
    --experiment "$EXP" --dataset_name "$DS" --data_root "$DATA_BASE/$DS_FOLDER" \
    --model_type L0 --batch_size 64 --max_steps 40000 --lr 5e-4 \
    --gpu_id "$GPU_ID" --seed 2333 \
    --save_dir "$SAVE_DIR" "$@"
}

# 矩阵（外挂模块实现后按需扩展，每数据集两臂：baseline vs plug-in）
run_one R0 LEVIR LEVIR-CD-256
run_one R1 LEVIR LEVIR-CD-256
run_one R0 SYSU SYSU-CD-256
run_one R1 SYSU SYSU-CD-256
run_one R0 WHU WHU-CD-256
run_one R1 WHU WHU-CD-256
run_one R0 CDD CDD-CD-256
run_one R1 CDD CDD-CD-256
