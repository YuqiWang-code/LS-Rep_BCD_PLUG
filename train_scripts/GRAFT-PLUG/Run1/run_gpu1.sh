#!/bin/bash
# GRAFT-PLUG Run1 — Phase A 机制消融矩阵（seed 2333），物理 GPU 1
# R0 / R1-T / R1-D / R1-U / R1-F / R1-L
# 用法：run_gpu1.sh <gpu_id> <group>   group ∈ {sysu, cdd, levir, whu, all}
#   拆两路并行占满 GPU 1：  tmux 各跑 run_gpu1.sh 1 <group_a> 与 run_gpu1.sh 1 <group_b>
set -euo pipefail

GPU_ID="${1:-1}"
GROUP="${2:-all}"
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

run_group() {
  local DS="$1"; local DS_FOLDER="$2"
  run_one R0   "$DS" "$DS_FOLDER"
  run_one R1-T "$DS" "$DS_FOLDER" --graft_ablation T
  run_one R1-D "$DS" "$DS_FOLDER" --graft_ablation D
  run_one R1-U "$DS" "$DS_FOLDER" --graft_ablation U
  run_one R1-F "$DS" "$DS_FOLDER" --graft_ablation F
  run_one R1-L "$DS" "$DS_FOLDER" --graft_ablation L
}

if [ "$GROUP" = "sysu" ] || [ "$GROUP" = "all" ]; then
  run_group SYSU SYSU-CD-256
fi
if [ "$GROUP" = "cdd" ] || [ "$GROUP" = "all" ]; then
  run_group CDD CDD-CD-256
fi
if [ "$GROUP" = "levir" ] || [ "$GROUP" = "all" ]; then
  run_group LEVIR LEVIR-CD-256
fi
if [ "$GROUP" = "whu" ] || [ "$GROUP" = "all" ]; then
  run_group WHU WHU-CD-256
fi
