#!/bin/bash
# GRAFT-PLUG Run1 — 顺序单跑（GPU 1 单路，避免 R1-D 起两路并行 OOM）
# 用法：run_sequential.sh <gpu_id>
#   等待当前 graft_sysu / graft_cdd 会话跑完后，顺序执行 CDD → LEVIR → WHU 三个数据集
set -euo pipefail

GPU_ID="${1:-1}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

for s in graft_sysu graft_cdd; do
  while tmux has-session -t "$s" 2>/dev/null; do
    echo "[wait] 会话 $s 仍在运行 ... $(date '+%F %T')"
    sleep 300
  done
  echo "[done] 会话 $s 已结束"
done

echo "[go] 开始顺序单跑：CDD → LEVIR → WHU ... $(date '+%F %T')"
bash "$SCRIPT_DIR/run_gpu1.sh" "$GPU_ID" cdd
bash "$SCRIPT_DIR/run_gpu1.sh" "$GPU_ID" levir
bash "$SCRIPT_DIR/run_gpu1.sh" "$GPU_ID" whu
echo "[all-done] 全部数据集跑完 ... $(date '+%F %T')"
