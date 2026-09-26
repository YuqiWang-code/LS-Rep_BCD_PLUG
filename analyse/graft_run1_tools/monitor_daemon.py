# -*- coding: utf-8 -*-
"""GRAFT-PLUG Run1 每 4 小时自动监控 daemon（只读，不重启训练）。

每 4 小时：SSH 抓取 tmux / GPU1 / 12 实验状态，把紧凑摘要追加到
docs/temporary/monitor_history.log，并打印到 stdout（后台 job 输出）。
用法（后台）：python docs/temporary/monitor_daemon.py
停止：对后台 job 用 job_kill。
"""
import datetime
import json
import os
import sys
import time

import paramiko

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SFTP_JSON = os.path.join(ROOT, ".vscode", "sftp.json")
HIST = os.path.join(ROOT, "docs", "temporary", "monitor_history.log")
INTERVAL = 4 * 3600  # 4 小时

REMOTE = r"""
set +e
date '+%Y-%m-%d %H:%M:%S CST'
echo "TMUX: $(tmux ls 2>&1 | tr '\n' ' ')"
echo "GPU1: $(nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader 2>/dev/null | sed -n '2p')"
BASE=/storage/yqwang/LS-Rep_BCD/saved_models/GRAFT-PLUG/Run1
for exp in R0 R1-T R1-D R1-U R1-F R1-L; do
  for ds in SYSU-CD-256 CDD-CD-256 LEVIR-CD-256 WHU-CD-256; do
    log="$BASE/$exp/$ds/train_log.txt"
    if [ -f "$log" ] && grep -q "Deploy checkpoint saved" "$log" 2>/dev/null; then
      f1=$(grep -E "^F1:" "$log" | tail -1 | sed 's/^F1:[[:space:]]*//')
      echo "[DONE]  $exp/$ds  F1=$f1"
    elif [ -f "$log" ] && grep -q "Test Results (Best Model)" "$log" 2>/dev/null; then
      echo "[FINALIZING] $exp/$ds"
    elif [ -f "$log" ]; then
      last=$(grep -E "^Epoch " "$log" | tail -1)
      err=$(grep -icE "traceback|cuda out of memory|out of memory|killed" "$log" 2>/dev/null)
      stale=$(find "$log" -mmin +120 2>/dev/null)
      if [ -n "$stale" ]; then
        echo "[STALLED] $exp/$ds  $last  (log unchanged >2h, err_hits=$err)"
      else
        echo "[TRAIN] $exp/$ds  $last  (err_hits=$err)"
      fi
    else
      echo "[NOT_STARTED] $exp/$ds"
    fi
  done
done
"""


def check_once(cfg):
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(cfg["host"], int(cfg.get("port", 22)), cfg["username"], cfg["password"], timeout=30)
    _i, o, e = c.exec_command(REMOTE, timeout=120)
    out = o.read().decode("utf-8", "replace")
    c.close()
    return out.strip()


def main():
    with open(SFTP_JSON, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    os.makedirs(os.path.dirname(HIST), exist_ok=True)
    print(f"[daemon] start, interval={INTERVAL}s, history={HIST}", flush=True)
    while True:
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        try:
            summary = check_once(cfg)
        except Exception as e:  # noqa: BLE001
            summary = f"[CHECK-ERROR] {e}"
        block = f"\n===== {ts} =====\n{summary}\n"
        with open(HIST, "a", encoding="utf-8") as fh:
            fh.write(block)
        print(block, flush=True)
        time.sleep(INTERVAL)


if __name__ == "__main__":
    main()
