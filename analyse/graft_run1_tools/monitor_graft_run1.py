# -*- coding: utf-8 -*-
"""GRAFT-PLUG Run1 远程训练监控脚本（只读，不重启训练）。

连接信息从 .vscode/sftp.json 读取。检查 tmux 会话、GPU 占用、12 个实验日志状态。
用法: python docs/temporary/monitor_graft_run1.py
"""
import json
import os
import sys

import paramiko

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
SFTP_JSON = os.path.join(ROOT, ".vscode", "sftp.json")

SAVE_BASE = "/storage/yqwang/LS-Rep_BCD/saved_models/GRAFT-PLUG/Run1"
EXPS = ["R0", "R1-T", "R1-D", "R1-U", "R1-F", "R1-L"]
DSS = ["SYSU-CD-256", "CDD-CD-256"]

REMOTE = r"""
set +e
echo "===== HOST / TIME ====="
hostname; date '+%Y-%m-%d %H:%M:%S %Z'

echo ""
echo "===== TMUX SESSIONS ====="
tmux ls 2>&1

echo ""
echo "===== TRAIN PROCESSES ====="
ps -eo pid,etime,cmd | grep -E "[m]odels.scripts.train" | head -n 20

echo ""
echo "===== NVIDIA-SMI ====="
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader 2>&1

echo ""
echo "===== EXPERIMENT STATUS ====="
BASE=""" + SAVE_BASE + """
for exp in """ + " ".join(EXPS) + """; do
  for ds in """ + " ".join(DSS) + """; do
    d="$BASE/$exp/$ds"
    log="$d/train_log.txt"
    if [ ! -d "$d" ]; then
      echo "[$exp/$ds] NOT_STARTED (dir missing)"
      echo ""
      continue
    fi
    if [ ! -f "$log" ]; then
      echo "[$exp/$ds] DIR_EXISTS_NO_LOG"
      echo "    dir: $d"
      ls -la "$d" 2>&1
      echo ""
      continue
    fi
    if grep -q "Deploy checkpoint saved & strict-loaded" "$log" 2>/dev/null; then
      echo "[$exp/$ds] DONE"
      grep -A10 -E "^Test Results \(Best Model\)$" "$log" 2>/dev/null
      echo ""
    elif grep -q "Test Results (Best Model)" "$log" 2>/dev/null; then
      echo "[$exp/$ds] FINALIZING (Test Results present, deploy step not yet logged)"
      tail -n 6 "$log"
      echo ""
    else
      echo "[$exp/$ds] TRAINING"
      echo "  -- last 4 epoch lines --"
      grep -E "^Epoch " "$log" | tail -n 4
      echo "  -- errors in log (if any) --"
      grep -inE "traceback|cuda out of memory|out of memory|oom|killed|error:" "$log" | tail -n 5
      echo ""
    fi
  done
done
"""


def main():
    with open(SFTP_JSON, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    host = cfg["host"]
    port = int(cfg.get("port", 22))
    user = cfg["username"]
    pw = cfg["password"]

    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        c.connect(host, port, user, pw, timeout=30)
    except Exception as e:  # noqa: BLE001
        print(f"[FATAL] SSH connect failed: {e}", file=sys.stderr)
        sys.exit(2)

    _i, o, e = c.exec_command(REMOTE, timeout=120)
    out = o.read().decode("utf-8", "replace")
    err = e.read().decode("utf-8", "replace")
    rc = o.channel.recv_exit_status()
    c.close()

    print(out)
    if err.strip():
        print("----- STDERR -----", file=sys.stderr)
        print(err, file=sys.stderr)
    print(f"[remote exit code: {rc}]")


if __name__ == "__main__":
    main()
