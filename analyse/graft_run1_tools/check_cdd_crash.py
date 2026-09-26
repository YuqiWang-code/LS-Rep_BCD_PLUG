# -*- coding: utf-8 -*-
"""Investigate graft_cdd disappearance + CDD R1-D stall."""
import json
import os
import paramiko

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
with open(os.path.join(ROOT, ".vscode", "sftp.json"), "r", encoding="utf-8") as f:
    cfg = json.load(f)

LOG = "/storage/yqwang/LS-Rep_BCD/saved_models/GRAFT-PLUG/Run1/R1-D/CDD-CD-256/train_log.txt"
REMOTE = r"""
set +e
echo "===== tmux ls ====="
tmux ls 2>&1
echo ""
echo "===== CDD R1-D log tail (last 40 lines) ====="
tail -n 40 "LOG" 2>&1
echo ""
echo "===== CDD R1-D: error markers ====="
grep -inE "traceback|cuda out of memory|out of memory|oom|killed|error|assert" "LOG" 2>&1 | tail -n 20
echo ""
echo "===== CDD R1-D dir listing ====="
ls -la /storage/yqwang/LS-Rep_BCD/saved_models/GRAFT-PLUG/Run1/R1-D/CDD-CD-256/ 2>&1
echo ""
echo "===== any train.py processes at all ====="
ps -eo pid,etime,cmd | grep -E "[m]odels.scripts.train" 2>&1
""".replace("LOG", LOG)

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect(cfg["host"], int(cfg.get("port", 22)), cfg["username"], cfg["password"], timeout=30)
_i, o, e = c.exec_command(REMOTE, timeout=60)
print(o.read().decode("utf-8", "replace"))
err = e.read().decode("utf-8", "replace")
if err.strip():
    print("STDERR:", err)
c.close()
