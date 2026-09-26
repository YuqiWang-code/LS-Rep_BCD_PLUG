# -*- coding: utf-8 -*-
"""Verify graft_seq is waiting (not training) + no unexpected train processes."""
import json
import os
import paramiko

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
with open(os.path.join(ROOT, ".vscode", "sftp.json"), "r", encoding="utf-8") as f:
    cfg = json.load(f)

REMOTE = r"""
set +e
echo "===== graft_seq pane content ====="
tmux capture-pane -t graft_seq -p 2>&1 | tail -n 20
echo ""
echo "===== tmux ls ====="
tmux ls 2>&1
echo ""
echo "===== train.py processes (should be ONLY SYSU R1-D) ====="
ps -eo pid,etime,cmd | grep -E "[m]odels.scripts.train" 2>&1
"""

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect(cfg["host"], int(cfg.get("port", 22)), cfg["username"], cfg["password"], timeout=30)
_i, o, e = c.exec_command(REMOTE, timeout=60)
print(o.read().decode("utf-8", "replace"))
err = e.read().decode("utf-8", "replace")
if err.strip():
    print("STDERR:", err)
c.close()
