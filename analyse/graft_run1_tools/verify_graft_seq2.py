# -*- coding: utf-8 -*-
"""Confirm run_sequential.sh process is alive & in sleep loop."""
import json
import os
import paramiko

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
with open(os.path.join(ROOT, ".vscode", "sftp.json"), "r", encoding="utf-8") as f:
    cfg = json.load(f)

REMOTE = r"""
set +e
echo "===== run_sequential.sh process ====="
ps -eo pid,ppid,etime,stat,cmd | grep -E "[r]un_sequential|[s]leep 300" 2>&1
echo ""
echo "===== graft_seq pane (full, raw) ====="
tmux capture-pane -t graft_seq -p -S -200 2>&1 | grep -v '^$' | tail -n 30
echo ""
echo "===== graft_seq window command (confirm it points to run_sequential) ====="
tmux list-panes -t graft_seq -F '#{pane_current_command} | #{pane_pid}' 2>&1
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
