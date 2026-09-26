# -*- coding: utf-8 -*-
"""Check uptime + GPU processes + any python/train leftovers."""
import json
import os
import paramiko

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
with open(os.path.join(ROOT, ".vscode", "sftp.json"), "r", encoding="utf-8") as f:
    cfg = json.load(f)

REMOTE = r"""
set +e
echo "===== uptime / who ====="
uptime
who 2>&1 | head -n 10
echo ""
echo "===== nvidia-smi (full, with processes) ====="
nvidia-smi 2>&1 | head -n 40
echo ""
echo "===== any python / train processes ====="
ps -eo pid,ppid,etime,cmd | grep -iE "[p]ython|[t]rain" | head -n 30
echo ""
echo "===== tmux server processes ====="
ps -eo pid,etime,cmd | grep -iE "[t]mux" | head -n 10
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
