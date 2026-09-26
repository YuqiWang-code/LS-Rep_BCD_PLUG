# -*- coding: utf-8 -*-
"""Check LEVIR/WHU dataset folders exist + current GPU/tmux snapshot."""
import json
import os
import paramiko

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
with open(os.path.join(ROOT, ".vscode", "sftp.json"), "r", encoding="utf-8") as f:
    cfg = json.load(f)

REMOTE = r"""
set +e
echo "===== /data/CD contents ====="
ls -la /data/CD/ 2>&1
echo ""
echo "===== LEVIR-CD-256 structure ====="
ls /data/CD/LEVIR-CD-256/ 2>&1
echo ""
echo "===== WHU-CD-256 structure ====="
ls /data/CD/WHU-CD-256/ 2>&1
echo ""
echo "===== LEVIR list counts ====="
wc -l /data/CD/LEVIR-CD-256/list/*.txt 2>&1
echo ""
echo "===== WHU list counts ====="
wc -l /data/CD/WHU-CD-256/list/*.txt 2>&1
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
