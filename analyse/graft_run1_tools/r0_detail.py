# -*- coding: utf-8 -*-
"""Extract R0 detail: Test Results block, deploy checkpoint files, strict-loaded marker."""
import json
import os
import paramiko

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
with open(os.path.join(ROOT, ".vscode", "sftp.json"), "r", encoding="utf-8") as f:
    cfg = json.load(f)

BASE = "/storage/yqwang/LS-Rep_BCD/saved_models/GRAFT-PLUG/Run1"
REMOTE = r"""
set +e
for exp in R0; do
  for ds in SYSU-CD-256 CDD-CD-256; do
    d=""" + BASE + r"""/$exp/$ds
    log="$d/train_log.txt"
    echo "########## $exp / $ds ##########"
    echo "--- Test Results block ---"
    grep -A11 -E "^Test Results \(Best Model\)$" "$log" 2>/dev/null
    echo "--- strict-loaded marker count ---"
    grep -c "Deploy checkpoint saved & strict-loaded" "$log" 2>/dev/null
    echo "--- dir listing (checkpoints) ---"
    ls -la "$d" 2>/dev/null | grep -E "bestF1|best_deploy|train_log"
  done
done
"""

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect(cfg["host"], int(cfg.get("port", 22)), cfg["username"], cfg["password"], timeout=30)
_i, o, e = c.exec_command(REMOTE, timeout=120)
print(o.read().decode("utf-8", "replace"))
err = e.read().decode("utf-8", "replace")
if err.strip():
    print("STDERR:", err)
c.close()
