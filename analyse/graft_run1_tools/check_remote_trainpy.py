# -*- coding: utf-8 -*-
"""Check remote train.py deploy-section ordering (fix present?) and run_gpu1.sh sanity."""
import json
import os
import paramiko

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
with open(os.path.join(ROOT, ".vscode", "sftp.json"), "r", encoding="utf-8") as f:
    cfg = json.load(f)

REMOTE = r"""
set +e
echo "===== train.py deploy-section relevant lines (with line numbers) ====="
grep -n "Deploy checkpoint saved\|strict-loaded\|count_flops\|torch.save(model.state_dict(), deploy_ckpt)\|count_deploy_params\|verify_deploy_equivalence\|assert max_err" \
  /home/yqwang/project/LS-Rep_BCD_PLUG/models/scripts/train.py 2>&1
echo ""
echo "===== run_gpu1.sh (first 50 lines) ====="
sed -n '1,50p' /home/yqwang/project/LS-Rep_BCD_PLUG/train_scripts/GRAFT-PLUG/Run1/run_gpu1.sh 2>&1
echo ""
echo "===== line endings check (CRLF would show ^M) ====="
file /home/yqwang/project/LS-Rep_BCD_PLUG/train_scripts/GRAFT-PLUG/Run1/run_gpu1.sh 2>&1
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
