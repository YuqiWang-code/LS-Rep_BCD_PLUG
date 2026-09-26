# -*- coding: utf-8 -*-
"""Verify remote source files carry the audit fixes (RNG / warmup / KGR / local_dim / dropout / deploy)."""
import json
import os
import paramiko

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
with open(os.path.join(ROOT, ".vscode", "sftp.json"), "r", encoding="utf-8") as f:
    cfg = json.load(f)

P = "/home/yqwang/project/LS-Rep_BCD_PLUG"
REMOTE = r"""
set +e
echo "===== [1] train.py: DataLoader RNG + warmup ====="
grep -n "generator=\|worker_init_fn\|warmup\|repr_warmup\|independent\|manual_seed" %P%/models/scripts/train.py 2>&1 | head -n 30

echo ""
echo "===== [2] cd_dataset.py: worker_init_fn / generator ====="
grep -n "worker_init_fn\|generator\|def seed_worker\|manual_seed" %P%/models/datasets/cd_dataset.py 2>&1 | head -n 30

echo ""
echo "===== [3] graft_plug.py: local_dim / dropout / class-balance ====="
grep -n "local_dim\|dropout\|class_balance\|cls_balance\|changed\|unchanged\|denom\|per-class\|0.5 \*" %P%/models/plugins/graft_plug.py 2>&1 | head -n 40

echo ""
echo "===== [4] a2net.py: switch_to_deploy / graft mount ====="
grep -n "switch_to_deploy\|use_graft\|self.graft\|graft_cfg" %P%/models/a2net.py 2>&1 | head -n 30

echo ""
echo "===== [5] graft_plug.py: local_dim literal value (search 204/192/128) ====="
grep -n "204\|192\|128" %P%/models/plugins/graft_plug.py 2>&1 | head -n 20
""".replace("%P%", P)

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect(cfg["host"], int(cfg.get("port", 22)), cfg["username"], cfg["password"], timeout=30)
_i, o, e = c.exec_command(REMOTE, timeout=60)
print(o.read().decode("utf-8", "replace"))
err = e.read().decode("utf-8", "replace")
if err.strip():
    print("STDERR:", err)
c.close()
