# -*- coding: utf-8 -*-
"""Deploy Run1 launcher scripts to server (LF-normalized) + bash -n verify, then launch graft_seq."""
import json
import os
import paramiko

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
with open(os.path.join(ROOT, ".vscode", "sftp.json"), "r", encoding="utf-8") as f:
    cfg = json.load(f)

REMOTE_DIR = "/home/yqwang/project/LS-Rep_BCD_PLUG/train_scripts/GRAFT-PLUG/Run1"
LOCAL_DIR = os.path.join(ROOT, "train_scripts", "GRAFT-PLUG", "Run1")
FILES = ["run_gpu1.sh", "run_sequential.sh"]

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect(cfg["host"], int(cfg.get("port", 22)), cfg["username"], cfg["password"], timeout=30)
sftp = c.open_sftp()

for fn in FILES:
    with open(os.path.join(LOCAL_DIR, fn), "rb") as f:
        data = f.read()
    data = data.replace(b"\r\n", b"\n")  # 强制 LF
    remote = f"{REMOTE_DIR}/{fn}"
    with sftp.file(remote, "wb") as f:
        f.write(data)
    print(f"[uploaded] {fn} ({len(data)} bytes, LF-normalized)")
sftp.close()

# 校验：bash -n + file（确认无 CRLF）
verify = f"cd {REMOTE_DIR} && bash -n run_gpu1.sh && bash -n run_sequential.sh && echo 'BASH-N-OK' && file run_gpu1.sh run_sequential.sh"
_i, o, e = c.exec_command(verify, timeout=60)
vout = o.read().decode("utf-8", "replace")
verr = e.read().decode("utf-8", "replace")
print(vout)
if verr.strip():
    print("STDERR:", verr)
if "BASH-N-OK" not in vout:
    print("[ABORT] bash -n failed, NOT launching")
    c.close()
    raise SystemExit(1)

# 启动 graft_seq（等 graft_sysu/graft_cdd 跑完后顺序跑 CDD→LEVIR→WHU）
launch = f"tmux new-session -d -s graft_seq 'bash {REMOTE_DIR}/run_sequential.sh 1'"
_i, o, e = c.exec_command(launch, timeout=30)
print("[launch]", o.read().decode("utf-8", "replace").strip(), e.read().decode("utf-8", "replace").strip())

# 验证
_i, o, e = c.exec_command("tmux ls 2>&1", timeout=30)
print("---- tmux ls ----")
print(o.read().decode("utf-8", "replace"))
c.close()
