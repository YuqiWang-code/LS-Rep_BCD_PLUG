# -*- coding: utf-8 -*-
"""GRAFT-PLUG Run1 训练结束后：SFTP 下载 train_log.txt + bestF1 checkpoint + best_deploy 到本地备份。

用法: python docs/temporary/download_run1_results.py
本地备份根: saved_models/GRAFT-PLUG/Run1/（已被 .gitignore 排除）
只下载，不删除/不改动远程任何文件。
"""
import json
import os
import sys

import paramiko

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
with open(os.path.join(ROOT, ".vscode", "sftp.json"), "r", encoding="utf-8") as f:
    cfg = json.load(f)

REMOTE_BASE = "/storage/yqwang/LS-Rep_BCD/saved_models/GRAFT-PLUG/Run1"
LOCAL_BASE = os.path.join(ROOT, "saved_models", "GRAFT-PLUG", "Run1")
EXPS = ["R0", "R1-T", "R1-D", "R1-U", "R1-F", "R1-L"]
DSS = ["SYSU-CD-256", "CDD-CD-256"]


def main():
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(cfg["host"], int(cfg.get("port", 22)), cfg["username"], cfg["password"], timeout=30)
    sftp = c.open_sftp()

    n_log, n_pth = 0, 0
    for exp in EXPS:
        for ds in DSS:
            rd = f"{REMOTE_BASE}/{exp}/{ds}"
            ld = os.path.join(LOCAL_BASE, exp, ds)
            os.makedirs(ld, exist_ok=True)
            try:
                names = sftp.listdir(rd)
            except IOError:
                print(f"[skip]  remote dir missing: {rd}")
                continue
            for name in names:
                if name == "train_log.txt":
                    sftp.get(f"{rd}/{name}", os.path.join(ld, name)); n_log += 1
                    print(f"[log]  {exp}/{ds}/train_log.txt")
                elif name.startswith("bestF1=") and name.endswith("_model.pth"):
                    sftp.get(f"{rd}/{name}", os.path.join(ld, name)); n_pth += 1
                    print(f"[pth]  {exp}/{ds}/{name}")
                elif name == "best_deploy_model.pth":
                    sftp.get(f"{rd}/{name}", os.path.join(ld, name)); n_pth += 1
                    print(f"[pth]  {exp}/{ds}/best_deploy_model.pth")

    sftp.close(); c.close()
    print(f"\nDONE: {n_log} logs, {n_pth} checkpoints -> {LOCAL_BASE}")


if __name__ == "__main__":
    main()
