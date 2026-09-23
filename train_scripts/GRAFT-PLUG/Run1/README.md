# GRAFT-PLUG Run1 — Phase A 机制消融

GRAFT-PLUG（梯度路由式辅助融合教师外挂）第一阶段机制消融。方法设计见 `docs/temporary/LS-Rep_BCD_PLUG_GRAFT_PLUG_Research_Design_v2.md`。

## 协议（固定）

batch 64、40000 steps、seed 2333、lr 5e-4、wd 1e-4、主损失 BCE+Dice（4 尺度）、测试集当验证集（每 epoch 在测试集选 best）。外挂只在训练期挂载，`switch_to_deploy()` 后部署参数回到 2,913,094 / FLOPs 2.7475G，主路输出与 baseline 逐 bit 一致（max error < 1e-6）。

## 消融矩阵（Phase A：SYSU + CDD）

| ID | 配置 | 唯一变量 | `--graft_ablation` | task | repr | detach | kgr | local |
|---|---|---|---|---|---|---|---|---|
| R0 | 干净 baseline | — | （不传） | — | — | — | — | — |
| R1-T | task gradient only | 只有 GT 深度监督 | T | ✓ | ✗ | ✗ | ✗ | ✗ |
| R1-D | distill gradient only | 只有 global→local 自蒸馏 | D | ✓ | ✓ | ✓ | ✗ | ✗ |
| R1-U | full, uniform | 两梯度无 KGR | U | ✓ | ✓ | ✗ | ✗ | ✗ |
| R1-F | full GRAFT | + KGR | F | ✓ | ✓ | ✗ | ✓ | ✗ |
| R1-L | capacity-matched local tutor | 去掉跨 stage 全局推理 | L | ✓ | ✓ | ✗ | ✓ | ✓ |

## 实验分层

- **Phase A**（本轮，SYSU + CDD）：R0 / R1-T / R1-D / R1-U / R1-F / R1-L，seed 2333，验证各机制贡献。
- **Phase B**（待 Phase A 通过）：R0 / R1-F / R-AFD 四数据集一致性。
- **Phase C**（待 Phase B 通过）：R0 / R1-F 多种子（2333/3407/7777），报 mean±std。

## 脚本

```text
run_gpu1.sh   物理 GPU 1 顺序跑 12 组（skip 已完成）
```

## 判据

拆除后的部署模型，六指标（R/P/OA/F1/IoU/Kappa）相对同批 R0 方向一致上涨；单种子/单数据集只能 preliminary。
