# GRAFT-PLUG Run1 — Phase A 机制消融

GRAFT-PLUG（梯度路由式辅助融合教师外挂）第一阶段机制消融。方法设计见 `docs/temporary/LS-Rep_BCD_PLUG_GRAFT_PLUG_Research_Design_v2.md`。

## 协议（固定）

batch 64、40000 nominal steps（epoch 取 ceil，实际 optimizer steps 略超，日志会记录 actual steps）、seed 2333、lr 5e-4、wd 1e-4、主损失 BCE+Dice（4 尺度）、测试集当验证集（每 epoch 在测试集选 best）。外挂只在训练期挂载，`switch_to_deploy()` 后部署参数回到 2,913,094 / FLOPs 2.7475G，拆除前后主路输出逐 bit 一致（max error < 1e-6，训练末尾自动校验）。

实验控制：R0 与各 R1-* 共享同一 DataLoader 随机流（独立 `generator` + 确定性 worker seed），representation KD 前 10% steps 线性 warmup。

## 消融矩阵（Phase A：SYSU + CDD）

| ID | 配置 | 唯一变量 | `--graft_ablation` | task | repr | detach | kgr | local |
|---|---|---|---|---|---|---|---|---|
| R0 | 干净 baseline | — | （不传） | — | — | — | — | — |
| R1-T | task gradient only | 只有 GT 深度监督 | T | ✓ | ✗ | ✗ | ✗ | ✗ |
| R1-D | distillation-only-to-backbone | teacher 受 GT，backbone 只收蒸馏梯度 | D | ✓ | ✓ | ✓ | ✗ | ✗ |
| R1-U | full, uniform | 两梯度无 KGR | U | ✓ | ✓ | ✗ | ✗ | ✗ |
| R1-F | full GRAFT | + KGR | F | ✓ | ✓ | ✗ | ✓ | ✗ |
| R1-L | capacity-matched local tutor | 去掉跨 stage 全局推理 | L | ✓ | ✓ | ✗ | ✓ | ✓ |

## 实验分层

- **Phase A**（本轮，SYSU + CDD）：R0 / R1-T / R1-D / R1-U / R1-F / R1-L，seed 2333，验证各机制贡献。
- **Phase B**（待 Phase A 通过）：R0 / R1-F / R-AFD 四数据集一致性。
- **Phase C**（待 Phase B 通过）：R0 / R1-F 多种子（2333/3407/7777），报 mean±std。

## 脚本

```text
run_gpu1.sh <gpu_id> <group>     group ∈ {sysu, cdd, all}
```

拆两路并行占满 GPU 1（SYSU / CDD 各一路，batch 64，峰值约 21.8 GB < 24 GB）：

```bash
tmux new-session -d -s graft_sysu 'bash run_gpu1.sh 1 sysu'
tmux new-session -d -s graft_cdd  'bash run_gpu1.sh 1 cdd'
```

> R1-L（local tutor）中间层已加宽到 204，train-only 参数 ≈ 3.8M，与 R1-F（3.8M）容量对齐，用于隔离「全局信息」vs「容量」。

## 判据

拆除后的部署模型，六指标（R/P/OA/F1/IoU/Kappa）相对同批 R0 方向一致上涨；单种子/单数据集只能 preliminary。
