# LS-Rep_BCD_PLUG

可无损拆除的编码器外挂模块（plug-in）研究项目：在全监督二值遥感图像变化检测上，训练期用外挂模块强化主路编码器，推理/部署时完整拆除、主路更强。

---

## GitHub 代码更新流程（置顶）

代码托管在 [YuqiWang-code/LS-Rep_BCD_PLUG](https://github.com/YuqiWang-code/LS-Rep_BCD_PLUG)。以后修改代码或文档后，在本地仓库根目录执行：

```bash
cd F:/Code_Repositories_2/CursorCode/LS-Rep_BCD_PLUG

git add .
git commit -m "更新代码"
git push
```

提交前确认只包含本次需要上传的代码与文档；训练数据、教师缓存、checkpoint、`docs/temporary/` 与大体积实验输出不通过此流程提交（见 `.gitignore`）。

---

## 项目简介

- **任务**：256×256 全监督二值遥感图像变化检测（binary change detection, BCD）。
- **主路部署学生**：A2Net + LWGANet-L0。`T1/T2 → 共享 LWGANet-L0 四阶段特征 → SWA → TFM → A2Net decoder → 四尺度 change probability`。
  - 部署参数 **2,913,094（2.9131M）**，256×256 FLOPs **2.7475G**。
- **核心研究方向**：设计挂在编码器每一层之外的**外挂模块（plug-in）**——只有输入、没有输出、不进入主特征流，训练期通过辅助损失反向增强 backbone；`switch_to_deploy()` 后完整拆除，部署参数回到 2.9131M，主路输出与未挂外挂模块时逐 bit 一致（max error < 1e-6）。目标是在四个数据集上，拆除后的编码器比不挂外挂模块的 baseline 更强。

## 研究定位

- **性质**：长期学术研究项目，以学术创新为主、不做工程化/生产项目；代码为实验服务。
- **创新边界**：不把损失函数（loss）本身作为创新点；创新体现在外挂模块的结构、机制或学习范式。
- **实验目标**：以证明方法有效性为主（四个数据集一致提升），不追求吞吐/显存/推理速度等工程指标。
- **迭代方式**：项目会不断更新版本、修改代码；目录结构与命名保持可维护、可扩展，便于消融与版本回溯。

## 数据集与评估协议

四个数据集：`LEVIR-CD-256`、`SYSU-CD-256`、`WHU-CD-256`、`CDD-CD-256`。

沿用本任务一脉相承的协议（ChangeMamba / CDMamba）：**直接用测试集当验证集，每个 epoch 在测试集上选 best**，不另设 infer 文件夹。正式结果以测试集六个指标为准：Recall / Precision / OA / F1 / IoU / Kappa。

### 干净 baseline 参考结果（test，单位 %，顺序 R/P/OA/F1/IoU/Kappa）

| Dataset | Recall | Precision | OA | F1 | IoU | Kappa |
|---|---:|---:|---:|---:|---:|---:|
| LEVIR-CD-256 | 89.03 | 89.13 | 98.89 | 89.08 | 80.32 | 88.50 |
| SYSU-CD-256 | 80.40 | 84.80 | 91.98 | 82.54 | 70.27 | 77.34 |
| WHU-CD-256 | 92.54 | 94.77 | 99.50 | 93.64 | 88.05 | 93.39 |
| CDD-CD-256 | 96.46 | 97.46 | 99.22 | 96.95 | 94.09 | 96.50 |

## 目录结构

```text
models/                  A2Net + LWGANet-L0 干净 baseline（部署模型）
train_scripts/
  A2Net/Run1/            干净 baseline 训练脚本（4 数据集）
  PLUG/Run1/             外挂模块消融矩阵模板（仿 SAGE-CD/Run2 简洁写法）
Model_Reproduction/      各工作关键创新点代码（每个只保留核心 .py）
docs/                    项目文档与实验指标（docs/temporary/ 不提交）
analyse/
  update_metrics.py      训练结束后把结果写回 experiment_metrics.xlsx
  generate_snapshot.py   在 docs/temporary/ 生成代码+指标快照
```

## 快速开始

### 训练（干净 baseline）

```bash
bash train_scripts/A2Net/Run1/run_gpu0.sh 0
```

### 更新指标表 / 生成快照

```bash
python analyse/update_metrics.py --log <path/to/train_log.txt> --experiment "PLUG Run1 R1"
python analyse/generate_snapshot.py --name A2Net_Run1 --include_metrics
```

## 研究约束

外挂模块是训练期辅助结构，必须满足：部署参数固定 2.9131M / 2.7475G；`switch_to_deploy()` 后主路输出逐 bit 不变；外挂模块开/关不改变主输出。详细约束见 `docs/ChatGPT_Project_Settings.md`。

## 参考文献

GRAFT-PLUG 的方法依据与 2024–2026 年 CCF-A 及以上 venue 的网络调研见 [docs/参考文献/文献索引.md](docs/参考文献/文献索引.md)（论文 PDF 在 `docs/参考文献/GRAFT相关/`，参考代码见 `Model_Reproduction/`）。
