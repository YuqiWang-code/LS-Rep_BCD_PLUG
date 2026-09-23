# PLUG Run1（模板）

可无损拆除的编码器外挂模块（plug-in）消融矩阵模板。写法仿 SAGE-CD/Run2：一个 `run_one()` 函数 + 一行一个实验，不再用 LSRepNet 那样逐个实验堆脚本。

## 实验矩阵

| ID | 配置 | 唯一变量 |
|---|---|---|
| R0 | Clean A2Net baseline | — |
| R1 | R0 + 外挂模块（待设计） | plug-in |

## 协议（固定）

batch 64、40000 steps、seed 2333、lr 5e-4、wd 1e-4、batch Dice、主损失 BCE+Dice、测试集当验证集（每 epoch 在测试集上选 best）。

外挂模块只在训练期挂载，`switch_to_deploy()` 后部署参数回到 2,913,094、FLOPs 2.7475G，主路输出与 baseline 逐 bit 一致（max error < 1e-6）。

## 脚本

```text
run_gpu0.sh   GPU 0 顺序跑 R0 -> R1（矩阵可扩展）
```
