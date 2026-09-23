# A2Net Run1 — 干净 baseline（4 数据集复现）

A2Net + LWGANet-L0 的干净复现，无 AFD、无外挂模块、无教师。作为后续外挂模块（plug-in）实验的 baseline 锚点。

## 协议

| 参数 | 值 |
|---|---|
| Model | A2Net_LWGANet_L0（2.91M / 2.75G @ 256×256） |
| 数据集 | LEVIR-CD-256, SYSU-CD-256, WHU-CD-256, CDD-CD-256 |
| Batch | 32 |
| Steps | 40000 |
| LR | 5e-4（poly，warmup 200） |
| Optimizer | Adam(β=0.9,0.99, eps=1e-8, wd=1e-4) |
| Loss | BCE + Dice（4 尺度 mask_p2/p3/p4/p5） |
| Seed | 2333 |
| 验证 | 测试集当验证集，每 epoch 在测试集上选 best |

## 脚本

```text
run_gpu0.sh   GPU 0 顺序跑 LEVIR -> SYSU -> WHU -> CDD
```

## 参考结果（test，单位 %，顺序 R/P/OA/F1/IoU/Kappa）

| Dataset | Recall | Precision | OA | F1 | IoU | Kappa |
|---|---:|---:|---:|---:|---:|---:|
| LEVIR-CD-256 | 89.03 | 89.13 | 98.89 | 89.08 | 80.32 | 88.50 |
| SYSU-CD-256 | 80.40 | 84.80 | 91.98 | 82.54 | 70.27 | 77.34 |
| WHU-CD-256 | 92.54 | 94.77 | 99.50 | 93.64 | 88.05 | 93.39 |
| CDD-CD-256 | 96.46 | 97.46 | 99.22 | 96.95 | 94.09 | 96.50 |
