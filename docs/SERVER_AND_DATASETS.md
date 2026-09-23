# 服务器与数据集文档

> 最后更新: 2026-09-23

---

## 工作流程说明

### 开发和上传流程

**重要**: 所有代码修改都在本地 Windows 环境完成，然后手动上传到服务器。

**本地环境**:
- 路径: `F:\Code_Repositories_2\CursorCode\LS-Rep_BCD`
- IDE: VSCode / Cursor
- AI助手: Claude 在本地进行代码修改和重构

**服务器环境**:
- 路径: `/home/yqwang/project/LS-Rep_BCD/`
- 用途: 训练和实验执行

**工作流程**:
```
1. 本地修改代码 (F:\Code_Repositories_2\CursorCode\LS-Rep_BCD)
   ↓
2. 确保换行符正确 (LF, 不是 CRLF)
   ↓
3. 手动上传到服务器 (scp / SFTP)
   ↓
4. 在服务器上运行训练
   ↓
5. 查看结果和日志
```

**上传命令示例**:
```bash
# 上传整个 models/ 目录
scp -r F:/Code_Repositories_2/CursorCode/LS-Rep_BCD/models/ yqwang@172.18.232.141:/home/yqwang/project/LS-Rep_BCD/

# 上传训练脚本
scp -r F:/Code_Repositories_2/CursorCode/LS-Rep_BCD/train_scripts/LSRepNet/Run4/ yqwang@172.18.232.141:/home/yqwang/project/LS-Rep_BCD/train_scripts/LSRepNet/

# 上传单个文件
scp F:/Code_Repositories_2/CursorCode/LS-Rep_BCD/models/scripts/train.py yqwang@172.18.232.141:/home/yqwang/project/LS-Rep_BCD/models/scripts/
```

**注意事项**:
- ✅ 本地修改后，所有换行符问题已在本地自动处理
- ✅ 上传前无需手动检查换行符
- ✅ 直接上传即可在服务器上运行
- ⚠️ 不要在服务器上直接修改代码（除非临时调试）

---

## 常见问题与解决方案

### Windows 换行符问题 (CRLF vs LF)

**问题现象:**
```bash
run_gpu0.sh: 行 2: $'\r': 未找到命令
run_gpu0.sh: 第 36 行： cd: $'E1_RepViT_Optimal\r': 没有那个文件或目录
```

**原因**: 
在 Windows 上创建的脚本文件使用 `\r\n` (CRLF) 作为换行符，而 Linux/Unix 系统需要 `\n` (LF)。

**已解决**: 
- ✅ 所有 .sh 文件在本地生成时已自动转换为 LF
- ✅ 使用 PowerShell 脚本批量修复
- ✅ 2026-08-10 已修复所有 35 个 .sh 文件

**如果服务器上仍有问题 (历史文件)**:

**方法1: 使用 dos2unix 转换 (推荐)**
```bash
# 在服务器上安装 dos2unix (如果没有)
sudo yum install dos2unix  # CentOS/RHEL
# 或
sudo apt-get install dos2unix  # Ubuntu/Debian

# 转换所有 .sh 文件
cd /home/yqwang/project/LS-Rep_BCD/train_scripts/LSRepNet/Run4
dos2unix run_gpu0.sh run_gpu1.sh
find . -name "*.sh" -exec dos2unix {} \;
```

**方法2: 使用 sed 转换**
```bash
cd /home/yqwang/project/LS-Rep_BCD/train_scripts/LSRepNet/Run4
sed -i 's/\r$//' run_gpu0.sh
sed -i 's/\r$//' run_gpu1.sh
find . -name "*.sh" -exec sed -i 's/\r$//' {} \;
```

**方法3: 使用 tr 命令**
```bash
tr -d '\r' < run_gpu0.sh > run_gpu0_fixed.sh
mv run_gpu0_fixed.sh run_gpu0.sh
chmod +x run_gpu0.sh
```

**预防措施 (在本地 Windows):**
- Git 配置: `git config --global core.autocrlf input`
- VSCode 设置: 文件右下角选择 "LF" 而非 "CRLF"
- 或在 `.gitattributes` 添加: `*.sh text eol=lf`

---

## 1. 服务器信息

### 1.1 连接方式

| 项目 | 信息 |
|------|------|
| **IP** | `172.18.232.141` |
| **端口** | `22` |
| **协议** | SFTP / SSH |
| **用户名** | `yqwang` |
| **Conda 环境（A2Net 训练）** | `lwganet` (Python 3.10, 服务器实测 PyTorch 2.6.0 + CUDA 12.6) |
| **Conda 环境（Run4 阶段一）** | `sam2cache` (Python 3.10, PyTorch 2.5.1 + torchvision 0.20.1 + CUDA 12.4) |
| **Conda 环境（旧）** | `lsrep` (Python 3.10) — LSRepNet v5 用，已弃用 |
| **远程项目路径** | `/home/yqwang/project/LS-Rep_BCD/` |

SSH 登录:
```bash
ssh yqwang@172.18.232.141
```

### 1.2 硬件配置

| 硬件 | 型号 / 规格 |
|------|-------------|
| **GPU × 2** | NVIDIA GeForce RTX 4090 (24 GiB 显存 × 2) |
| **显存总计** | 48 GiB |
| **系统内存** | 约 4 TiB (实际 ~3.756 TiB) |
| **CPU** | 多核 (Load Avg ~3.9) |
| **NVIDIA 驱动** | 570.211.01 |
| **CUDA 版本** | 12.8 |

### 1.3 GPU 当前状态 (2026-08-03)

| GPU | 温度 | 功耗 | 显存占用 | 利用率 |
|-----|------|------|----------|--------|
| GPU 0 | 51°C | 34W / 450W | 134.60 MiB / 23.99 GiB | 0.5% |
| GPU 1 | 54°C | 19W / 450W | 14.75 MiB / 23.99 GiB | 0.1% |

> 闲时状态，两张卡基本空闲，可用于训练。

### 1.4 系统负载 (2026-08-03)

| 指标 | 数值 |
|------|------|
| CPU 使用率 | 12.1% |
| 内存占用 | 7.1% (3.756 TiB 中) |
| SWAP 占用 | 0.7% |
| 平均 GPU 显存占用 | 0.3% |
| 平均 GPU 利用率 | 0.0% |

### 1.5 Conda 环境 `lsrep` (2026-08-04)

| 包 | 版本 | 用途 |
|----|------|------|
| python | 3.10 | 运行环境 |
| torch | 2.1.2+cu121 | 深度学习框架 |
| torchvision | 0.16.2+cu121 | 图像预处理 |
| torchaudio | 2.1.2+cu121 | (随 torch 安装) |
| numpy | 1.26.4 | 数值计算 |
| opencv-python | 4.10.0.84 | 图像读写 |
| pillow | 12.2.0 | 图像加载 |
| scikit-learn | 1.7.2 | 混淆矩阵 / Kappa |
| scipy | 1.15.3 | 科学计算 |
| tensorboardX | 2.6.5 | 训练可视化 |
| thop | 0.1.1 | FLOPs / 参数量统计 |
| tqdm | 4.70.0 | 进度条 |
| triton | 2.1.0 | (torch 依赖) |

查看完整列表:
```bash
conda activate lsrep && pip list --format=columns
```

### 1.6 Conda 环境 `lwganet` (2026-08-18，当前 A2Net 主力)

> 该环境为 A2Net_LWGANet 变化检测专用，含 mmcv-full（LWGANet backbone 依赖）。
> 完整安装步骤见 `Model_Reproduction/LWGANet/change_detection/A2Net_LWGANet/docs/MMCV_INSTALL_FIX.md`。

| 包 | 版本 | 用途 |
|----|------|------|
| python | 3.10 | 运行环境 |
| torch | 2.6.0（CUDA 12.6，以 2026-08-24 smoke 输出为准） | 深度学习框架 |
| torchvision | 需在服务器复核 | 曾出现 `image.so undefined symbol` 警告；A2Net 当前不使用 `torchvision.io` |
| torchaudio | 非 A2Net 必需 | 不用于当前任务 |
| numpy | 1.26.4 | 数值计算 |
| opencv-python | 4.10.0.84 | 图像读写 |
| pillow | latest | 图像加载 |
| scikit-learn | latest | 混淆矩阵 / Kappa |
| scipy | latest | 科学计算 |
| tensorboardX | latest | 训练可视化 |
| thop | latest | FLOPs / 参数量统计 |
| tqdm | latest | 进度条 |
| timm | 1.0.28 | 预训练模型库（LWGANet 依赖 DropPath/trunc_normal_） |
| antialiased-cnns | 0.3 | 抗混叠卷积（LWGANet MRA 模块依赖 BlurPool） |
| mmcv-full | 1.7.2 | OpenMMLab 基础库（LWGANet 依赖 build_norm_layer） |

**用途**: A2Net_LWGANet 变化检测（当前主力）
**路径**: `/home/yqwang/miniconda3/envs/lwganet/`

查看完整列表:
```bash
conda activate lwganet && pip list --format=columns
```

### 1.7 Conda 环境 `sam2cache`（A2Net Run4 阶段一专用）

#### 环境分工

| 阶段 | 环境 | 是否加载 SAM2 | 用途 |
|------|------|:------------:|------|
| Run4 阶段一（Stage A） | `sam2cache` | 是 | 离线生成 SYSU/WHU 的 SAM2.1 结构教师缓存 |
| Run4 阶段二（Stage B） | `lwganet` | 否 | 训练 E0/E1/E2/E3，E1/E3 只读已生成的 `.pt` 缓存 |

`models/third_party/sam2` 不是一个已存在的 Conda 环境，而是项目内置的 Meta SAM2 源码。Run4 脚本约定把它安装到名为 `sam2cache` 的独立环境。不建议把 SAM2 直接安装到 `lwganet`：SAM2 要求 `torch>=2.5.1` 和 `torchvision>=0.20.1`，而 `lwganet` 还有 mmcv/LWGANet 的二进制依赖，原地升级容易破坏已通过的 A2Net 训练环境。

版本依据：项目内置 [SAM2 README](../models/third_party/sam2/README.md) 和 [SAM2 INSTALL](../models/third_party/sam2/INSTALL.md)，以及 [PyTorch 官方历史版本安装矩阵](https://docs.pytorch.org/get-started/previous-versions/)。

#### 一次性创建（服务器执行，推荐）

以下使用 SAM2 官方支持的 PyTorch 2.5.1，`torch`/`torchvision` 严格配对。使用 Conda CUDA 12.4 Runtime 可避开服务器访问 `pypi.nvidia.com` 时的 SSL 证书问题；NVIDIA 570 驱动可以正常运行 CUDA 12.4 Runtime：

```bash
source /home/yqwang/miniconda3/etc/profile.d/conda.sh
conda create -n sam2cache python=3.10 pip -y
conda activate sam2cache

python -m pip install --upgrade pip setuptools wheel
conda install -y \
  pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
  pytorch-cuda=12.4 \
  "mkl<2024.1" "intel-openmp<2024.1" \
  -c pytorch -c nvidia -c defaults

# 缓存构建器使用 cv2；服务器无需 GUI，因此用 headless 版
python -m pip install opencv-python-headless==4.10.0.84

# 安装项目内置 SAM2.1 源码。阶段一不依赖其可选 CUDA 后处理扩展，
# 先显式关闭编译，可避免系统 nvcc 与 PyTorch CUDA 轮子不一致。
cd /home/yqwang/project/LS-Rep_BCD/models/third_party/sam2
SAM2_BUILD_CUDA=0 python -m pip install --no-build-isolation -e .

python -m pip check
```

> 当前缓存工具自身会用 OpenCV 移除过小区域，所以不编译 SAM2 的可选 connected-components CUDA 扩展不会阻止阶段一。第一次推理可能看到“Skipping the post-processing step”警告，它不是缓存构建失败。

#### 环境验证

```bash
conda activate sam2cache
cd /home/yqwang/project/LS-Rep_BCD

python -c "import torch, torchvision, cv2; from sam2.build_sam import build_sam2; print('torch=', torch.__version__, 'torchvision=', torchvision.__version__, 'cuda=', torch.version.cuda, 'gpu=', torch.cuda.get_device_name(0)); print('SAM2 import OK')"

test -f models/third_party/sam2/configs/sam2.1/sam2.1_hiera_l.yaml
test -f pre-trained_weights/sam2.1_hiera_large.pt
```

预期至少看到：`torch=2.5.1`、`torchvision=0.20.1`、`cuda=12.4`、RTX 4090 名称和 `SAM2 import OK`。两条 `test` 命令均应返回 0；若权重不存在，需把 `sam2.1_hiera_large.pt` 上传到：

```text
/home/yqwang/project/LS-Rep_BCD/pre-trained_weights/sam2.1_hiera_large.pt
```

#### 可选：启用 SAM2 CUDA 后处理扩展

当服务器已安装与 PyTorch CUDA 版本匹配的 CUDA Toolkit/NVCC 时才需要这一步；它不是 Run4 必需项：

```bash
conda activate sam2cache
nvcc --version
python -c "import torch; from torch.utils.cpp_extension import CUDA_HOME; print(torch.version.cuda, CUDA_HOME)"

cd /home/yqwang/project/LS-Rep_BCD/models/third_party/sam2
SAM2_BUILD_ALLOW_ERRORS=0 python -m pip install -v --no-build-isolation -e .
python -c "from sam2 import _C; print('SAM2 CUDA extension OK')"
```

若编译报 CUDA Toolkit/NVCC 版本不匹配，不要改动 `lwganet`；重新执行上面 `SAM2_BUILD_CUDA=0` 的稳定安装即可。

#### 已知安装问题：`iJIT_NotifyEvent`

如果 Conda 求解出 `mkl 2024.1+`（例如 `mkl 2025.0.0`），`import torch` 可能报：

```text
libtorch_cpu.so: undefined symbol: iJIT_NotifyEvent
```

这是 MKL/OpenMP 兼容性问题，不是 CUDA 或 GPU 问题。无需重新下载 PyTorch，在当前环境降级两个包即可：

```bash
conda activate sam2cache
conda install -y "mkl<2024.1" "intel-openmp<2024.1" -c defaults

conda list | grep -E '^(mkl|intel-openmp|pytorch|torchvision|pytorch-cuda)[[:space:]]'
python -c "import torch, torchvision; print(torch.__version__, torchvision.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

修复依据：[PyTorch issue #123097](https://github.com/pytorch/pytorch/issues/123097)。不要通过 `LD_PRELOAD`、替换系统动态库或重装 NVIDIA 驱动来规避该错误。

#### Run4 阶段一：生成 SAM2 结构缓存

阶段一的两个构建脚本会自动激活 `sam2cache`，因此从 `base` 环境启动也可以：

```bash
cd /home/yqwang/project/LS-Rep_BCD

# SYSU 默认用物理 GPU 1：12000 个 train 样本
bash train_scripts/A2Net/Run4/build_sam_cache_sysu.sh

# WHU 默认用物理 GPU 1：5947 个 train 样本（与 SYSU 顺序执行）
bash train_scripts/A2Net/Run4/build_sam_cache_whu.sh
```

两个脚本内部都设置 `CUDA_VISIBLE_DEVICES=${SAM_CACHE_GPU_ID:-1}`，因此物理 GPU 1 在 Python 进程中是逻辑 `cuda:0`，脚本统一传入 `--gpu_id 0`。不要在同一张物理 GPU 上并行启动两个 SAM2 Hiera-L 构建器；一个完成后再启动另一个。如需临时改用物理 GPU 0：

```bash
SAM_CACHE_GPU_ID=0 bash train_scripts/A2Net/Run4/build_sam_cache_sysu.sh
```

两个脚本都会先构建缓存，然后自动执行 schema/覆盖率/数值统计验证并生成 100 张可视化预览。构建可断点续跑；只有权重、配置或数据划分已变更时才使用：

```bash
bash train_scripts/A2Net/Run4/build_sam_cache_sysu.sh --rebuild
bash train_scripts/A2Net/Run4/build_sam_cache_whu.sh --rebuild
```

预期产物：

```text
/data/CD_teacher_cache/SAMStruct/sam2.1_hiera_large/
├── SYSU-CD-256/manifest.json + validation_summary.json + train/*.pt
└── WHU-CD-256/manifest.json  + validation_summary.json + train/*.pt

/data/CD_teacher_cache/SAMStruct/preview/
├── SYSU/
└── WHU/
```

生成完后检查 `validation_summary.json` 不得报缺样本/损坏 pack，并人工查看预览图的小建筑、道路和边界质量。缓存保存 T1/T2 的 `int32 instance_id + fp16 boundary + fp16 quality`，两个数据集建议预留至少 25 GiB 空间。

---

## 2. 数据集概览

数据集根路径: `/data/CD/`

| # | 数据集名称 | 路径 | 图像数 | 文件格式 | 子目录结构 |
|---|-----------|------|--------|----------|------------|
| 1 | LEVIR-CD-256 | `/data/CD/LEVIR-CD-256/` | 10,192 | `.png` | A / B / label / list |
| 2 | SYSU-CD-256 | `/data/CD/SYSU-CD-256/` | 20,000 | `.png` | A / B / label / list |
| 3 | WHU-CD-256 | `/data/CD/WHU-CD-256/` | 7,434 | `.png` | A / B / label / list |
| 4 | CDD-CD-256 | `/data/CD/CDD-CD-256/` | 15,998 | `.jpg` | A / B / label / list |
| 5 | LEVIR-CD+256 | `/data/CD/LEVIR-CD+256/` | 15,760 | `.png` | A / B / label / list |
| 6 | DSIFN | `/data/CD/DSIFN/` | 3,988 | `.png` | A / B / label / list |

> **总计**: 6 个数据集，共 **73,372** 对图像

### 2.0 四个主力数据集的共同特点

尽管四个数据集（LEVIR/SYSU/WHU/CDD）在场景与变化类型上各有侧重，但作为"双时相、高分辨率、256×256 裁剪"的遥感变化检测基准，它们共享以下核心图像特点：

| 共性特点 | 影响 | 建模对策 |
|---------|------|---------|
| **双时相配准良好但辐射差异显著** | 光照/色调/阴影/季节植被差异→伪变化 | 时相不变特征 + 边缘/梯度引导 |
| **变化像素占比低、类别严重不平衡** | 正负样本失衡（WHU 4.26%, LEVIR 4.65%, CDD 0.147 比例） | Focal/Dice Loss + 难样本挖掘 + 像素重加权 |
| **目标尺度跨度大** | 单车辆→大型仓库、单树→整片森林 | 多尺度特征融合 + 空洞卷积扩大感受野 |
| **256×256 裁剪边界效应** | 变化目标被裁切、碎片化 | 对不完整目标的鲁棒性设计 |
| **建筑是主体但非全部** | LEVIR/WHU 建筑为核心，SYSU/CDD 多元 | 通用变化建模而非过度建筑先验 |

---

### 2.0.1 LEVIR-CD-256 详细特性

**基本信息**:
- **来源**: 北京航空航天大学，637 对 Google Earth 影像
- **原始尺寸**: 1024×1024 → 裁剪 256×256 (7120/1024/2048 = 7:1:2)
- **分辨率**: 0.5 m/像素 | **时间跨度**: 5–14 年 | **区域**: 美国得州 20 个区域
- **变化类型**: 建筑增长/衰退（别墅/公寓/车库/仓库），约 31,333 个独立实例
- **变化像素比**: ~1:20（正样本极缺）

**核心难点**: 极端不平衡 + 刻意引入季节/光照伪变化

**建模要点**: Dice/Focal Loss + 伪变化抑制 + 多尺度特征融合 + 严格 7:1:2 划分

---

### 2.0.2 SYSU-CD-256 详细特性（最难）

**基本信息**:
- **来源**: 中山大学，20,000 对航空影像
- **原始尺寸**: 1024×1024 → 裁剪 256×256 (12000/4000/4000 = 6:2:2)
- **分辨率**: 0.5 m/像素 | **时间**: 2007–2014 年 | **区域**: 香港及周边（城市+郊区）
- **变化类型**: **最复杂**——城市扩张/建筑/道路/植被/填海/海域/场地整理
- **难度**: 四个数据集中最难，当前最佳 F1 仅 80.54

**核心难点**: 变化类型杂 + 标注粗 + 植被/海域受光照影响大

**建模要点**: 通用变化建模（避免建筑先验）+ 标签噪声鲁棒 + 光照不变特征 + 增强克制

---

### 2.0.3 WHU-CD-256 详细特性

**基本信息**:
- **来源**: 武汉大学，一对新西兰基督城震后影像（2012 & 2016）
- **原始尺寸**: 32507×15354 → 裁剪 256×256 (6096/762/762 ≈ 7:1:2)
- **分辨率**: 0.2–0.3 m/像素 | **覆盖**: 20.5 km² 2011 年 6.3 级地震重建区
- **变化统计**: 12,796 → 16,077 座建筑
- **变化像素比**: ~4.26%（最不平衡之一）

**核心难点**: 极端不平衡 + 建筑密集相似 + 原始仅一对大图（划分随机性强）

**建模要点**: 重加权/过采样 + 边缘约束 + 固定随机种子多次实验 + 轻量化优先

---

### 2.0.4 CDD-CD-256 详细特性

**基本信息**:
- **来源**: Google Earth 多源，11 对原始（7 季节 + 4 灾害）
- **原始尺寸**: 混合（4725×2200 & 1900×1000）→ 裁剪+旋转增广 16,000 对
- **分辨率**: 3 cm–1 m/像素（跨度大）| **划分**: 10000/3000/3000
- **场景**: 城市/乡村/山地/水体
- **变化像素比**: ~0.147 | **目标**: 97.0 F1（最高要求）

**核心难点**: 伪变化干扰最强（季节/光照/云影/旋转）+ 场景最杂 + 跨尺度目标

**建模要点**: 时相不变特征 + 多尺度差异融合 + 均衡多场景采样 + 同步增广

---

### 2.0.5 四数据集横向对比与统一建议

| 数据集 | 核心场景 | 分辨率 | 主要难点 | 建模首要抓手 |
|--------|---------|--------|---------|-------------|
| LEVIR | 建筑增长（得州） | 0.5m | 正样本极缺+伪变化 | 不平衡损失+伪变化抑制 |
| SYSU | 通用城市（香港） | 0.5m | 类型杂+标注粗+最难 | 通用建模+噪声鲁棒 |
| WHU | 震后重建（新西兰） | 0.2–0.3m | 极端不平衡+相似建筑 | 重加权+边缘约束 |
| CDD | 多源季节/灾害 | 3cm–1m | 强伪变化+跨尺度 | 时相不变+多尺度融合 |

**统一改进建议**:
1. **损失**: Dice/BCE-Dice 主干，按正样本比例动态调权，WHU/LEVIR 优先难样本挖掘
2. **差异建模**: 通道拼接+差分+注意力，时相分支可重参数化多分支
3. **多尺度**: 空洞卷积/金字塔池化/跨尺度跳跃，应对车辆→仓库、单树→森林
4. **增广**: 双时相同步（翻转/旋转/色彩抖动），光照/季节归一化
5. **评估**: 严格标准划分+256无重叠裁剪，多次随机种子报告均值±标准差

---

### 2.1 目录结构说明

所有数据集采用统一的 Change Detection 目录结构:

```
数据集根目录/
├── A/          # 时相 1 图像
├── B/          # 时相 2 图像
├── label/      # 变化标签（二值图: 0=未变化, 255=变化）
└── list/       # 训练/验证/测试集划分文件（.txt）
```

### 2.2 各数据集详细

#### LEVIR-CD-256

| 子目录 | 文件数 | 命名示例 |
|--------|--------|----------|
| A | 10,192 | `test_100_10.png`, `train_100_10.png` |
| B | 10,192 | `test_100_10.png`, `train_100_10.png` |
| label | 10,192 | `test_100_10.png`, `train_100_10.png` |
| list | 3 | `train.txt`, `val.txt`, `test.txt` |

#### SYSU-CD-256

| 子目录 | 文件数 | 命名示例 |
|--------|--------|----------|
| A | 20,000 | `00000.png`, `00001.png` |
| B | 20,000 | `00000.png`, `00001.png` |
| label | 20,000 | `00000.png`, `00001.png` |
| list | 3 | `train.txt`, `val.txt`, `test.txt` |

#### WHU-CD-256

| 子目录 | 文件数 | 命名示例 |
|--------|--------|----------|
| A | 7,434 | `whucd_00001.png`, `whucd_00002.png` |
| B | 7,434 | `whucd_00001.png`, `whucd_00002.png` |
| label | 7,434 | `whucd_00001.png`, `whucd_00002.png` |
| list | 26 | 含半监督划分：`N_train_supervised.txt`, `N_train_unsupervised.txt` + `train.txt`, `val.txt`, `test.txt` |

#### CDD-CD-256

| 子目录 | 文件数 | 命名示例 |
|--------|--------|----------|
| A | 15,998 | `test_00000.jpg`, `train_00000.jpg` |
| B | 15,998 | `test_00000.jpg`, `train_00000.jpg` |
| label | 15,998 | `test_00000.jpg`, `train_00000.jpg` |
| list | 3 | `train.txt`, `val.txt`, `test.txt` |

> ⚠️ CDD-CD-256 使用 `.jpg` 格式，其余数据集为 `.png`

#### LEVIR-CD+256

| 子目录 | 文件数 | 命名示例 |
|--------|--------|----------|
| A | 15,760 | `train_100_00.png`, `train_100_01.png` |
| B | 15,760 | `train_100_00.png`, `train_100_01.png` |
| label | 15,760 | `train_100_00.png`, `train_100_01.png` |
| list | 3 | `train.txt`, `val.txt`, `test.txt` |

#### DSIFN

| 子目录 | 文件数 | 命名示例 |
|--------|--------|----------|
| A | 3,988 | `test_0.png`, `train_1.png` |
| B | 3,988 | `test_0.png`, `train_1.png` |
| label | 3,988 | `test_0.png`, `train_1.png` |
| list | 4 | `train.txt`, `trainval.txt`, `val.txt`, `test.txt` |

> DSIFN 有 4 个 list 文件，其中 `trainval.txt` 为额外划分。训练使用 `train.txt`，list 条目含完整文件名（如 `train_1.png`）。图像原始尺寸为 **512×512**（其余数据集为 256×256）。

---

## 3. 快速命令

### 3.1 数据集核查

```bash
# 生成完整审计报告（在服务器上运行）
cd /home/yqwang/project/LS-Rep_BCD/docs/DATA/
# 使用之前的一键命令生成 CD_dataset_audit.txt
```

### 3.2 查看 GPU 状态

```bash
nvitop           # 实时 GPU 监控
nvidia-smi       # NVIDIA 官方监控
```

### 3.3 环境管理

```bash
conda activate lwganet         # Run4 阶段二：A2Net 训练
conda activate sam2cache       # Run4 阶段一：离线 SAM2 缓存
pip list --format=columns      # 查看已安装包
```

---

## 4. 项目路径

| 位置 | 路径 |
|------|------|
| **本地项目** | `f:\Code_Repositories_2\CursorCode\LS-Rep_BCD\` |
| **服务器项目** | `/home/yqwang/project/LS-Rep_BCD/` |
| **数据集根** | `/data/CD/` |
| **Conda 环境（A2Net 训练）** | `/home/yqwang/miniconda3/envs/lwganet/` |
| **Conda 环境（SAM2 缓存）** | `/home/yqwang/miniconda3/envs/sam2cache/` |
