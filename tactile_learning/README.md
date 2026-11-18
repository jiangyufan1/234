# ST-CTE-HGNN：三指触觉时空异构图神经网络

## 1. 项目简介

本项目面向三指触觉物体识别，基于现有 CTE-HGNN 时间建模架构，进一步研究空间接触结构与动态触觉信息的联合建模。

核心研究目标：

- RGB 点观测与 Optical Flow 区间观测的异构时间建模。
- 触觉趋势信息与瞬态事件信息分解。
- 三指空间接触关系建模。
- 空间结构与时间动态特征融合。
- 在线物体识别与不同采样频率下的鲁棒性。

项目最终目标是建立一个具有明确物理意义、可进行模块消融和可复现实验的三指触觉时空识别网络。

## 2. 项目总体架构

项目采用双分支设计。

### 2.1 Temporal Branch（已有基础）

基于 CTE-HGNN，主要包括：

- RGB Encoder：逐帧外观特征编码。
- Flow Encoder：逐区间运动特征编码。
- Trend–Event Decomposition：趋势与瞬态事件分解。
- Temporal Heterogeneous Graph：RGB 与 Flow 时间异构图。
- Bidirectional Message Passing：双向异构消息传播。
- Dynamic Pseudo-Memory Routing：动态跨模态路由。
- Classification Head：物体分类。

时间建模必须遵守：

- RGB 是点观测，表示为 RGB(t)。
- Flow 是区间观测，表示为 Flow([t_start,t_end])。
- Flow 只有在区间结束后才能用于在线推理。
- RGB 与 Flow 不允许仅通过帧索引强制对齐。
- 保留真实时间戳和独立序列长度。

### 2.2 Spatial Branch（待开发）

在现有时间分支基础上增加空间结构建模。

计划包含：

- Spatial Encoder：局部触觉空间特征提取。
- Spatial Graph：三指空间关系图。
- Spatiotemporal Fusion：空间与时间特征融合。

重点研究：

- 三指相对位置与方向。
- 局部接触区域分布。
- 接触区域随时间的变化。
- 三指之间的空间协同关系。
- 空间结构与瞬态事件之间的关联。

空间输入必须区分真实可观测信息和仿真专用真值。

### 2.3 最终目标结构

Temporal Branch + Spatial Branch → Spatiotemporal Fusion → Object Classification

最终模型暂定名称：ST-CTE-HGNN。

## 3. 工程目录

```text
tactile_learning/
│
├── configs/
│   ├── default.json
│   ├── debug.json
│   └── experiments/
│
├── data/                       # 本地数据，不上传 Git
│   ├── raw/
│   │   └── MCB_A/
│   ├── collected/
│   ├── processed/
│   ├── derived/
│   ├── indexes/
│   └── splits/
│
├── docs/
│   ├── architecture/
│   ├── data/
│   ├── experiments/
│   └── collection/
│
├── scripts/
│   ├── collection/
│   ├── data/
│   └── experiments/
│
├── tactile/
│   ├── data/
│   ├── models/
│   ├── evaluation/
│   ├── losses/
│   └── training/
│
├── tests/
│   ├── test_cte/
│   ├── test_spatial/
│   ├── test_collection/
│   └── test_training/
│
├── artifacts/                  # 本地缓存
├── runs/                       # 训练结果
├── checkpoints/                # 模型权重
│
├── train.py
├── evaluate.py
├── build_dataset_index.py
├── precompute_flow.py
├── precompute_geometry.py
│
├── .gitignore
├── .gitattributes
├── AGENTS.md
└── README.md
```

## 4. 数据集

当前物体几何模型来源为 MCB_A。

本地路径：

```text
data/raw/MCB_A/dataset_org_norm/
├── train/
└── test/
```

数据规模：

- 68 个物体类别。
- Train：46,980 个 OBJ 文件。
- Test：11,716 个 OBJ 文件。
- 合计：58,696 个 OBJ 文件。

MCB_A 是物体三维模型库，不是已经采集完成的触觉 RGB/Flow 数据集。

它主要用于 MuJoCo 场景生成和后续三指触觉采集。

后续新采集数据统一存放在：

```text
data/collected/
```

处理后的数据存放在：

```text
data/processed/
```

不同采样频率的派生数据存放在：

```text
data/derived/
```

原始数据不得直接修改，派生数据必须保留来源 Episode 和处理参数。

## 5. 数据采集目标

新的采集程序围绕时空网络需求设计。

计划重点采集：

1. 不同物体几何结构下的三指接触。
2. 不同接触位置和物体姿态。
3. 法向加载与卸载过程。
4. 切向接触运动。
5. 接触区域迁移。
6. 三指接触顺序和空间协同。
7. 接触、保持、释放的完整过程。

每个 Episode 尽可能保留：

- 三指 RGB 原始图像。
- 图像时间戳。
- 光流区间起止时间。
- 三指接触状态。
- 接触区域信息。
- 传感器空间位姿。
- 仿真物理状态和实验配置。

仿真真值与网络实际输入必须明确区分。

## 6. 开发环境

主要开发环境：

- Windows 11
- Python 3.12
- PyTorch
- CUDA
- NVIDIA GPU

辅助环境：

- Ubuntu 22.04
- ROS 2 Humble（机械臂集成需要时使用）

进入工程：

```powershell
cd D:\robotics\234\tactile_learning
conda activate tactile
```

检查环境：

```powershell
python check_environment.py
```

## 7. 测试

运行全部测试：

```powershell
python -m pytest -v
```

新模块开发必须提供对应测试。

重点检查：

- 不同 RGB / Flow 时间长度。
- 点—区间时间关系。
- 在线因果性。
- 缺失模态。
- 三指独立性。
- 空间坐标一致性。
- 空间图消息传播。
- 时空融合梯度。
- 原有基线兼容性。

合成数据测试只证明软件行为，不代表真实识别性能。

## 8. Git 与数据管理

Git 管理：

- 网络源码。
- 采集程序。
- 数据处理程序。
- 测试代码。
- 实验配置。
- 技术文档。

Git 不管理：

- 原始 OBJ 数据集。
- 原始触觉图像。
- 大型光流和深度缓存。
- 模型权重。
- 训练日志和运行产物。

根目录 `data/`、`artifacts/`、`runs/` 和 `checkpoints/` 已纳入 `.gitignore`。

## 9. 研究实验计划

后续实验围绕以下问题展开：

1. Temporal Branch 是否能够有效提取趋势与瞬态事件？
2. Spatial Branch 是否能够学习三指空间接触结构？
3. 时空融合是否优于单独时间建模？
4. 不同采样频率下模型性能如何变化？
5. 模型能否在部分手指缺失时保持识别能力？
6. 在线早期识别能否减少所需观测时间？

主要消融：

- Temporal-only
- Spatial-only
- Temporal + Spatial
- 去除 Trend–Event
- 去除异构时间边
- 去除双向消息传播
- 去除 Dynamic Routing
- 去除空间关系编码

所有性能结论必须通过实际实验验证。

## 10. 当前开发状态

| 模块 | 状态 |
|---|---|
| 原始 CTE-HGNN 时间分支 | 已有实现 |
| Trend–Event | 已有实现 |
| RGB-Flow 时间异构图 | 已有实现 |
| 双向消息传播 | 已有实现 |
| Dynamic Routing | 已有实现 |
| 原始数据编码接口 | 已有实现 |
| 在线因果性 | 已有专项测试 |
| MCB_A 数据迁入 | 已完成 |
| Spatial Encoder | 待开发 |
| Spatial Graph | 待开发 |
| Spatiotemporal Fusion | 待开发 |
| 新版动态采集程序 | 待开发 |
| 新数据集正式采集 | 待开展 |

---

项目开发遵循根目录 `AGENTS.md` 中的工程规范。

核心原则：保留已有时间网络，在其基础上增加具有明确物理意义的空间结构建模，并通过严格的消融实验验证时空联合建模的实际价值。
