# 三指触觉物体识别：模块化训练工程

本工程读取现有采集脚本生成的 episode，实现共享静态编码、光流时序编码、传感器姿态条件化对齐、三节点图融合和分类。采集程序独立运行；训练程序不启动 MuJoCo viewer，也不执行抓取采集。

## 1. 文件职责

| 文件 | 职责与调试入口 |
|---|---|
| `build_dataset_index.py` | 扫描采集目录，检查图像、时间、接触和标签，记录被拒绝的 episode |
| `make_splits.py` | 按抓取条件、位置区域或物体设置姿态分组划分 |
| `precompute_flow.py` | 连续相邻帧光流，Farneback 或 RAFT-small，断点跳过有效缓存 |
| `precompute_geometry.py` | 从关节角和 XML 重建夹爪坐标系中的传感器位姿 |
| `inspect_dataset.py` | 打印张量形状、有效帧；可单独检查前向、反向 |
| `train.py` | 训练、验证选模、保存最佳模型和断点续训 |
| `evaluate.py` | 加载冻结模型，输出指标、逐 episode 预测和混淆矩阵 |
| `check_environment.py` | 检查依赖和 CUDA 运算 |
| `smoke_pipeline.py` | 用临时合成数据验证完整训练、续训和评估流程 |
| `configs/default.json` | 正式训练的尺寸、模型和优化器配置 |
| `configs/debug.json` | 小模型、少帧、单进程，便于定位错误 |
| `tactile/data/index.py` | 文件校验、标签映射、采样条件提取 |
| `tactile/data/splits.py` | 分组和防数据泄漏检查 |
| `tactile/data/dataset.py` | 读取 RGB、光流、几何，统一时序、掩码及 padding |
| `tactile/features.py` | 光流算法和缓存源文件校验 |
| `tactile/geometry.py` | 前向运动学和图边空间属性 |
| `tactile/models/encoders.py` | RGB CNN、光流 CNN、GRU，三指共享参数 |
| `tactile/models/alignment.py` | 传感器位姿条件化的残差特征对齐 |
| `tactile/models/graph.py` | 纯 PyTorch 三节点图注意力与有效节点池化 |
| `tactile/models/classifier.py` | 组装模型、各项消融开关和 MLP 分类器 |
| `tactile/engine.py` | 训练/验证循环、随机触点丢失和时序打乱 |
| `tactile/metrics.py` | 准确率、宏 F1、平衡准确率及分类别结果 |
| `tests/` | 数据泄漏、缓存、几何、掩码、梯度等单元测试 |

建议从 `inspect_dataset.py` 调试数据，再在各模块内部打断点。Windows 排查 DataLoader 问题时先用 `--workers 0`。

## 2. 环境

在 PowerShell 中进入工程目录，使用你已有的 MuJoCo 环境：

```powershell
conda activate mujoco
cd E:\project\p1\deepgelsight\tactile_learning
python check_environment.py
```

本机已实测 Python 3.12、torch 2.11.0+cu128、torchvision 0.26.0+cu128，可调用 RTX 5060。不要为了本工程重新覆盖已经工作的 CUDA PyTorch。

`cv2` 在当前环境的默认导入路径中缺失；测试时使用了项目已有的依赖目录。日常使用建议只把缺失依赖装进所选环境：

```powershell
python -m pip install opencv-python
```

`requirements.txt` 列出依赖。新环境中 torch/torchvision 应一起安装匹配的 CUDA 版本。工程不依赖 PyTorch Geometric、pandas 或 sklearn；ResNet18 默认 `pretrained=false`，不会自动下载权重。若改为 true，首次运行需要下载 torchvision 权重。RAFT 首次运行也会下载自己的权重。

## 3. 输入数据与分组

每个 episode 应有：

```text
物体类别/episode_xxxx/
    metadata.json
    signals.csv
    tactile_rgb.npz       finger1/2/3，各 [T,H,W,3] uint8
    tactile_depth.npz     可选；当前用于完整性检查，不输入分类模型
```

一次 episode 是一个样本。索引默认仅接受有效且具有稳定三点接触的完整采集；老数据没有 `collection_complete` 时，也必须通过实际文件校验。`metadata.object` 决定标签，类别 ID 排序后写入索引和 checkpoint。

不要把同次抓取的帧、同条件重复采集拆到不同集合。索引保留拒绝原因，便于统计各类采集可用率。改变原始数据后需重新索引、划分；缓存也会检查源文件大小和修改时间。

### 建立索引

将下面 `$dataRoot` 改成你希望用于该实验的采集 run 目录。推荐一次实验固定一个明确的数据版本。

```powershell
$dataRoot = 'E:\project\p1\deepgelsight\dataset_underactuated\runs\你的多物体run目录'
python build_dataset_index.py --root $dataRoot --output artifacts/index.json
```

单类别数据可以建立索引并检查网络，正式分类训练至少需要两个类别。此限制防止把只有一个类别的 100% 准确率误当成识别结果。

### 三种划分

```powershell
python make_splits.py --index artifacts/index.json --output artifacts/split_grouped.json --protocol grouped --seed 42
python make_splits.py --index artifacts/index.json --output artifacts/split_position.json --protocol position --block-mm 15 --buffer-mm 5 --seed 42
python make_splits.py --index artifacts/index.json --output artifacts/split_pose.json --protocol pose --seed 42
```

- `grouped`：同物体、设置姿态、抓取偏移、接近角度和夹爪角度的重复采集同组。适合一般识别对比。
- `position`：同一物体坐标系下的目标偏移划为空间块，同块所有姿态/角度同组；buffer 会排除不同集合中过近的目标位置。它衡量目标抓取区域变化，不能称为真实接触质心划分。
- `pose`：按设置的物体四元数分组，q 与 -q 视为相同姿态。当前不根据落稳后的真实旋转自动聚类；正式声称“未见实际姿态”前，还需核查落稳姿态是否重叠。

默认约 70/15/15 按组分配，各集合保留每个类别。每类至少需要 3 个不同组；条件不够或 buffer 排空某类集合时明确报错，不会自动退回随机 episode 划分。只采 upright 时不能进行跨姿态划分。

## 4. 先跑静态基线

它不需要光流或几何缓存，适合首先检查数据。

```powershell
python inspect_dataset.py --index artifacts/index.json --variant static_concat --forward --device cuda
python train.py --index artifacts/index.json --split artifacts/split_grouped.json --config configs/default.json --variant static_concat --output runs/static_concat_seed42 --device cuda
python evaluate.py --checkpoint runs/static_concat_seed42/best.pt --output runs/static_concat_seed42/test --device cuda
```

`inspect_dataset.py` 不需要 split，也允许只有一类；它的输出仅用于检查形状和计算流程。

## 5. 完整模型：先生成两类缓存

### 光流

先用计算成本较低的 Farneback 验证整体流程：

```powershell
python precompute_flow.py --index artifacts/index.json --output artifacts/flow --backend farneback --height 128 --width 96 --workers 2
```

它计算三路所有原始相邻帧，不先跳帧；保存二维归一化位移 `u/width,v/height` 和实际帧间隔，不保存彩色可视化作为训练输入。缓存为 float16，加载时转 float32。

可选 GPU RAFT-small：

```powershell
python precompute_flow.py --index artifacts/index.json --output artifacts/flow_raft --backend raft --device cuda --workers 1 --batch-size 8
```

然后在自己的配置中同时改 `data.flow_cache` 为 `artifacts/flow_raft`、`data.flow_backend` 为 `raft`。避免把不同算法的缓存混在同一个实验里。RAFT 有 GPU 实现，但在当前触觉数据上是否优于 Farneback，需要实验比较。本次未下载或运行预训练 RAFT 权重。

### 传感器几何

```powershell
python precompute_geometry.py --index artifacts/index.json --output artifacts/geometry --xml ..\xml\ur3_with_threefinger_gelsight_table_tomato_underactuated.xml
```

使用 `q_f1/q_f2/q_f3` 回放运动学，只调用 `mj_forward`，不推进物理仿真。输出三个 GelSight site 相对 `gripper_mount` 的位置、旋转，以及两两位移、距离、法向夹角余弦。这里的 site 是传感器中心，不是真实接触质心。

只要各 YCB 场景的夹爪安装结构一致，就可以共享这个机器人 XML。夹爪标定、site 方向或运动学改变后必须重新生成缓存。网络不读取 `gel*_obj_*`、物体真实中心/姿态、目标抓取偏移或仿真法向力。

两种预处理都支持重复运行以跳过源文件和配置匹配的缓存；`--limit 2` 适合首次检查，`--force` 强制重算。正式训练前要覆盖当前索引中的全部 episode。

### 检查并训练完整模型

```powershell
python inspect_dataset.py --index artifacts/index.json --config configs/default.json --variant full --backward --device cuda
python train.py --index artifacts/index.json --split artifacts/split_grouped.json --config configs/default.json --variant full --output runs/full_seed42 --device cuda
python evaluate.py --checkpoint runs/full_seed42/best.pt --output runs/full_seed42/test --device cuda
```

完整默认结构：共享 ResNet18 静态编码 128 维 + 共享光流 CNN/GRU 128 维 → 姿态条件化残差对齐 → 两层、四头的边特征图注意力 → 有效节点加权池化 → MLP。图模块是普通 PyTorch 实现的 GATv2 风格模块，便于直接调试；没有声称与 PyG 的具体实现逐参数一致。

保持阶段默认选 4 帧，光流最多选 31 个相邻帧对。所有指使用相同时间索引，padding 和无效接触显式屏蔽；GRU 不更新填充步。对齐保留不同局部接触内容，没有强制三路特征相等。

默认 batch=4、AMP、AdamW、最多 60 轮，验证宏 F1 连续 12 轮不提升则停止。8GB 显存不足时先改 `--batch-size 2`。默认图像 128×96 保持当前 320×240 图像的宽高比；提高分辨率时同步考虑光流预处理设置。

## 6. 对比和消融

只改 `--variant` 与输出目录；固定同一 split、输入设置与随机种子。

| variant | 含义 |
|---|---|
| `static_single1/2/3` | 三个单指静态基线，分别训练与报告 |
| `static_concat` | 三指静态特征直接拼接 |
| `dynamic_concat` | 三指静态+动态直接拼接，无对齐/几何 |
| `full` | 完整模型 |
| `no_flow` | 去动态分支 |
| `no_alignment` | 去姿态条件化对齐，保留图边几何 |
| `no_edge_geometry` | 去图边几何，仍保留对齐所用传感器位姿 |
| `no_geometry_input` | 同时去对齐和图边几何 |
| `mean` | 同样的静态/动态/对齐特征，改成均值聚合 |
| `concat` | 同样的静态/动态/对齐特征，改成拼接聚合 |
| `temporal_shuffle` | 打乱光流时序；验证/测试按 episode 固定排列 |

`no_geometry` 是 `no_edge_geometry` 的兼容别名，不能解释为去掉全部几何输入。

每个方案可以分别用 `--seed 42`、`--seed 43`、`--seed 44` 运行。当前脚本保存参数量，但没有强制不同融合器参数量相等；论文中应报告参数量，必要时再设计容量匹配对照。各 variant 在默认 full 配置上应用；不要用已经关闭分支的自定义基础配置再套另一消融。

`contact_source=signals` 默认使用仿真接触标志作有效性掩码，应在实验描述中注明。设为 `all` 可做不使用接触真值掩码的对照；真实部署需要图像接触检测或实际传感信号，本工程没有把仿真标志伪装成真实可测输入。

缺失触点评估示例：

```powershell
python evaluate.py --checkpoint runs/full_seed42/best.pt --output runs/full_seed42/drop_finger3 --drop-fingers 3 --device cuda
```

也可在训练配置中设置 `finger_dropout=0.15`，模拟随机缺失，至少保留一个有效指。单指基线不启用此增强。真实缺失接触的数据和人为屏蔽实验应分别报告。

## 7. 输出、续训与调试

每次训练写入独立目录：

```text
run.json          完整配置、类别、划分摘要、参数量和设备信息
history.jsonl     每轮 train/val loss 与分类指标
last.pt           模型、优化器、AMP、随机状态、已完成轮数
best.pt           验证宏 F1 最佳模型
test/metrics.json  总体、分类别和姿态组指标
test/predictions.csv
test/confusion_matrix.csv
```

测试集不参与训练中的早停和选模。评估保存实际支持样本数；指标不以空类别填充来伪造结果。

在原输出目录续训，使用相同配置和 variant：

```powershell
python train.py --index artifacts/index.json --split artifacts/split_grouped.json --config configs/default.json --variant full --output runs/full_seed42 --resume runs/full_seed42/last.pt --epochs 100 --device cuda
```

只允许扩大总轮数；不允许悄悄更换数据划分、标签、特征或优化器。路径在索引/checkpoint中保存为绝对路径，移动数据/工程后应建立新索引和新实验。恢复历史工程时保留原路径。

调试命令：

```powershell
python inspect_dataset.py --index artifacts/index.json --config configs/debug.json --variant full --backward --device cuda
python -m unittest discover -s tests -v
python smoke_pipeline.py --xml ..\xml\ur3_with_threefinger_gelsight_table_tomato_underactuated.xml --device cuda --workers 2
```

合成 smoke 数据在临时目录自动清理，不能用于报告识别精度。已有番茄罐头数据只做过完整 ResNet18 网络的计算检查，没有进行正式多类别训练。

## 8. 实验结论的边界

- 光流首先是渲染图像上的运动估计；应比较无接触光流噪声与接触变化，不直接当作已测得的真实胶体位移。
- 一个 YCB 模型对应一个类别时，结果是这些物体在新接触条件下的识别；同类新实例、仿真到真实传感器的迁移需要单独数据和实验。
- 当前区域/姿态划分依赖采样命令；物体发生滑动或落稳到同一姿态时，应额外审计实际接触/落稳条件再解释泛化结果。
