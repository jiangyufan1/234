# Three-Finger Underactuated Tactile Robotics

三指欠驱动夹爪、MuJoCo仿真、GelSight触觉数据处理与物体识别研究工程。

## Project Modules

| Directory | Description |
|---|---|
| `采集代码/` | 三指触觉数据采集程序 |
| `tactile_learning/` | 触觉特征提取、数据集构建和模型训练 |
| `xml/` | 机器人、夹爪和MuJoCo场景模型 |
| `mujoco_xml/` | YCB物体仿真模型 |
| `mujoco_xml_diverse/` | 多类别物体仿真模型 |

## Learning Pipeline

1. Load existing tactile RGB, optical flow and signal data.
2. Build episode-level dataset index.
3. Extract static and dynamic tactile features.
4. Construct train, validation and test splits.
5. Train three-finger tactile recognition models.
6. Evaluate classification and generalization performance.

## Development Environment

- Python
- NumPy
- PyTorch
- TorchVision
- OpenCV
- MuJoCo

Training dependencies are maintained in `tactile_learning/requirements.txt`.

## Data Management

Raw datasets, optical-flow archives, feature caches, checkpoints and training outputs are stored outside the Git repository.

Git tracks source code, configuration, tests and documentation.

## Git Workflow

- `main`: stable versions
- `develop`: integrated development
- `feature/*`: new features
- `fix/*`: bug fixes

Use descriptive commit messages:

- `feat:` new functionality
- `fix:` bug fixes
- `refactor:` code restructuring
- `test:` tests
- `docs:` documentation
- `chore:` maintenance

## Training

See `tactile_learning/README.md` for dataset preparation, feature extraction, training and evaluation commands.
