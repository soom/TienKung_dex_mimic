# Dex EVT / Walker C1 — Whole-Body Motion Tracking

基于 NVIDIA Isaac Lab 的双机器人全身运动跟踪强化学习框架，目前支持：

- **Dex EVT**：`Tracking-Flat-DexEVT-Simple-v0`
- **Walker C1**：`Tracking-Flat-WalkerC1-Simple-v0`

策略以残差关节位置形式输出动作：

```text
target_joint_pos = reference_joint_pos + action × scale
```

训练仅使用各机器人的 Simple tracking 环境。推理时分别加载 tracking 模型与预训练 standing 模型，并合并导出双头 ONNX；本项目不提供 standing 模型训练任务。

## 环境要求

- NVIDIA GPU 与 CUDA 驱动
- Isaac Sim 5.0 / Isaac Lab
- Conda

## 安装

```bash
conda env create -f env.yaml
conda activate mimic
pip install -e source/whole_body_tracking/
```

## 模型布局

两台机器人的模型必须分开存放，不能交叉使用：

```text
policy/
├── dex/
│   ├── wbt_model_46400.pt       # Dex tracking 基础模型
│   └── standing_model_2000.pt  # Dex play 合并用 standing 模型
└── c1/
    ├── wbt_model_26600.pt       # C1 tracking 基础模型
    └── standing_model_2600.pt  # C1 play 合并用 standing 模型
```

standing checkpoint 仅供 `play` 合并，不参与本项目训练。

## 训练

### 从头训练

```bash
# Dex
RESUME=false bash train_dex.sh smoke
RESUME=false bash train_dex.sh full dataset/npz_dex 1

# C1
RESUME=false bash train_c1.sh smoke
RESUME=false bash train_c1.sh full dataset/npz_c1 1
```

### 继承基础模型

`LOAD_CKPT` 只加载模型权重，优化器重新初始化，并创建新的训练 run：

```bash
RESUME=false LOAD_CKPT=policy/dex/wbt_model_46400.pt \
  bash train_dex.sh full dataset/npz_dex 1

RESUME=false LOAD_CKPT=policy/c1/wbt_model_26600.pt \
  bash train_c1.sh full dataset/npz_c1 1
```

### 恢复中断训练

`RESUME=true` 从 `logs/rsl_rl/` 恢复模型、优化器及训练状态：

```bash
RESUME=true bash train_dex.sh full dataset/npz_dex 1 <run_name> model_500.pt
RESUME=true bash train_c1.sh full dataset/npz_c1 1 <run_name> model_500.pt
```

`RESUME=true` 与 `LOAD_CKPT` 互斥，不能同时设置。

### 训练规模

| 模式 | 环境数 | 迭代数 | 用途 |
| --- | ---: | ---: | --- |
| `smoke` | 256 | 500 | 环境与配置冒烟测试 |
| `medium` | 1024 | 5000 | 中等规模验证 |
| `full` | 12000 | 50000 | 全量训练 |

多 GPU 示例：

```bash
RESUME=false bash train_dex.sh full dataset/npz_dex 4
RESUME=false bash train_c1.sh full dataset/npz_c1 4
```

训练日志分别写入：

```text
logs/rsl_rl/dex_evt_fix/
logs/rsl_rl/walker_c1_fix/
```

## 播放与 ONNX 合并

### 默认播放

```bash
bash play_dex.sh
bash play_c1.sh
```

默认模型对应关系：

| 脚本 | Tracking 模型 | Standing 模型 | 任务 |
| --- | --- | --- | --- |
| `play_dex.sh` | `policy/dex/wbt_model_46400.pt` | `policy/dex/standing_model_2000.pt` | `Tracking-Flat-DexEVT-Simple-v0` |
| `play_c1.sh` | `policy/c1/wbt_model_26600.pt` | `policy/c1/standing_model_2600.pt` | `Tracking-Flat-WalkerC1-Simple-v0` |

两个模型都存在时，`scripts/rsl_rl/play.py` 自动导出：

```text
policy/<robot>/exported/policy.onnx
policy/<robot>/exported/policy_merged.onnx
```

部署应使用 `policy_merged.onnx`：

```text
phase_mode = 0 (STAND)    → standing head
phase_mode = 1 (TRACKING) → tracking head
```

任一 tracking 或 standing checkpoint 缺失时，播放脚本会直接报错，不会静默导出不完整模型。

### 覆盖模型或动作

```bash
CHECKPOINT_PATH=/path/to/tracking.pt \
STANDING_CHECKPOINT=/path/to/standing.pt \
MOTION_FILE=/path/to/motion.npz \
bash play_dex.sh
```

C1 使用相同的环境变量调用 `play_c1.sh`。

常用播放参数：

```bash
HEADLESS=1 MAX_STEPS=1000 bash play_dex.sh
VIDEO=1 VIDEO_LENGTH=500 bash play_c1.sh
EXPORT_ROLLOUT=1 bash play_dex.sh
```

## Sim2Sim

使用合并后的 ONNX 在 MuJoCo 中运行：

```bash
bash sim2sim_dex.sh
bash sim2sim_c1.sh
```

## 数据准备

```bash
# 批量转换 Dex 数据
bash batch_pkl_to_npz.sh

# 单文件转换
python scripts/pkl_to_npz.py input.pkl -o dataset/npz_dex/
```

数据目录按机器人区分：

```text
dataset/npz_dex/
dataset/npz_c1/
```

## 项目结构

```text
dex_wbt/
├── train_dex.sh
├── train_c1.sh
├── play_dex.sh
├── play_c1.sh
├── sim2sim_dex.sh
├── sim2sim_c1.sh
├── dataset/
│   ├── npz_dex/
│   └── npz_c1/
├── policy/
│   ├── dex/
│   └── c1/
├── scripts/
│   ├── rsl_rl/
│   ├── sim2sim_dex.py
│   └── sim2sim_c1.py
└── source/whole_body_tracking/whole_body_tracking/
    ├── assets/
    │   ├── dex_evt/
    │   └── walker_c1/
    ├── robots/
    │   ├── dex_evt.py
    │   └── walker_c1.py
    └── tasks/tracking/config/
        ├── dex_evt/
        │   └── simple_env_cfg.py
        └── walker_c1/
            └── simple_env_cfg.py
```

## 训练架构

- 动作：参考关节位置上的 residual joint position
- 观测：运动相位、关节残差、lookahead 参考帧及 critic 特权信息
- 奖励：锚点、身体、关节跟踪误差，接触惩罚和动作正则化
- PPO 网络：`[768, 384, 192]`，ELU
- Noise curriculum：`0.35 → 0.05`
- 双头推理：standing 与 tracking 根据 `phase_mode` 路由

## 监控与分析

```bash
bash log.sh
python scripts/analyze_training_log.py logs/rsl_rl/<experiment>/<run>
```
