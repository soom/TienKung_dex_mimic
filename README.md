# Dex EVT — Whole-Body Motion Tracking

基于 NVIDIA Isaac Lab 的全身运动跟踪强化学习训练框架。策略网络以**残差关节位置**（residual joint position）的形式输出动作，叠加在参考运动数据上，通过 PPO 训练使仿人机器人复现参考动作。

<video controls width="100%" src="https://github.com/user-attachments/assets/3f8b4aa4-edd3-4b71-be1f-2cdad4333444"></video>

## 环境要求

- **GPU** + NVIDIA 驱动（支持 CUDA 12.8+）
- **Omniverse 许可证**（Isaac Sim 5.0 需要）
- **Conda** 环境管理

## 快速开始

```bash
# 1. 创建环境（仅首次）
conda env create -f env.yaml
conda activate mimic

# 2. 安装本项目的 Python 包
pip install -e source/whole_body_tracking/

# 3. 数据准备：将 PKL 原始数据转为 NPZ
bash batch_pkl_to_npz.sh

# 4. 冒烟测试（小规模，验证环境）
bash train_dex.sh smoke

# 5. 全量训练
bash train_dex.sh full
```

## 训练

```bash
# 基本用法
bash train_dex.sh [smoke|medium|full] [motion_file] [num_gpus]

# 从预训练模型继承权重继续训练（仅加载模型权重，优化器重新初始化）
LOAD_CKPT=policy/wbt_model_46400.pt bash train_dex.sh full

# 恢复中断的训练（完整恢复，含优化器状态）
RESUME=true bash train_dex.sh full motions/dex.npz 2 run_name model_500.pt

# 替换任务或调整回合长度
TASK_ID=Tracking-Flat-DexEVT-Simple-v0 EPISODE_LENGTH_CAP_S=12 bash train_dex.sh full
```

### 训练规模

| 模式 | 环境数 | 迭代数 | 用途 |
|------|--------|--------|------|
| `smoke` | 256 | 500 | 快速冒烟测试 |
| `medium` | 1024 | 5000 | 中等规模验证 |
| `full` | 12000 | 50000 | 全量训练 |

训练日志和模型保存于 `logs/rsl_rl/dex_evt_fix/{时间戳}_{run_name}/`，每 200 轮保存一次 checkpoint。

### 多 GPU

```bash
bash train_dex.sh full dataset/npz_dex 4
```

## 评估与导出

```bash
# 播放训练好的策略（自动导出 ONNX）
bash play_dex.sh

# 指定 checkpoint
LOAD_RUN=<run_name> CHECKPOINT=model_1400.pt bash play_dex.sh
```

## Sim2Sim（MuJoCo 独立推理）

无需 Isaac Lab，纯 MuJoCo 运行导出的 ONNX 策略：

```bash
bash sim2sim_dex.sh
```

## 监控与分析

```bash
# 实时监控训练指标
bash log.sh

# 分析日志
python scripts/analyze_training_log.py logs/rsl_rl/<run_dir>
```

## 项目结构

```
dex_wbt/
├── train_dex.sh              # 训练入口
├── play_dex.sh               # 评估 & ONNX 导出
├── sim2sim_dex.sh            # MuJoCo Sim2Sim 部署
├── env.yaml                  # Conda 环境锁定文件
├── dataset/                  # 运动数据（PKL → NPZ）
├── source/whole_body_tracking/  # Python 包
│   └── whole_body_tracking/
│       ├── robots/           # 机器人配置（Dex EVT, 29 DOF）
│       ├── tasks/tracking/   # 任务定义
│       │   ├── config/dex_evt/  # 环境 & PPO 配置
│       │   └── mdp/             # 奖励 / 观测 / 终止 / 动作
│       └── utils/            # ONNX 导出、自定义 Runner
├── scripts/
│   ├── rsl_rl/               # 训练 & 评估脚本
│   ├── pkl_to_npz.py         # 数据格式转换
│   ├── analyze_training_log.py # TensorBoard 日志分析
│   └── sim2sim_dex.py        # MuJoCo 推理
├── logs/                     # 训练日志 & checkpoint
└── policy/                   # 预训练模型
```

## 预训练

当前预训练模型使用 **~200 个动作片段**（`dataset/npz_dex/`）进行训练，涵盖行走、转身、伸展、蹲起、推拉、交互等日常全身动作。

预训练策略暂未开源，感兴趣可以联系微信: soommm

## 双头策略架构

推理时采用**双头模型**（standing + tracking），由 `utils/exporter.py` 合并导出为单一 ONNX：

```
phase_mode = 0 (STAND)    →  action = standing_head(obs)
phase_mode = 1 (TRACKING) →  action = tracking_head(obs)
```

- **Standing 头**：在站立/待机阶段保持稳定姿态，低噪声（init_std=0.05）
- **Tracking 头**：在运动阶段全身跟踪参考动作，noise curriculum 逐步收敛
- 两个头共享观测输入，根据 `MotionCommand` 的 phase_mode 信号动态切换
- 网络结构相同：`[768, 384, 192]` 隐藏层，ELU 激活

## 架构简介

- **任务**：`Tracking-Flat-DexEVT-Simple-v0`（Gym 环境注册）
- **动作空间**：残差关节位置 — `target = ref + action × scale`
- **观测**：运动相位、关节残差、lookahead 参考帧（策略）/ 特权信息（critic）
- **奖励**：锚点/身体/关节跟踪误差（指数核）+ 接触惩罚 + 正则化
- **PPO**：`[768, 384, 192]` 隐藏层，ELU，学习率 `7e-4`，自适应调度
- **Noise Curriculum**：`0.35 → 0.05`（8000 轮指数衰减），熵同步衰减
- **双头合并**：Standing + Tracking 双头合并为单个 `policy_merged.onnx`，根据 phase_mode 路由

## 机器人

**Dex EVT** — 29 DOF 人形机器人：

- 腿部 × 2：髋 pitch/roll/yaw、膝 pitch、踝 pitch/roll（各 6 DOF）
- 腰部 × 3：yaw/roll/pitch
- 手臂 × 2：肩 pitch/roll/yaw、肘 pitch/yaw、腕 pitch/roll（各 7 DOF）

腕部和肘部 yaw 关节的 `action_scale=0`，通过 PD 控制器被动跟随参考运动。
