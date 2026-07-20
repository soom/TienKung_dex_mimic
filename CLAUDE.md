# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Dex EVT whole-body motion tracking — training humanoid robots to imitate full-body motion from reference NPZ data. Built on **NVIDIA Isaac Lab 0.44.9** + **RSL RL 2.3.3**. The policy outputs residual joint position deltas on top of reference motion, learned via PPO.

```
source/whole_body_tracking/   # Python package (whole_body_tracking)
scripts/rsl_rl/               # Training, playback, CLI
scripts/                      # Data conversion, analysis, sim2sim
dataset/                      # Motion data (PKL raw → NPZ processed)
```

## Environment Setup

```bash
conda env create -f env.yaml          # one-time, creates env named "mimic"
conda activate mimic
pip install -e source/whole_body_tracking/
```

Requires a GPU with NVIDIA drivers and an Omniverse license for Isaac Sim 5.0.

## Key Commands

### Training

```bash
# Quick smoke test (256 envs, 500 iter)
bash train_dex.sh smoke

# Medium scale (1024 envs, 5000 iter)
bash train_dex.sh medium

# Full training (12000 envs, 50000 iter)
bash train_dex.sh full [motion_npz_or_dir] [num_gpus]

# Inherit weights from a pre-trained model (loads model_state_dict only, no optimizer)
RESUME=false LOAD_CKPT=policy/dex/wbt_model_46400.pt bash train_dex.sh full

# Resume from a previous run (loads full checkpoint including optimizer)
RESUME=true bash train_dex.sh full motions/dex.npz 2 run_name model_500.pt

# Override task, episode length, curriculum stage
TASK_ID=Tracking-Flat-DexEVT-Simple-v0 EPISODE_LENGTH_CAP_S=12 CURRICULUM_INITIAL_STAGE=2 bash train_dex.sh full
```

### Playback / ONNX Export

```bash
bash play_dex.sh                    # uses default checkpoint
LOAD_RUN=... CHECKPOINT=model_1400.pt bash play_dex.sh
```

### Sim2Sim (MuJoCo, no Isaac Lab dependency)

```bash
bash sim2sim_dex.sh                 # runs ONNX policy in standalone MuJoCo
```

### Monitoring & Analysis

```bash
bash log.sh                         # real-time TensorBoard metrics (watch -n1)
python scripts/analyze_training_log.py logs/rsl_rl/<run_dir>
```

### Data Preparation

```bash
bash batch_pkl_to_npz.sh            # PKL under dataset/pkl_dex/ → dataset/npz_dex/
python scripts/pkl_to_npz.py dataset/pkl_dex/file.pkl -o dataset/npz_dex/
```

## Architecture

### Task Registration Pattern

Gym environments are registered in `tasks/tracking/config/dex_evt/__init__.py` via `gym.register()`. The environment is composed through Isaac Lab's `@configclass` pattern:

```
DexEVTSimpleEnvCfg (simple_env_cfg.py)
├── scene         — ArticulationCfg + terrain + contact sensors
├── commands      — MotionCommand (loads NPZ motion data)
├── actions       — ResidualJointPositionAction
├── observations  — policy group + critic (privileged) group
├── rewards       — tracking errors (exp kernel) + contact penalties
├── terminations  — anchor/body position thresholds + timeout
└── events        — domain randomization (mass, CoM, friction, push)
```

The currently registered task is **`Tracking-Flat-DexEVT-Simple-v0`**. Older task variants (flat_env_cfg, teleop_teacher_env_cfg, standing_env_cfg) are no longer registered — only their `.pyc` cache files remain.

### MDP Layer (`tasks/tracking/mdp/`)

Each module defines functions with the Isaac Lab signature `(env: ManagerBasedRLEnv, ...) -> torch.Tensor`. These are wired into the env config via `RewTerm(func=mdp.<name>, weight=...).`

| Module | Role |
|---------|------|
| `commands.py` | `MotionCommand` — loads NPZ, manages curriculum phases, resamples motion |
| `actions.py` | `ResidualJointPositionAction` — `target = ref + action * scale`, with EMA filter |
| `actuators.py` | `DelayedImplicitActuator` — adds 0-3 step random communication delay |
| `rewards.py` | Tracking errors (exp kernel) + contact/regularization penalties |
| `observations.py` | Policy obs (phase, residuals, lookahead) + critic obs (privileged reference info) |
| `terminations.py` | `bad_anchor_pos_z_only`, `bad_anchor_ori`, `bad_anchor_ori_full`, `bad_motion_body_pos_z_only` |
| `events.py` | COM offset, actuator gain, body mass, physics material, push randomization |

### Residual Action Flow

The policy outputs `raw_action` (delta values per joint). The action term computes:
```
target_joint_pos = ref_joint_pos + raw_action * scale
```
Wrist and elbow_yaw joints have `action_scale=0` (passive PD tracking only) with high stiffness values (500/200).

### Noise Curriculum

A custom `MotionOnPolicyRunner` (`utils/my_on_policy_runner.py`) implements a noise std schedule:
- `init_std=0.35` → `final_std=0.05` (exponential decay over 8000 iterations)
- Entropy coefficient decays in sync: `0.012` → `0.004`

### Motion Data Pipeline

1. Raw PKL (pose sequences) → `scripts/pkl_to_npz.py` → NPZ with FK, velocities, transition frames
2. NPZ → `MotionLoader` (in `commands.py`) → concatenated, remapped to robot joint/body names
3. `MotionCommand` → env steps → policy receives residuals + lookahead → action computed

### PPO Agent Config (`agents/rsl_rl_ppo_cfg.py`)

- Actor/Critic: `[768, 384, 192]` hidden dims, ELU activation
- `num_steps_per_env=24`, `learning_rate=7e-4` (adaptive), `gamma=0.99`, `lam=0.95`
- 5 epochs, 4 mini-batches, `clip_param=0.2`, `desired_kl=0.01`

## Key Conventions

- **`train_dex.sh` configures via environment variables**, not positional args (except mode + motion + gpus). See header comment for all vars.
- **`RESUME=true`** restores full training state (model + optimizer). **`LOAD_CKPT`** loads model weights only (strict=False), fresh optimizer.
- **Logs** go to `logs/rsl_rl/dex_evt_fix/{timestamp}_{run_name}/`. Checkpoints saved every 200 iterations.
- **Robot is 29 DOF** after fixed joint merging: 6/leg, 3 waist, 7/arm. Joint names in `robots/dex_evt.py`.
- **No test suite exists** in this repository.
- **Not a git repository** — no version control, no `.gitignore`.
