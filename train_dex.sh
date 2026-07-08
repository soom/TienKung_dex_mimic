#!/usr/bin/env bash
# Dex EVT 动作跟踪训练脚本（Pro 训练策略，支持 TeleopTeacher）
# 用法：bash train_dex.sh [smoke|medium|full] [motion_npz_or_dir] [num_gpus] [resume_run] [checkpoint]
#
# 示例：
#   bash train_dex.sh smoke                              # 冒烟测试（256 envs, 500 iter）
#   bash train_dex.sh medium                             # 中等规模（1024 envs, 5000 iter）
#   bash train_dex.sh full                               # 全量训练（默认 dataset/npz_dex）
#   bash train_dex.sh full motions/dex_motion.npz 2      # 显式传 motion + GPU 数
#   RESUME=true bash train_dex.sh full motions/dex_motion.npz 2 run_name model_500.pt
#   TASK_ID=Tracking-Flat-DexEVT-TeleopTeacher-v0 bash train_dex.sh full motions/dex_motion.npz 2
#   EPISODE_LENGTH_CAP_S=12 bash train_dex.sh full motions/dex_motion.npz 2
#
# 说明：
#   - 默认任务是 `Tracking-Flat-DexEVT-v0`
#   - TeleopTeacher 任务 `Tracking-Flat-DexEVT-TeleopTeacher-v0`
#   - 支持 smoke / medium / full 三种规模
#   - 多卡模式下使用 torchrun，将总 env 数均分
#   - 可通过 EPISODE_LENGTH_S / EPISODE_LENGTH_CAP_S / EPISODE_LENGTH_SCALE 控制回合长度
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

DEFAULT_MOTION_FILE="dataset/npz_dex/body_check_101__A206_with_transition.npz"

MODE="${1:-full}"
MOTION_FILE="${2:-$DEFAULT_MOTION_FILE}"
NUM_GPUS="${3:-${NUM_GPUS:-1}}"
TASK_ID="${TASK_ID:-Tracking-Flat-DexEVT-Simple-v0}"
RUN_PREFIX="${RUN_PREFIX:-dex_evt}"
RESUME="${RESUME:-false}"
RESUME_RUN="${4:-${RESUME_RUN:-}}"
RESUME_CHECKPOINT="${5:-${RESUME_CHECKPOINT:-}}"
EPISODE_LENGTH_S="${EPISODE_LENGTH_S:-}"
EPISODE_LENGTH_CAP_S="${EPISODE_LENGTH_CAP_S:-24.64}"
EPISODE_LENGTH_SCALE="${EPISODE_LENGTH_SCALE:-1.0}"
CURRICULUM_INITIAL_STAGE="${CURRICULUM_INITIAL_STAGE:-2}"
LOAD_CKPT="${LOAD_CKPT:-policy/wbt_model_46400.pt}"

if [ ! -f "$MOTION_FILE" ] && [ ! -d "$MOTION_FILE" ]; then
    echo "[ERROR] Motion file not found: $MOTION_FILE"
    echo "        Please provide a valid motion NPZ file or directory."
    exit 1
fi

if [[ "$TASK_ID" == *"TeleopTeacher"* && -z "$EPISODE_LENGTH_CAP_S" && -z "$EPISODE_LENGTH_S" ]]; then
    EPISODE_LENGTH_CAP_S="25.0"
fi

if [[ "$RESUME" != "true" && "$RESUME" != "false" ]]; then
    echo "[ERROR] RESUME must be true or false, got: $RESUME"
    exit 1
fi

if [[ "$RESUME" == "true" && ( -z "$RESUME_RUN" || -z "$RESUME_CHECKPOINT" ) ]]; then
    echo "[ERROR] RESUME=true requires resume_run and checkpoint arguments."
    exit 1
fi

if ! [[ "$NUM_GPUS" =~ ^[1-9][0-9]*$ ]]; then
    echo "[ERROR] Invalid num_gpus: $NUM_GPUS"
    echo "        Use a positive integer, e.g. 1 / 2 / 4."
    exit 1
fi

case "$MODE" in
    smoke)
        NUM_ENVS=256
        MAX_ITER=500
        RUN_NAME="smoke"
        ;;
    medium)
        NUM_ENVS=1024
        MAX_ITER=5000
        RUN_NAME="medium"
        ;;
    full)
        NUM_ENVS=12000
        MAX_ITER=50000
        RUN_NAME="full"
        ;;
    *)
        echo "[ERROR] Unknown mode: $MODE. Use smoke / medium / full."
        exit 1
        ;;
esac

if (( NUM_ENVS % NUM_GPUS != 0 )); then
    echo "[ERROR] Total envs ($NUM_ENVS) must be divisible by num_gpus ($NUM_GPUS)."
    exit 1
fi

if [[ -n "$EPISODE_LENGTH_S" ]] && ! [[ "$EPISODE_LENGTH_S" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
    echo "[ERROR] EPISODE_LENGTH_S must be a positive number, got: $EPISODE_LENGTH_S"
    exit 1
fi
if [[ -n "$EPISODE_LENGTH_CAP_S" ]] && ! [[ "$EPISODE_LENGTH_CAP_S" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
    echo "[ERROR] EPISODE_LENGTH_CAP_S must be a positive number, got: $EPISODE_LENGTH_CAP_S"
    exit 1
fi
if ! [[ "$EPISODE_LENGTH_SCALE" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
    echo "[ERROR] EPISODE_LENGTH_SCALE must be a positive number, got: $EPISODE_LENGTH_SCALE"
    exit 1
fi

NUM_ENVS_PER_GPU=$((NUM_ENVS / NUM_GPUS))
RUN_TIMESTAMP="$(date +"%m%d_%H%M%S")"

echo "========================================"
echo " Dex EVT Training — $MODE (Pro Strategy)"
echo "========================================"
echo " Task       : $TASK_ID"
echo " Motion     : $MOTION_FILE"
echo " Envs(total): $NUM_ENVS"
echo " Envs/GPU   : $NUM_ENVS_PER_GPU"
echo " GPUs       : $NUM_GPUS"
echo " Iterations : $MAX_ITER"
echo " Run Prefix : ${RUN_PREFIX}_${RUN_NAME}"
if [[ -n "$EPISODE_LENGTH_S" ]]; then
    echo " Episode(s) : override=${EPISODE_LENGTH_S}"
elif [[ -n "$EPISODE_LENGTH_CAP_S" ]]; then
    echo " Episode(s) : cap=${EPISODE_LENGTH_CAP_S}, scale=${EPISODE_LENGTH_SCALE}"
else
    echo " Episode(s) : auto-from-motion, scale=${EPISODE_LENGTH_SCALE}"
fi
if [[ "$RESUME" == "true" ]]; then
    echo " Resume     : $RESUME_RUN / $RESUME_CHECKPOINT"
else
    echo " Resume     : disabled (from scratch)"
fi
if [[ -n "$LOAD_CKPT" ]]; then
    echo " Load CKPT  : $LOAD_CKPT"
fi
echo "========================================"

RESUME_ARGS=()
if [[ "$RESUME" == "true" ]]; then
    RESUME_ARGS+=(
        --resume true
        --load_run "$RESUME_RUN"
        --checkpoint "$RESUME_CHECKPOINT"
    )
fi

LOAD_CKPT_ARGS=()
if [[ -n "$LOAD_CKPT" ]]; then
    LOAD_CKPT_ARGS+=(--load_ckpt "$LOAD_CKPT")
fi

EPISODE_ARGS=(--episode_length_scale "$EPISODE_LENGTH_SCALE")
if [[ -n "$EPISODE_LENGTH_CAP_S" ]]; then
    EPISODE_ARGS+=(--episode_length_cap_s "$EPISODE_LENGTH_CAP_S")
fi
if [[ -n "$EPISODE_LENGTH_S" ]]; then
    EPISODE_ARGS+=(--episode_length_s "$EPISODE_LENGTH_S")
fi
if [[ -n "$CURRICULUM_INITIAL_STAGE" ]]; then
    EPISODE_ARGS+=(--curriculum_initial_stage "$CURRICULUM_INITIAL_STAGE")
fi

if (( NUM_GPUS > 1 )); then
    WBT_RUN_TIMESTAMP="$RUN_TIMESTAMP" \
    python -m torch.distributed.run --nnodes=1 --nproc_per_node="$NUM_GPUS" \
        scripts/rsl_rl/train.py \
        --task "$TASK_ID" \
        --motion_file "$MOTION_FILE" \
        --num_envs "$NUM_ENVS_PER_GPU" \
        --max_iterations "$MAX_ITER" \
        --run_name "${RUN_PREFIX}_${RUN_NAME}" \
        --distributed \
        "${EPISODE_ARGS[@]}" \
        --headless \
        "${RESUME_ARGS[@]}" \
        "${LOAD_CKPT_ARGS[@]}"
else
    WBT_RUN_TIMESTAMP="$RUN_TIMESTAMP" \
    python scripts/rsl_rl/train.py \
        --task "$TASK_ID" \
        --motion_file "$MOTION_FILE" \
        --num_envs "$NUM_ENVS_PER_GPU" \
        --max_iterations "$MAX_ITER" \
        --run_name "${RUN_PREFIX}_${RUN_NAME}" \
        "${EPISODE_ARGS[@]}" \
        --headless \
        "${RESUME_ARGS[@]}" \
        "${LOAD_CKPT_ARGS[@]}"
fi

echo ""
echo "[DONE] Training complete. Logs: logs/rsl_rl/"
