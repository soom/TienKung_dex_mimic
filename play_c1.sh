#!/usr/bin/env bash
# Walker C1 回放与导出脚本
# 用法：bash play_c1.sh
#
# 环境变量覆盖：
#   TASK                   — Gym 任务 ID（默认 Tracking-Flat-WalkerC1-Simple-v0）
#   MOTION_FILE            — 动作 NPZ 文件路径
#   CHECKPOINT_PATH        — 基础模型路径
#   NUM_ENVS               — 并行 env 数（默认 1）
#   START_FIRST_FRAME=1    — 强制从 motion 首帧开始
#   EXPORT_ROLLOUT=1       — 导出 rollout 为 NPZ
#   COMPARE_AFTER_EXPORT=1 — 导出后自动对比
#   VIDEO=1                — 录制视频
#   MAX_STEPS=<N>          — 最大步数限制
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

TASK="${TASK:-Tracking-Flat-WalkerC1-Simple-v0}"
MOTION_FILE="${MOTION_FILE:-dataset/npz_c1/take004_chr01_with_transition.npz}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-logs/rsl_rl/walker_c1_fix/0720_042712_walker_c1_full/model_2000.pt}"
NUM_ENVS="${NUM_ENVS:-1}"
START_FIRST_FRAME="${START_FIRST_FRAME:-1}"
HEADLESS="${HEADLESS:-0}"
VIDEO="${VIDEO:-0}"
VIDEO_LENGTH="${VIDEO_LENGTH:-200}"
MAX_STEPS="${MAX_STEPS:-}"
EXPORT_ROLLOUT="${EXPORT_ROLLOUT:-0}"
COMPARE_AFTER_EXPORT="${COMPARE_AFTER_EXPORT:-0}"

STANDING_CHECKPOINT="${STANDING_CHECKPOINT:-policy/c1/standing_model_2600.pt}"

ANALYSIS_DIR="${ANALYSIS_DIR:-outputs/play_c1}"
EXPORT_ROLLOUT_NPZ="${EXPORT_ROLLOUT_NPZ:-$ANALYSIS_DIR/rollout_$(basename "${CHECKPOINT_PATH%.pt}").npz}"

if [[ ! -f "$MOTION_FILE" ]]; then
    echo "[ERROR] Motion file not found: $MOTION_FILE"
    exit 1
fi

if [[ ! -f "$CHECKPOINT_PATH" ]]; then
    echo "[ERROR] Tracking checkpoint not found: $CHECKPOINT_PATH"
    exit 1
fi

if [[ -n "$STANDING_CHECKPOINT" && ! -f "$STANDING_CHECKPOINT" ]]; then
    echo "[ERROR] Standing checkpoint not found: $STANDING_CHECKPOINT"
    exit 1
fi

echo "========================================"
echo " Walker C1 Playback"
echo "========================================"
echo " Task      : $TASK"
echo " Motion    : $MOTION_FILE"
echo " Tracking  : $CHECKPOINT_PATH"
echo " Num envs  : $NUM_ENVS"
echo " Start@0   : $START_FIRST_FRAME"
echo " Export    : $EXPORT_ROLLOUT"
echo " Standing  : $STANDING_CHECKPOINT"
echo "========================================"

PLAY_ARGS=(
    --task "$TASK"
    --motion_file "$MOTION_FILE"
    --checkpoint_path "$CHECKPOINT_PATH"
    --num_envs "$NUM_ENVS"
)

if [[ "$START_FIRST_FRAME" == "1" ]]; then
    PLAY_ARGS+=(--start_first_frame)
fi

if [[ "$HEADLESS" == "1" ]]; then
    PLAY_ARGS+=(--headless)
fi

if [[ "$VIDEO" == "1" ]]; then
    PLAY_ARGS+=(--video --video_length "$VIDEO_LENGTH")
fi

if [[ -n "$MAX_STEPS" ]]; then
    PLAY_ARGS+=(--max_steps "$MAX_STEPS")
fi

PLAY_ARGS+=(--standing_checkpoint "$STANDING_CHECKPOINT")

if [[ "$EXPORT_ROLLOUT" == "1" ]]; then
    mkdir -p "$ANALYSIS_DIR"
    PLAY_ARGS+=(--export_rollout_npz "$EXPORT_ROLLOUT_NPZ") 
fi

python scripts/rsl_rl/play.py "${PLAY_ARGS[@]}" --headless

if [[ "$EXPORT_ROLLOUT" == "1" && "$COMPARE_AFTER_EXPORT" == "1" ]]; then
    echo ""
    echo "========================================"
    echo " Rollout Comparison"
    echo "========================================"
    python scripts/compare_rollout_npz.py "$EXPORT_ROLLOUT_NPZ" --motion_file "$MOTION_FILE" --top_joints 0
fi
