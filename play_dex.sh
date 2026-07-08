#!/usr/bin/env bash
# Dex EVT 回放与导出脚本
# 用法：bash play_dex.sh
#
# 环境变量覆盖：
#   TASK                   — Gym 任务 ID（默认 Tracking-Flat-DexEVT-v0）
#   MOTION_FILE            — 动作 NPZ 文件路径
#   LOAD_RUN               — 训练运行的 run 名称
#   CHECKPOINT             — 模型文件名（如 model_500.pt）
#   NUM_ENVS               — 并行 env 数（默认 1）
#   START_FIRST_FRAME=1    — 强制从 motion 首帧开始
#   EXPORT_ROLLOUT=1       — 导出 rollout 为 NPZ
#   COMPARE_AFTER_EXPORT=1 — 导出后自动对比
#   VIDEO=1                — 录制视频
#   MAX_STEPS=<N>          — 最大步数限制
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

TASK="${TASK:-Tracking-Flat-DexEVT-Simple-v0}"
MOTION_FILE="${MOTION_FILE:-dataset/npz_dex/body_check_101__A206_with_transition.npz}"
LOAD_RUN="${LOAD_RUN:-0707_230600_dex_evt_full}"
CHECKPOINT="${CHECKPOINT:-model_1400.pt}"
NUM_ENVS="${NUM_ENVS:-1}"
START_FIRST_FRAME="${START_FIRST_FRAME:-1}"
HEADLESS="${HEADLESS:-0}"
VIDEO="${VIDEO:-0}"
VIDEO_LENGTH="${VIDEO_LENGTH:-200}"
MAX_STEPS="${MAX_STEPS:-}"
EXPORT_ROLLOUT="${EXPORT_ROLLOUT:-0}"
COMPARE_AFTER_EXPORT="${COMPARE_AFTER_EXPORT:-0}"

STANDING_CHECKPOINT="${STANDING_CHECKPOINT:-policy/standing_model_2000.pt}"

ANALYSIS_DIR="${ANALYSIS_DIR:-logs/rsl_rl/dex_evt_fix/$LOAD_RUN/analysis}"
EXPORT_ROLLOUT_NPZ="${EXPORT_ROLLOUT_NPZ:-$ANALYSIS_DIR/rollout_${CHECKPOINT%.pt}.npz}"

if [[ ! -f "$MOTION_FILE" ]]; then
    echo "[ERROR] Motion file not found: $MOTION_FILE"
    exit 1
fi

if [[ -z "$LOAD_RUN" ]]; then
    echo "[ERROR] LOAD_RUN must be set (run name in logs/rsl_rl/dex_evt_fix/)."
    exit 1
fi

if [[ ! -f "logs/rsl_rl/dex_evt_fix/$LOAD_RUN/$CHECKPOINT" ]]; then
    echo "[ERROR] Checkpoint not found: logs/rsl_rl/dex_evt_fix/$LOAD_RUN/$CHECKPOINT"
    exit 1
fi

echo "========================================"
echo " Dex EVT Playback"
echo "========================================"
echo " Task      : $TASK"
echo " Motion    : $MOTION_FILE"
echo " Load run  : $LOAD_RUN"
echo " Checkpoint: $CHECKPOINT"
echo " Num envs  : $NUM_ENVS"
echo " Start@0   : $START_FIRST_FRAME"
echo " Export    : $EXPORT_ROLLOUT"
echo " Standing  : ${STANDING_CHECKPOINT:-}"
echo "========================================"

PLAY_ARGS=(
    --task "$TASK"
    --motion_file "$MOTION_FILE"
    --load_run "$LOAD_RUN"
    --checkpoint "$CHECKPOINT"
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

if [[ -n "$STANDING_CHECKPOINT" ]]; then
    PLAY_ARGS+=( --standing_checkpoint "$STANDING_CHECKPOINT" )
fi

if [[ "$EXPORT_ROLLOUT" == "1" ]]; then
    mkdir -p "$ANALYSIS_DIR"
    PLAY_ARGS+=(--export_rollout_npz "$EXPORT_ROLLOUT_NPZ")
fi

python scripts/rsl_rl/play.py "${PLAY_ARGS[@]}" --headless
