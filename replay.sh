#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# MJCF="${MJCF:-ros_deploy_pro/src/tienkung2_pro/assets/mjcf/tienkung_new.xml}"
MJCF="${MJCF:-source/whole_body_tracking/whole_body_tracking/assets/walker_c1/mjcf/walker_astron.xml}"
NPZ_DIR="${NPZ_DIR:-dataset/npz_c1}"
START_INDEX="${START_INDEX:-0}"

python scripts/replay_npz_mujoco.py \
    --mjcf "$MJCF" \
    --npz-dir "$NPZ_DIR" \
    --start-index "$START_INDEX" \
    "$@"
