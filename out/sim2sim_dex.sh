#!/usr/bin/env bash
# Dex EVT sim2sim 运行脚本
#
# 用法：
#   bash sim2sim.sh                          # 默认：playlist 模式，加载 dataset/npz_pro，按 B 切换动作
#   bash sim2sim.sh --headless               # 无界面跑（用于 CI / 调试）
#   POLICY=path/to/policy.onnx bash sim2sim.sh
#   NPZ_DIR=dataset/npz_pro bash sim2sim.sh
#
# 按键：
#   B  — 切换到下一个 NPZ 动作（playlist 模式）

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
POLICY="${POLICY:-policy_merged.onnx}"

# ── 默认 NPZ 目录 ──────────────────────────────────────────────────────────
DEFAULT_NPZ_DIR="npz"
NPZ_DIR="${NPZ_DIR:-$DEFAULT_NPZ_DIR}"

# ── 默认 MJCF ──────────────────────────────────────────────────────────────
DEFAULT_MJCF="dex_evt/urdf/evt2.xml"
MJCF="${MJCF:-$DEFAULT_MJCF}"

NPZ_ARGS=(--npz-dir "$NPZ_DIR")
for arg in "$@"; do
    if [[ "$arg" == "--npz" || "$arg" == "--npz-dir" || "$arg" == --npz=* || "$arg" == --npz-dir=* ]]; then
        NPZ_ARGS=()
        break
    fi
done

# ── 验证路径 ────────────────────────────────────────────────────────────────
if [[ -z "$POLICY" ]] || [[ ! -f "$POLICY" ]]; then
    echo "[ERROR] Policy ONNX not found: ${POLICY:-<empty>}"
    echo "        Export a policy first, or set POLICY=path/to/policy.onnx"
    exit 1
fi
if [[ ${#NPZ_ARGS[@]} -gt 0 && ! -d "$NPZ_DIR" ]]; then
    echo "[ERROR] NPZ directory not found: $NPZ_DIR"
    exit 1
fi
if [[ ! -f "$MJCF" ]]; then
    echo "[ERROR] MJCF not found: $MJCF"
    exit 1
fi

echo "========================================"
echo " Dex EVT sim2sim — playlist mode"
echo "========================================"
echo " Policy  : $POLICY"
echo " NPZ dir : $NPZ_DIR"
echo " MJCF    : $MJCF"
echo " 按键 B  : 切换到下一个动作"
echo "========================================"

python sim2sim_dex.py \
    --policy "$POLICY" \
    --mjcf   "$MJCF" \
    --sim-dt 0.0025 \
    "${NPZ_ARGS[@]}" \
    "$@"
