#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

DATA_ROOT="${DATA_ROOT:-$ROOT/data}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-$ROOT/checkpoints}"
BASE_DATASET="${BASE_DATASET:-$DATA_ROOT/bridge_grounding}"
BASE_MODEL="${BASE_MODEL:-$CHECKPOINT_ROOT/yolo_world_siglip2_aligned_s.pt}"
FLAT_MODEL="${FLAT_MODEL:-$CHECKPOINT_ROOT/hagr_ovd_flat_s.pt}"
HAGR_MODEL="${HAGR_MODEL:-$CHECKPOINT_ROOT/hagr_ovd_s.pt}"
OUTPUT="${OUTPUT:-$ROOT/runs/world_granularity_router/hagr_efficiency.json}"

case "${HALF:-1}" in
  1|true|TRUE|yes|YES) precision=(--half) ;;
  0|false|FALSE|no|NO) precision=(--no-half) ;;
  *) echo "HALF must be 1/0, true/false, or yes/no" >&2; exit 2 ;;
esac
case "${PROFILE_FLOPS:-1}" in
  1|true|TRUE|yes|YES) flops=(--profile-flops) ;;
  0|false|FALSE|no|NO) flops=(--no-profile-flops) ;;
  *) echo "PROFILE_FLOPS must be 1/0, true/false, or yes/no" >&2; exit 2 ;;
esac

python3 tools/eval_bridge_granularity_efficiency.py \
  --base-model "$BASE_MODEL" \
  --flat-model "$FLAT_MODEL" \
  --hagr-model "$HAGR_MODEL" \
  --base-dataset "$BASE_DATASET" \
  --category-data "${CATEGORY_DATA:-$BASE_DATASET/bridge_qwen_category_det.yaml}" \
  --attribute-data "${ATTRIBUTE_DATA:-$BASE_DATASET/bridge_qwen_attribute_phrase_det.yaml}" \
  --affordance-data "${AFFORDANCE_DATA:-$BASE_DATASET/bridge_qwen_affordance_phrase_det.yaml}" \
  --device "${DEVICE:-0}" \
  --imgsz "${IMGSZ:-640}" \
  --text-batch "${TEXT_BATCH:-64}" \
  --warmup "${WARMUP:-100}" \
  --iterations "${ITERATIONS:-500}" \
  --output "$OUTPUT" \
  "${precision[@]}" \
  "${flops[@]}"
