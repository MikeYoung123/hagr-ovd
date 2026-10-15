#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

ACTION="${1:-}"
DATA_ROOT="${DATA_ROOT:-$ROOT/data}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-$ROOT/checkpoints}"
BASE_DATASET="${BASE_DATASET:-$DATA_ROOT/bridge_grounding}"
TRI_DATASET="${TRI_DATASET:-$DATA_ROOT/bridge_grounding_tri_granular}"
PROJECT="${PROJECT:-$ROOT/runs/world_granularity_router}"
DIAG_DIR="${DIAG_DIR:-$PROJECT/diagnostics}"
STRATA="${STRATA:-$TRI_DATASET/bridge_eval_frequency_strata.json}"
FLAT_MODEL="${FLAT_MODEL:-$CHECKPOINT_ROOT/hagr_ovd_flat_s.pt}"
HAGR_MODEL="${HAGR_MODEL:-$CHECKPOINT_ROOT/hagr_ovd_s.pt}"
FLAT_OUTPUT="${FLAT_OUTPUT:-$DIAG_DIR/flat_diagnostics.json}"
HAGR_OUTPUT="${HAGR_OUTPUT:-$DIAG_DIR/hagr_diagnostics.json}"

build_strata() {
  python3 tools/build_bridge_eval_strata.py \
    --train-json "$TRI_DATASET/bridge_qwen_tri_granular_train.json" \
    --category-data "$BASE_DATASET/bridge_qwen_category_det.yaml" \
    --attribute-data "$BASE_DATASET/bridge_qwen_attribute_phrase_det.yaml" \
    --affordance-data "$BASE_DATASET/bridge_qwen_affordance_phrase_det.yaml" \
    --frequency-unit "${FREQUENCY_UNIT:-video}" \
    --rare-max "${RARE_MAX:-10}" \
    --common-max "${COMMON_MAX:-100}" \
    --output "$STRATA"
}

evaluate_model() {
  local model="$1"
  local output="$2"
  local cache_dir="$3"
  local extra=()
  [[ "${REBUILD_CACHE:-0}" == "1" ]] && extra+=(--rebuild-cache)
  [[ -f "$STRATA" ]] || build_strata
  python3 tools/eval_bridge_granularity_diagnostics.py \
    --model "$model" \
    --summary "$TRI_DATASET/summary_tri_granular.json" \
    --strata "$STRATA" \
    --base-dataset "$BASE_DATASET" \
    --category-data "$BASE_DATASET/bridge_qwen_category_det.yaml" \
    --attribute-data "$BASE_DATASET/bridge_qwen_attribute_phrase_det.yaml" \
    --affordance-data "$BASE_DATASET/bridge_qwen_affordance_phrase_det.yaml" \
    --device "${EVAL_DEVICE:-0}" \
    --batch "${EVAL_BATCH:-4}" \
    --workers "${EVAL_WORKERS:-4}" \
    --imgsz "${IMGSZ:-640}" \
    --text-batch "${TEXT_BATCH:-64}" \
    --conf "${CONF:-0.001}" \
    --iou "${NMS_IOU:-0.7}" \
    --max-det "${MAX_DET:-300}" \
    --cross-iou "${CROSS_IOU:-0.5}" \
    --gt-align-iou "${GT_ALIGN_IOU:-0.99}" \
    --cache-dir "$cache_dir" \
    --output "$output" \
    ${extra[@]+"${extra[@]}"}
}

compare_reports() {
  python3 tools/compare_bridge_granularity_diagnostics.py \
    --baseline "$FLAT_OUTPUT" \
    --candidate "$HAGR_OUTPUT" \
    --baseline-label flat \
    --candidate-label HAGR-v5 \
    --output-json "$DIAG_DIR/flat_vs_hagr.json" \
    --output-md "$DIAG_DIR/flat_vs_hagr.md"
}

case "$ACTION" in
  strata) build_strata ;;
  flat) evaluate_model "$FLAT_MODEL" "$FLAT_OUTPUT" "$DIAG_DIR/cache_flat" ;;
  hagr) evaluate_model "$HAGR_MODEL" "$HAGR_OUTPUT" "$DIAG_DIR/cache_hagr" ;;
  compare) compare_reports ;;
  all) build_strata; evaluate_model "$FLAT_MODEL" "$FLAT_OUTPUT" "$DIAG_DIR/cache_flat"; evaluate_model "$HAGR_MODEL" "$HAGR_OUTPUT" "$DIAG_DIR/cache_hagr"; compare_reports ;;
  *)
    echo "usage: bash tools/run_bridge_granularity_diagnostics.sh {strata|flat|hagr|compare|all}"
    echo "recommended: bash tools/run_bridge_granularity_diagnostics.sh all"
    echo "resume safely: run flat, hagr and compare separately; verified caches are reused"
    echo "override models: FLAT_MODEL=/path/flat.pt HAGR_MODEL=/path/hagr.pt bash tools/run_bridge_granularity_diagnostics.sh all"
    exit 2
    ;;
esac
