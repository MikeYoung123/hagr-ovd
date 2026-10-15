#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

ACTION="${1:-}"
DATA_ROOT="${DATA_ROOT:-$ROOT/data}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-$ROOT/checkpoints}"
BASE_DATASET="${BASE_DATASET:-$DATA_ROOT/bridge_grounding}"
TRI_DATASET="${TRI_DATASET:-$DATA_ROOT/bridge_grounding_tri_granular}"
BASE_MODEL="${BASE_MODEL:-$ROOT/runs/bridge_roles/siglip_bridge_attribute/weights/best.pt}"
PROJECT="${PROJECT:-$ROOT/runs/world_granularity_router}"
VARIANT="${VARIANT:-full}"
case "$VARIANT" in
  full) ;;
  flat) CATEGORY_PRIOR=0.0 ;;
  no_stop_gradient) STOP_GRADIENT=0 ;;
  no_retention) ATTRIBUTE_PRESERVE_GAIN=0.0 ;;
  *) echo "VARIANT must be full, flat, no_stop_gradient or no_retention" >&2; exit 2 ;;
esac
RUN_NAME="${RUN_NAME:-siglip2_bridge_hagr_$VARIANT}"
MODEL="${MODEL:-$PROJECT/$RUN_NAME/weights/best.pt}"

train() {
  local extra=()
  local stop_gradient=()
  [[ "${EXIST_OK:-0}" == "1" ]] && extra+=(--exist-ok)
  [[ "${COS_LR:-0}" == "1" ]] && extra+=(--cos-lr)
  case "${STOP_GRADIENT:-1}" in
    1|true|TRUE|yes|YES) stop_gradient=(--stop-gradient) ;;
    0|false|FALSE|no|NO) stop_gradient=(--no-stop-gradient) ;;
    *) echo "STOP_GRADIENT must be 1/0, true/false, or yes/no" >&2; exit 2 ;;
  esac
  python3 tools/train_yolo_world_siglip2_granularity_router.py \
    --model "$BASE_MODEL" \
    --train-json "$TRI_DATASET/bridge_qwen_tri_granular_train.json" \
    --train-vocab "$TRI_DATASET/bridge_qwen_tri_granular_train_vocab.json" \
    --image-dir "$BASE_DATASET/images" \
    --category-data "$BASE_DATASET/bridge_qwen_category_det.yaml" \
    --attribute-data "$BASE_DATASET/bridge_qwen_attribute_phrase_det.yaml" \
    --affordance-data "$BASE_DATASET/bridge_qwen_affordance_phrase_det.yaml" \
    --device "${DEVICE:-0,1,2,3}" \
    --epochs "${EPOCHS:-40}" \
    --batch "${BATCH:-64}" \
    --workers "${WORKERS:-8}" \
    --imgsz "${IMGSZ:-640}" \
    --project "$PROJECT" \
    --name "$RUN_NAME" \
    --hidden "${HIDDEN:-192}" \
    --shared-lr-mult "${SHARED_LR_MULT:-0.1}" \
    --train-stage "${TRAIN_STAGE:-full}" \
    --selection-role "${SELECTION_ROLE:-auto}" \
    --category-prior "${CATEGORY_PRIOR:-1.0}" \
    --inheritance-max "${INHERITANCE_MAX:-0.5}" \
    --category-gain "${CATEGORY_GAIN:-1.0}" \
    --attribute-gain "${ATTRIBUTE_GAIN:-1.0}" \
    --affordance-gain "${AFFORDANCE_GAIN:-1.0}" \
    --attribute-preserve-gain "${ATTRIBUTE_PRESERVE_GAIN:-1.0}" \
    --max-supervision "${MAX_SUPERVISION:-48}" \
    --lr0 "${LR0:-0.0003}" \
    --lrf "${LRF:-0.1}" \
    --patience "${PATIENCE:-15}" \
    --save-period "${SAVE_PERIOD:-5}" \
    "${stop_gradient[@]}" \
    ${extra[@]+"${extra[@]}"}
}

evaluate() {
  python3 tools/eval_bridge_tri_granular.py \
    --model "$MODEL" \
    --base-dataset "$BASE_DATASET" \
    --tri-dataset "$TRI_DATASET" \
    --train-vocab "$TRI_DATASET/bridge_qwen_tri_granular_train_vocab.json" \
    --category-data "$BASE_DATASET/bridge_qwen_category_det.yaml" \
    --attribute-data "$BASE_DATASET/bridge_qwen_attribute_phrase_det.yaml" \
    --affordance-data "$BASE_DATASET/bridge_qwen_affordance_phrase_det.yaml" \
    --device "${EVAL_DEVICE:-0}" \
    --batch "${EVAL_BATCH:-4}" \
    --workers "${EVAL_WORKERS:-4}" \
    --imgsz "${IMGSZ:-640}" \
    --output "${EVAL_OUTPUT:-$PROJECT/$RUN_NAME/eval_tri_granular.json}"
}

case "$ACTION" in
  train) train ;;
  eval) evaluate ;;
  all) train; evaluate ;;
  *)
    echo "usage: VARIANT={full|flat|no_stop_gradient|no_retention} bash tools/run_bridge_granularity_router.sh {train|eval|all}"
    echo "Set BASE_MODEL to the SAME Bridge-adapted SigLIP2 checkpoint for all matched variants."
    exit 2
    ;;
esac
