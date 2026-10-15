#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

ACTION="${1:-}"
DATA_ROOT="${DATA_ROOT:-$ROOT/data}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-$ROOT/checkpoints}"
BASE_DATASET="${BASE_DATASET:-$DATA_ROOT/bridge_grounding}"
TRI_DATASET="${TRI_DATASET:-$DATA_ROOT/bridge_grounding_tri_granular}"
PROPAGATED_DATASET="${PROPAGATED_DATASET:-$DATA_ROOT/bridge_temporal_propagated}"
TEMPORAL_DATASET="${TEMPORAL_DATASET:-$DATA_ROOT/bridge_grounding_temporal_joint}"
PROJECT="${PROJECT:-$ROOT/runs/world_temporal_joint}"
RUN_NAME="${RUN_NAME:-siglip2_bridge_hagr_temporal_joint}"
BASE_MODEL="${BASE_MODEL:-$CHECKPOINT_ROOT/hagr_ovd_s.pt}"
MODEL="${MODEL:-$PROJECT/$RUN_NAME/weights/best.pt}"

train() {
  local extra=()
  [[ "${EXIST_OK:-0}" == "1" ]] && extra+=(--exist-ok)
  [[ "${COS_LR:-0}" == "1" ]] && extra+=(--cos-lr)
  python3 tools/train_yolo_world_siglip2_temporal_joint.py \
    --model "$BASE_MODEL" \
    --original-train-json "$TRI_DATASET/bridge_qwen_tri_granular_train.json" \
    --train-vocab "$TRI_DATASET/bridge_qwen_tri_granular_train_vocab.json" \
    --temporal-train-json "$TEMPORAL_DATASET/bridge_qwen_tri_granular_temporal_train.json" \
    --original-image-dir "$BASE_DATASET/images" \
    --temporal-image-dir "$PROPAGATED_DATASET/images/train" \
    --category-data "$BASE_DATASET/bridge_qwen_category_det.yaml" \
    --attribute-data "$BASE_DATASET/bridge_qwen_attribute_phrase_det.yaml" \
    --affordance-data "$BASE_DATASET/bridge_qwen_affordance_phrase_det.yaml" \
    --selection-role "${SELECTION_ROLE:-attribute}" \
    --device "${DEVICE:-0,1,2,3}" \
    --epochs "${EPOCHS:-50}" \
    --batch "${BATCH:-64}" \
    --workers "${WORKERS:-8}" \
    --imgsz "${IMGSZ:-640}" \
    --project "$PROJECT" \
    --name "$RUN_NAME" \
    --hidden "${HIDDEN:-192}" \
    --shared-lr-mult "${SHARED_LR_MULT:-0.1}" \
    --box-lr-mult "${BOX_LR_MULT:-0.25}" \
    --category-prior "${CATEGORY_PRIOR:-1.0}" \
    --inheritance-max "${INHERITANCE_MAX:-0.5}" \
    --category-gain "${CATEGORY_GAIN:-1.0}" \
    --attribute-gain "${ATTRIBUTE_GAIN:-1.0}" \
    --affordance-gain "${AFFORDANCE_GAIN:-1.0}" \
    --attribute-preserve-gain "${ATTRIBUTE_PRESERVE_GAIN:-1.0}" \
    --max-supervision "${MAX_SUPERVISION:-48}" \
    --lr0 "${LR0:-0.0001}" \
    --lrf "${LRF:-0.1}" \
    --warmup-epochs "${WARMUP_EPOCHS:-0.5}" \
    --weight-decay "${WEIGHT_DECAY:-0.0001}" \
    --close-mosaic "${CLOSE_MOSAIC:-5}" \
    --patience "${PATIENCE:-6}" \
    --save-period "${SAVE_PERIOD:-1}" \
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
  *) echo "usage: bash tools/run_bridge_temporal_joint.sh {train|eval|all}"; exit 2 ;;
esac
