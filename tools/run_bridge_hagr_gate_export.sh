#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MODEL="${MODEL:-$ROOT/checkpoints/hagr_ovd_s.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/world_granularity_router/hagr_gate_export}"
SCALE_NAMES="${SCALE_NAMES:-P3,P4,P5}"
NEAR_ZERO="${NEAR_ZERO:-1e-6}"
DPI="${DPI:-300}"

python3 tools/export_bridge_hagr_gates.py \
  --model "$MODEL" \
  --output-dir "$OUTPUT_DIR" \
  --scale-names "$SCALE_NAMES" \
  --near-zero "$NEAR_ZERO" \
  --dpi "$DPI"
