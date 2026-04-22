#!/usr/bin/env bash
set -euo pipefail

CONFIG="config.method1_detection.toml"
DATE_TAG="$(date +%Y%m%d)"
PRED_OUT="outputs/yolo_pred_bboxes_segmentation_${DATE_TAG}"
HANDOFF_JSON="outputs/handoff/bbox_manifest_predicted_${DATE_TAG}.json"
HANDOFF_CSV="outputs/handoff/bbox_manifest_predicted_${DATE_TAG}.csv"
SKIP_TRAIN=0
SKIP_EVAL=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      CONFIG="$2"
      shift 2
      ;;
    --pred-out)
      PRED_OUT="$2"
      shift 2
      ;;
    --handoff-json)
      HANDOFF_JSON="$2"
      shift 2
      ;;
    --handoff-csv)
      HANDOFF_CSV="$2"
      shift 2
      ;;
    --skip-train)
      SKIP_TRAIN=1
      shift
      ;;
    --skip-eval)
      SKIP_EVAL=1
      shift
      ;;
    *)
      echo "Unknown arg: $1"
      exit 1
      ;;
  esac
done

if [[ $SKIP_TRAIN -eq 0 ]]; then
  echo "[1/4] Training YOLO detector with ${CONFIG}"
  uv run main.py --train --config "${CONFIG}"
else
  echo "[1/4] Skipped training"
fi

if [[ $SKIP_EVAL -eq 0 ]]; then
  echo "[2/4] Evaluating YOLO detector with ${CONFIG}"
  uv run main.py --eval --config "${CONFIG}"
else
  echo "[2/4] Skipped evaluation"
fi

echo "[3/4] Exporting predicted boxes on segmentation train/val/test -> ${PRED_OUT}"
uv run scripts/export_yolo_bboxes_for_segmentation.py \
  --config "${CONFIG}" \
  --splits train val test \
  --output-dir "${PRED_OUT}"

echo "[4/4] Building handoff manifests -> ${HANDOFF_JSON} and ${HANDOFF_CSV}"
uv run scripts/build_bbox_manifest.py \
  --pred-root "${PRED_OUT}" \
  --seg-root data/preprocessed \
  --output-json "${HANDOFF_JSON}" \
  --output-csv "${HANDOFF_CSV}" \
  --include-empty-samples

gzip -kf "${HANDOFF_JSON}" "${HANDOFF_CSV}"

echo
echo "Pipeline complete."
echo "Predicted labels root: ${PRED_OUT}"
echo "Handoff JSON: ${HANDOFF_JSON}"
echo "Handoff CSV: ${HANDOFF_CSV}"
echo "Compressed JSON: ${HANDOFF_JSON}.gz"
echo "Compressed CSV: ${HANDOFF_CSV}.gz"
