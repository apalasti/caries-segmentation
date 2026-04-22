# BBox Handoff Workflow (Slim + Reproducible)

This workflow creates a small transferable bbox package (JSON/CSV) and reconstructs image+mask crops on another machine using the same prepared dataset.

## Guarantees

- Uses predicted YOLO boxes directly (no augmentation in export).
- Stores normalized YOLO coordinates (`xc`, `yc`, `w`, `h`) and score.
- Reconstruction uses local `data/preprocessed` generated from the same `download.py` + `preprocess.py` pipeline.

## 1) Sender: build slim manifest

Run from repository root:

```bash
uv run scripts/build_bbox_manifest.py \
  --pred-root outputs/yolo_pred_bboxes_segmentation \
  --seg-root data/preprocessed \
  --output-json outputs/handoff/bbox_manifest_predicted.json \
  --output-csv outputs/handoff/bbox_manifest_predicted.csv \
  --include-empty-samples
```

Send one of these files:

- `outputs/handoff/bbox_manifest_predicted.json` (recommended)
- `outputs/handoff/bbox_manifest_predicted.csv`

Optional extra-slim compressed files are also usable directly:

- `outputs/handoff/bbox_manifest_predicted.json.gz`
- `outputs/handoff/bbox_manifest_predicted.csv.gz`

## 2) Receiver: prepare same local dataset

On the receiver machine, run your normal data setup:

1. Download data via `scripts/download.py`
2. Prepare data via `scripts/preprocess.py`

Expected local segmentation root:

- `data/preprocessed/{train,val,test}/images/*.png`
- `data/preprocessed/{train,val,test}/masks/*.png`

## 3) Receiver: reconstruct crops from manifest

Using JSON manifest:

```bash
uv run scripts/materialize_crops_from_bbox_manifest.py \
  --manifest /path/to/bbox_manifest_predicted.json \
  --seg-root data/preprocessed \
  --splits train val test \
  --output-dir outputs/segmentation_crops_from_manifest
```

Using CSV manifest:

```bash
uv run scripts/materialize_crops_from_bbox_manifest.py \
  --manifest /path/to/bbox_manifest_predicted.csv \
  --seg-root data/preprocessed \
  --splits train val test \
  --output-dir outputs/segmentation_crops_from_manifest
```

Using compressed manifest directly:

```bash
uv run scripts/materialize_crops_from_bbox_manifest.py \
  --manifest /path/to/bbox_manifest_predicted.json.gz \
  --seg-root data/preprocessed \
  --splits train val test \
  --output-dir outputs/segmentation_crops_from_manifest
```

## Output structure

- `outputs/segmentation_crops_from_manifest/{split}/images/*.png`
- `outputs/segmentation_crops_from_manifest/{split}/masks/*.png`
- `outputs/segmentation_crops_from_manifest/{split}/metadata.csv`
- `outputs/segmentation_crops_from_manifest/summary.json`

## Notes

- Default reconstruction reference size is 640, matching the manifest metadata.
- No random transforms are applied by reconstruction.
- If needed, set `--fallback-full-image` to keep samples that have no valid boxes after filtering.
