# caries-segmentation

## Quick Start
### Download the raw dataset with the following script: 
```bash
uv run scripts/download.py
```

## Training Modes

The project now has explicit training modes via training.task in config.toml:

- tooth_detection: trains YOLO detector on data/preprocessed_detection
- unet_with_yolo_boxes: trains U-Net on crops generated from YOLO predicted boxes
- yolo_unet_conjunction (or end2end_joint): trains the true joint YOLO + U-Net model end-to-end on segmentation data
- segmentation: trains baseline U-Net on full images

### 1) Train YOLO detector
Set training.task to tooth_detection, then run:

```bash
uv run main.py --train
```

This writes detector checkpoints to checkpoints/detection by default.

### 2) Train U-Net with YOLO box predictions
Set training.task to unet_with_yolo_boxes and ensure tooth_detection.detector_checkpoint points to a trained YOLO checkpoint, then run:

```bash
uv run main.py --train
```

This uses YOLO to pick crop regions and trains U-Net on the cropped image/mask pairs.

### 3) Train true end-to-end YOLO + U-Net
Set training.task to yolo_unet_conjunction (or end2end_joint), then run:

```bash
uv run main.py --train
```

This jointly optimizes detector and segmenter and saves checkpoints under checkpoints/end2end by default.

### 4) Export YOLO predicted bboxes for segmentation train/val/test
After detector training, export predicted tooth bboxes for segmentation splits:

```bash
uv run scripts/export_yolo_bboxes_for_segmentation.py \
	--config config.method1_detection.toml \
	--splits train val test \
	--output-dir outputs/yolo_pred_bboxes_segmentation_YYYYMMDD
```

This writes YOLO label files under split-specific labels directories and also writes split metadata CSV files.

### 5) One-command pipeline (train + eval + export + handoff manifests)
Use the convenience script:

```bash
./scripts/run_yolo_bbox_pipeline.sh
```

Optional flags:

```bash
./scripts/run_yolo_bbox_pipeline.sh --skip-train --skip-eval
```


| Model/System                                                                                 | Imaging Type    | Key Features                                                                                        | Performance Metrics                                                                                                                                                        |
| -------------------------------------------------------------------------------------------- | --------------- | --------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| CariesNet [pmc.ncbi.nlm.nih](https://pmc.ncbi.nlm.nih.gov/articles/PMC8736291/)              | Panoramic X-ray | U-shape network with the additional full-scale axial attention module to segment three caries types | Mean 93.64% Dice coefficient and 93.61% accuracy.                                                                                                                          |
| CariSeg [sciencedirect](https://www.sciencedirect.com/science/article/pii/S2405844024068671) | Panoramic X-ray | Ensemble of U-net, Feature Pyramid Network (FPN), and DeeplabV3                                     | This results in 94.895% accuracy and a Dice score of 88.5% for teeth segmentation, as well as a 99.42% accuracy and a mean 68.2% Dice coefficient for caries segmentation. |
| U-Net + ResNet-50 [pmc.ncbi.nlm.nih](https://pmc.ncbi.nlm.nih.gov/articles/PMC12659894/)     | RVG radiographs | Pixel-wise segmentation (U-Net), multi-class (ResNet-50) (no/enamel/dentin caries)                  | Dice 0.89, Accuracy 93.2%                                                                                                                                                  |


Studies favor intraoral and panoramic X-rays, with CNNs enabling classification and segmentation across dentition stages.

---

1. **Class Imbalance:** A radiograph is 95% background/healthy tooth and only <5% caries. Loss functions like **Focal Loss** or **Dice Loss** are essential to prevent the model from just predicting "healthy" everywhere.
2. **The "Mach Band" Effect:** An optical illusion in X-rays that mimics decay at the enamel-dentin junction. Models often produce False Positives here.
3. **Restoration Artifacts:** Metallic fillings cause bright streaks (beam hardening) that obscure adjacent caries, confusing segmentation models.
4. **Enamel vs. Dentin:** Distinguishing between reversible (enamel) and irreversible (dentin) caries is clinically vital but computationally difficult due to low contrast.

