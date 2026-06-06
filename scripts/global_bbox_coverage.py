import pandas as pd
import numpy as np
from PIL import Image
import os


def calculate_global_outside_percentage():
    csv_path = (
        "outputs/yolo_pred_bboxes_segmentation_ultralytics/train_box_metadata.csv"
    )
    df = pd.read_csv(csv_path)

    # Group by sample_id
    grouped = df.groupby("sample_id")

    total_gt_area = 0
    total_outside_area = 0

    count = 0

    for sample_id, group in grouped:
        mask_path = group.iloc[0]["mask_path"]

        if not os.path.exists(mask_path):
            continue

        # Load mask
        mask_img = Image.open(mask_path).convert("L")
        mask = np.array(mask_img)

        # Convert mask to binary (GT)
        gt_mask = mask > 0
        gt_area = np.sum(gt_mask)

        if gt_area == 0:
            continue  # No caries in GT

        # Create BBox mask
        bbox_mask = np.zeros_like(mask, dtype=bool)

        for _, row in group.iterrows():
            x1, y1, x2, y2 = (
                int(row["x1"]),
                int(row["y1"]),
                int(row["x2"]),
                int(row["y2"]),
            )
            # Ensure within bounds
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(mask.shape[1], x2), min(mask.shape[0], y2)
            bbox_mask[y1:y2, x1:x2] = True

        # Calculate intersection and outside
        outside_mask = gt_mask & (~bbox_mask)
        outside_area = np.sum(outside_mask)

        total_gt_area += gt_area
        total_outside_area += outside_area

        count += 1
        if count % 200 == 0:
            print(f"Processed {count} images...")

    if total_gt_area > 0:
        global_percent = (total_outside_area / total_gt_area) * 100.0
        print(f"\n--- GLOBAL METRICS ---")
        print(f"Total GT Caries Area (pixels): {total_gt_area}")
        print(f"Total Caries Area Outside BBoxes (pixels): {total_outside_area}")
        print(f"Global Percentage of Caries Outside BBoxes: {global_percent:.4f}%")
    else:
        print("No GT caries found in the dataset.")


if __name__ == "__main__":
    calculate_global_outside_percentage()
