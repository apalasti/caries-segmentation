import pandas as pd
import numpy as np
from PIL import Image
import os
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def calculate_outside_percentage():
    csv_path = (
        "outputs/yolo_pred_bboxes_segmentation_ultralytics/train_box_metadata.csv"
    )
    df = pd.read_csv(csv_path)

    # Group by sample_id
    grouped = df.groupby("sample_id")

    results = []

    # Create output dir
    out_dir = "outputs/caries_outside_bboxes"
    os.makedirs(out_dir, exist_ok=True)

    count = 0

    for sample_id, group in grouped:
        mask_path = group.iloc[0]["mask_path"]
        image_path = group.iloc[0]["image_path"]

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

        percent_outside = (outside_area / gt_area) * 100.0

        results.append(
            {
                "sample_id": sample_id,
                "image_path": image_path,
                "mask_path": mask_path,
                "percent_outside": percent_outside,
                "group": group,
            }
        )

        count += 1
        if count % 100 == 0:
            print(f"Processed {count} images...")

    # Sort by percent_outside descending
    results.sort(key=lambda x: x["percent_outside"], reverse=True)

    print(f"\nTop 10 images with highest percentage of caries outside bboxes:")
    for i in range(min(10, len(results))):
        res = results[i]
        print(f"{i + 1}. {res['sample_id']}: {res['percent_outside']:.2f}%")

        # Plotting
        img = np.array(Image.open(res["image_path"]).convert("RGB"))

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.imshow(img)

        # Plot GT mask (semi-transparent red)
        gt_mask_rgb = np.zeros_like(img)
        gt_mask_rgb[np.array(Image.open(res["mask_path"]).convert("L")) > 0] = [
            255,
            0,
            0,
        ]
        ax.imshow(gt_mask_rgb, alpha=0.5)

        # Plot BBoxes (green rectangles)
        for _, row in res["group"].iterrows():
            x1, y1, x2, y2 = row["x1"], row["y1"], row["x2"], row["y2"]
            rect = plt.Rectangle(
                (x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor="green", linewidth=2
            )
            ax.add_patch(rect)

        ax.set_title(
            f"Sample: {res['sample_id']} | Caries outside: {res['percent_outside']:.2f}%"
        )
        ax.axis("off")

        save_path = os.path.join(out_dir, f"outside_{i + 1:02d}_{res['sample_id']}.png")
        plt.savefig(save_path, bbox_inches="tight")
        plt.close(fig)

    print(f"\nSaved visualization images to {out_dir}/")


if __name__ == "__main__":
    calculate_outside_percentage()
