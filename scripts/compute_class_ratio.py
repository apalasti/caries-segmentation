import os
import csv
from PIL import Image
import numpy as np
from tqdm import tqdm


def load_train_mask_paths(preprocessed_path):
    csv_path = os.path.join(preprocessed_path, "data.csv")
    pairs = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["split"] != "train":
                continue
            img_id = row["id"]
            mask_path = os.path.join(
                preprocessed_path, "train", "masks", f"{img_id}.png"
            )
            pairs.append(mask_path)
    return pairs


def compute_class_ratio(mask_paths, size=(256, 256)):
    total_bg = 0
    total_caries = 0
    images_with_caries = 0
    zero_caries_images = 0
    caries_pixel_counts = []

    for mask_path in tqdm(mask_paths, desc="Processing masks"):
        mask = Image.open(mask_path).convert("L")
        mask = mask.resize(size, resample=Image.NEAREST)
        mask = np.array(mask)
        mask = (mask > 0).astype(np.float32)

        n_bg = np.sum(mask == 0)
        n_caries = np.sum(mask == 1)

        total_bg += n_bg
        total_caries += n_caries

        if n_caries > 0:
            images_with_caries += 1
            caries_pixel_counts.append(n_caries)
        else:
            zero_caries_images += 1

    return (
        total_bg,
        total_caries,
        images_with_caries,
        zero_caries_images,
        caries_pixel_counts,
    )


def main():
    preprocessed_path = "data/preprocessed"
    size = (256, 256)

    print("Loading training mask paths...")
    mask_paths = load_train_mask_paths(preprocessed_path)
    print(f"Found {len(mask_paths)} training images\n")

    (
        total_bg,
        total_caries,
        images_with_caries,
        zero_caries_images,
        caries_pixel_counts,
    ) = compute_class_ratio(mask_paths, size)

    total_pixels = total_bg + total_caries
    caries_pct = (total_caries / total_pixels) * 100
    bg_pct = (total_bg / total_pixels) * 100
    ratio = total_bg / total_caries if total_caries > 0 else float("inf")

    print("=" * 50)
    print("CLASS DISTRIBUTION ANALYSIS")
    print("=" * 50)
    print(f"Total images:              {len(mask_paths)}")
    print(
        f"Images with caries:        {images_with_caries} ({100 * images_with_caries / len(mask_paths):.1f}%)"
    )
    print(
        f"Images without caries:     {zero_caries_images} ({100 * zero_caries_images / len(mask_paths):.1f}%)"
    )
    print()
    print(f"Total pixels per image:    {size[0]} x {size[1]} = {size[0] * size[1]:,}")
    print(f"Total pixels (dataset):   {total_pixels:,}")
    print()
    print(f"Background pixels:         {total_bg:,} ({bg_pct:.2f}%)")
    print(f"Caries pixels:             {total_caries:,} ({caries_pct:.2f}%)")
    print()
    print("=" * 50)
    print("RECOMMENDED LOSS WEIGHTS")
    print("=" * 50)
    print(f"Class ratio (bg/caries):  {ratio:.1f}")
    print()
    print(
        f"BCE pos_weight:           {ratio:.1f}  # Use this in BCEWithLogitsLoss(pos_weight=...)"
    )
    print(
        f"Dice weight:              [1.0, {ratio:.1f}]  # Use this in DiceLoss(weight=[1.0, {ratio:.1f}])"
    )
    print()
    print("Alternative (conservative, ~5x):")
    print(f"BCE pos_weight:            5.0")
    print(f"Dice weight:               [1.0, 5.0]")
    print()

    if caries_pixel_counts:
        avg_caries_pixels = np.mean(caries_pixel_counts)
        median_caries_pixels = np.median(caries_pixel_counts)
        max_caries_pixels = np.max(caries_pixel_counts)
        min_caries_pixels = np.min(caries_pixel_counts)
        print("=" * 50)
        print("CARIES PIXEL STATS (per image with caries)")
        print("=" * 50)
        print(f"Min:     {min_caries_pixels:,} pixels")
        print(f"Max:     {max_caries_pixels:,} pixels")
        print(f"Mean:    {avg_caries_pixels:,.0f} pixels")
        print(f"Median:  {median_caries_pixels:,.0f} pixels")
        print(
            f"Max caries % of image: {100 * max_caries_pixels / (size[0] * size[1]):.2f}%"
        )


if __name__ == "__main__":
    main()
