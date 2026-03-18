import hashlib
import pathlib
import shutil
from itertools import chain

import pandas as pd
from PIL import Image, ImageDraw
from tqdm import tqdm


DATA_DIR = pathlib.Path(__file__).parent.parent / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PREPROCESSED_DIR = DATA_DIR / "preprocessed"


def get_dc1000_mask_path(image_path: pathlib.Path) -> pathlib.Path:
    return pathlib.Path(
        str(image_path)
            .replace("org_test_dataset/images", "org_test_dataset/colors")
            .replace("org_train_dataset/images", "org_train_dataset/colors_clean")
    )


def get_roboflow_label_path(image_path: pathlib.Path) -> pathlib.Path:
    label_path = str(image_path).replace("/images/", "/labels/")
    label_path = label_path.replace(".jpg", ".txt").replace(".png", ".txt")
    return pathlib.Path(label_path)


def parse_yolo_polygon(line: str) -> tuple[int, list[float]]:
    parts = line.strip().split()
    class_id = int(parts[0])
    coords = [float(x) for x in parts[1:]]
    return class_id, coords


def create_mask_from_polygons(
    polygons: list[list[tuple[float, float]]], width: int, height: int
) -> Image.Image:
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    for polygon in polygons:
        scaled_polygon = [(x * width, y * height) for x, y in polygon]
        draw.polygon(scaled_polygon, fill=255)
    return mask


def copy_to_split(
    image_path: pathlib.Path, mask_path: pathlib.Path, split: str, id_: str
) -> None:
    split_dir = PREPROCESSED_DIR / split
    images_dir = split_dir / "images"
    masks_dir = split_dir / "masks"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(image_path, images_dir / f"{id_}.png")
    shutil.copy2(mask_path, masks_dir / f"{id_}.png")


def handle_dc1000(row) -> bool:
    image_path = DATA_DIR / row["original"]
    mask_path = get_dc1000_mask_path(image_path)

    if not image_path.exists() or not mask_path.exists():
        return True

    try:
        copy_to_split(image_path, mask_path, row["split"], row["id"])
        return False
    except Exception:
        return True


def handle_roboflow(row) -> bool:
    image_path = DATA_DIR / row["original"]
    label_path = get_roboflow_label_path(image_path)

    if not image_path.exists() or not label_path.exists():
        return True

    caries_polygons = []
    with open(label_path, "r") as f:
        for line in f:
            class_id, coords = parse_yolo_polygon(line)
            if class_id == 0:
                coords_pairs = list(zip(coords[::2], coords[1::2]))
                caries_polygons.append(coords_pairs)

    if not caries_polygons:
        return True

    try:
        with Image.open(image_path) as img:
            width, height = img.size
        mask = create_mask_from_polygons(caries_polygons, width, height)

        split_dir = PREPROCESSED_DIR / row["split"]
        images_dir = split_dir / "images"
        masks_dir = split_dir / "masks"
        images_dir.mkdir(parents=True, exist_ok=True)
        masks_dir.mkdir(parents=True, exist_ok=True)

        shutil.copy2(image_path, images_dir / f"{row['id']}.png")
        mask.save(masks_dir / f"{row['id']}.png")
        return False
    except Exception:
        return True


def main():
    shutil.rmtree(PREPROCESSED_DIR, ignore_errors=True)
    PREPROCESSED_DIR.mkdir(exist_ok=True)

    df = pd.DataFrame(
        {
            "original": [
                str(fp.relative_to(DATA_DIR))
                for fp in chain(
                    RAW_DATA_DIR.glob("vzrad2-6/train/images/*"),
                    RAW_DATA_DIR.glob("vzrad2-6/valid/images/*"),
                    RAW_DATA_DIR.glob("vzrad2-6/test/images/*"),
                    RAW_DATA_DIR.glob("DC1000_dataset/org_train_dataset/images/*.png"),
                    RAW_DATA_DIR.glob("DC1000_dataset/org_test_dataset/images/*.png"),
                )
            ]
        }
    )
    df["source"] = df["original"].apply(
        lambda fp: "DC1000" if "DC1000" in fp else "roboflow"
    )
    df["original_type"] = df["original"].apply(
        lambda fp: next((t for t in ["train", "valid", "test"] if t in fp), "unknown")
    )
    df["id"] = df["original"].apply(lambda fp: hashlib.md5(fp.encode()).hexdigest())

    def deterministic_split(id_str: str, train: float = 0.7, val: float = 0.15) -> str:
        u = int(id_str[:8], 16) / (16**8)
        if u < train:
            return "train"
        if u < train + val:
            return "val"
        return "test"

    df["split"] = df["id"].apply(deterministic_split)

    skipped_ids = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Preprocessing images"):
        if row["source"] == "DC1000":
            skipped = handle_dc1000(row)
        elif row["source"] == "roboflow":
            skipped = handle_roboflow(row)
        else:
            skipped = True

        if skipped:
            skipped_ids.append(row["id"])

    # Remove rows with ids in skipped_ids from the dataframe
    df = df[~df["id"].isin(skipped_ids)]
    df.to_csv(PREPROCESSED_DIR / "data.csv", index=False)

    split_order = ["train", "val", "test"]
    split_totals = df["split"].value_counts().reindex(split_order, fill_value=0)
    source_counts = pd.crosstab(df["split"], df["source"]).reindex(split_order, fill_value=0)

    print("\nSummary of resulting data split:")
    for split in split_order:
        split_dir = PREPROCESSED_DIR / split
        images_dir = split_dir / "images"
        masks_dir = split_dir / "masks"
        total = int(split_totals[split])

        mix_parts = []
        for source in sorted(source_counts.columns):
            n = int(source_counts.at[split, source])
            pct = (n / total * 100) if total else 0.0
            mix_parts.append(f"{source} {n} ({pct:.1f}%)")

        folders = f"{images_dir.relative_to(DATA_DIR)}/ | {masks_dir.relative_to(DATA_DIR)}/"
        print(f"- {split}: {total} images ({total / split_totals.sum()*100:.1f}%)")
        print(f"  folders: {folders}")
        print(f"  source: {', '.join(mix_parts) if mix_parts else '-'}")


if __name__ == "__main__":
    main()
