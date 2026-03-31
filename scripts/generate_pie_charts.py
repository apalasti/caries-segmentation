# visualization_utils.py
import csv
import os
import random
from collections import defaultdict
from PIL import Image, ImageDraw

import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import confusion_matrix, precision_score, recall_score, jaccard_score


def sample_test_predictions(model, dataloader, max_samples=5, device="auto"):
    """
    Sample up to max_samples images from test dataloader and get predictions
    """
    all_samples = []

    for batch in dataloader:
        images, masks = batch
        images = images.to(device)
        masks = masks.to(device)

        with torch.no_grad():
            preds = model(images)
            preds = torch.sigmoid(preds) > 0.5  # binary mask

        for img, gt_mask, pred_mask in zip(images, masks, preds):
            all_samples.append((img.cpu(), gt_mask.cpu(), pred_mask.cpu()))

    return random.sample(all_samples, min(max_samples, len(all_samples)))


def visualize_prediction(img, gt_mask, pred_mask, save_path=None):
    """
    Visualize original, GT mask, pred mask, difference map
    """
    img = img.squeeze().numpy()  # assuming single channel
    gt_mask = gt_mask.squeeze().numpy()
    pred_mask = pred_mask.squeeze().numpy()

    # Difference map: FP=red, FN=blue
    diff = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    diff[(pred_mask == 1) & (gt_mask == 0)] = [255, 0, 0]  # FP red
    diff[(pred_mask == 0) & (gt_mask == 1)] = [0, 0, 255]  # FN blue

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(img, cmap="gray")
    axes[0].set_title("Original")
    axes[1].imshow(gt_mask, cmap="gray")
    axes[1].set_title("Ground Truth")
    axes[2].imshow(pred_mask, cmap="gray")
    axes[2].set_title("Prediction")
    axes[3].imshow(diff)
    axes[3].set_title("Difference (FP=Red, FN=Blue)")

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_pixel_metrics(gt_masks, pred_masks, save_path=None):
    """
    Compute pixel-level confusion matrix, Dice, IoU, precision, recall
    """
    y_true = np.concatenate([m.flatten() for m in gt_masks])
    y_pred = np.concatenate([m.flatten() for m in pred_masks])

    cm = confusion_matrix(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    iou = jaccard_score(y_true, y_pred)
    dice = 2 * (precision * recall) / (precision + recall + 1e-8)

    print("Pixel-level metrics:")
    print("Confusion matrix:\n", cm)
    print(f"Precision: {precision:.3f}, Recall: {recall:.3f}, Dice: {dice:.3f}, IoU: {iou:.3f}")

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.figure()
        plt.imshow(cm, cmap="Blues")
        plt.title("Pixel-level Confusion Matrix")
        plt.colorbar()
        plt.xlabel("Predicted")
        plt.ylabel("Ground Truth")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()


def plot_training_curves(logs, save_dir):
    """
    Plot training curves: loss, Dice, IoU
    logs: dict with keys: 'train_loss', 'val_loss', 'val_dice', 'val_iou'
    """
    os.makedirs(save_dir, exist_ok=True)
    epochs = range(1, len(logs['train_loss']) + 1)

    # Loss curves
    plt.figure()
    plt.plot(epochs, logs['train_loss'], label="Train Loss")
    plt.plot(epochs, logs['val_loss'], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.savefig(os.path.join(save_dir, "loss.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # Metrics curves
    plt.figure()
    plt.plot(epochs, logs['val_dice'], label="Val Dice")
    plt.plot(epochs, logs['val_iou'], label="Val IoU")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.title("Validation Metrics")
    plt.legend()
    plt.savefig(os.path.join(save_dir, "metrics.png"), dpi=300, bbox_inches="tight")
    plt.close()

def get_split_counts(preprocessed_path):
    csv_path = os.path.join(preprocessed_path, "data.csv")

    split_counts = defaultdict(int)

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            split_counts[row["split"]] += 1

    return split_counts


def compute_pixel_ratio(preprocessed_path, split, size=(256, 256)):
    csv_path = os.path.join(preprocessed_path, "data.csv")

    total_bg = 0
    total_caries = 0

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)

        for row in reader:
            if row["split"] != split:
                continue

            img_id = row["id"]
            mask_path = os.path.join(
                preprocessed_path, split, "masks", f"{img_id}.png"
            )

            mask = Image.open(mask_path).convert("L")
            mask = mask.resize(size, resample=Image.NEAREST)
            mask = np.array(mask)
            mask = (mask > 0).astype(np.uint8)

            total_bg += np.sum(mask == 0)
            total_caries += np.sum(mask == 1)

    return total_bg, total_caries


def save_dataset_split_pie(split_counts, save_path):
    labels = list(split_counts.keys())
    sizes = list(split_counts.values())

    plt.figure()
    plt.pie(
        sizes,
        labels=labels,
        autopct="%1.1f%%",
        startangle=90
    )
    plt.title("Dataset Split")
    plt.axis("equal")

    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def save_class_ratio_pie(bg, caries, title, save_path):
    sizes = [bg, caries]
    labels = ["Background", "Caries"]

    plt.figure()
    plt.pie(
        sizes,
        labels=labels,
        autopct="%1.2f%%",
        startangle=90
    )
    plt.title(title)
    plt.axis("equal")

    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def generate_all_pies(preprocessed_path, output_dir, size=(256,256)):

    os.makedirs(output_dir, exist_ok=True)

    # 1️⃣ dataset split pie
    split_counts = get_split_counts(preprocessed_path)

    save_dataset_split_pie(
        split_counts,
        os.path.join(output_dir, "dataset_split_pie.png")
    )
def load_train_pairs_by_source(preprocessed_path, source):
    csv_path = os.path.join(preprocessed_path, "data.csv")

    pairs = []

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)

        for row in reader:
            if row["split"] != "train":
                continue
            if row["source"] != source:
                continue

            img_id = row["id"]

            img_path = os.path.join(
                preprocessed_path, "train", "images", f"{img_id}.png"
            )

            mask_path = os.path.join(
                preprocessed_path, "train", "masks", f"{img_id}.png"
            )

            pairs.append((img_path, mask_path))

    return pairs


def create_overlay(image, mask, alpha=0.4):
    image = np.array(image.convert("RGB"))
    mask = np.array(mask)

    overlay = image.copy()

    red = np.zeros_like(image)
    red[:, :, 0] = 255

    mask_binary = mask > 0

    overlay[mask_binary] = (
        (1 - alpha) * overlay[mask_binary] + alpha * red[mask_binary]
    )

    return overlay.astype(np.uint8)


def save_annotation_grid(pairs, source_name, save_path, n_samples=3):

    n_total = len(pairs)
    samples = random.sample(pairs, min(n_samples, n_total))

    fig, axes = plt.subplots(len(samples), 3, figsize=(9, 3 * len(samples)))

    if len(samples) == 1:
        axes = [axes]

    for i, (img_path, mask_path) in enumerate(samples):

        image = Image.open(img_path).convert("L")
        mask = Image.open(mask_path).convert("L")

        overlay = create_overlay(image, mask)

        axes[i][0].imshow(image, cmap="gray")
        axes[i][0].set_title("Original")
        axes[i][0].axis("off")

        axes[i][1].imshow(mask, cmap="gray")
        axes[i][1].set_title("Mask")
        axes[i][1].axis("off")

        axes[i][2].imshow(overlay)
        axes[i][2].set_title("Overlay")
        axes[i][2].axis("off")

    # cím a train képszámmal
    fig.suptitle(f"{source_name} (train, n={n_total})", fontsize=14)

    plt.tight_layout()
    plt.subplots_adjust(top=0.90)

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def generate_annotation_figures(preprocessed_path, output_dir):

    os.makedirs(output_dir, exist_ok=True)

    for source in ["roboflow", "DC1000"]:

        pairs = load_train_pairs_by_source(preprocessed_path, source)

        if len(pairs) == 0:
            print(f"No samples found for {source}")
            continue

        save_path = os.path.join(
            output_dir, f"annotation_examples_{source}.png"
        )

        save_annotation_grid(
            pairs,
            source_name=source,
            save_path=save_path
        )

        print(f"Saved: {save_path}")

def get_source_split_counts(preprocessed_path):
    """
    Returns nested dict: counts[split][source] = number of images
    """
    csv_path = os.path.join(preprocessed_path, "data.csv")
    counts = defaultdict(lambda: defaultdict(int))

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            split = row["split"]
            source = row.get("source", "unknown")
            counts[split][source] += 1

    return counts


def save_source_distribution_pies(counts, output_dir):
    """
    Creates one pie chart per split (train/val/test) showing Roboflow vs DC1000 distribution
    """
    os.makedirs(output_dir, exist_ok=True)
    splits = ["train", "val", "test"]

    for split in splits:
        sources = ["roboflow", "DC1000"]
        sizes = [counts[split].get(s, 0) for s in sources]
        labels = [f"{s} (n={counts[split].get(s,0)})" for s in sources]

        plt.figure(figsize=(5,5))
        plt.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=90)
        plt.title(f"Source distribution – {split.capitalize()} (total n={sum(sizes)})")
        plt.axis("equal")

        save_path = os.path.join(output_dir, f"source_distribution_{split}.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved {save_path}")


def generate_source_distribution_pies(preprocessed_path, output_dir):
    counts = get_source_split_counts(preprocessed_path)
    save_source_distribution_pies(counts, output_dir)

if __name__ == "__main__":
    generate_all_pies(
        preprocessed_path="../data/preprocessed",
        output_dir="../docs/typst/figures/dataset_introduction"
    )

    generate_source_distribution_pies(
        preprocessed_path="../data/preprocessed",
        output_dir="../docs/typst/figures/dataset_introduction"
    )

    generate_annotation_figures(
        preprocessed_path="../data/preprocessed",
        output_dir="../docs/typst/figures/annotations"
    )
