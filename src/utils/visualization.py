import random

import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import confusion_matrix, precision_score, recall_score, jaccard_score
import os
import seaborn as sns

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

def plot_metrics_bars(metrics, save_dir):
    """
    metrics: dict with keys ['precision','recall','dice','iou']
    """
    os.makedirs(save_dir, exist_ok=True)
    keys = ['precision','recall','dice','iou']
    values = [metrics[k] for k in keys]

    plt.figure(figsize=(6,4))
    plt.bar(keys, values, color=['skyblue','orange','green','red'])
    plt.ylim(0,1)
    plt.title("Final Test Metrics")
    plt.ylabel("Score")
    for i,v in enumerate(values):
        plt.text(i, v+0.02, f"{v:.2f}", ha='center')
    plt.savefig(os.path.join(save_dir, "metrics_barplot.png"), dpi=300, bbox_inches="tight")
    plt.close()

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


def plot_training_curves(logs, save_path):
    epochs = range(1, len(logs.get("train_loss", [])) + 1)
    plt.figure()

    if "train_loss" in logs:
        plt.plot(epochs, logs["train_loss"], label="Train Loss")
    if "val_loss" in logs:
        plt.plot(range(1, len(logs["val_loss"]) + 1), logs["val_loss"], label="Val Loss")
    if "val_dice" in logs:
        plt.plot(range(1, len(logs["val_dice"]) + 1), logs["val_dice"], label="Val Dice")

    plt.xlabel("Epoch")
    plt.ylabel("Metric")
    plt.legend()
    plt.grid(True)

    try:
        plt.savefig(save_path)
        print(f" Saved training curves to: {save_path}")
    except Exception as e:
        print(f" Failed to save figure: {e}")
    finally:
        plt.close()

def plot_confusion_matrix(metrics, save_path):
    cm = np.array([[metrics["TP"], metrics["FP"]],
                   [metrics["FN"], metrics["TN"]]])
    plt.figure(figsize=(4,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Ground Truth")
    plt.title("Pixel-level Confusion Matrix")
    plt.savefig(save_path)
    plt.close()