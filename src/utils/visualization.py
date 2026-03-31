import random

import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import confusion_matrix, precision_score, recall_score, jaccard_score
import os

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
