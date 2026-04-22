import os
import torch
import numpy as np
import cv2
from PIL import Image

def visualize_evaluation(image_tensor, pred_boxes, pred_mask, gt_mask, output_path, gt_boxes=None):
    """
    Overlays the predictions and ground truths onto a test image.
    - image_tensor: [H, W] or [1, H, W] float
    - pred_boxes: [N, 4] float
    - pred_mask: [H, W] or [1, H, W] float
    - gt_mask: [H, W] or [1, H, W] float
    - gt_boxes: [N, 4] float (optional)
    """
    
    # 1. Convert Base Image to Grayscale format [0, 255]
    if image_tensor.ndim == 3:
        image_tensor = image_tensor[0]
    img_gray = (image_tensor.detach().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    
    # Convert exactly back to 3 channels for coloring
    img_color = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)

    # 2. Add Ground Truth Mask (Green color)
    if gt_mask.ndim == 3:
        gt_mask = gt_mask[0]
    gt_np = gt_mask.detach().cpu().numpy()
    green_mask = np.zeros_like(img_color)
    green_mask[gt_np > 0.5] = [0, 255, 0] # BGR format: Green is index 1

    # Overlay with alpha
    cv2.addWeighted(green_mask, 0.4, img_color, 1.0, 0, img_color)

    # 3. Add Predicted Segmentations (Yellow color)
    if pred_mask is not None and pred_mask.ndim == 3:
        pred_mask = pred_mask[0]
    if pred_mask is not None:
        pred_np = pred_mask.detach().cpu().numpy()
        yellow_mask = np.zeros_like(img_color)
        yellow_mask[pred_np > 0.5] = [0, 255, 255] # BGR format: Yellow is (Cyan 0, Green 255, Red 255)
        cv2.addWeighted(yellow_mask, 0.4, img_color, 1.0, 0, img_color)

    # 4. Add Predicted Bounding Boxes (Red color)
    if pred_boxes is not None:
        boxes_np = pred_boxes.detach().cpu().numpy()
        for box in boxes_np:
            x1, y1, x2, y2 = map(int, box[:4])
            cv2.rectangle(img_color, (x1, y1), (x2, y2), (0, 0, 255), 2) # Red Bounding box

    # 5. Add GT Bounding Boxes (Blue color)
    if gt_boxes is not None:
        gt_boxes_np = gt_boxes.detach().cpu().numpy()
        for box in gt_boxes_np:
            x1, y1, x2, y2 = map(int, box[:4])
            cv2.rectangle(img_color, (x1, y1), (x2, y2), (255, 0, 0), 2) # Blue Bounding box

    # Save logic
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, img_color)
