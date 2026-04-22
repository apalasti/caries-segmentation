import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F

# Import models & dataset
from src.data.dataset import BaseKariesDataset, load_split_pairs
from src.models.bbox.yolo import YOLOv5
from src.models.unet import UNet
from src.models.end2end import EndToEndCariesModel
from src.utils.visualize_evaluation import visualize_evaluation

def collate_fn(batch):
    images = []
    targets = []
    for img, target in batch:
        images.append(img)
        targets.append(target)
    return torch.stack(images, dim=0), targets

def train_and_visualize():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. Dataset & Loaders
    preprocessed_path = os.path.join("data", "preprocessed")
    if not os.path.exists(preprocessed_path):
        print("Preprocessed data path doesn't exist. Please run standard preprocessing first.")
        return

    train_pairs = load_split_pairs(preprocessed_path, "train")
    # Take a small subset for quick debugging and training (e.g. 50 items)
    train_pairs = train_pairs[:50]
    train_dataset = BaseKariesDataset(
        train_pairs,
        size=(256, 256),
        bbox_padding=10.0,
        return_targets=True,
    )
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)

    val_pairs = load_split_pairs(preprocessed_path, "test")
    # Only need 10 for the visualization
    val_pairs = val_pairs[:10]
    val_dataset = BaseKariesDataset(
        val_pairs,
        size=(256, 256),
        bbox_padding=10.0,
        return_targets=True,
    )
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    # 2. Models
    # YOLO only for class 1
    yolo = YOLOv5(num_classes=1, conf_threshold=0.1, iou_threshold=0.45).to(device)
    # UNet
    unet = UNet(n_channels=3, n_classes=1).to(device)
    # Joint
    model = EndToEndCariesModel(detector=yolo, segmenter=unet, unet_input_size=(64, 64)).to(device)

    # 3. Optimizer
    # We optimize both YOLO and UNet
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    print("Starting Debug Training over 3 epochs...")
    model.train()
    for config_epoch in range(3):
        total_loss = 0.0
        iters = 0
        for images, targets in tqdm(train_loader, desc=f"Epoch {config_epoch+1}/3"):
            images = images.to(device)
            # move targets to device
            for t in targets:
                t['boxes'] = t['boxes'].to(device)
                t['labels'] = t['labels'].to(device)
                t['masks'] = t['masks'].to(device)

            optimizer.zero_grad()
            losses_dict = model(images, targets)
            
            loss = 0.0
            for name, l_val in losses_dict.items():
                # sum all losses up
                if isinstance(l_val, torch.Tensor) and l_val.requires_grad:
                    loss += l_val
                    
            if torch.is_tensor(loss) and loss.item() > 0:
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                iters += 1
                
        print(f"Epoch {config_epoch+1} / 3 Average Loss: {total_loss/max(1, iters):.4f}")

    # 4. Evaluation and Visualization
    print("Generating 10 visualizations...")
    model.eval()
    os.makedirs("visualizations", exist_ok=True)
    
    with torch.no_grad():
        for i, (images, targets) in enumerate(val_loader):
            if i >= 10:
                break
                
            images = images.to(device)
            gt_masks = targets[0]['masks']
            
            # Forward pass inference (YOLO conf threshold set artificially low at 0.05 to see boxes)
            model.detector.conf_threshold = 0.05 
            outputs = model(images)
            
            preds = outputs['detections'][0]
            pred_boxes = preds['boxes'].cpu()
            
            print(f"Image {i}: Found {len(pred_boxes)} predicted bounding boxes vs {len(targets[0]['boxes'])} ground truth boxes.")
            
            pred_masks = outputs['masks'][0, 0].cpu() # shape [H,W]
            
            image_cpu = images[0, 0].cpu() # shape [H,W]
            
            # Save Visualization
            out_file = os.path.join("visualizations", f"test_pred_{i}.png")
            visualize_evaluation(
                image_tensor=image_cpu,
                pred_boxes=pred_boxes,
                pred_mask=pred_masks,
                gt_mask=gt_masks[0],
                output_path=out_file,
                gt_boxes=targets[0]['boxes'].cpu()
            )
            print(f"Saved visualization for image {i} to {out_file}")

if __name__ == "__main__":
    train_and_visualize()
