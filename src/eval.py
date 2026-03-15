import torch
import torch.nn as nn
import os
import wandb
from config import load_config
from data.dataloader import get_dataloaders
from models import get_model
from utils.metrics import dice_coeff, DiceLoss

def evaluate():
    config = load_config()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    _, _, test_loader = get_dataloaders(config)
    
    model = get_model(config).to(device)
    best_model_path = os.path.join(config['training']['output_dir'], 'best_model.pth')
    
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        print("Loaded best model successfully.")
    else:
        print("Best model not found! Please run train.py first.")
        return
        
    model.eval()
    
    bce_loss = nn.BCEWithLogitsLoss()
    dice_loss_fn = DiceLoss()
    
    test_loss = 0
    test_dice = 0
    
    print("Evaluating...")
    
    with torch.no_grad():
        for images, masks in test_loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            
            loss = bce_loss(outputs, masks) + dice_loss_fn(outputs, masks)
            test_loss += loss.item()
            test_dice += dice_coeff(outputs, masks).item()
            
    avg_test_loss = test_loss / len(test_loader)
    avg_test_dice = test_dice / len(test_loader)
    
    print(f"Test Loss: {avg_test_loss:.4f} | Test Dice: {avg_test_dice:.4f}")

if __name__ == '__main__':
    evaluate()
