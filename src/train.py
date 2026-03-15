import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import os
from config import load_config
from data.dataloader import get_dataloaders
from models import get_model
from utils.metrics import dice_coeff, DiceLoss
from tqdm import tqdm

def train():
    config = load_config()
    
    wandb.init(project=config['wandb']['project'], config=config)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    train_loader, val_loader, _ = get_dataloaders(config)
    
    model = get_model(config).to(device)
    
    learning_rate = config['training'].get('learning_rate', 5e-4)
    weight_decay = config['training'].get('weight_decay', 1e-4)
    epochs = config['training'].get('epochs', 50)
    patience = config['training'].get('early_stopping_patience', 10)
    
    bce_loss = nn.BCEWithLogitsLoss()
    dice_loss_fn = DiceLoss()
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    
    best_val_dice = 0.0
    epochs_no_improve = 0
    os.makedirs(config['training']['output_dir'], exist_ok=True)
    best_model_path = os.path.join(config['training']['output_dir'], 'best_model.pth')
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        train_dice = 0
        
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for images, masks in loop:
            images, masks = images.to(device), masks.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = bce_loss(outputs, masks) + dice_loss_fn(outputs, masks)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_dice += dice_coeff(outputs, masks).item()
            loop.set_postfix({"loss": loss.item()})
            
        model.eval()
        val_loss = 0
        val_dice = 0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                loss = bce_loss(outputs, masks) + dice_loss_fn(outputs, masks)
                val_loss += loss.item()
                val_dice += dice_coeff(outputs, masks).item()
                
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        avg_val_dice = val_dice / len(val_loader)
        avg_train_dice = train_dice / len(train_loader)
        
        print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Dice: {avg_val_dice:.4f}")
        
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "train_dice": avg_train_dice,
            "val_dice": avg_val_dice,
            "learning_rate": optimizer.param_groups[0]['lr']
        })
        
        scheduler.step(avg_val_dice)
        
        if avg_val_dice > best_val_dice:
            best_val_dice = avg_val_dice
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved! Dice: {best_val_dice:.4f}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered after {patience} epochs with no improvement.")
                break
                
    wandb.finish()

if __name__ == '__main__':
    train()
