import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

class BaseKariesDataset(Dataset):
    def __init__(self, data_pairs, size=(256, 256)):
        """
        data_pairs: list of tuples (img_path, mask_path)
        """
        self.data_pairs = data_pairs
        self.size = size

    def __len__(self):
        return len(self.data_pairs)

    def __getitem__(self, i):
        img_path, mask_path = self.data_pairs[i]

        img = Image.open(img_path).convert('L')
        mask = Image.open(mask_path).convert('L')

        img = img.resize(self.size, resample=Image.BILINEAR)
        mask = mask.resize(self.size, resample=Image.NEAREST)

        img = np.array(img).astype(np.float32) / 255.0
        mask = np.array(mask)
        mask = (mask > 127).astype(np.float32)

        img = torch.from_numpy(img).unsqueeze(0)
        mask = torch.from_numpy(mask).unsqueeze(0)
        return img, mask

def load_dc1000(config):
    images_dir = config['data']['dc1000_train_img']
    masks_dir = config['data']['dc1000_train_lbl']
    ids = [f for f in os.listdir(images_dir) if os.path.isfile(os.path.join(images_dir, f))]
    return [(os.path.join(images_dir, img_id), os.path.join(masks_dir, img_id)) for img_id in ids]

def load_dental(config):
    pairs = []
    
    adult_img = config['data']['adult_img']
    adult_lbl = config['data']['adult_lbl']
    if os.path.exists(adult_img):
        ids = [f for f in os.listdir(adult_img) if f.endswith(('.jpg', '.png'))]
        for f in ids:
             base = os.path.splitext(f)[0]
             pairs.append((os.path.join(adult_img, f), os.path.join(adult_lbl, base + '.png'))) # adjust extension if needed
             
    children_img = config['data']['children_img']
    children_lbl = config['data']['children_lbl']
    if os.path.exists(children_img):
        ids = [f for f in os.listdir(children_img) if f.endswith(('.jpg', '.png'))]
        for f in ids:
             base = os.path.splitext(f)[0]
             pairs.append((os.path.join(children_img, f), os.path.join(children_lbl, base + '.png')))
             
    return pairs

def get_dataset(config):
    dataset_type = config['data']['dataset_type']
    size = tuple(config['data'].get('images_size', [256, 256]))
    
    if dataset_type == 'DC1000':
        pairs = load_dc1000(config)
    elif dataset_type == 'Dental':
        pairs = load_dental(config)
    elif dataset_type == 'Mixed':
        pairs = load_dc1000(config) + load_dental(config)
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
        
    return BaseKariesDataset(pairs, size=size)
