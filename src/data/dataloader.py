import torch
from torch.utils.data import DataLoader, random_split
from .dataset import load_dc1000, load_dental, BaseKariesDataset
import hashlib
import os

def check_leakage(train_pairs, val_pairs, test_pairs):
    def get_filenames(pairs):
        return {os.path.basename(p[0]) for p in pairs}
        
    train_files = get_filenames(train_pairs)
    val_files = get_filenames(val_pairs)
    test_files = get_filenames(test_pairs)
    
    assert len(train_files.intersection(val_files)) == 0, "Data leakage between train and val!"
    assert len(train_files.intersection(test_files)) == 0, "Data leakage between train and test!"
    assert len(val_files.intersection(test_files)) == 0, "Data leakage between val and test!"

def get_pairs(dataset_type, config):
    if dataset_type == 'DC1000':
        return load_dc1000(config)
    elif dataset_type == 'Dental':
        return load_dental(config)
    elif dataset_type == 'Mixed':
        return load_dc1000(config) + load_dental(config)
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")

def deduplicate(pairs):
    unique_pairs = {}
    for img, mask in pairs:
        fname = os.path.basename(img)
        if fname not in unique_pairs:
            unique_pairs[fname] = (img, mask)
    return list(unique_pairs.values())

def get_dataloaders(config):
    train_dtype = config['data'].get('train_dataset_type', 'DC1000')
    test_dtype = config['data'].get('test_dataset_type', 'Same')

    generator = torch.Generator().manual_seed(config['training'].get('seed', 42))
    
    if test_dtype == 'Same' or test_dtype == train_dtype:
        all_pairs = deduplicate(get_pairs(train_dtype, config))
        
        total_len = len(all_pairs)
        val_split = config['data']['val_split']
        test_split = config['data']['test_split']
        
        num_val = int(total_len * val_split)
        num_test = int(total_len * test_split)
        num_train = total_len - num_val - num_test
        
        splits = random_split(all_pairs, [num_train, num_val, num_test], generator=generator)
        
        train_pairs = [all_pairs[i] for i in splits[0].indices]
        val_pairs = [all_pairs[i] for i in splits[1].indices]
        test_pairs = [all_pairs[i] for i in splits[2].indices]
    else:
        train_val_pairs = deduplicate(get_pairs(train_dtype, config))
        test_pairs = deduplicate(get_pairs(test_dtype, config))
        
        total_len = len(train_val_pairs)
        val_split = config['data']['val_split']
        
        num_val = int(total_len * val_split)
        num_train = total_len - num_val
        
        splits = random_split(train_val_pairs, [num_train, num_val], generator=generator)
        train_pairs = [train_val_pairs[i] for i in splits[0].indices]
        val_pairs = [train_val_pairs[i] for i in splits[1].indices]

    check_leakage(train_pairs, val_pairs, test_pairs)
    
    size = tuple(config['data'].get('images_size', [256, 256]))
    train_ds = BaseKariesDataset(train_pairs, size=size)
    val_ds = BaseKariesDataset(val_pairs, size=size)
    test_ds = BaseKariesDataset(test_pairs, size=size)
    
    bs = config['training'].get('batch_size', 32)
    nw = config['training'].get('num_workers', 4)
    
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=nw)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=nw)
    test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False, num_workers=nw)
    
    return train_loader, val_loader, test_loader
