from .unet import UNet, LargeUNet

def get_model(config):
    model_type = config['training'].get('model_type', 'unet')
    if model_type == 'unet':
        return UNet(n_channels=1, n_classes=1)
    elif model_type == 'large_unet':
        return LargeUNet(n_channels=1, n_classes=1)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
