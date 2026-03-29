from .unet import UNet


def get_model(config):
    model_config = config.get("model", {})
    return UNet(
        n_channels=model_config.get("n_channels", 1),
        n_classes=model_config.get("n_classes", 1),
        depth=model_config.get("depth", 4),
        base_channels=model_config.get("base_channels", 64),
    )
