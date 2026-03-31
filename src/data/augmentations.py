import albumentations as A
import numpy as np


def get_train_transforms(config=None):
    if config is None:
        config = {}

    transforms = []

    hflip_prob = config.get("horizontal_flip_prob", 0.5)
    if hflip_prob > 0:
        transforms.append(A.HorizontalFlip(p=hflip_prob))

    scale_limit = config.get("scale_limit", 0.2)
    scale_prob = config.get("scale_prob", 0.5)
    if scale_prob > 0 and scale_limit > 0:
        transforms.append(A.RandomScale(scale_limit=scale_limit, p=scale_prob))

    elastic_alpha = config.get("elastic_alpha", 30)
    elastic_sigma = config.get("elastic_sigma", 5)
    elastic_prob = config.get("elastic_prob", 0.3)
    if elastic_prob > 0 and elastic_alpha > 0:
        transforms.append(
            A.ElasticTransform(
                alpha=elastic_alpha,
                sigma=elastic_sigma,
                p=elastic_prob,
            )
        )

    brightness_limit = config.get("brightness_limit", 0.2)
    contrast_limit = config.get("contrast_limit", 0.2)
    bc_prob = config.get("brightness_contrast_prob", 0.5)
    if bc_prob > 0 and (brightness_limit > 0 or contrast_limit > 0):
        transforms.append(
            A.RandomBrightnessContrast(
                brightness_limit=brightness_limit,
                contrast_limit=contrast_limit,
                p=bc_prob,
            )
        )

    return A.Compose(transforms) if transforms else None


def get_val_transforms():
    return None
