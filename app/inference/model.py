import random
from pathlib import Path

import torch
import cv2
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from src.models import UNet
import gdown

class MockSegmentationModel:

    def __init__(self):
        self.name = "Mock Dental Caries U-Net"

    def predict(self, image: np.ndarray):

        h, w = image.shape[:2]

        mask = np.zeros((h, w), dtype=np.uint8)

        number_of_regions = random.randint(2, 6)

        for _ in range(number_of_regions):

            center_x = random.randint(
                int(w * 0.1),
                int(w * 0.9),
            )

            center_y = random.randint(
                int(h * 0.1),
                int(h * 0.9),
            )

            axis_x = random.randint(
                max(10, int(w * 0.02)),
                max(20, int(w * 0.08)),
            )

            axis_y = random.randint(
                max(10, int(h * 0.02)),
                max(20, int(h * 0.08)),
            )

            angle = random.randint(0, 180)

            cv2.ellipse(
                mask,
                (center_x, center_y),
                (axis_x, axis_y),
                angle,
                0,
                360,
                255,
                -1,
            )

        overlay = image.copy()

        color_mask = np.zeros_like(image)

        color_mask[:, :, 0] = mask

        overlay = cv2.addWeighted(
            overlay,
            1.0,
            color_mask,
            0.4,
            0,
        )

        return mask, overlay



#model = MockSegmentationModel()
FILE_ID = "1HJL3H8buXZJQGewJ-frNZouG_eAn9rzl"

BASE_DIR = Path(__file__).parent.parent
MODEL_PATH = BASE_DIR / "checkpoints" / "solar-field-59.ckpt"

MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

if not MODEL_PATH.exists():
    print("Downloading model from Google Drive...")
    gdown.download(
        id=FILE_ID,
        output=str(MODEL_PATH),
        quiet=False,
    )

ckpt = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)

hparams = ckpt["hyper_parameters"]

model = UNet(
    n_channels=hparams["model"]["n_channels"],
    n_classes=hparams["model"]["n_classes"],
    depth=hparams["model"]["depth"],
    base_channels=hparams["model"]["base_channels"],
    dropout=hparams["model"]["dropout"],
)
print("n_channels",hparams["model"]["n_channels"])
state_dict = ckpt["state_dict"]

clean_state_dict = {
    k.replace("model.", ""): v
    for k, v in state_dict.items()
    if k.startswith("model.")
}

missing, unexpected = model.load_state_dict(clean_state_dict, strict=False)

print("Missing keys:", missing)
print("Unexpected keys:", unexpected)

model.eval()

print("Weights loaded successfully.")