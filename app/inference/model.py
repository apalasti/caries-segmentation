import random

import cv2
import numpy as np


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


model = MockSegmentationModel()