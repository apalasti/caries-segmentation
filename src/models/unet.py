import cv2
import numpy as np
import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, dropout=0.0):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
        )

    def forward(self, x):
        return self.double_conv(x)

class UNet(nn.Module):

    def __init__(
        self,
        n_channels=1,
        n_classes=1,
        depth=4,
        base_channels=64,
        dropout=0.2,
    ):
        super().__init__()

        self.n_channels = n_channels
        self.n_classes = n_classes
        self.name = "UNet Caries Segmentation"
        channels = [base_channels * (2 ** i) for i in range(depth)]

        self.inc = DoubleConv(n_channels, channels[0], dropout)

        self.downs = nn.ModuleList()

        for i in range(depth - 1):
            self.downs.append(
                nn.Sequential(
                    nn.MaxPool2d(2),
                    DoubleConv(
                        channels[i],
                        channels[i + 1],
                        dropout,
                    ),
                )
            )

        self.ups = nn.ModuleList()
        self.up_convs = nn.ModuleList()

        for i in range(depth - 1, 0, -1):

            self.ups.append(
                nn.ConvTranspose2d(
                    channels[i],
                    channels[i - 1],
                    kernel_size=2,
                    stride=2,
                )
            )

            self.up_convs.append(
                DoubleConv(
                    channels[i],
                    channels[i - 1],
                    dropout,
                )
            )

        self.outc = nn.Conv2d(
            channels[0],
            n_classes,
            kernel_size=1,
        )

    def forward(self, x):

        x_features = [self.inc(x)]

        for down in self.downs:
            x_features.append(down(x_features[-1]))

        x = x_features[-1]

        for i, (up, up_conv) in enumerate(
            zip(self.ups, self.up_convs)
        ):
            x = up(x)

            x = torch.cat(
                [x, x_features[-(i + 2)]],
                dim=1,
            )

            x = up_conv(x)

        return self.outc(x)

    @torch.no_grad()
    def predict(self, image: np.ndarray, device="cpu", threshold=0.5, return_prob=False):

        self.eval()

        original_h, original_w = image.shape[:2]

        gray = image

        resized = cv2.resize(gray, (256, 256))
        resized = resized.astype(np.float32) / 255.0

        tensor = torch.from_numpy(resized).unsqueeze(0).unsqueeze(0).to(device)

        logits = self(tensor)
        probs = torch.sigmoid(logits).squeeze().cpu().numpy()

        probs = cv2.resize(probs, (original_w, original_h), interpolation=cv2.INTER_LINEAR)

        if return_prob:
            return probs

        mask = (probs > threshold).astype(np.uint8)

        overlay = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if len(image.shape) == 2 else image.copy()

        color_mask = np.zeros_like(overlay)
        color_mask[:, :, 2] = mask * 255

        overlay = cv2.addWeighted(overlay, 1.0, color_mask, 0.4, 0)

        return mask, overlay