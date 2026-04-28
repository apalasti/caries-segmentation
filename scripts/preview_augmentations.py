#!/usr/bin/env python3
"""Visualize BaseKariesDataset with train augmentations and mask overlay."""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.config import load_config
from src.data.augmentations import get_train_transforms
from src.data.dataset import BaseKariesDataset, load_split_pairs


def overlay_mask_on_image(
    img: np.ndarray, mask: np.ndarray, alpha: float = 0.45
) -> np.ndarray:
    """img, mask: 2D float HxW (typically [0, 1]). Returns HxWx3 RGB in [0, 1]."""
    g = np.clip(img.astype(np.float64), 0.0, 1.0)
    rgb = np.stack([g, g, g], axis=-1)
    m = np.clip(mask.astype(np.float64), 0.0, 1.0)
    red = np.stack([np.ones_like(g), np.zeros_like(g), np.zeros_like(g)], axis=-1)
    w = alpha * m[..., None]
    return np.clip(rgb * (1.0 - w) + red * w, 0.0, 1.0)


def parse_indices(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Show augmented images from BaseKariesDataset with mask overlay."
    )
    parser.add_argument("--config", type=str, default="config.toml")
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "val", "test"],
    )
    parser.add_argument("--n", type=int, default=3, help="with --start: how many samples")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument(
        "--indices",
        type=str,
        default=None,
        help="comma-separated dataset indices (overrides --start/--n)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="if set, save figure to this path (PNG recommended)",
    )
    parser.add_argument("--no-show", action="store_true")
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.45,
        help="mask overlay blend strength",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    config = load_config(args.config)
    preprocessed = config["data"]["preprocessed_path"]
    raw_sources = config["data"].get("sources", [])
    sources = raw_sources if raw_sources else None
    size = tuple(config["data"].get("images_size", [640, 640]))
    aug_cfg = config.get("augmentation", {})

    transform = get_train_transforms(aug_cfg, size)
    pairs = load_split_pairs(preprocessed, args.split, sources)
    if not pairs:
        raise SystemExit(
            f"No samples for split={args.split!r} under {preprocessed!r} (check data.csv)."
        )

    if args.indices is not None:
        indices = parse_indices(args.indices)
    else:
        end = min(args.start + args.n, len(pairs))
        indices = list(range(args.start, end))

    for i in indices:
        if i < 0 or i >= len(pairs):
            raise SystemExit(f"Index {i} out of range [0, {len(pairs) - 1}]")

    dataset = BaseKariesDataset(pairs, size=size, transform=transform, patch_size=128)

    ncols = len(indices)
    fig, axes = plt.subplots(1, ncols, figsize=(4 * ncols, 4), squeeze=False)
    for ax, idx in zip(axes[0], indices):
        img_t, mask_t = dataset[idx]
        img = img_t.squeeze().numpy()
        mask = mask_t.squeeze().numpy()
        vis = overlay_mask_on_image(img, mask, alpha=args.alpha)
        ax.imshow(vis, interpolation="nearest")
        ax.set_title(f"Index {idx}")
        ax.axis("off")
    fig.suptitle(
        f"split={args.split} (same pipeline as BaseKariesDataset.__getitem__)",
        fontsize=10,
    )
    plt.tight_layout()
    if args.save:
        fig.savefig(args.save, dpi=150, bbox_inches="tight")
    if not args.no_show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
