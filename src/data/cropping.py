"""Train-time random vs mask-focused patch crops on a fixed canvas."""

from typing import Literal

import cv2
import numpy as np
import torch


def random_patch_top_left(h: int, w: int, ph: int, pw: int, rng: np.random.Generator) -> tuple[int, int]:
    y0 = int(rng.integers(0, h - ph + 1))
    x0 = int(rng.integers(0, w - pw + 1))
    return y0, x0


def center_patch_top_left(h: int, w: int, ph: int, pw: int) -> tuple[int, int]:
    return (h - ph) // 2, (w - pw) // 2


def clamp_yolo_bbox(
    xc: float, yc: float, bw: float, bh: float
) -> tuple[float, float, float, float]:
    """Shrink (if needed) a normalized YOLO box so x±w/2 and y±h/2 lie in [0, 1].

    Preprocessing and Albumentations reject boxes that extend past the canvas;
    CSV annotations can slightly exceed 1.0 due to resize/rounding.
    """
    xc = float(np.clip(xc, 0.0, 1.0))
    yc = float(np.clip(yc, 0.0, 1.0))
    bw = max(0.0, float(bw))
    bh = max(0.0, float(bh))
    bw = min(bw, 2.0 * xc, 2.0 * (1.0 - xc))
    bh = min(bh, 2.0 * yc, 2.0 * (1.0 - yc))
    return (xc, yc, bw, bh)


def normed_box_to_pixels(
    xc: float, yc: float, bw: float, bh: float, h: int, w: int
) -> tuple[int, int, int, int]:
    """Convert normalized YOLO center-size box to clamped inclusive pixel bounds."""
    if h <= 0 or w <= 0:
        raise ValueError(f"Invalid canvas shape h={h}, w={w}")
    xc = float(np.clip(xc, 0.0, 1.0))
    yc = float(np.clip(yc, 0.0, 1.0))
    bw = max(0.0, float(bw))
    bh = max(0.0, float(bh))
    x0 = (xc - bw / 2.0) * w
    x1 = (xc + bw / 2.0) * w
    y0 = (yc - bh / 2.0) * h
    y1 = (yc + bh / 2.0) * h
    min_c = int(np.floor(x0))
    max_c = int(np.ceil(x1) - 1)
    min_r = int(np.floor(y0))
    max_r = int(np.ceil(y1) - 1)
    min_r = int(np.clip(min_r, 0, h - 1))
    max_r = int(np.clip(max_r, 0, h - 1))
    min_c = int(np.clip(min_c, 0, w - 1))
    max_c = int(np.clip(max_c, 0, w - 1))
    if max_r < min_r:
        min_r = max_r = int(np.clip(round(yc * (h - 1)), 0, h - 1))
    if max_c < min_c:
        min_c = max_c = int(np.clip(round(xc * (w - 1)), 0, w - 1))
    return min_r, min_c, max_r, max_c


def bbox_covering_patch_top_left(
    h: int,
    w: int,
    ph: int,
    pw: int,
    min_r: int,
    min_c: int,
    max_r: int,
    max_c: int,
) -> tuple[int, int]:
    """Deterministic top-left patch origin that covers the bbox when possible."""
    if ph <= 0 or pw <= 0:
        raise ValueError(f"Patch size must be positive, got ph={ph}, pw={pw}")
    if h < ph or w < pw:
        raise ValueError(f"Canvas ({h}, {w}) must be >= patch ({ph}, {pw})")
    bh = max_r - min_r + 1
    bw = max_c - min_c + 1
    if bh <= ph and bw <= pw:
        y_lo = max(0, max_r + 1 - ph)
        y_hi = min(min_r, h - ph)
        x_lo = max(0, max_c + 1 - pw)
        x_hi = min(min_c, w - pw)
        if y_lo <= y_hi:
            y0 = (y_lo + y_hi) // 2
        else:
            cy = (min_r + max_r) // 2
            y0 = max(0, min(cy - ph // 2, h - ph))
        if x_lo <= x_hi:
            x0 = (x_lo + x_hi) // 2
        else:
            cx = (min_c + max_c) // 2
            x0 = max(0, min(cx - pw // 2, w - pw))
        return y0, x0

    cy = (min_r + max_r) // 2
    cx = (min_c + max_c) // 2
    y0 = max(0, min(cy - ph // 2, h - ph))
    x0 = max(0, min(cx - pw // 2, w - pw))
    return y0, x0


def _connected_component_bboxes(mask: np.ndarray) -> list[tuple[int, int, int, int]]:
    """4-connected foreground blobs; each item is (min_r, min_c, max_r, max_c)."""
    m = (mask > 0).astype(np.uint8)
    if int(m.sum()) == 0:
        return []
    n, _, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=4)
    out: list[tuple[int, int, int, int]] = []
    for i in range(1, n):
        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        bw = int(stats[i, cv2.CC_STAT_WIDTH])
        bh = int(stats[i, cv2.CC_STAT_HEIGHT])
        min_r, min_c = y, x
        max_r, max_c = y + bh - 1, x + bw - 1
        out.append((min_r, min_c, max_r, max_c))
    return out


def _patch_top_left_for_bbox(
    h: int,
    w: int,
    ph: int,
    pw: int,
    min_r: int,
    min_c: int,
    max_r: int,
    max_c: int,
    rng: np.random.Generator,
) -> tuple[int, int]:
    """Place a ph x pw window to cover the given bbox; random jitter when the bbox fits in the patch."""
    bh = max_r - min_r + 1
    bw = max_c - min_c + 1

    if bh <= ph and bw <= pw:
        y_lo = max(0, max_r + 1 - ph)
        y_hi = min(min_r, h - ph)
        x_lo = max(0, max_c + 1 - pw)
        x_hi = min(min_c, w - pw)
        if y_lo <= y_hi:
            y0 = int(rng.integers(y_lo, y_hi + 1))
        else:
            cy = (min_r + max_r) // 2
            y0 = max(0, min(cy - ph // 2, h - ph))
        if x_lo <= x_hi:
            x0 = int(rng.integers(x_lo, x_hi + 1))
        else:
            cx = (min_c + max_c) // 2
            x0 = max(0, min(cx - pw // 2, w - pw))
        return y0, x0

    cy = (min_r + max_r) // 2
    cx = (min_c + max_c) // 2
    y0 = max(0, min(cy - ph // 2, h - ph))
    x0 = max(0, min(cx - pw // 2, w - pw))
    return y0, x0


def focused_patch_top_left(
    mask: np.ndarray, ph: int, pw: int, rng: np.random.Generator
) -> tuple[int, int]:
    """Pick a random connected lesion, then top-left of a patch that covers that blob (with jitter)."""
    h, w = mask.shape[:2]
    bboxes = _connected_component_bboxes(mask)
    if not bboxes:
        return random_patch_top_left(h, w, ph, pw, rng)
    comp = bboxes[int(rng.integers(0, len(bboxes)))]
    return _patch_top_left_for_bbox(h, w, ph, pw, *comp, rng)


def apply_patch_crop(
    img: np.ndarray, mask: np.ndarray, y0: int, x0: int, ph: int, pw: int
) -> tuple[np.ndarray, np.ndarray]:
    return img[y0 : y0 + ph, x0 : x0 + pw], mask[y0 : y0 + ph, x0 : x0 + pw]


def iter_tile_top_lefts(h: int, w: int, ph: int, pw: int) -> list[tuple[int, int]]:
    coords: list[tuple[int, int]] = []
    for y in range(0, h, ph):
        for x in range(0, w, pw):
            coords.append((y, x))
    return coords


def cut_patches_from_canvas(
    image: torch.Tensor,
    origins_yx: torch.Tensor,
    ph: int,
    pw: int,
) -> torch.Tensor:
    """Cut N patches of size (ph, pw) from a [C, H, W] canvas at given top-left origins.

    Returns a tensor of shape [N, C, ph, pw]. If N == 0, returns an empty tensor
    of shape [0, C, ph, pw].
    """
    if image.ndim != 3:
        raise ValueError(
            f"image must have shape [C, H, W], got {tuple(image.shape)}"
        )
    if origins_yx.ndim != 2 or origins_yx.shape[1] != 2:
        raise ValueError(
            f"origins_yx must have shape [N, 2], got {tuple(origins_yx.shape)}"
        )

    c, h, w = int(image.shape[0]), int(image.shape[1]), int(image.shape[2])
    n = int(origins_yx.shape[0])

    if n == 0:
        return image.new_zeros((0, c, ph, pw))

    if h < ph or w < pw:
        raise ValueError(f"Canvas ({h}, {w}) must be >= patch ({ph}, {pw})")

    patches = image.new_empty((n, c, ph, pw))
    origins = origins_yx.detach().to(dtype=torch.long).cpu()
    for i in range(n):
        y0 = int(origins[i, 0].item())
        x0 = int(origins[i, 1].item())
        if y0 < 0 or x0 < 0 or y0 + ph > h or x0 + pw > w:
            raise ValueError(
                f"patch {i} at ({y0}, {x0}) of size ({ph}, {pw}) "
                f"falls outside canvas ({h}, {w})"
            )
        patches[i] = image[:, y0 : y0 + ph, x0 : x0 + pw]
    return patches


def clamp_fill_for_dtype(fill_value: float, dtype: torch.dtype) -> float | int:
    """Clamp fill so ``Tensor.new_full(..., fill)`` is valid (e.g. float16 AMP)."""
    if not dtype.is_floating_point:
        return fill_value
    finfo = torch.finfo(dtype)
    f = float(fill_value)
    return max(finfo.min, min(finfo.max, f))


def stitch_patches(
    patch_values: torch.Tensor,
    origins_yx: torch.Tensor,
    canvas_hw: tuple[int, int] | torch.Tensor,
    reduce: Literal["max"] = "max",
    fill_value: float = -1e9,
) -> torch.Tensor:
    """Stitch [N, C, ph, pw] (or [N, ph, pw]) patches back to a canvas.

    Pixels not covered by any patch retain `fill_value`. Overlapping patches
    are combined with `reduce` ("max" only for now). Returns [C, H, W] for 4D
    inputs or [H, W] for 3D inputs.
    """
    if reduce != "max":
        raise ValueError(f"Unsupported reduce mode: {reduce!r}")

    if isinstance(canvas_hw, torch.Tensor):
        h = int(canvas_hw[0].item())
        w = int(canvas_hw[1].item())
    else:
        h, w = int(canvas_hw[0]), int(canvas_hw[1])

    fill = clamp_fill_for_dtype(fill_value, patch_values.dtype)
    if patch_values.ndim == 3:
        n, ph, pw = patch_values.shape
        out = patch_values.new_full((h, w), fill)
    elif patch_values.ndim == 4:
        n, c, ph, pw = patch_values.shape
        out = patch_values.new_full((c, h, w), fill)
    else:
        raise ValueError(
            f"patch_values must be 3D or 4D, got shape {tuple(patch_values.shape)}"
        )

    if origins_yx.shape[0] != n:
        raise ValueError(
            f"origins count {origins_yx.shape[0]} does not match patches {n}"
        )

    if n == 0:
        return out

    origins = origins_yx.detach().to(dtype=torch.long).cpu()
    for i in range(n):
        y0 = int(origins[i, 0].item())
        x0 = int(origins[i, 1].item())
        y1 = y0 + ph
        x1 = x0 + pw

        if patch_values.ndim == 3:
            out[y0:y1, x0:x1] = torch.maximum(
                out[y0:y1, x0:x1], patch_values[i]
            )
        else:
            out[:, y0:y1, x0:x1] = torch.maximum(
                out[:, y0:y1, x0:x1], patch_values[i]
            )
    return out
