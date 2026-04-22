"""Train-time random vs mask-focused patch crops on a fixed canvas."""

import cv2
import numpy as np
import torch


def random_patch_top_left(h: int, w: int, ph: int, pw: int, rng: np.random.Generator) -> tuple[int, int]:
    y0 = int(rng.integers(0, h - ph + 1))
    x0 = int(rng.integers(0, w - pw + 1))
    return y0, x0


def center_patch_top_left(h: int, w: int, ph: int, pw: int) -> tuple[int, int]:
    return (h - ph) // 2, (w - pw) // 2


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


def pad_to_tile_grid(
    img: np.ndarray, mask: np.ndarray, ph: int, pw: int
) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    """Pad bottom/right with zeros so height and width are multiples of ph and pw."""
    h, w = img.shape[:2]
    H = ((h + ph - 1) // ph) * ph
    W = ((w + pw - 1) // pw) * pw
    if H == h and W == w:
        return img, mask, (h, w)
    out_i = np.zeros((H, W), dtype=img.dtype)
    out_m = np.zeros((H, W), dtype=mask.dtype)
    out_i[:h, :w] = img
    out_m[:h, :w] = mask
    return out_i, out_m, (h, w)


def iter_tile_top_lefts(h: int, w: int, ph: int, pw: int) -> list[tuple[int, int]]:
    coords: list[tuple[int, int]] = []
    for y in range(0, h, ph):
        for x in range(0, w, pw):
            coords.append((y, x))
    return coords


def padded_canvas_hw(orig_h: int, orig_w: int, ph: int, pw: int) -> tuple[int, int]:
    H = ((orig_h + ph - 1) // ph) * ph
    W = ((orig_w + pw - 1) // pw) * pw
    return H, W


def stitch_tiles(
    tiles: torch.Tensor,
    canvas_h: int,
    canvas_w: int,
    tile_h: int,
    tile_w: int,
) -> torch.Tensor:
    """Paste tile batch into a canvas (same order as ``iter_tile_top_lefts``).

    Args:
        tiles: ``(B, N, C, tile_h, tile_w)`` row-major over ``(y0, x0)``.
        canvas_h, canvas_w: Padded canvas size (multiples of ``tile_h``, ``tile_w``).
    """
    B, N, C, th, tw = tiles.shape
    if th != tile_h or tw != tile_w:
        raise ValueError(
            f"tile spatial shape ({th}, {tw}) != ({tile_h}, {tile_w})"
        )
    out = tiles.new_zeros((B, C, canvas_h, canvas_w))
    k = 0
    for y0 in range(0, canvas_h, tile_h):
        for x0 in range(0, canvas_w, tile_w):
            out[:, :, y0 : y0 + tile_h, x0 : x0 + tile_w] = tiles[:, k, :, :, :]
            k += 1
    if k != N:
        raise ValueError(
            f"tile count N={N} does not match canvas grid "
            f"{canvas_h // tile_h}x{canvas_w // tile_w}={k}"
        )
    return out


def tiled_eval_layout_tensors(
    orig_h: int, orig_w: int, ph: int, pw: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fixed grid for val/test when every sample is resized to (orig_h, orig_w) before padding."""
    H, W = padded_canvas_hw(orig_h, orig_w, ph, pw)
    coords = iter_tile_top_lefts(H, W, ph, pw)
    tile_origins = torch.tensor(coords, dtype=torch.long)
    canvas_hw = torch.tensor([H, W], dtype=torch.long)
    orig_hw = torch.tensor([orig_h, orig_w], dtype=torch.long)
    return tile_origins, canvas_hw, orig_hw
