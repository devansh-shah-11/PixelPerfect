"""Path overlay and full-resolution corridor crop for the VLM judge (PLAN_DETAILED.md D9, chunk M2.2).

The VLM is shown two images: the frame with the flight path drawn as a
highlighted region, and a full-resolution crop of that region. Downscaling
erases 1 to 2 px wires, which is why the crop exists.

MVP region: the fixed forward-corridor window that the labels use
(``perception.depth_utils.CORRIDOR_ROWS`` / ``CORRIDOR_COLS``, rows 40 to 60%
and columns 45 to 55% of the image). Asking the VLM about exactly the region
the L1 label is computed from keeps the two comparable. The projected flight
tube over the next ``T_h`` seconds replaces this window once the camera
intrinsics are fixed (TASKS.md, parked).

Pure numpy and Pillow; no ground truth is used anywhere here.
"""
from __future__ import annotations

import numpy as np
from PIL import Image

# Must equal perception.depth_utils.CORRIDOR_ROWS / CORRIDOR_COLS. Duplicated so
# that vlm/ imports nothing from perception/; tests/test_vlm.py checks they agree.
CORRIDOR_ROWS = (0.4, 0.6)
CORRIDOR_COLS = (0.45, 0.55)

Box = tuple[int, int, int, int]  # r0, r1, c0, c1 (half-open)


def corridor_box(shape: tuple[int, ...]) -> Box:
    """Pixel box of the forward corridor for an image of ``shape`` (H, W, ...)."""
    h, w = shape[:2]
    return (
        int(CORRIDOR_ROWS[0] * h),
        int(CORRIDOR_ROWS[1] * h),
        int(CORRIDOR_COLS[0] * w),
        int(CORRIDOR_COLS[1] * w),
    )


def draw_overlay(
    rgb: np.ndarray,
    box: Box,
    color: tuple[int, int, int] = (255, 40, 40),
    thickness: int = 2,
    fill_alpha: float = 0.15,
) -> np.ndarray:
    """Copy of ``rgb`` with ``box`` outlined in ``color`` and lightly tinted inside.

    The tint is light on purpose: a strong fill would hide the thin wire we are
    asking about.
    """
    _check_rgb(rgb)
    r0, r1, c0, c1 = box
    out = rgb.astype(np.float32).copy()
    col = np.array(color, dtype=np.float32)
    if fill_alpha > 0:
        out[r0:r1, c0:c1] = (1 - fill_alpha) * out[r0:r1, c0:c1] + fill_alpha * col
    t = max(1, thickness)
    h, w = rgb.shape[:2]
    # Outline drawn just outside the box so the region's own pixels stay visible.
    top, bot = max(0, r0 - t), min(h, r1 + t)
    left, right = max(0, c0 - t), min(w, c1 + t)
    out[top:r0, left:right] = col
    out[r1:bot, left:right] = col
    out[top:bot, left:c0] = col
    out[top:bot, c1:right] = col
    return np.clip(out, 0, 255).astype(np.uint8)


def crop_region(
    rgb: np.ndarray,
    box: Box,
    margin: float = 0.5,
    upscale: int = 4,
    resample: int = Image.BICUBIC,
) -> np.ndarray:
    """Full-resolution crop of ``box`` plus ``margin`` (fraction of the box size), enlarged ``upscale`` times.

    On a 225 x 400 frame the corridor box is 45 x 40 px; with the default
    margin and upscale the crop is 360 x 320 px, so a 1 px wire becomes 4 px.
    """
    _check_rgb(rgb)
    r0, r1, c0, c1 = box
    h, w = rgb.shape[:2]
    mr, mc = int(round(margin * (r1 - r0))), int(round(margin * (c1 - c0)))
    rr0, rr1 = max(0, r0 - mr), min(h, r1 + mr)
    cc0, cc1 = max(0, c0 - mc), min(w, c1 + mc)
    crop = rgb[rr0:rr1, cc0:cc1]
    if upscale > 1:
        im = Image.fromarray(crop).resize((crop.shape[1] * upscale, crop.shape[0] * upscale), resample)
        crop = np.asarray(im)
    return np.ascontiguousarray(crop)


def render(rgb: np.ndarray, box: Box | None = None, **crop_kwargs) -> tuple[np.ndarray, np.ndarray]:
    """``(path_overlay, crop)`` for one frame, as ``interfaces.VLMJudge.ask`` expects them."""
    if box is None:
        box = corridor_box(rgb.shape)
    return draw_overlay(rgb, box), crop_region(rgb, box, **crop_kwargs)


def _check_rgb(rgb: np.ndarray) -> None:
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError(f"expected an (H, W, 3) uint8 RGB image, got shape {rgb.shape} dtype {rgb.dtype}")
