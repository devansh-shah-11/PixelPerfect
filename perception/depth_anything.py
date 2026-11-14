"""Fast stack: Depth Anything V2-Small, metric checkpoint (PLAN.md D7, task W-B1).

Needs the ``perception`` extra: ``pip install -e ".[perception]"``.
"""
from __future__ import annotations

import numpy as np

from interfaces import DepthEstimate
from perception.depth_utils import MAX_RANGE_M, clip_depth, corridor_features

# Outdoor metric checkpoint, Small encoder (Hugging Face, transformers format).
DEFAULT_CHECKPOINT = "depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf"


class DepthAnythingV2:
    """Implements ``interfaces.FastStack``.

    ``scale`` and ``shift`` are the frozen global correction from the scale check
    (task W-B2): ``depth = scale * raw + shift``. They are fitted once on train
    episodes and never from ground truth at run time.
    """

    def __init__(
        self,
        checkpoint: str = DEFAULT_CHECKPOINT,
        device: str | None = None,
        scale: float = 1.0,
        shift: float = 0.0,
        max_range: float = MAX_RANGE_M,
    ) -> None:
        self.checkpoint = checkpoint
        self.device = device
        self.scale = scale
        self.shift = shift
        self.max_range = max_range
        self._model = None
        self._processor = None

    def _load(self) -> None:
        import torch
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        if self.device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        self._processor = AutoImageProcessor.from_pretrained(self.checkpoint)
        self._model = AutoModelForDepthEstimation.from_pretrained(self.checkpoint).to(self.device).eval()

    def _infer(self, rgb: np.ndarray) -> np.ndarray:
        """Raw model depth in metres, (H, W) float32 at the input resolution."""
        import torch

        if self._model is None:
            self._load()
        inputs = self._processor(images=rgb, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self._model(**inputs)
        post = self._processor.post_process_depth_estimation(outputs, target_sizes=[rgb.shape[:2]])
        return post[0]["predicted_depth"].float().cpu().numpy()

    def perceive(self, rgb: np.ndarray) -> DepthEstimate:
        """``rgb`` is (H, W, 3) uint8 in RGB order (cv2.imread gives BGR: convert first)."""
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError(f"expected an (H, W, 3) RGB image, got shape {rgb.shape}")
        raw = self._infer(rgb)
        # A corrected estimate at or below zero means "very near", not "invalid".
        corrected = np.maximum(self.scale * raw + self.shift, 1e-3)
        depth = clip_depth(corrected, self.max_range)
        return DepthEstimate(depth=depth, features=corridor_features(depth, self.max_range))
