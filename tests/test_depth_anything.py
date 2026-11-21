"""Fast-stack wrapper logic, with the network replaced by a stub (task W-B1)."""
import numpy as np
import pytest

from interfaces import DepthEstimate, FastStack
from perception.depth_anything import DepthAnythingV2

H, W = 225, 400


class StubDepth(DepthAnythingV2):
    """Returns a fixed raw depth map instead of running the model."""

    def __init__(self, raw: np.ndarray, **kwargs) -> None:
        super().__init__(**kwargs)
        self.raw = raw

    def _infer(self, rgb: np.ndarray) -> np.ndarray:
        return self.raw


def rgb() -> np.ndarray:
    return np.zeros((H, W, 3), dtype=np.uint8)


def test_satisfies_the_fast_stack_interface():
    assert isinstance(StubDepth(np.ones((H, W), dtype=np.float32)), FastStack)


def test_perceive_returns_clipped_depth_and_corridor_features():
    raw = np.full((H, W), 500.0, dtype=np.float32)
    raw[100:120, 190:210] = 6.0
    est = StubDepth(raw).perceive(rgb())
    assert isinstance(est, DepthEstimate)
    assert est.depth.shape == (H, W) and est.depth.dtype == np.float32
    assert est.depth.max() == 100.0
    assert est.features["corridor_min"] == 6.0


def test_scale_and_shift_are_applied_before_clipping():
    raw = np.full((H, W), 10.0, dtype=np.float32)
    est = StubDepth(raw, scale=2.0, shift=1.0).perceive(rgb())
    assert est.features["corridor_min"] == 21.0


def test_non_positive_corrected_depth_is_near_not_far():
    raw = np.full((H, W), 1.0, dtype=np.float32)
    est = StubDepth(raw, shift=-5.0).perceive(rgb())
    assert est.features["corridor_min"] < 0.01


def test_rejects_a_grayscale_image():
    with pytest.raises(ValueError):
        StubDepth(np.ones((H, W), dtype=np.float32)).perceive(np.zeros((H, W), dtype=np.uint8))
