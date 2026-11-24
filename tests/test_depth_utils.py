"""Depth clipping and corridor statistics (task W-S2)."""
import numpy as np
import pytest

from perception.depth_utils import (
    MAX_RANGE_M,
    clip_depth,
    corridor_features,
    corridor_min,
    corridor_window,
    valid_depth_mask,
)

H, W = 225, 400  # frame-bank resolution


def far_scene() -> np.ndarray:
    return np.full((H, W), 60.0, dtype=np.float32)


@pytest.mark.parametrize("sky", [65504.0, 16416.0, np.inf])
def test_saturated_sky_is_clipped_to_max_range(sky):
    # Frame-bank depth is float16; 65504 is its largest finite value.
    depth = np.full((H, W), sky, dtype=np.float16)
    depth[150:, :] = 12.0  # ground
    clipped = clip_depth(depth)
    assert clipped.dtype == np.float32
    assert clipped[:150].max() == MAX_RANGE_M
    assert np.allclose(clipped[150:], 12.0)
    assert not valid_depth_mask(depth)[:150].any()
    assert valid_depth_mask(depth)[150:].all()


def test_invalid_pixels_count_as_far_not_near():
    depth = far_scene()
    rows, cols = corridor_window(depth.shape)
    depth[rows, cols][:] = [[np.nan, 0.0, -1.0] + [60.0] * (cols.stop - cols.start - 3)]
    assert corridor_min(depth) == 60.0


def test_clip_depth_does_not_modify_its_input():
    depth = np.full((H, W), 65504.0, dtype=np.float16)
    clip_depth(depth)
    assert depth.max() == 65504.0


def test_corridor_window_matches_the_frame_bank_definition():
    rows, cols = corridor_window((H, W))
    assert (rows.start, rows.stop) == (int(0.4 * H), int(0.6 * H))
    assert (cols.start, cols.stop) == (int(0.45 * W), int(0.55 * W))


def test_corridor_min_sees_an_obstacle_inside_the_corridor_only():
    depth = far_scene()
    depth[100:120, 190:210] = 4.0  # centre of the image
    assert corridor_min(depth) == 4.0

    off_path = far_scene()
    off_path[100:120, 20:60] = 4.0  # far left, outside the corridor
    assert corridor_min(off_path) == 60.0


def test_corridor_min_of_open_sky_is_max_range():
    assert corridor_min(np.full((H, W), 65504.0, dtype=np.float16)) == MAX_RANGE_M


def test_corridor_features_are_plain_floats_and_react_to_an_obstacle():
    clear = corridor_features(far_scene())
    assert all(isinstance(v, float) for v in clear.values())
    assert clear["corridor_var"] == 0.0 and clear["corridor_grad_mean"] == 0.0

    depth = far_scene()
    depth[100:120, 195:205] = 4.0
    blocked = corridor_features(depth)
    assert blocked["corridor_min"] == 4.0
    assert blocked["corridor_var"] > 0 and blocked["corridor_grad_mean"] > 0
    assert blocked["corridor_median"] == 60.0  # a thin obstacle barely moves the median
