"""L1 gate labels (task W-B3)."""
import numpy as np
import pytest

from gate.labels import Outcome, l1_labels, outcomes

D_SAFE = 8.0


@pytest.mark.parametrize(
    "gt, est, expected",
    [
        (30.0, 25.0, Outcome.AGREE_CLEAR),
        (5.0, 20.0, Outcome.MISSED),  # e.g. a wire the depth model does not see
        (30.0, 6.0, Outcome.FALSE_STOP),
        (5.0, 4.0, Outcome.AGREE_AVOID),
    ],
)
def test_each_outcome(gt, est, expected):
    assert outcomes(gt, est, D_SAFE) == expected


def test_only_a_missed_obstacle_is_an_l1_positive():
    gt = np.array([30.0, 5.0, 30.0, 5.0])
    est = np.array([25.0, 20.0, 6.0, 4.0])
    assert l1_labels(gt, est, D_SAFE).tolist() == [0, 1, 0, 0]


def test_boundary_at_d_safe_counts_as_clear():
    # "obstacle" means strictly closer than d_safe (PLAN_DETAILED.md §4.3).
    assert outcomes(D_SAFE, D_SAFE, D_SAFE) == Outcome.AGREE_CLEAR
    assert outcomes(D_SAFE - 0.01, D_SAFE, D_SAFE) == Outcome.MISSED


def test_label_depends_on_which_side_of_d_safe_not_on_error_size():
    # A large depth error far from the drone is not a missed obstacle.
    assert l1_labels(60.0, 20.0, D_SAFE) == 0
    # A small error that crosses d_safe is.
    assert l1_labels(7.5, 8.5, D_SAFE) == 1


def test_nan_is_rejected():
    with pytest.raises(ValueError):
        outcomes(np.array([np.nan]), np.array([5.0]), D_SAFE)
