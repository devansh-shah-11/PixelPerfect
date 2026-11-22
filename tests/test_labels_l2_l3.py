import numpy as np
import pytest

from gate.labels import ReplayOutcome, l2_labels, l3_labels, replay_outcomes


def test_l2_marks_frames_within_k_seconds_before_the_crash():
    t = np.arange(0.0, 3.01, 0.5)  # 0.0 .. 3.0
    assert l2_labels(t, crash_time=3.0, k=1.0).tolist() == [0, 0, 0, 0, 1, 1, 1]
    assert l2_labels(t, crash_time=None, k=1.0).sum() == 0
    with pytest.raises(ValueError):
        l2_labels(t, 3.0, k=0.0)


def test_l3_is_positive_only_when_asking_prevents_the_crash():
    without = [False, True, False, True]
    with_vlm = [False, False, True, True]
    assert replay_outcomes(without, with_vlm).tolist() == [
        ReplayOutcome.NO_EFFECT_SAFE, ReplayOutcome.SAVED, ReplayOutcome.CAUSED, ReplayOutcome.NO_EFFECT_CRASH]
    assert l3_labels(without, with_vlm).tolist() == [0, 1, 0, 0]
    with pytest.raises(ValueError):
        l3_labels([True], [True, False])
