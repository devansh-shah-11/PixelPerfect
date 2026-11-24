"""Bare-map scan: map scenery is hit in every pass, a bad capture in one."""
import io

import numpy as np

from eval.pose_clearance import report, scan
from sim.raycast import Observation


class FakeMap:
    """A surface at x = 5 above 4 m on pose 1; one blank capture on pose 0 in the first pass."""

    def __init__(self):
        self.pose, self.calls = -1, 0

    def empty_map(self, start):
        self.pose = int(start[0])

    def observe(self, s):
        self.calls += 1
        d = np.full((45, 80), 50.0, dtype=np.float32)
        if self.pose == 1 and -s.z >= 4.0 and abs(s.x - 5.0) < 0.1:
            d[:] = 0.1
        if self.pose == 0 and self.calls == 1:
            d[:] = 0.0
        return Observation(rgb=np.zeros((45, 80, 3), np.uint8), depth=d, seg=np.zeros((45, 80, 3), np.uint8))


def test_scan_separates_map_hits_from_bad_captures():
    rows = scan(FakeMap(), [[0, 0, 0], [1, 0, 0]], [0.0, 5.0], [2.0, 4.0], 2, io.StringIO())
    assert len(rows) == 2 * 2 * 2 * 2
    text = "\n".join(report(rows, 0.3))
    assert "| 1 | 4 | 2 | 1 | 5 | 0 | none | 1 |" in text  # solid in both passes
    assert "| 0 | 2 | 2 | 0 | none | 1 | 0 | 0 |" in text  # first capture only
    assert "with zero depth in the window: 1" in text
