"""vlm.eval_flight_frames: labels from true distances, resumable driver, per-cut yes-rates (fake judge, no GPU)."""
import json

import numpy as np

from interfaces import VLMAnswer
from vlm import eval_flight_frames as F


class FakeJudge:
    """Scores a frame by its mean pixel value, so the test controls the confidence through the picture."""

    calls = 0

    def ask(self, rgb, overlay, crop):
        FakeJudge.calls += 1
        c = float(rgb.mean()) - 10.0
        return VLMAnswer(c > 0, c, 0.01)


def test_frame_class_rules():
    assert F.frame_class(5.0, 5.0, "box", 15, 30) == "placed"
    assert F.frame_class(5.0, 100.0, None, 15, 30) == "scenery"
    assert F.frame_class(5.0, 5.0, None, 15, 30) == "scenery"  # no family name: not a placed obstacle
    assert F.frame_class(40.0, 100.0, None, 15, 30) == "clear"
    assert F.frame_class(20.0, 100.0, None, 15, 30) is None


def _flight(tmp_path, specs):
    d = tmp_path / "f1"
    d.mkdir()
    lines = []
    for i, (d_all, d_placed, fam, pix) in enumerate(specs):
        np.save(d / f"{i}_depth.npy", np.full((60, 80), d_all, dtype=np.float32))
        np.save(d / f"{i}_rgb.npy", np.full((60, 80, 3), pix, dtype=np.uint8))
        lines.append(json.dumps({"episode_id": "e1", "t": float(i), "rgb_path": str(d / f"{i}_rgb.npy"),
                                 "depth_gt_path": str(d / f"{i}_depth.npy"), "corridor_min_gt": d_placed,
                                 "obstacle_family": fam, "action_taken": "continue"}))
    (d / "flight.jsonl").write_text("\n".join(lines) + "\n")
    return d


def test_driver_labels_resumes_and_summarizes(tmp_path):
    d = _flight(tmp_path, [(5.0, 5.0, "wire", 30), (6.0, 100.0, None, 20), (50.0, 100.0, None, 5),
                           (50.0, 100.0, None, 25), (20.0, 100.0, None, 15)])
    out = tmp_path / "rows.jsonl"
    args = dict(stride=1, d_pos=15.0, d_neg=30.0, load_rgb=lambda p: np.load(p), load_depth=lambda p: np.load(p))
    FakeJudge.calls = 0
    rows = F.evaluate(FakeJudge(), "fake", [d], out, **args)
    assert [r["cls"] for r in rows] == ["placed", "scenery", "clear", "clear"]  # the 20 m frame is ambiguous: skipped
    assert FakeJudge.calls == 4
    F.evaluate(FakeJudge(), "fake", [d], out, **args)  # resume: nothing new is asked
    assert FakeJudge.calls == 4
    s = F.summarize(F.read_rows(out))
    assert s["n"] == {"placed": 1, "scenery": 1, "clear": 2}
    assert s["yes_rate"]["clear"]["0"] == 0.5 and s["yes_rate"]["placed"]["0"] == 1.0
    assert s["yes_rate"]["clear"]["-6"] == 1.0  # a low cut says yes to everything
    assert "| cut |" in F.table(s)
