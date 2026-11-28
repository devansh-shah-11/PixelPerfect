"""vlm.features: hidden-state extraction over saved flight pictures (fake judge, no GPU)."""
import json

import numpy as np

from vlm import features as F
from vlm.judge import QwenVLJudge


class FakeJudge:
    """Score = mean pixel - 10; the 'hidden state' of each layer is a vector filled with the pixel value plus the layer."""

    def __init__(self):
        self.calls = 0

    def features(self, rgb, overlay, crop, layers):
        self.calls += 1
        m = float(rgb.mean())
        return m - 10.0, {i: np.full(4, m + i, dtype=np.float32) for i in layers}


def _logs(tmp_path, n=7, episode="val-101000-never"):
    lines = []
    for i in range(n):
        np.save(tmp_path / f"{episode}_{i}_d.npy", np.full((60, 80), 5.0 + 10 * i, dtype=np.float32))
        np.save(tmp_path / f"{episode}_{i}_rgb.npy", np.full((60, 80, 3), i, dtype=np.uint8))
        saved = i % 2 == 0  # save_every: only some frames have a picture
        lines.append(json.dumps({"episode_id": episode, "split": "val", "seed": 101000, "t": float(i), "pose": [0, 0, -2.0, 0],
                                 "rgb_path": str(tmp_path / f"{episode}_{i}_rgb.npy") if saved else "",
                                 "depth_gt_path": str(tmp_path / f"{episode}_{i}_d.npy") if saved else "",
                                 "corridor_min_gt": 100.0, "obstacle_family": None,
                                 "scenery_min_gt": 3.0 + i}))
    p = tmp_path / f"{episode}.jsonl"
    p.write_text("\n".join(lines) + "\n")
    return p


def _run(tmp_path, log, judge, shard_size=2):
    return F.extract(judge, [log], tmp_path / "out", layers=(-1, -9), shard_size=shard_size,
                     load_rgb=lambda p: np.load(p), load_depth=lambda p: np.load(p))


def test_extracts_saved_frames_only_and_loads_back_in_order(tmp_path):
    log = _logs(tmp_path)
    assert _run(tmp_path, log, FakeJudge()) == 4  # frames 0, 2, 4, 6
    index, X = F.load(tmp_path / "out", -9)
    assert [r["t"] for r in index] == [0.0, 2.0, 4.0, 6.0] and X.shape == (4, 4)
    assert np.allclose(X[:, 0], [0 - 9, 2 - 9, 4 - 9, 6 - 9])
    assert index[0]["alt"] == 2.0 and index[1]["d_all"] < index[3]["d_all"] and index[2]["score"] == -6.0
    assert index[1]["d_scenery"] == 5.0
    assert len(list((tmp_path / "out").glob("shard_*.npz"))) == 2


def test_resume_skips_frames_already_extracted(tmp_path):
    log = _logs(tmp_path)
    _run(tmp_path, log, FakeJudge())
    j = FakeJudge()
    assert _run(tmp_path, log, j) == 0 and j.calls == 0
    log2 = _logs(tmp_path, episode="val-101001-always")
    assert _run(tmp_path, log2, FakeJudge()) == 4
    index, X = F.load(tmp_path / "out", -1)
    assert len(index) == 8 and X.shape == (8, 4) and len({r["shard"] for r in index}) == 4


def test_main_refuses_test_split_logs(tmp_path):
    log = _logs(tmp_path, episode="test-101500-never")
    try:
        F.main(["--logs", str(log), "--out", str(tmp_path / "o")])
    except SystemExit as e:
        assert "test" in str(e)
    else:
        raise AssertionError("test-split logs were accepted")


def test_judge_features_uses_the_ask_inputs_and_score():
    class Fake(QwenVLJudge):
        def __init__(self):
            super().__init__()
            self._yes_ids, self._no_ids = [1], [3]
            self.n_images = None

        def _hidden(self, images, layers):
            self.n_images = len(images)
            logits = np.zeros(8)
            logits[1], logits[3] = 2.0, 0.0
            return logits, {i: np.ones(3) * i for i in layers}

    j = Fake()
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    score, v = j.features(img, img, img, layers=(-1, -2))
    assert j.n_images == 2 and score > 0 and set(v) == {-1, -2}
