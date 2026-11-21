"""Frame-bank reader, episode split (W-S1) and bank runner (W-B1) on a small fake bank."""
import json

import numpy as np
import pytest
import yaml
from PIL import Image

from interfaces import DepthEstimate
from perception import run_bank as run_bank_module
from perception.bank_split import SPLITS, check_split, load_split, make_split, split_of_episode
from perception.depth_utils import corridor_features
from perception.frame_bank import episode_rows, frame_rows, load_depth_gt, load_rgb, read_jsonl
from perception.run_bank import run_bank

H, W = 225, 400


def write_bank(root, n_episodes=4, steps=3):
    """A bank in the layout of hpc/capability/capability_test.py. Episode 1 has an obstacle at 5 m."""
    (root / "frames").mkdir(parents=True)
    with open(root / "frames.jsonl", "w") as frames, open(root / "episodes.jsonl", "w") as episodes:
        for ep in range(n_episodes):
            obstacle = "pole" if ep == 1 else "none"
            for k in range(steps):
                stem = f"{ep:04d}_{k:03d}"
                rgb = np.zeros((H, W, 3), dtype=np.uint8)
                rgb[..., 0] = 200  # red in RGB order
                Image.fromarray(rgb).save(root / "frames" / f"{stem}_rgb.png")
                depth = np.full((H, W), 65504.0, dtype=np.float16)  # sky
                depth[150:] = 20.0
                if obstacle != "none":
                    depth[100:120, 190:210] = 5.0
                np.save(root / "frames" / f"{stem}_depth.npy", depth)
                row = {"episode": ep, "step": k, "stem": stem, "condition": "clear", "obstacle": obstacle}
                if obstacle != "none":
                    row.update(gt_along_m=5.0, gt_lateral_m=0.0)
                frames.write(json.dumps(row) + "\n")
            frames.write(json.dumps({"episode": ep, "step": steps, "error": "missing channel"}) + "\n")
            episodes.write(json.dumps({"episode": ep, "condition": "clear", "obstacle": obstacle, "frames": steps}) + "\n")
    return root


class BlindModel:
    """Fast stack that never sees an obstacle: every pixel is 50 m away."""

    def __init__(self):
        self.calls = 0

    def perceive(self, rgb):
        self.calls += 1
        depth = np.full(rgb.shape[:2], 50.0, dtype=np.float32)
        return DepthEstimate(depth=depth, features=corridor_features(depth))


@pytest.fixture
def bank(tmp_path):
    return write_bank(tmp_path / "bank")


def test_reader_skips_error_rows_and_loads_rgb_and_float32_depth(bank):
    assert len(episode_rows(bank)) == 4
    rows = frame_rows(bank)
    assert len(rows) == 12 and all("stem" in r for r in rows)
    rgb = load_rgb(bank, "0000_000")
    assert rgb.shape == (H, W, 3) and rgb[0, 0].tolist() == [200, 0, 0]
    depth = load_depth_gt(bank, "0000_000")
    assert depth.dtype == np.float32 and depth.max() == 65504.0


def test_runner_writes_gt_and_estimated_corridor_minimum(bank, tmp_path):
    out = tmp_path / "runs" / "out.jsonl"
    counts = run_bank(bank, out, BlindModel())
    assert counts == {"done": 12, "skipped": 0}
    rows = {r["stem"]: r for r in read_jsonl(out)}
    assert rows["0000_000"]["corridor_min_gt"] == 100.0  # open sky, clipped
    assert rows["0001_000"]["corridor_min_gt"] == 5.0  # the pole
    assert rows["0001_000"]["corridor_min_est"] == 50.0
    assert rows["0001_000"]["gt_along_m"] == 5.0 and rows["0000_000"]["gt_along_m"] is None
    assert "corridor_var" in rows["0000_000"]["features"]


def test_runner_resumes_without_redoing_finished_frames(bank, tmp_path):
    out = tmp_path / "out.jsonl"
    model = BlindModel()
    assert run_bank(bank, out, model, limit=5) == {"done": 5, "skipped": 0}
    assert run_bank(bank, out, model) == {"done": 7, "skipped": 5}
    assert run_bank(bank, out, model) == {"done": 0, "skipped": 12}
    assert model.calls == 12
    stems = [r["stem"] for r in read_jsonl(out)]
    assert len(stems) == len(set(stems)) == 12


def test_runner_can_save_estimated_depth(bank, tmp_path):
    run_bank(bank, tmp_path / "out.jsonl", BlindModel(), save_depth_dir=tmp_path / "depth", limit=1)
    saved = np.load(tmp_path / "depth" / "0000_000_depth_est.npy")
    assert saved.shape == (H, W) and saved.dtype == np.float16


def test_runner_never_passes_ground_truth_to_the_model(bank, tmp_path, monkeypatch):
    seen = []

    class Recorder(BlindModel):
        def perceive(self, rgb):
            seen.append(rgb.shape)
            return super().perceive(rgb)

    monkeypatch.setattr(run_bank_module, "load_rgb", lambda b, s: np.zeros((H, W, 3), dtype=np.uint8))
    run_bank(bank, tmp_path / "out.jsonl", Recorder(), limit=2)
    assert seen == [(H, W, 3)] * 2  # only the RGB image reaches perceive()


def episodes(n=400):
    obstacles = ["none", "pole", "glass", "box"]
    conditions = ["clear", "dusk", "rain", "fog", "snow"]
    return [{"episode": i, "obstacle": obstacles[i % 4], "condition": conditions[(i // 4) % 5]} for i in range(n)]


def test_split_is_disjoint_complete_and_excludes_fog_and_snow():
    split = make_split(episodes())
    check_split(split)
    assigned = split_of_episode(split)
    assert len(assigned) == 240 and len(split["excluded"]) == 160
    assert set(assigned) | set(split["excluded"]) == set(range(400))
    by_id = {e["episode"]: e for e in episodes()}
    assert all(by_id[i]["condition"] in ("fog", "snow") for i in split["excluded"])


def test_split_is_close_to_70_15_15_and_every_stratum_reaches_val_and_test():
    split = make_split(episodes())
    n = split["n_episodes"]
    assert 0.65 <= n["train"] / 240 <= 0.75
    assert 0.10 <= n["val"] / 240 <= 0.20 and 0.10 <= n["test"] / 240 <= 0.20
    by_id = {e["episode"]: e for e in episodes()}
    for name in ("val", "test"):
        strata = {(by_id[i]["obstacle"], by_id[i]["condition"]) for i in split["splits"][name]}
        assert len(strata) == 12  # 4 obstacles x 3 kept conditions


def test_split_is_deterministic_and_seed_dependent():
    assert make_split(episodes(), seed=0) == make_split(list(reversed(episodes())), seed=0)
    assert make_split(episodes(), seed=0)["checksum"] != make_split(episodes(), seed=1)["checksum"]


def test_check_split_catches_leakage_and_tampering(tmp_path):
    split = make_split(episodes())
    leaked = json.loads(json.dumps(split))
    leaked["splits"]["test"].append(leaked["splits"]["train"][0])
    with pytest.raises(ValueError, match="both"):
        check_split(leaked)

    tampered = json.loads(json.dumps(split))
    tampered["splits"]["test"].pop()
    path = tmp_path / "split.yaml"
    path.write_text(yaml.safe_dump(tampered))
    with pytest.raises(ValueError, match="checksum"):
        load_split(path)


def test_split_names():
    assert SPLITS == ("train", "val", "test")
