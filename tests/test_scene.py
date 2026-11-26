"""vlm.scene and vlm.eval_scene: parsing, truth from true depth (placed + scenery), scoring, boxes to a move, resumable driver (no GPU)."""
import json

import numpy as np

from interfaces import VLMAnswer
from vlm import eval_scene as E
from vlm import scene as S


def test_parse_objects_fenced_clipped_and_bad():
    text = 'Sure:\n```json\n[{"box": [900, -5, 1200, 300], "kind": "wire", "range": "near"}, {"box": [1,2], "kind": "x"}, {"box": [10, 10, 10, 50]}]\n```'
    objs, ok = S.parse_objects(text)
    assert ok and len(objs) == 1  # a short box and a zero-width box are dropped
    assert objs[0]["box"] == [900.0, 0.0, 1000.0, 300.0] and objs[0]["kind"] == "wire"
    assert S.parse_objects("[]") == ([], True)
    assert S.parse_objects("I see a wall.") == ([], False)
    assert S.parse_objects('[{"box": [0,0,10,10], "kind": "ufo", "range": "soon"}]')[0][0]["range"] == "near"  # unreadable range: the safe side


def test_prompt_has_no_ground_truth_and_names_the_format():
    p = S.scene_prompt()
    assert "JSON" in p and "0 to 1000" in p and "close" in p
    assert S.cell_name(0, 1) == "upper centre" and "upper centre" in S.grid_prompt(0, 1)


def _depth(h=90, w=160, far=60.0):
    return np.full((h, w), far, dtype=np.float32)


def test_truth_scenery_placed_and_ground_excluded():
    d = _depth()
    d[20:50, 100:140] = 8.0  # a building face at 8 m, right of centre, above the horizon (row 45)
    d[75:, :] = 4.0  # flat ground close to the camera, below the ground rows: must not count
    mask = np.zeros(d.shape, bool)
    mask[30:70, 40] = True  # a 1-px wide vertical wire
    t = S.truth_objects(d, mask, alt_m=3.0, placed_dist_m=11.0, family="wire_1cm_axis_z", hfov_deg=90.0)
    src = {o["src"]: o for o in t}
    assert set(src) == {"placed", "scenery"} and len(t) == 2
    assert src["placed"]["kind"] == "wire" and src["placed"]["dist"] == 11.0 and src["placed"]["bucket"] == "near"
    assert src["scenery"]["bucket"] == "near" and 600 < src["scenery"]["box"][0] < 700  # x0 = 100/160 of 1000
    assert S.ground_row(3.0, 15.0, (90, 160), 90.0) > 45


def test_score_frame_found_supported_all_ones():
    truth = [{"src": "placed", "box": [400, 300, 420, 700], "dist": 9.0, "bucket": "near", "kind": "wire", "depth_misses": True},
             {"src": "scenery", "box": [700, 100, 900, 500], "dist": 30.0, "bucket": "far", "kind": "scenery"}]
    good = [{"box": [380, 280, 440, 720], "kind": "wire", "range": "near"}]
    s = S.score_frame(good, truth)
    assert s["positive"] and s["flagged"] and not s["all_ones"]
    assert s["objects"][0]["found"] and s["objects"][0]["range_exact"] and s["n_supported"] == 1
    wrong = [{"box": [0, 0, 100, 100], "kind": "wall", "range": "near"}]
    sw = S.score_frame(wrong, truth)
    assert sw["flagged"] and not sw["objects"][0]["found"] and sw["n_supported"] == 0  # a false alarm: near box with nothing near in it
    everything = [{"box": [0, 0, 1000, 1000], "kind": "wall", "range": "close"}]
    assert S.score_frame(everything, truth)["all_ones"]
    many = [{"box": [i * 100, 0, i * 100 + 50, 50], "kind": "other", "range": "near"} for i in range(4)]
    assert S.is_all_ones(many) and not S.is_all_ones(many[:3])
    assert not S.score_frame([], [truth[1]])["positive"]  # a far object is not a positive


def test_boxes_to_move_cases():
    bx = lambda x0, x1, y0=300, y1=700, r="near": {"box": [x0, y0, x1, y1], "kind": "other", "range": r}  # noqa: E731
    assert S.boxes_to_move([]) == "straight"
    assert S.boxes_to_move([bx(700, 900)]) == "straight"  # near but off the corridor
    assert S.boxes_to_move([bx(420, 560, r="far")]) == "straight"  # far boxes do not block
    assert S.boxes_to_move([bx(420, 900)]) == "left"  # blocked right of centre: the left side is wider
    assert S.boxes_to_move([bx(100, 580)]) == "right"
    assert S.boxes_to_move([bx(380, 620)], goal_deg=-25.0) == "left"  # a tie: the goal side wins
    assert S.boxes_to_move([bx(380, 620)], goal_deg=25.0) == "right"
    assert S.boxes_to_move([bx(0, 1000)]) == "over"  # covers the width, the top is free
    assert S.boxes_to_move([bx(0, 1000, y0=0)]) in ("left", "right")  # touches the top: no way over
    assert S.boxes_to_move([bx(300, 500), bx(500, 760)], goal_deg=0.0) in ("left", "right")  # two objects join into one group


def test_grid_objects_and_summary_baselines():
    scores = {S.cell_name(r, c): -1.0 for r in range(2) for c in range(3)}
    scores["lower right"] = 2.0
    g = S.grid_objects(scores, 0.0)
    assert len(g) == 1 and g[0]["box"][0] > 600
    assert S.cell_box(0, 0) == [0.0, 0.0, 1000 / 3, 500.0]


class FakeJudge:
    """JSON reply by frame brightness; yes/no and grid scores likewise. Counts calls to check that a resume asks nothing."""

    calls = 0
    _yes_ids, _no_ids = [0], [1]

    def generate(self, images, prompt, max_new_tokens=200):
        FakeJudge.calls += 1
        hit = float(images[0].mean()) > 100
        return ('[{"box": [380, 300, 620, 700], "kind": "wall", "range": "near"}]' if hit else "[]"), 0.5

    def score_yes_no(self, images, prompt):
        FakeJudge.calls += 1
        return (3.0 if float(images[0].mean()) > 100 else -3.0), 0.1

    def ask(self, image, overlay, crop):
        FakeJudge.calls += 1
        c = 3.0 if float(image.mean()) > 100 else -6.0
        return VLMAnswer(c > -4.02, c, 0.05)


def _frames():
    base = {"split": "val", "condition": "clear", "obstacle": "box_1m_black", "pose": [0, 0, -3.0, 0], "gt_lateral_m": 0.0}
    return [{**base, "episode": 1, "step": 0, "stem": "a", "gt_along_m": 9.0},   # an obstacle on course: avoid
            {**base, "episode": 1, "step": 1, "stem": "b", "gt_along_m": 40.0}]  # clear


def test_driver_end_to_end_resumes_and_scores(tmp_path, monkeypatch):
    depth = {"a": _depth(far=60.0), "b": _depth(far=60.0)}
    depth["a"][20:50, 40:120] = 8.0
    rgb = {"a": np.full((90, 160, 3), 200, np.uint8), "b": np.full((90, 160, 3), 20, np.uint8)}
    mask = np.zeros((90, 160), bool)
    monkeypatch.setattr(E, "ground_truth", lambda bank, fr, hfov: {"corridor_min_gt_obstacle": 100.0})
    kw = dict(load_rgb=lambda b, s: rgb[s], load_depth=lambda b, s: depth[s], load_mask=lambda b, s: mask)
    out = tmp_path / "scene.jsonl"
    FakeJudge.calls = 0
    rows = E.evaluate(FakeJudge(), "fake", "bank", _frames(), out, E.CONDITIONS, **kw)
    assert len(rows) == 6 and FakeJudge.calls == 2 * (1 + 6 + 1)
    E.evaluate(FakeJudge(), "fake", "bank", _frames(), out, E.CONDITIONS, **kw)
    assert FakeJudge.calls == 16  # resume: nothing new is asked
    summ = E.summarize(E.read_rows(out), yesno_cut=-4.02, grid_cut=0.0)
    js = summ["scene"]["fake|json_raw"]
    assert js["n_pos"] == 1 and js["n_neg"] == 1 and js["recall"] == 1.0 and js["false_alarm"] == 0.0
    assert summ["scene"]["oracle|-"]["recall"] == 1.0 and summ["scene"]["all_near|-"]["false_alarm"] == 1.0
    assert "scene_json_raw|boxes" in summ["steering"] and "oracle|-" in summ["steering"] and "depth_gt_choose|-" in summ["steering"]
    assert "| condition |" in summ["scene_table"] and "| model |" in summ["steering_table"]
    json.dumps(summ, default=float)
