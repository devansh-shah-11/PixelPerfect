"""Tests for the closed loop: geometry, drone, scenarios and seeds, controller, raycast backend, flight loop, sweep (M4)."""
from __future__ import annotations

import json
import math
from dataclasses import replace

import numpy as np
import pytest
import yaml

from control.rules import RuleController
from eval.run_episode import LoopConfig, run_episode
from interfaces import Action, Controller, DepthEstimate, DroneState
from perception.depth_utils import corridor_features
from perception.frame_bank import OBSTACLE_SEG_RGB
from perception.stand_ins import BlindStack, TrueDepthStack
from policies.base import Always, Never, Oracle
from sim.drone import DroneParams, KinematicState, hold, step
from sim.geometry import OBB, hits_ground, ray_ground, ray_obb, segment_hits
from sim.raycast import Camera, RaycastSim, corridor_truth
from sim.scenario import Obstacle, Scenario, generate, half_extents, load_seed_lists, seed_lists, write_seed_lists
from vlm.surrogate import SurrogateJudge

CAM = Camera(160, 90)


def scenario(*obstacles: Obstacle, alt=2.0, goal=40.0, seed=1, split="val") -> Scenario:
    return Scenario(seed=seed, split=split, alt_m=alt, speed_mps=5.0, goal_m=goal, condition="clear", obstacles=list(obstacles))


# ---- geometry ------------------------------------------------------------------------------


def test_ray_box_and_ground():
    box = OBB(center=(10.0, 0.0, -2.0), half=(0.5, 0.5, 0.5))
    dirs = np.array([[1.0, 0, 0], [0, 1.0, 0], [-1.0, 0, 0]])
    t = ray_obb(np.array([0.0, 0, -2.0]), dirs, box)
    assert t[0] == pytest.approx(9.5) and np.isinf(t[1]) and np.isinf(t[2])
    rotated = OBB(center=(10.0, 0.0, -2.0), half=(0.5, 0.5, 0.5), yaw=math.pi / 4)
    assert ray_obb(np.array([0.0, 0, -2.0]), dirs[:1], rotated)[0] == pytest.approx(10 - 0.5 * math.sqrt(2), abs=1e-6)
    g = ray_ground(np.array([0.0, 0, -2.0]), np.array([[1.0, 0, 1.0]]) / math.sqrt(2))
    assert g[0] == pytest.approx(2 * math.sqrt(2))


def test_swept_collision_catches_thin_wires_and_vertical_moves():
    wire = OBB(center=(10.0, 0.0, -2.0), half=(0.005, 3.0, 0.005))  # 1 cm wire across the path
    assert segment_hits(np.array([9.8, 0, -2.0]), np.array([10.3, 0, -2.0]), wire, 0.3)  # passes through within one step
    assert not segment_hits(np.array([9.8, 0, -3.0]), np.array([10.3, 0, -3.0]), wire, 0.3)  # 1 m above: clear
    box = OBB(center=(10.0, 0.0, -2.0), half=(0.5, 0.5, 0.5))
    # Regression: a vertical step parallel to two slabs and far outside them must not collide.
    assert not segment_hits(np.array([2.0, 0, -2.0]), np.array([2.0, 0, -2.2]), box, 0.3)
    assert segment_hits(np.array([10.0, 0, -4.0]), np.array([10.0, 0, -2.0]), box, 0.3)
    assert hits_ground(np.array([0, 0, -0.5]), np.array([0, 0, -0.2]), 0.3) and not hits_ground(np.array([0, 0, -1.0]), np.array([0, 0, -1.0]), 0.3)


# ---- drone and controller -------------------------------------------------------------------


def test_drone_actions():
    p = DroneParams(v=5.0, climb_rate=2.0, alt_max=4.0)
    s = KinematicState(t=0.0, x=0.0, y=0.0, z=-2.0, yaw=0.0, speed=0.0)
    s1 = step(s, Action.CONTINUE, 0.1, p)
    assert (s1.x, s1.t, s1.speed) == (pytest.approx(0.5), pytest.approx(0.1), 5.0)
    s2 = step(s, Action.CLIMB, 0.1, p)
    assert s2.alt == pytest.approx(2.2) and s2.x == 0.0
    top = step(replace(s, z=-3.95), Action.CLIMB, 0.1, p)
    assert top.alt == pytest.approx(4.0)  # capped
    r = step(s, Action.YAW_RIGHT, 0.1, p)
    assert r.yaw == pytest.approx(math.radians(4.5)) and r.x == 0.0  # turns in place, NED positive yaw is right
    assert step(s1, Action.STOP, 0.1, p).x == pytest.approx(0.5)
    h = hold(s1, 1.0)
    assert h.t == pytest.approx(1.1) and h.x == s1.x and h.speed == 0.0


def est(cm: float, w: int = 40) -> DepthEstimate:
    d = np.full((20, w), cm, np.float32)
    return DepthEstimate(depth=d, features=corridor_features(d))


def test_controller_hysteresis_and_cap():
    c = RuleController(d_safe=8.0, hysteresis=2.0, alt_max=10.0)
    assert isinstance(c, Controller)
    s = DroneState(t=0, position=(0, 0, -2.0), yaw=0.0, speed=5.0)
    assert c.act(est(20.0), s) == Action.CONTINUE
    assert c.act(est(7.0), s) == Action.CLIMB
    assert c.act(est(9.0), s) == Action.CLIMB  # still blocked until above 10 m
    assert c.act(est(10.5), s) == Action.CONTINUE
    top = DroneState(t=0, position=(0, 0, -9.5), yaw=0.0, speed=5.0)
    d = np.full((20, 40), 5.0, np.float32)
    d[:, 20:] = 30.0  # more room on the right
    assert c.act(DepthEstimate(depth=d, features=corridor_features(d)), top) == Action.YAW_RIGHT
    off = DroneState(t=0, position=(0, 0, -2.0), yaw=-0.5, speed=5.0)
    assert c.act(est(50.0), off) == Action.YAW_RIGHT  # steer back to the goal bearing


# ---- scenarios and seeds ---------------------------------------------------------------------


def test_scenarios_are_deterministic_and_respect_holdout():
    from sim.scenario import ScenarioConfig

    assert generate(5, "val").to_json() == generate(5, "val").to_json()
    assert generate(5, "val").to_json() != generate(6, "val").to_json()
    cfg = ScenarioConfig(holdout_family="wire_1cm_axis_z")
    train = [o.family for s in range(300) for o in generate(s, "train", cfg).obstacles]
    val = [o.family for s in range(300) for o in generate(s, "val", cfg).obstacles]
    assert "wire_1cm_axis_z" not in train and "wire_1cm_axis_z" in val
    assert half_extents("wire_1cm_axis_z") == (0.005, 3.0, 0.005)  # horizontal bar across the path
    assert half_extents("glass_pane_alpha20") == (0.025, 1.25, 1.5)  # facing the camera
    sc = generate(5, "val")
    assert Scenario.from_dict(json.loads(sc.to_json())).to_json() == sc.to_json()


def test_seed_lists_freeze(tmp_path):
    lists = seed_lists(10, 5, 5)
    path = tmp_path / "seeds.yaml"
    write_seed_lists(path, lists)
    assert load_seed_lists(path) == lists
    with pytest.raises(FileExistsError):
        write_seed_lists(path, lists)
    d = yaml.safe_load(path.read_text())
    d["val"][0] = d["train"][0]  # tamper
    path.write_text(yaml.safe_dump(d))
    with pytest.raises(ValueError):
        load_seed_lists(path)


# ---- raycast backend -------------------------------------------------------------------------


def test_raycast_depth_seg_and_glass():
    sim = RaycastSim(CAM)
    sim.reset(scenario(Obstacle("box_1m_grey", 10.0, 0.0, 2.0), Obstacle("glass_pane_alpha20", 5.0, 0.0, 2.0)))
    s = KinematicState(t=0, x=0, y=0, z=-2.0, yaw=0.0, speed=0)
    obs = sim.observe(s)
    h, w = obs.depth.shape
    assert obs.depth[h // 2, w // 2] == pytest.approx(9.5, abs=0.05)  # sees the box *through* the glass, like Project AirSim
    assert (obs.seg[h // 2, w // 2] == OBSTACLE_SEG_RGB).all()
    assert obs.depth[0, w // 2] > 1000  # sky saturated
    d, fam = corridor_truth(s, sim.boxes, CAM)
    assert d == pytest.approx(4.975, abs=0.05) and fam == "glass_pane_alpha20"  # labels use the true geometry
    plain = RaycastSim(CAM)
    plain.reset(scenario(Obstacle("box_1m_grey", 10.0, 0.0, 2.0)))
    assert not np.array_equal(obs.rgb, plain.observe(s).rgb)  # the pane tints the image


# ---- flight loop -----------------------------------------------------------------------------


def fly(sc, stack, policy, judge=None, delay=1.0, **kw):
    return run_episode(sc, RaycastSim(CAM), stack, RuleController(), policy, judge, LoopConfig(delay_s=delay, **kw))


def test_true_depth_climbs_over_and_reaches_goal():
    recs, res = fly(scenario(Obstacle("box_1m_grey", 15.0, 0.0, 2.0)), TrueDepthStack(), Never())
    assert not res.crashed and res.reached_goal and res.n_calls == 0
    assert any(r.action_taken == Action.CLIMB for r in recs)


def test_true_depth_avoids_glass_the_camera_cannot_see():
    _, res = fly(scenario(Obstacle("glass_pane_alpha20", 15.0, 0.0, 2.0)), TrueDepthStack(), Never())
    assert not res.crashed and res.reached_goal


def test_blind_stack_crashes_and_the_asking_loop_saves_it():
    sc = scenario(Obstacle("pole_10cm_axis_z", 15.0, 0.0, 2.0))
    _, never = fly(sc, BlindStack(), Never())
    assert never.crashed and never.obstacle_family == "pole_10cm_axis_z"
    perfect = SurrogateJudge(default_tpr=1.0, default_fpr=0.0)
    _, timed_out = fly(sc, BlindStack(), Always(cooldown_s=0.0), perfect)
    assert not timed_out.crashed and not timed_out.reached_goal  # 1 s per frame: safe, but out of mission time (t_max 60 s)
    recs, always = fly(sc, BlindStack(), Always(cooldown_s=0.0), perfect, t_max=300.0)
    assert not always.crashed and always.reached_goal
    assert always.n_calls == len(recs) and always.delay_charged_s == pytest.approx(always.n_calls * 1.0)
    assert always.mission_time_s > never.mission_time_s  # asking costs time (brake and hold)
    _, oracle = fly(sc, BlindStack(), Oracle(), perfect)
    assert not oracle.crashed and 0 < oracle.n_calls < always.n_calls / 4  # asks only when the fast stack is wrong


def test_flights_replay_exactly_and_delay_must_be_positive():
    sc = generate(100_501, "val")
    judge = lambda: SurrogateJudge(default_tpr=0.7, default_fpr=0.1, seed=3)  # noqa: E731
    a, ra = fly(sc, BlindStack(), Always(cooldown_s=0.5), judge())
    b, rb = fly(sc, BlindStack(), Always(cooldown_s=0.5), judge())
    strip = lambda rs: [{**json.loads(r.to_json()), "vlm_latency_real_s": None} for r in rs]  # noqa: E731
    assert strip(a) == strip(b) and ra.to_json() == rb.to_json()
    with pytest.raises(ValueError):
        LoopConfig(delay_s=0.0)


def test_sweep_resumes_and_reports(tmp_path):
    from eval.sweep import main as sweep_main

    cfg = {"backend": "raycast", "backend_kw": {"width": 120, "height": 68}, "stack": "blind", "judge": "surrogate",
           "judge_kw": {"default_tpr": 0.9, "default_fpr": 0.05}, "delay_s": 1.0, "split": "val", "seeds": list(range(100_500, 100_504)),
           "policies": [{"name": "never", "policy": "never"}, {"name": "always", "policy": "always", "cooldown_s": 0.5},
                        {"name": "random5", "policy": "random", "p": 0.05, "cooldown_s": 0.5}],
           "out": str(tmp_path / "sweep"), "flight_logs": True}
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(cfg))
    sweep_main(["--config", str(path), "--no-log"])
    lines = (tmp_path / "sweep" / "results.jsonl").read_text().splitlines()
    assert len(lines) == 12
    sweep_main(["--config", str(path), "--no-log"])  # resume: nothing new
    assert len((tmp_path / "sweep" / "results.jsonl").read_text().splitlines()) == 12
    table = (tmp_path / "sweep" / "table.md").read_text()
    assert "surrogate VLM" in table and "| random5 |" in table and "raycast" in table
    assert len(list((tmp_path / "sweep" / "flights").glob("*.jsonl"))) == 12


def test_airsim_path_to_world_transform():
    from sim.airsim_backend import path_to_world

    assert path_to_world(10.0, 0.0, -2.0, 0.0, (5.0, 5.0, 0.0)) == pytest.approx((15.0, 5.0, -2.0, 0.0))
    x, y, z, yaw = path_to_world(10.0, 1.0, -2.0, 0.1, (0.0, 0.0, math.pi / 2))  # path heading east
    assert (x, y, z, yaw) == pytest.approx((-1.0, 10.0, -2.0, math.pi / 2 + 0.1))


def test_scenario_cli_freezes_seeds(tmp_path, capsys):
    from sim.scenario import main as scenario_main

    scenario_main(["--write-seeds", str(tmp_path / "seeds.yaml"), "--n", "5", "3", "3", "--show", "100500"])
    assert "checksum" in capsys.readouterr().out and len(load_seed_lists(tmp_path / "seeds.yaml")["test"]) == 3
