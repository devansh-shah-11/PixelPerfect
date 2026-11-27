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


def test_descend_sinks_in_place_and_stops_at_the_landing_height():
    p = DroneParams(climb_rate=2.0, alt_min=1.0)
    s = KinematicState(t=0.0, x=3.0, y=0.0, z=-2.0, yaw=0.0, speed=5.0)
    d = step(s, Action.DESCEND, 0.1, p)
    assert d.alt == pytest.approx(1.8) and d.x == 3.0 and d.speed == 0.0  # in place by default
    low = step(replace(s, z=-1.1), Action.DESCEND, 0.1, p)
    assert low.alt == pytest.approx(1.0)  # floored at the landing height, never ground contact
    assert step(replace(s, z=-0.8), Action.DESCEND, 0.1, p).alt == pytest.approx(0.8)  # a low drone is not pushed up
    assert not hits_ground(s.position, low.position, p.radius)


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


def banded(centre: float, left: float, right: float, up: float, h: int = 40, w: int = 80) -> DepthEstimate:
    """A depth estimate with set values in the corridor, the side bands and the upper band (control.rules.free_space)."""
    d = np.full((h, w), centre, np.float32)
    d[int(0.35 * h): int(0.65 * h), : int(0.4 * w)] = left
    d[int(0.35 * h): int(0.65 * h), int(0.6 * w):] = right
    d[: int(0.3 * h), int(0.4 * w): int(0.6 * w)] = up
    return DepthEstimate(depth=d, features=corridor_features(d))


def pos(x=0.0, y=0.0, alt=2.0, yaw=0.0) -> DroneState:
    return DroneState(t=0, position=(x, y, -alt), yaw=yaw, speed=5.0)


def test_choose_mode_picks_the_freer_side_keeps_it_and_rescores_when_it_closes():
    c = RuleController(avoid_mode="choose", goal_bias_m=0.0)
    assert c.act(banded(5.0, left=12.0, right=40.0, up=90.0), pos()) == Action.YAW_RIGHT  # sky reads far, but a free side wins
    assert c.phase == "avoid"
    assert c.act(banded(5.0, left=30.0, right=20.0, up=90.0), pos()) == Action.YAW_RIGHT  # kept while it stays free
    assert c.act(banded(5.0, left=30.0, right=6.0, up=90.0), pos()) == Action.YAW_LEFT  # re-scored once it reads blocked
    c2 = RuleController(avoid_mode="choose")
    assert c2.act(banded(5.0, left=6.0, right=7.0, up=90.0), pos()) == Action.CLIMB  # both sides blocked: go over
    assert c2.act(banded(5.0, left=6.0, right=7.0, up=90.0), pos(alt=24.5)) == Action.YAW_RIGHT  # at the cap: better side


def test_choose_mode_prefers_the_side_towards_the_goal_and_follows_the_vlm():
    c = RuleController(avoid_mode="choose", goal_bias_m=5.0)
    c.set_mission((50.0, -10.0), cruise_alt=2.0, land_alt=1.0)  # goal ahead and to the left
    assert c.act(banded(5.0, left=20.0, right=22.0, up=90.0), pos()) == Action.YAW_LEFT  # 20 + 5 beats 22
    c.reset()
    assert c.avoid_action(banded(50.0, left=10.0, right=40.0, up=90.0), pos(), direction="left") == Action.YAW_LEFT
    assert c.avoid_action(banded(50.0, left=10.0, right=40.0, up=90.0), pos(), direction=None) == Action.YAW_LEFT  # VLM's side held
    c.reset()
    assert c.avoid_action(banded(50.0, left=40.0, right=40.0, up=90.0), pos(), direction="over") == Action.CLIMB
    with pytest.raises(ValueError):
        c.avoid_action(banded(50.0, left=40.0, right=40.0, up=90.0), pos(), direction="down")
    with pytest.raises(ValueError):
        RuleController(avoid_mode="random")


def test_mission_steers_to_the_goal_bypasses_returns_to_cruise_and_lands():
    c = RuleController(avoid_mode="choose")
    c.set_mission((40.0, 40.0), cruise_alt=2.0, land_alt=1.0)
    clear = banded(60.0, left=60.0, right=60.0, up=90.0)
    assert c.act(clear, pos()) == Action.YAW_RIGHT  # goal bearing +45 deg, positive yaw turns right (NED)
    assert c.act(clear, pos(yaw=math.radians(45))) == Action.CONTINUE
    c.reset()
    c.set_mission((60.0, 0.0), cruise_alt=2.0, land_alt=1.0)
    assert c.act(banded(5.0, left=6.0, right=7.0, up=90.0), pos(x=10.0)) == Action.CLIMB  # obstacle estimated 5 m ahead: over
    assert c.act(clear, pos(x=10.0, alt=6.0)) == Action.CONTINUE and c.phase == "bypass"  # hold the heading until past it
    assert c.act(clear, pos(x=16.5, alt=6.0)) == Action.CONTINUE and c.phase == "bypass"  # 6.5 m ahead < 5 + 2
    assert c.act(clear, pos(x=17.1, alt=6.0)) == Action.DESCEND and c.phase == "return"  # past the obstacle: back down
    assert c.act(clear, pos(x=22.5, alt=2.1)) == Action.CONTINUE and c.phase == "cruise"
    assert c.act(clear, pos(x=58.5, alt=2.0)) == Action.DESCEND and c.phase == "landing"  # within 2 m of the goal
    assert c.act(banded(1.0, left=1.0, right=1.0, up=1.0), pos(x=58.5, alt=1.1)) == Action.DESCEND  # down to the landing height
    assert c.act(banded(1.0, left=1.0, right=1.0, up=1.0), pos(x=58.5, alt=1.0)) == Action.STOP  # landed; obstacles ignored


def test_climb_over_is_bounded_so_a_tall_object_is_taken_on_a_side():
    """Both sides and the way ahead read blocked: climb, but only ``avoid_climb_max_m`` above cruise height, then the better side."""
    c = RuleController(avoid_mode="choose", goal_bias_m=0.0, avoid_climb_max_m=6.0)
    c.set_mission((60.0, 0.0), cruise_alt=2.0, land_alt=1.0)
    wall = banded(5.0, left=6.0, right=7.0, up=90.0)
    assert c.act(wall, pos(alt=2.0)) == Action.CLIMB
    assert c.act(wall, pos(alt=7.5)) == Action.CLIMB  # still inside the budget (cruise 2 + 6 = 8)
    c2 = RuleController(avoid_mode="choose", goal_bias_m=0.0, avoid_climb_max_m=6.0)
    c2.set_mission((60.0, 0.0), cruise_alt=2.0, land_alt=1.0)
    assert c2.act(wall, pos(alt=8.5)) == Action.YAW_RIGHT  # budget used up: the better side, not further up
    c3 = RuleController(avoid_mode="choose", avoid_climb_max_m=6.0)  # no mission: measured from where the avoidance began
    assert c3.act(wall, pos(alt=10.0)) == Action.CLIMB
    assert c3.act(wall, pos(alt=16.5)) == Action.YAW_RIGHT


def test_bypass_ends_past_the_vlm_range_or_after_a_sideways_detour_and_restarts_on_a_new_obstacle():
    c = RuleController(avoid_mode="choose", d_judge=15.0)
    c.set_mission((60.0, 0.0), cruise_alt=2.0, land_alt=1.0)
    clear = banded(60.0, left=60.0, right=60.0, up=90.0)
    assert c.avoid_action(clear, pos(x=5.0), direction="left") == Action.YAW_LEFT  # VLM only: depth saw nothing
    assert c.act(clear, pos(x=15.0, y=-3.0, yaw=-0.3)) == Action.CONTINUE and c.phase == "bypass"  # 10 m < 15 + 2
    assert c.act(banded(5.0, left=6.0, right=40.0, up=90.0), pos(x=17.0, y=-3.5, yaw=-0.3)) == Action.YAW_RIGHT  # new obstacle
    assert c.phase == "avoid"
    c.reset()
    c.set_mission((60.0, 0.0), cruise_alt=2.0, land_alt=1.0)
    c.act(banded(5.0, left=6.0, right=40.0, up=90.0), pos(x=10.0))  # obstacle 5 m ahead, turns right
    assert c.act(clear, pos(x=10.5, y=12.0, yaw=1.5)) == Action.CONTINUE  # sideways: only 0.5 m further along, 12 m flown
    assert c.act(clear, pos(x=10.5, y=14.0, yaw=1.5)) == Action.YAW_LEFT and c.phase == "cruise"  # 14 m >= 2 x 7: turn back to the goal


def test_soft_geofence_blocks_the_outward_side_overrides_the_vlm_and_cuts_the_hold():
    c = RuleController(avoid_mode="choose", goal_bias_m=0.0, fence_margin_m=3.0)
    c.set_mission((60.0, 0.0), cruise_alt=2.0, land_alt=1.0, area_half_width_m=15.0)
    blocked = banded(5.0, left=12.0, right=40.0, up=90.0)  # right is freer
    assert c.act(blocked, pos(x=20.0, y=5.0)) == Action.YAW_RIGHT  # well inside the area: free choice
    c.reset()
    c.set_mission((60.0, 0.0), cruise_alt=2.0, land_alt=1.0, area_half_width_m=15.0)
    assert c.act(blocked, pos(x=20.0, y=13.0)) == Action.YAW_LEFT  # 13 m right of the line, 2 m from the edge: right is out
    c.reset()
    c.set_mission((60.0, 0.0), cruise_alt=2.0, land_alt=1.0, area_half_width_m=15.0)
    assert c.avoid_action(est(50.0), pos(x=20.0, y=-13.0), direction="left") == Action.CLIMB  # VLM's outward side becomes "over"
    c.reset()
    c.set_mission((60.0, 0.0), cruise_alt=2.0, land_alt=1.0, area_half_width_m=15.0)
    c.act(banded(5.0, left=6.0, right=40.0, up=90.0), pos(x=10.0))  # avoids to the right, obstacle 5 m ahead
    clear = banded(60.0, left=60.0, right=60.0, up=90.0)
    assert c.act(clear, pos(x=11.0, y=12.5, yaw=1.2)) == Action.YAW_LEFT and c.phase == "cruise"  # near the edge: hold ends


def test_side_choice_does_not_flip_back_and_forth():
    """Job B1 seed 101068: turning in place moved the obstacle between the side bands, so the choice flipped every turn."""
    c = RuleController(avoid_mode="choose", goal_bias_m=0.0)
    right_free, left_free = banded(5.0, left=6.0, right=30.0, up=90.0), banded(5.0, left=30.0, right=6.0, up=90.0)
    assert c.act(right_free, pos()) == Action.YAW_RIGHT
    assert c.act(left_free, pos()) == Action.YAW_LEFT  # right closed: one switch allowed
    assert c.act(right_free, pos()) == Action.CLIMB  # left closed too: both sides tried, go over
    assert c.act(left_free, pos(alt=5.0)) == Action.CLIMB  # over is kept
    c.reset()
    assert c.act(right_free, pos(alt=24.5)) == Action.YAW_RIGHT  # at the height cap
    assert c.act(left_free, pos(alt=24.5)) == Action.YAW_LEFT
    assert c.act(right_free, pos(alt=24.5)) == Action.YAW_LEFT  # cannot climb: keeps the side instead of flipping
    c.reset()
    assert c.act(right_free, pos()) == Action.YAW_RIGHT  # a new avoidance episode gets its switch back


def test_fence_does_not_cut_the_hold_after_going_over():
    """Job B1 seeds 101071, 101077: near the edge the drone climbed over, the fence ended the hold at once, it descended
    in front of the obstacle, saw it again and climbed again, until it was marked lost."""
    c = RuleController(avoid_mode="choose", goal_bias_m=0.0, fence_margin_m=3.0)
    c.set_mission((55.0, 0.0), cruise_alt=2.0, land_alt=1.0, area_half_width_m=15.0)
    clear = banded(60.0, left=60.0, right=60.0, up=90.0)
    assert c.act(banded(5.0, left=6.0, right=6.0, up=90.0), pos(x=20.0, y=13.0)) == Action.CLIMB  # both sides blocked: over
    assert c.act(clear, pos(x=20.0, y=13.0, alt=5.0)) == Action.CONTINUE and c.phase == "bypass"  # still holds near the edge
    assert c.act(clear, pos(x=27.1, y=13.0, alt=5.0)) == Action.DESCEND and c.phase == "return"  # past it: back down
    c.reset()
    c.set_mission((55.0, 0.0), cruise_alt=2.0, land_alt=1.0, area_half_width_m=15.0)
    assert c.avoid_action(est(50.0), pos(x=20.0, y=-13.0), direction="left") == Action.CLIMB  # outward VLM side becomes over
    assert c.act(clear, pos(x=20.0, y=-13.0, alt=4.0)) == Action.CONTINUE and c.phase == "bypass"


def test_cone_courses_carry_the_mission_area():
    from sim.scenario import ScenarioConfig

    sc = generate(3, "val", ScenarioConfig(layout="cone", area_half_width_m=12.0))
    assert sc.area_half_width_m == 12.0 and generate(3, "val").area_half_width_m is None


def test_best_side_is_the_smallest_move_past_the_box():
    from sim.geometry import best_side

    me = np.array([0.0, 0.0, -2.0])
    post = OBB(center=(10.0, 0.2, -2.5), half=(0.05, 0.05, 2.5))  # vertical post 0.2 m to the right, 5 m tall
    assert best_side(me, 0.0, post, 0.3) == "left"  # 0.15 m left beats 0.55 m right
    bar = OBB(center=(10.0, 0.0, -2.0), half=(0.05, 2.5, 0.05))  # horizontal bar across the path
    assert best_side(me, 0.0, bar, 0.3) == "over"
    wall = OBB(center=(10.0, -3.0, -2.5), half=(0.1, 4.0, 2.5))  # 5 m tall wall mostly on the left: 1.3 m right vs 3.3 m over
    assert best_side(me, 0.0, wall, 0.3) == "right"
    east_post = OBB(center=(0.2, 10.0, -2.5), half=(0.05, 0.05, 2.5))  # heading east (NED): north of the line is the left
    assert best_side(me, math.pi / 2, east_post, 0.3) == "right"


def test_line_courses_keep_the_original_controller():
    old, new = RuleController(), RuleController()
    seq = [(est(20.0), pos()), (est(7.0), pos()), (est(9.0), pos(alt=3.0)), (est(12.0), pos(alt=4.0)), (est(50.0), pos(alt=4.0, yaw=-0.5))]
    assert [new.act(e, s) for e, s in seq] == [Action.CONTINUE, Action.CLIMB, Action.CLIMB, Action.CONTINUE, Action.YAW_RIGHT]
    assert new.config()["avoid_mode"] == "climb_first" and new.goal is None


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
    assert sc.goal_xy is None and sc.land_alt_m is None  # line courses carry no goal point


def test_cone_courses_place_the_goal_and_space_the_obstacles():
    from sim.scenario import ScenarioConfig

    cfg = ScenarioConfig(layout="cone", n_obstacles=(1, 5), p_in_path=0.5)
    assert generate(7, "val", cfg).to_json() == generate(7, "val", cfg).to_json()
    counts = []
    for s in range(300):
        sc = generate(s, "val", cfg)
        assert Scenario.from_dict(json.loads(sc.to_json())).to_json() == sc.to_json()
        gx, gy = sc.goal_xy
        assert 50.0 <= sc.goal_m <= 60.0 and abs(gy) <= 10.0 and math.hypot(gx, gy) == pytest.approx(sc.goal_m, abs=1e-2)
        assert sc.land_alt_m == 1.0
        ux, uy = gx / sc.goal_m, gy / sc.goal_m
        along = sorted(o.along_m * ux + o.lateral_m * uy for o in sc.obstacles)  # projection on the start-to-goal line
        assert 1 <= len(along) <= 5
        assert along[0] >= 12.0 - 1e-2 and along[-1] <= sc.goal_m - 8.0 + 1e-2
        assert all(b - a >= 8.0 - 1e-2 for a, b in zip(along, along[1:]))  # positions are stored to the millimetre
        assert all(math.hypot(o.along_m - gx, o.lateral_m - gy) >= 8.0 - 1e-2 for o in sc.obstacles)  # landing spot clear
        assert all(o.yaw == pytest.approx(math.atan2(uy, ux), abs=1e-4) for o in sc.obstacles)  # facing along the line
        counts.append(len(along))
    assert set(counts) == {1, 2, 3, 4, 5}  # every count occurs (5 only when the goal is far enough)
    assert generate(7, "val").to_json() != generate(7, "val", cfg).to_json()
    with pytest.raises(ValueError):
        generate(7, "val", ScenarioConfig(layout="spiral"))


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


def test_scenery_guard_counts_unspawned_walls_as_crashes():
    from sim.raycast import Observation

    class WallSim(RaycastSim):
        """Raycast backend with an invisible-to-the-scenario wall: depth says 0.1 m everywhere."""

        def observe(self, state):
            obs = super().observe(state)
            return Observation(rgb=obs.rgb, depth=np.full_like(obs.depth, 0.1), seg=obs.seg)

    sc = scenario()  # no spawned obstacles at all
    _, res = run_episode(sc, WallSim(CAM), TrueDepthStack(), RuleController(), Never(), None, LoopConfig())
    assert res.crashed and res.extra["collision_source"] == "scenery"
    assert res.extra["guard_hit"]["depth_min"] < 0.3 and {"x", "alt", "frac_zero", "frac_below_radius"} <= set(res.extra["guard_hit"])
    _, ok = fly(scenario(Obstacle("box_1m_grey", 15.0, 0.0, 2.0)), TrueDepthStack(), Never())
    assert not ok.crashed and ok.extra["collision_source"] is None


def test_airsim_configs_parse_and_expand(monkeypatch, tmp_path):
    import eval.sweep as sw
    from sim.airsim_backend import pick_start
    from sim.scenario import ScenarioConfig, generate

    cfg = yaml.safe_load(open("configs/sweeps/airsim_smoke.yaml"))  # the 20 m smoke test that already ran: one start pose
    scs = [generate(s, "smoke", ScenarioConfig(**cfg["scenario"])) for s in range(1, 41)]
    assert all(len(s.obstacles) == 1 and s.obstacles[0].along_m <= 12.0 and s.goal_m == 20.0 for s in scs)
    for name in ("airsim_starts_smoke", "airsim_true_depth", "airsim_bench"):  # 40 m flights from several start poses (D24)
        cfg = yaml.safe_load(open(f"configs/sweeps/{name}.yaml"))
        starts = cfg["backend_kw"]["starts"]
        assert len(starts) == 5 and len({(x, y, yaw) for x, y, yaw, _ in starts}) == 5
        scs = [generate(s, "val", ScenarioConfig(**cfg["scenario"])) for s in range(1, 41)]
        assert all(len(s.obstacles) == 1 and 15.0 <= s.obstacles[0].along_m <= 25.0 and s.goal_m == 40.0 for s in scs)  # room for a 2 s delay at 5 m/s
        assert {pick_start(starts, s.seed, s.goal_m)[0] for s in scs} == set(range(5))  # every pose is used, and every flight fits
    assert yaml.safe_load(open("configs/sweeps/airsim_starts_smoke.yaml"))["seeds"] == list(range(1, 11))  # two flights per pose
    seen = {}
    monkeypatch.setenv("PP_SIM_CONFIGS", "/scratch/x/sim_config")
    monkeypatch.setattr(sw, "make_sim", lambda name, **kw: seen.update(kw) or (_ for _ in ()).throw(SystemExit("stop")))
    cfg = yaml.safe_load(open("configs/sweeps/airsim_smoke.yaml"))
    cfg["out"] = str(tmp_path / "o")
    with pytest.raises(SystemExit):
        sw.run(cfg)
    assert seen["configs"] == "/scratch/x/sim_config" and seen["start"] == [150.0, -150.0, 0.0]


# ---- frozen seeds, start poses, frozen scenario settings (D24) --------------------------------


def test_frozen_seed_file_is_intact_and_disjoint():
    lists = load_seed_lists("configs/seeds.yaml")  # raises if the checksum fails or a seed is in two splits
    assert {k: len(v) for k, v in lists.items()} == {"train": 1000, "val": 500, "test": 500}
    assert lists == seed_lists()  # the defaults regenerate the frozen file exactly


def test_start_pose_follows_the_seed_and_must_fit_the_flight():
    from sim.airsim_backend import AirSimBackend, pick_start

    starts = [[150.0, 150.0, 0.0, 145.0], [-150.0, 50.0, 180.0, 140.0], [50.0, -150.0, 90.0, 30.0]]
    assert 101_002 % 3 == 1 and pick_start(starts, 101_002, 40.0) == (1, (-150.0, 50.0, math.pi))
    assert pick_start(starts, 101_002, 40.0) == pick_start(starts, 101_002, 40.0)  # same seed, same pose: every policy flies it from one spot
    assert {pick_start(starts, s, 20.0)[0] for s in range(101_000, 101_006)} == {0, 1, 2}
    with pytest.raises(ValueError):
        pick_start(starts, 2, 40.0)  # pose 2 has 30 m clear: a 40 m flight (+5 m margin) does not fit
    sim = AirSimBackend(configs="unused", starts=starts)
    with pytest.raises(ValueError):  # refused before the simulator is touched
        sim.reset(scenario(seed=2, goal=40.0))
    assert sim._world is None


def test_scenario_settings_freeze_once_and_detect_edits(tmp_path, capsys):
    from sim.scenario import ScenarioConfig, load_scenario_config, write_scenario_config
    from sim.scenario import main as scenario_main

    path = tmp_path / "scenario.yaml"
    starts = [[150.0, 150.0, 0.0, 145.0]]
    write_scenario_config(path, {"goal_m": 40.0, "n_obstacles": [1, 1]}, starts)
    frozen = load_scenario_config(path)
    assert frozen["starts"] == starts and frozen["scenario"]["goal_m"] == 40.0
    assert frozen["scenario"]["speed_mps"] == 5.0  # defaults are written out too, so a later code default cannot change them
    assert generate(7, "val", ScenarioConfig(**frozen["scenario"])).to_json() == generate(7, "val", ScenarioConfig(goal_m=40.0, n_obstacles=(1, 1))).to_json()
    with pytest.raises(FileExistsError):
        write_scenario_config(path, {"goal_m": 50.0})
    d = yaml.safe_load(path.read_text())
    d["scenario"]["goal_m"] = 60.0  # tamper
    path.write_text(yaml.safe_dump(d))
    with pytest.raises(ValueError):
        load_scenario_config(path)
    scenario_main(["--freeze-scenario", "configs/sweeps/airsim_true_depth.yaml", "--to", str(tmp_path / "cli.yaml")])
    assert "checksum" in capsys.readouterr().out and len(load_scenario_config(tmp_path / "cli.yaml")["starts"]) == 5


def _sweep_cfg(tmp_path, **over):
    cfg = {"backend": "raycast", "backend_kw": {"width": 120, "height": 68}, "stack": "true-depth", "judge": "none", "delay_s": 1.0,
           "split": "val", "seeds": [101_000, 101_001], "policies": [{"name": "never", "policy": "never"}], "out": str(tmp_path / "out")}
    return {**cfg, **over}


def test_test_seeds_fly_only_from_frozen_scenario_settings(tmp_path):
    import eval.sweep as sw
    from sim.scenario import write_seed_lists as wsl
    from sim.scenario import write_scenario_config

    seeds = tmp_path / "seeds.yaml"
    wsl(seeds, seed_lists(4, 2, 2))
    base = {"split": "test", "seeds": {"file": str(seeds), "n": 2}}
    with pytest.raises(SystemExit, match="unfrozen scenario"):  # inline settings, even with the final-run flag
        sw.run(_sweep_cfg(tmp_path, **base, final_test_run=True, scenario={"goal_m": 30.0}))
    frozen = tmp_path / "scenario.yaml"
    write_scenario_config(frozen, {"goal_m": 30.0})
    with pytest.raises(SystemExit, match="final_test_run"):  # frozen settings, but not declared as the final run
        sw.run(_sweep_cfg(tmp_path, **base, scenario={"file": str(frozen)}))
    assert not (tmp_path / "out").exists()  # a refused run leaves nothing behind
    val = sw.run(_sweep_cfg(tmp_path, scenario={"file": str(frozen)}))  # val seeds may use the frozen file too
    assert len(val) == 2 and all(r.distance_m >= 30.0 for r in val)


def test_sweep_overrides():
    from eval.sweep import apply_overrides

    cfg = apply_overrides({"stack": "da2", "judge_kw": {"model_id": "a", "tau": 0.0}},
                          ["stack=clear", "judge_kw.model_id=Qwen/Qwen3-VL-8B-Instruct@4bit", "judge_kw.tau=-3.5", "backend_kw.width=120", "out=results/x"])
    assert cfg == {"stack": "clear", "judge_kw": {"model_id": "Qwen/Qwen3-VL-8B-Instruct@4bit", "tau": -3.5}, "backend_kw": {"width": 120}, "out": "results/x"}
    with pytest.raises(SystemExit):
        apply_overrides({}, ["no-equals-sign"])
    with pytest.raises(SystemExit):
        apply_overrides({"stack": "da2"}, ["stack.inner=1"])


def test_vlm_only_system_flies_on_the_judge_alone():
    from perception.stand_ins import ClearStack

    sc = scenario(Obstacle("box_1m_grey", 15.0, 0.0, 2.0))
    _, floor = fly(sc, ClearStack(), Never())  # no depth model and no VLM: flies straight into the box
    assert floor.crashed and floor.extra["stack"] == "none (no depth model)"
    perfect = SurrogateJudge(default_tpr=1.0, default_fpr=0.0)
    recs, vlm_only = fly(sc, ClearStack(), Always(cooldown_s=0.5), perfect, t_max=300.0)
    assert not vlm_only.crashed and vlm_only.reached_goal and vlm_only.n_calls > 0
    from perception.depth_utils import MAX_RANGE_M

    assert all(r.corridor_min_est == pytest.approx(MAX_RANGE_M) for r in recs)  # the "depth" never reports anything: every avoid came from the judge
    assert vlm_only.extra["start_index"] is None  # the raycast backend has no start poses


def test_resolve_seeds_skip_gives_a_disjoint_slice_of_the_frozen_list(tmp_path):
    from eval.sweep import resolve_seeds

    path = tmp_path / "seeds.yaml"
    lists = seed_lists(10, 200, 5)
    write_seed_lists(path, lists)
    tune = resolve_seeds({"split": "val", "seeds": {"file": str(path), "n": 60}})
    score = resolve_seeds({"split": "val", "seeds": {"file": str(path), "skip": 60, "n": 100}})
    assert tune == lists["val"][:60] and score == lists["val"][60:160]
    assert not set(tune) & set(score)
    assert resolve_seeds({"split": "val", "seeds": {"file": str(path), "skip": 190}}) == lists["val"][190:]
    with pytest.raises(SystemExit):
        resolve_seeds({"split": "val", "seeds": {"file": str(path), "skip": -1}})


# ---- cone courses in the flight loop (plan v7 chunk 4) ------------------------------------------


def cone(*obstacles: Obstacle, goal=(50.0, 0.0), area=15.0, alt=2.0, seed=1) -> Scenario:
    return Scenario(seed=seed, split="val", alt_m=alt, speed_mps=5.0, goal_m=math.hypot(*goal), condition="clear",
                    obstacles=list(obstacles), goal_xy=goal, land_alt_m=1.0, area_half_width_m=area)


def fly_cone(sc, stack, policy=None, judge=None, controller=None, **loop):
    return run_episode(sc, RaycastSim(CAM), stack, controller or RuleController(avoid_mode="choose"), policy or Never(), judge,
                       LoopConfig(**{"t_max": 120.0, **loop}))[1]


def test_cone_course_flies_to_the_goal_and_lands():
    for alt in (1.83, 1.9, 2.07, 2.2):  # regression: landing used to stop at 1.2 m for most start heights (smoke run 2026-11-25)
        assert fly_cone(cone(alt=alt), TrueDepthStack()).extra["outcome"] == "landed"
    r = fly_cone(cone(goal=(45.0, 8.0)), TrueDepthStack())
    assert r.reached_goal and not r.crashed and r.extra["outcome"] == "landed"
    assert r.extra["goal_dist_m"] <= 2.0 and r.extra["end_alt_m"] <= 1.1
    assert [p for _, p in r.extra["phases"]][-1] == "landing"


def test_cone_course_goes_around_a_box_with_true_depth_and_comes_back_down():
    r = fly_cone(cone(Obstacle("box_1m_grey", 20.0, 0.0, 2.0)), TrueDepthStack())
    assert r.extra["outcome"] == "landed", r.extra
    names = [p for _, p in r.extra["phases"]]
    assert "avoid" in names and names[-1] == "landing"


def test_avoidance_steps_log_the_free_distances_the_choice_used():
    r = fly_cone(cone(Obstacle("box_1m_grey", 20.0, 0.0, 2.0)), TrueDepthStack())
    rows = r.extra["avoid_bands"]
    assert rows and all(len(x) == 6 for x in rows)  # t, left, right, over, side, action
    t, left, right, over, side, action = rows[0]
    assert left > 0 and right > 0 and over > 0 and action in ("yaw_left", "yaw_right", "climb")
    assert fly_cone(cone(), TrueDepthStack()).extra["avoid_bands"] == []  # nothing to avoid


def test_every_crash_records_phase_action_and_the_obstacle_hit():
    sc = cone(Obstacle("box_1m_grey", 30.0, 9.0, 2.0), Obstacle("wire_3cm_axis_z", 20.0, 0.0, 2.0))
    r = fly_cone(sc, BlindStack())
    c = r.extra["crash"]
    assert r.extra["outcome"] == "crashed" and c["kind"] == "obstacle"
    assert c["obstacle"] == 1 and c["family"] == "wire_3cm_axis_z" and c["obstacle_off_line_m"] < 0.5
    assert c["phase"] == "cruise" and c["action"] == "continue" and 18.0 < c["x"] < 21.0
    assert "crash" not in fly_cone(cone(), TrueDepthStack()).extra  # landed: no crash record


def test_vlm_direction_steers_past_a_wire_depth_cannot_see():
    sc = cone(Obstacle("wire_3cm_axis_z", 20.0, 0.0, 2.0))
    assert fly_cone(sc, BlindStack()).extra["outcome"] == "crashed"  # never asks: flies into it
    judge = SurrogateJudge(default_tpr=1.0, default_fpr=0.0, seed=0, p_dir=1.0)
    r = fly_cone(sc, BlindStack(), policy=Oracle(), judge=judge)
    assert r.extra["outcome"] == "landed", r.extra
    assert "over" in r.extra["vlm_directions"]  # a 6 m horizontal wire: going over is the shortest way past


def test_lost_left_area_and_timeout_outcomes():
    class Hover:
        phase = "cruise"

        def reset(self):
            pass

        def act(self, depth, state):
            return Action.STOP

    assert fly_cone(cone(), TrueDepthStack(), controller=Hover()).extra["outcome"] == "lost"  # no progress for 30 s

    class Straight(Hover):
        def act(self, depth, state):
            return Action.CONTINUE

    assert fly_cone(cone(goal=(50.0, 10.0), area=2.0), TrueDepthStack(), controller=Straight()).extra["outcome"] == "left_area"
    r = fly_cone(cone(), TrueDepthStack(), t_max=3.0)
    assert r.extra["outcome"] == "timeout" and not r.reached_goal


def test_line_courses_report_no_outcome_field_and_fly_as_before():
    r = run_episode(scenario(Obstacle("box_1m_grey", 20.0, 0.0, 2.0)), RaycastSim(CAM), TrueDepthStack(), RuleController(), Never())[1]
    assert r.reached_goal and "outcome" not in r.extra


# ---- map scenery in the truth of the surrogate VLM and the oracle (sim.raycast.scenery_truth) ----------


class MapTreeSim(RaycastSim):
    """Raycast backend with a 'map' tree that is rendered but is not a placed obstacle (not in scenario.obbs())."""

    def __init__(self, tree: OBB):
        super().__init__(CAM)
        self.tree = tree

    def reset(self, scenario):
        super().reset(scenario)
        self.boxes = self.boxes + [self.tree]


class ClearStack:
    """A fast stack that sees nothing at all (every corridor reads far): misses map scenery like a blind depth model."""

    name = "clear (stand-in)"

    def perceive_obs(self, obs, state=None, sim=None):
        d = np.full(obs.depth.shape, 60.0, np.float32)
        return DepthEstimate(depth=d, features=corridor_features(d))


TREE = OBB(center=(20.0, 0.6, -3.0), half=(0.6, 0.6, 3.0), family="building_block")  # map stand-in, 6 m tall, just right of the line


def test_scenery_truth_finds_a_map_tree_but_not_ground_or_placed_obstacles():
    from sim.raycast import scenery_truth

    sim = MapTreeSim(TREE)
    sim.reset(cone())
    st = KinematicState(t=0.0, x=8.0, y=0.0, z=-2.0, yaw=0.0, speed=5.0)
    d, side = scenery_truth(sim.observe(st).depth, st, [], sim.cam)
    assert 10.5 < d < 12.0 and side == "left"  # the tree's mass is right of the heading: pass it on the left
    assert scenery_truth(sim.observe(st).depth, st, [TREE], sim.cam)[1] is None  # a placed obstacle is not scenery
    open_sim = RaycastSim(CAM)
    open_sim.reset(cone())
    for alt in (1.0, 2.0, 6.0):  # flat ground below the drone is not in its path
        low = KinematicState(t=0.0, x=0.0, y=0.0, z=-alt, yaw=0.0, speed=5.0)
        assert scenery_truth(open_sim.observe(low).depth, low, [], open_sim.cam) == (float("inf"), None)


def test_oracle_and_surrogate_avoid_a_map_tree_the_fast_stack_misses():
    def fly(**loop):
        judge = SurrogateJudge(default_tpr=1.0, default_fpr=0.0, seed=0, p_dir=1.0)
        return run_episode(cone(), MapTreeSim(TREE), ClearStack(), RuleController(avoid_mode="choose"), Oracle(), judge,
                           LoopConfig(**{"t_max": 120.0, **loop}))[1]

    blind = fly(scenery_tube_m=None)  # before: the truth knew placed obstacles only, so nobody asked
    assert blind.extra["outcome"] == "crashed" and blind.n_calls == 0
    assert blind.extra["crash"]["kind"] == "guard_scenery"
    r = fly()
    assert r.extra["outcome"] == "landed" and r.n_calls >= 1 and r.extra["scenery_truth_frames"] > 0


def test_drone_height_cap_follows_the_controller():
    from eval.run_episode import default_params
    from sim.scenario import ScenarioConfig, generate

    sc = generate(101063, "val", ScenarioConfig(layout="cone"))
    assert default_params(sc, RuleController()).alt_max == 25.0  # the default is unchanged
    p = default_params(sc, RuleController(alt_max=50.0))
    assert p.alt_max == 50.0 and p.alt_min == sc.land_alt_m


# ---- evasion of a VLM "yes" that depth does not show (plan v8, step 2) ----------------------------------------


def test_evasion_climbs_in_place_until_two_clear_answers_then_bypasses_at_height():
    clear = banded(40.0, left=40.0, right=40.0, up=90.0)
    c = RuleController(avoid_mode="choose")
    c.set_mission((50.0, 0.0), cruise_alt=2.0, land_alt=1.0)
    assert c.act(clear, pos()) == Action.CONTINUE and not c.evading
    c.alert(pos())
    assert c.evading and c.act(clear, pos(alt=2.0)) == Action.CLIMB and c.phase == "evade"
    c.alert(pos(alt=2.4))  # another yes: still climbing, nothing restarts
    assert c.evade_start_alt == 2.0 and c.act(clear, pos(alt=2.4)) == Action.CLIMB
    c.clear()
    assert c.evading and c.act(clear, pos(alt=3.0)) == Action.CLIMB  # one clear answer is not enough
    c.clear()
    assert c.act(clear, pos(alt=3.2)) == Action.CONTINUE  # two in a row: ended, the bypass flies on over the object
    assert not c.evading and c.phase == "bypass" and c.bypass_over and c.bypass_from[3] == c.evade_bypass_m
    assert c.act(clear, pos(x=5.0, alt=3.2)) == Action.CONTINUE and c.phase == "bypass"  # holds the height: no descent in front of it


def test_evasion_turns_to_a_committed_side_only_when_the_climb_limit_is_reached():
    clear = banded(40.0, left=40.0, right=40.0, up=90.0)
    c = RuleController(avoid_mode="choose", evade_climb_max_m=3.0, evade_yaw_steps=5)
    c.set_mission((50.0, 20.0), cruise_alt=2.0, land_alt=1.0)  # the goal lies to the right
    c.alert(pos())
    assert c.evade_side == "right"
    assert c.act(clear, pos(alt=4.9)) == Action.CLIMB
    assert [c.act(clear, pos(alt=5.1)) for _ in range(5)] == [Action.YAW_RIGHT] * 5  # 3 m up and still yes: a wall. Turn, one side
    c.alert(pos(y=-5.0, alt=5.1))  # a new yes, the goal is on the other side now: the side stays
    assert c.evade_side == "right" and c.act(clear, pos(y=-5.0, alt=5.1)) == Action.CONTINUE


def _yes_forever_run(c, n=400, **kw):
    """A VLM that says yes on every step and a blind depth: the drone climbs, turns, sidesteps and must hand back to the goal."""
    clear = banded(40.0, left=40.0, right=40.0, up=90.0)
    alt, x, y, acts = 2.0, 0.0, 0.0, []
    for _ in range(n):
        c.alert(pos(x=x, y=y, alt=alt))
        a = c.act(clear, pos(x=x, y=y, alt=alt))
        acts.append((a, c.phase))
        if a == Action.CLIMB:
            alt += 0.2
        if a == Action.CONTINUE:
            x += 0.5
        if a == Action.YAW_RIGHT or a == Action.YAW_LEFT:
            pass
    return acts


def test_a_vlm_that_always_says_yes_cannot_keep_the_drone_evading():
    c = RuleController(avoid_mode="choose", evade_climb_max_m=1.0, evade_total_steps=40, evade_side_m=2.0, evade_cooldown_m=100.0)
    c.set_mission((50.0, 0.0), cruise_alt=2.0, land_alt=1.0)
    acts = _yes_forever_run(c, n=200)
    evade_steps = sum(1 for _, ph in acts if ph == "evade")
    assert not c.evading and evade_steps <= 40  # the budget ends it, and the cooldown keeps it from starting again
    assert c.evade_cooldown_from is not None


def test_the_sidestep_is_bounded_and_then_the_cooldown_ignores_a_standing_yes():
    clear = banded(40.0, left=40.0, right=40.0, up=90.0)
    c = RuleController(avoid_mode="choose", evade_climb_max_m=0.5, evade_yaw_steps=2, evade_side_m=3.0, evade_cooldown_m=10.0)
    c.set_mission((50.0, 0.0), cruise_alt=2.0, land_alt=1.0)
    c.alert(pos(alt=2.0))
    assert c.act(clear, pos(alt=2.0)) == Action.CLIMB
    turn = Action.YAW_LEFT if c.evade_side == "left" else Action.YAW_RIGHT
    assert [c.act(clear, pos(alt=3.0)) for _ in range(2)] == [turn] * 2
    assert c.act(clear, pos(x=0.0, alt=3.0)) == Action.CONTINUE  # the sideways leg starts here
    c.act(clear, pos(x=3.5, alt=3.0))  # 3.5 m on: done, the controller takes over (no CONTINUE-for-ever)
    assert not c.evading and c.phase != "evade"
    c.alert(pos(x=5.0, alt=3.0))
    assert not c.evading  # inside the cooldown a standing yes is ignored
    c.alert(pos(x=20.0, alt=3.0))
    assert c.evading  # far enough on: a new yes counts again


def test_the_evasion_ends_at_the_edge_of_the_mission_area():
    clear = banded(40.0, left=40.0, right=40.0, up=90.0)
    c = RuleController(avoid_mode="choose", fence_margin_m=3.0)
    c.set_mission((50.0, 0.0), cruise_alt=2.0, land_alt=1.0, area_half_width_m=15.0)
    c.alert(pos(y=12.5, alt=2.0))  # inside the fence margin
    assert c.act(clear, pos(y=12.5, alt=2.0)) in (Action.CONTINUE, Action.YAW_LEFT, Action.YAW_RIGHT) and not c.evading
    assert c.phase != "evade" and c.evade_cooldown_from is not None


def test_evasion_respects_the_height_cap_and_a_silent_policy_ends_it_by_timeout():
    clear = banded(40.0, left=40.0, right=40.0, up=90.0)
    c = RuleController(avoid_mode="choose", evade_timeout_steps=12)
    c.alert(pos(alt=24.5))
    assert c.act(clear, pos(alt=24.5)) == Action.YAW_RIGHT  # no room above: straight to the turn
    c.reset()
    c.alert(pos())
    acts = [c.act(clear, pos(alt=2.0 + 0.4 * i)) for i in range(14)]
    assert not c.evading and acts[0] == Action.CLIMB  # no new yes and no clear answers: the timeout ends it


def test_depth_showing_the_obstacle_hands_over_to_the_normal_avoidance():
    c = RuleController(avoid_mode="choose")
    c.alert(pos())
    assert c.act(banded(5.0, left=12.0, right=40.0, up=90.0), pos()) in (Action.YAW_RIGHT, Action.YAW_LEFT, Action.CLIMB)
    assert not c.evading and c.phase == "avoid"


def test_evasion_off_by_default_and_the_loop_config_checks_its_mode():
    with pytest.raises(ValueError):
        LoopConfig(vlm_response="steer")
    assert LoopConfig().vlm_response == "override"
    assert RuleController().config()["evade_climb_max_m"] == 6.0


def test_evasion_saves_a_blind_stack_from_a_wire_a_pole_and_glass_without_any_direction():
    perfect = SurrogateJudge(default_tpr=1.0, default_fpr=0.0)  # no p_dir: it says yes, never a side
    for family in ("wire_3cm_axis_z", "wire_1cm_axis_z", "pole_10cm_axis_z", "glass_pane_alpha20"):
        sc = cone(goal=(50.0, 0.0), alt=2.0)
        sc.obstacles.append(Obstacle(family, 15.0, 0.0, 2.0))
        never = fly_cone(sc, BlindStack())
        assert never.crashed, family
        res = fly_cone(sc, BlindStack(), Always(cooldown_s=0.0), perfect, controller=RuleController(avoid_mode="choose"),
                       vlm_response="sidestep", t_max=400.0)
        assert not res.crashed and res.extra["outcome"] == "landed", (family, res.extra.get("crash"), res.extra.get("outcome"))


def test_fence_guard_turns_an_outward_heading_back_at_the_edge_and_leaves_inward_headings_alone():
    clear = banded(40.0, left=40.0, right=40.0, up=90.0)
    c = RuleController(avoid_mode="choose")
    c.set_mission((50.0, 0.0), cruise_alt=2.0, land_alt=1.0, area_half_width_m=15.0)
    # 14 m to the right of the line (y > 0 is right), heading right (yaw +45 deg): outward, so turn back toward the goal (left)
    assert c.act(clear, pos(y=14.0, yaw=math.radians(45.0))) == Action.YAW_LEFT and c.phase == "fence"
    # same place, heading along the line: not outward, the normal controller flies on
    assert c.act(clear, pos(y=14.0, yaw=0.0)) != Action.YAW_LEFT or c.phase != "fence"
    # well inside the area: no guard
    assert c.act(clear, pos(y=5.0, yaw=math.radians(45.0))) != Action.YAW_LEFT or c.phase != "fence"


def test_always_yes_vlm_and_blind_depth_never_take_the_drone_out_of_the_area():
    always_yes = SurrogateJudge(default_tpr=1.0, default_fpr=1.0)
    sc = cone(goal=(50.0, 0.0), alt=2.0)
    sc.obstacles.append(Obstacle("wall_white_featureless", 15.0, 0.0, 2.0))
    res = fly_cone(sc, BlindStack(), Always(cooldown_s=0.0), always_yes, controller=RuleController(avoid_mode="choose"),
                   vlm_response="sidestep", t_max=200.0)
    assert res.extra["outcome"] != "left_area", res.extra.get("outcome")
