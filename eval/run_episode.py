"""The closed flight loop (PLAN_DETAILED.md §4.1, D10, D11; chunks M4.5 and M4.6).

Every step (``dt`` simulated seconds):

1. observe at the current pose; the fast stack estimates depth; temporal features are added;
2. the controller proposes an action from the estimate;
3. the policy decides whether to ask (the oracle and the surrogate judge get the truth from here);
4. if it asks: the simulator is paused for the call, then simulated time advances by the delay ``d``
   (never zero, D10) while the drone brakes and holds (D11 MVP); a "yes" makes the drone take the
   controller's avoidance action for ``vlm_avoid_s`` seconds;
5. the drone steps; our own swept collision check runs against the true geometry (glass included, D5);
6. one ``FrameRecord`` is logged.

Line courses end on a collision, at ``x >= goal_m``, or at ``t_max``. Cone courses (a goal point, plan v7) end with
``extra["outcome"]``: "landed" (within ``land_radius_m`` of the goal at the landing height), "crashed", "lost" (the
distance to the goal has not shrunk by ``progress_m`` for ``lost_s``), "left_area" (more than ``area_half_width_m``
from the start-to-goal line: the hard geofence) or "timeout" (``t_max``). On cone courses the controller gets the
mission, and a VLM "yes" passes the judge's direction (``last_direction``) to ``avoid_action``. Wall-clock time
never enters the dynamics, so a seed and a delay replay the same flight
exactly; the real VLM latency is only logged.

    python -m eval.run_episode --policy never --seed 101000 --stack true-depth
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from eval.metrics import EpisodeResult
from gate.features import IMAGE_KEYS, TemporalFeatures
from interfaces import Action, DepthEstimate, FrameRecord, RuntimeStep
from sim.drone import DroneParams, KinematicState, hold, step
from sim.geometry import best_side, hits_ground, segment_hits
from perception.depth_utils import MAX_RANGE_M
from control.rules import free_space
from sim.raycast import corridor_truth, scenery_truth
from sim.scenario import Scenario


MAX_AVOID_BANDS = 400  # avoidance steps logged per flight in extra['avoid_bands'] (0.1 s each)


@dataclass(frozen=True)
class ForkPoint:
    """Everything the loop carries from one step to the next, taken just before observing step ``k``.

    The simulator needs no snapshot: kinematic stepping sets the pose every step (D4), so
    ``observe(state)`` depends only on the scenario and the state. Controller and temporal
    features are deep copies, so a fork can never change the flight it came from.
    """

    k: int
    state: KinematicState
    controller: Any
    temporal: TemporalFeatures
    avoid_until: float
    distance: float


@dataclass
class LoopConfig:
    dt: float = 0.1
    t_max: float = 60.0
    delay_s: float = 1.0  # simulated seconds charged per VLM call (D10: never zero)
    vlm_avoid_s: float = 1.5  # how long a "yes" overrides a proposed CONTINUE
    # What a VLM "yes" does. "override" (the original): the controller avoids for ``vlm_avoid_s`` on the VLM's direction or its own band choice.
    # "sidestep" (plan v8): no direction is used. A yes that depth backs (the estimate is under d_safe) is left to the controller's normal
    # avoidance; a yes that depth does not back starts the evasion (``RuleController.alert``: climb in place, ask again), which two "no" answers end.
    vlm_response: str = "override"
    d_safe: float = 8.0  # for the L1 label handed to the oracle; match the controller
    d_clear: float = 30.0  # scoring the VLM calls: nothing closer than this is "clear" (D23); 8 to 30 m is left out
    d_judge: float = 15.0  # the surrogate's truth: a surface in the window closer than this "crosses" it
    # Map scenery in the truth of the surrogate VLM and the oracle (sim.raycast.scenery_truth): a true-depth surface
    # within this many metres of the drone's height and heading line. None: placed obstacles only (before 2026-11-25).
    scenery_tube_m: float | None = 0.8
    save_dir: str | None = None  # write rgb/depth/seg per frame here (large); off by default
    save_every: int = 1  # with save_dir: keep every n-th frame only (data flights for the VLM probe: 5 = one picture per 0.5 s)
    # cone courses only (plan v7, DS 2026-11-25); all tunable on val
    land_radius_m: float = 2.0  # "landed" needs the drone this close to the goal, horizontally
    land_tol_m: float = 0.1  # and at most this far above the landing height
    lost_s: float = 30.0  # "lost": no progress towards the goal for this long
    progress_m: float = 0.5  # what counts as progress

    def __post_init__(self) -> None:
        if self.delay_s <= 0:
            raise ValueError("delay_s must be > 0 (PLAN_DETAILED.md D10)")
        if self.vlm_response not in ("override", "sidestep"):
            raise ValueError(f"vlm_response must be 'override' or 'sidestep', got {self.vlm_response!r}")


def default_params(scenario: Scenario, controller: Any) -> DroneParams:
    """The drone's physics for this flight. Its height cap follows the controller's ``alt_max`` (config ``controller: {alt_max: 50}``):
    a controller that plans a 50 m climb on a drone that stops at 25 m would climb forever."""
    alt_max = float(getattr(controller, "alt_max", DroneParams.alt_max))
    if scenario.goal_xy is not None:
        return DroneParams(v=scenario.speed_mps, alt_min=scenario.land_alt_m, alt_max=alt_max)
    return DroneParams(v=scenario.speed_mps, alt_max=alt_max)


def call_truth(truth_min: float, family: str | None, d_safe: float, d_clear: float) -> str | None:
    """Offline class of one VLM call (D23): "placed" or "scenery" closer than ``d_safe``, "clear" when nothing is within
    ``d_clear``, None in between. Ground truth, so for scoring only: the loop never reads it."""
    if truth_min < d_safe and family not in (None, "ground"):
        return "scenery" if family == "scenery" else "placed"
    return "clear" if truth_min >= d_clear else None


def run_episode(scenario: Scenario, sim: Any, stack: Any, controller: Any, policy: Any, judge: Any = None,
                cfg: LoopConfig | None = None, params: DroneParams | None = None, judge_name: str = "none",
                snapshots: list[ForkPoint] | None = None, resume: ForkPoint | None = None,
                t_stop: float | None = None) -> tuple[list[FrameRecord], EpisodeResult]:
    """Fly one episode. For replay labels (M6.1): ``snapshots`` collects a ``ForkPoint`` per step;
    ``resume`` continues from one (the simulator must already hold this scenario; ``controller`` is
    replaced by the snapshot's copy) and ``t_stop`` ends the flight at that simulated time."""
    from vlm.overlay import render

    cone = scenario.goal_xy is not None
    cfg = cfg or LoopConfig()
    params = params or default_params(scenario, controller)
    if resume is None:
        sim.reset(scenario)
        controller.reset()
        if cone and hasattr(controller, "set_mission"):
            controller.set_mission(scenario.goal_xy, cruise_alt=scenario.alt_m, land_alt=scenario.land_alt_m,
                                   area_half_width_m=scenario.area_half_width_m)
    else:
        controller = copy.deepcopy(resume.controller)
    if hasattr(policy, "reset"):
        policy.reset(scenario.seed)
    temporal = TemporalFeatures() if resume is None else copy.deepcopy(resume.temporal)
    # RGB image features only when the gate was trained on them (gate.image_features; RGB only, rule 5).
    use_image = bool(set(IMAGE_KEYS) & set(getattr(policy, "feature_names", ()) or ()))
    if use_image:
        from gate.image_features import image_features
    image_ms: list[float] = []
    boxes = scenario.obbs()
    state = KinematicState(t=0.0, x=0.0, y=0.0, z=-scenario.alt_m, yaw=0.0, speed=0.0) if resume is None else resume.state
    history: list[RuntimeStep] = []
    records: list[FrameRecord] = []
    latencies: list[float] = []
    avoid_until, distance, delay_total, crashed, reached = -1.0, 0.0, 0.0, False, False
    avoid_dir: str | None = None  # the VLM's direction for the current avoidance window (not carried by a fork)
    if resume is not None:
        avoid_until, distance = resume.avoid_until, resume.distance
    episode_id = f"{scenario.split}-{scenario.seed}-{getattr(policy, 'name', 'policy')}"
    save = Path(cfg.save_dir) / episode_id if cfg.save_dir else None
    if save:
        save.mkdir(parents=True, exist_ok=True)

    from perception.depth_utils import corridor_window

    collision_source, guard_hit = None, None
    outcome: str | None = None
    directions: list[str | None] = []
    calls: list[dict[str, Any]] = []  # one row per VLM call, with an offline truth class (scoring only, never read by the loop)
    phases: list[tuple[float, str]] = []  # (t, phase) whenever the controller's phase changes
    avoid_bands: list[list[Any]] = []  # [t, left, right, over, side, action] on every avoidance step: what the controller saw (capped)

    def goal_dist(st: KinematicState) -> float:
        return math.hypot(scenario.goal_xy[0] - st.x, scenario.goal_xy[1] - st.y)

    def cross_track(st: KinematicState) -> float:
        gx, gy = scenario.goal_xy
        return (gx * st.y - gy * st.x) / math.hypot(gx, gy)

    def crash_record(st: KinematicState, act: Action | None, kind: str, box_i: int | None) -> dict[str, Any]:
        """Where and how a flight ended in a crash: every crash gets one (analysis only, never read by the drone)."""
        rec: dict[str, Any] = {"kind": kind, "x": round(st.x, 2), "y": round(st.y, 2), "alt": round(st.alt, 2), "yaw": round(st.yaw, 3),
                               "t": round(st.t, 2), "phase": getattr(controller, "phase", None),
                               "action": act.value if act is not None else None, "start_index": getattr(sim, "start_index", None)}
        if box_i is not None:
            b = boxes[box_i]
            rec.update({"obstacle": box_i, "family": scenario.obstacles[box_i].family,
                        "obstacle_off_line_m": round(abs(cross_track_xy(b.center[0], b.center[1])), 2) if cone else None})
        return rec

    def cross_track_xy(x: float, y: float) -> float:
        gx, gy = scenario.goal_xy
        return (gx * y - gy * x) / math.hypot(gx, gy)

    crash: dict[str, Any] | None = None
    scenery_frames = 0  # frames whose truth was map scenery within d_judge
    last_action: Action | None = None
    best_dist, best_t = (goal_dist(state), state.t) if cone else (0.0, 0.0)
    for k in range(0 if resume is None else resume.k, int(math.ceil(cfg.t_max / cfg.dt)) + 1):
        if snapshots is not None:
            snapshots.append(ForkPoint(k, state, copy.deepcopy(controller), copy.deepcopy(temporal), avoid_until, distance))
        obs = sim.observe(state)
        # Scenery guard: our swept check only knows the spawned obstacles. On Project AirSim the map has its own
        # buildings, so a ground-truth surface closer than the drone's radius straight ahead also counts as a crash
        # (one step late at most; the camera sits about 0.3 m ahead of the drone origin, job 33776).
        rows, cols = corridor_window(obs.depth.shape)
        window = obs.depth[rows, cols]
        if float(np.min(window)) < params.radius:
            # The camera sits about 0.3 m ahead, so this often fires on a placed obstacle just before our swept check
            # does: call it an obstacle when the known geometry has one that close, scenery otherwise.
            gt_near, fam_near = corridor_truth(state, boxes, sim.cam)
            crashed = True
            collision_source = "obstacle" if gt_near < 1.0 and fam_near not in (None, "ground") else "scenery"
            r_, c_ = np.unravel_index(int(np.argmin(window)), window.shape)
            seg_px = np.asarray(obs.seg)[rows, cols][r_, c_][:3]
            # what triggered it: a real surface fills part of the window, a bad capture shows up as zeros (analysis only)
            guard_hit = {"x": round(state.x, 2), "y": round(state.y, 2), "alt": round(state.alt, 2), "yaw": round(state.yaw, 3),
                         "phase": getattr(controller, "phase", None), "start_index": getattr(sim, "start_index", None),
                         "seg_rgb": [int(v) for v in seg_px], "corridor_gt": round(gt_near, 2), "family": fam_near,
                         "depth_min": round(float(np.min(window)), 3), "depth_median": round(float(np.median(window)), 2),
                         "frac_below_radius": round(float(np.mean(window < params.radius)), 4),
                         "frac_zero": round(float(np.mean(window <= 0.0)), 4), "image_depth_median": round(float(np.median(obs.depth)), 2)}
            near_i = None
            if collision_source == "obstacle":  # the placed obstacle of that family nearest to the drone
                cand = [i for i, b in enumerate(boxes) if b.family == fam_near]
                near_i = min(cand, key=lambda i: float(np.linalg.norm(np.asarray(boxes[i].center) - state.position))) if cand else None
            crash = crash_record(state, last_action, f"guard_{collision_source}", near_i)
            break
        est = stack.perceive_obs(obs, state, sim)
        feats = temporal.update(est.features, state.position)  # change per metre flown: same scale as the bank
        if use_image:
            t0 = time.perf_counter()
            feats = {**feats, **image_features(obs.rgb)}
            image_ms.append((time.perf_counter() - t0) * 1e3)
        est = DepthEstimate(depth=est.depth, features=feats)
        public = state.public()
        proposed = controller.act(est, public)
        ph = getattr(controller, "phase", None)
        if ph is not None and (not phases or phases[-1][1] != ph):
            phases.append((round(state.t, 2), ph))
        if ph == "avoid" and len(avoid_bands) < MAX_AVOID_BANDS:  # the free distances the side choice is based on (job B2: why no climb?)
            fs = free_space(est.depth)
            avoid_bands.append([round(state.t, 2), round(fs["left"], 1), round(fs["right"], 1), round(fs["over"], 1),
                                getattr(controller, "side", None), proposed.value if hasattr(proposed, "value") else str(proposed)])

        gt_min, family = corridor_truth(state, boxes, sim.cam)
        truth_min, truth_family, scenery_side = gt_min, family, None
        scn_min = None
        if cfg.scenery_tube_m is not None:  # a map tree or building nearer than any placed obstacle in the path
            d_scn, side_scn = scenery_truth(obs.depth, state, boxes, sim.cam, tube_m=cfg.scenery_tube_m)
            scn_min = round(float(min(d_scn, MAX_RANGE_M)), 2)  # logged for offline labels (VLM probe); capped like the depth range
            if d_scn < gt_min or (family in (None, "ground") and d_scn < MAX_RANGE_M):
                truth_min, truth_family, scenery_side = d_scn, "scenery", side_scn
                scenery_frames += d_scn < cfg.d_judge
        est_min = float(feats["corridor_min"])
        if hasattr(policy, "set_truth"):
            policy.set_truth(truth_min < cfg.d_safe <= est_min)
        asked = bool(policy.should_ask(state.t, feats, public, history))
        answer = conf = lat = None
        charged = 0.0
        if asked:
            if judge is None:
                raise ValueError(f"policy {getattr(policy, 'name', policy)} asked but no judge was given")
            overlay, crop = render(obs.rgb)
            if hasattr(judge, "set_truth"):
                crosses = truth_min < cfg.d_judge and truth_family not in (None, "ground")
                if getattr(judge, "p_dir", None) is not None:  # the surrogate names a side: give it the true shortest way past
                    if truth_family == "scenery":
                        side = scenery_side if crosses else None
                    else:
                        near = [b for b in boxes if b.family == truth_family]
                        box = min(near, key=lambda b: float(np.linalg.norm(np.asarray(b.center[:2]) - state.position[:2]))) if near else None
                        side = best_side(state.position, state.yaw, box, params.radius) if box and crosses else None
                    judge.set_truth(truth_family, crosses, side=side)
                else:
                    judge.set_truth(truth_family, crosses)
            a = judge.ask(obs.rgb, overlay, crop)
            answer, conf, lat = bool(a.answer), float(a.confidence), float(a.latency_s)
            latencies.append(lat)
            state = hold(state, cfg.delay_s)  # paused-time delay: brake and hold for d (D10, D11)
            charged = cfg.delay_s
            delay_total += charged
            direction = getattr(judge, "last_direction", None) if answer else None
            directions.append(direction)
            calls.append({"t": round(state.t, 2), "answer": answer, "score": round(conf, 4), "gate": getattr(policy, "last_score", None),
                          "est_min": round(est_min, 2), "truth_min": round(float(truth_min), 2),
                          "truth": call_truth(truth_min, truth_family, cfg.d_safe, cfg.d_clear),
                          "phase": getattr(controller, "phase", None), "alt": round(state.alt, 2)})
            if cfg.vlm_response == "sidestep":
                if answer and est_min >= controller.d_safe:  # depth does not show it: the VLM is the only sensor that does
                    controller.alert(state.public())
                    proposed = controller.act(est, state.public())
                elif answer:
                    pass  # depth backs the yes: the controller's own avoidance (act above) already handles it
                else:
                    controller.clear()
            elif answer:
                avoid_until, avoid_dir = state.t + cfg.vlm_avoid_s, direction
        action = proposed
        if cfg.vlm_response == "override" and state.t < avoid_until and proposed == Action.CONTINUE:
            action = controller.avoid_action(est, state.public(), avoid_dir) if avoid_dir else controller.avoid_action(est, state.public())

        new = step(state, action, cfg.dt, params)
        p0, p1 = state.position, new.position
        hit_ground = hits_ground(p0, p1, params.radius)
        hit_i = next((i for i, b in enumerate(boxes) if segment_hits(p0, p1, b, params.radius)), None)
        collision = hit_ground or hit_i is not None
        if collision:
            collision_source = "ground" if hit_ground else "obstacle"
            crash = crash_record(state, action, collision_source, None if hit_ground else hit_i)
        last_action = action
        distance += float(np.linalg.norm(p1 - p0))
        paths = _save(save, k, obs, est) if save and k % max(1, cfg.save_every) == 0 else ("", "", "", "")
        records.append(FrameRecord(
            episode_id=episode_id, split=scenario.split, seed=scenario.seed, t=round(state.t, 4),
            rgb_path=paths[0], depth_gt_path=paths[1], seg_gt_path=paths[2], depth_est_path=paths[3],
            pose=(state.x, state.y, state.z, state.yaw), velocity=(state.speed * math.cos(state.yaw), state.speed * math.sin(state.yaw), 0.0),
            corridor_min_gt=gt_min, corridor_min_est=est_min, action_proposed=proposed, asked=asked, vlm_answer=answer,
            vlm_conf=conf, vlm_latency_real_s=lat, delay_charged_s=charged, action_taken=action, collision=collision,
            obstacle_family=family if family not in (None, "ground") else None, gate_score=getattr(policy, "last_score", None),
            scenery_min_gt=scn_min))
        history.append(RuntimeStep(t=state.t, asked=asked, vlm_answer=answer, action_taken=action, gate_score=getattr(policy, "last_score", None)))
        state = new
        if collision:
            crashed, outcome = True, "crashed"
            break
        if cone:
            d_goal = goal_dist(state)
            if d_goal <= cfg.land_radius_m and state.alt <= scenario.land_alt_m + cfg.land_tol_m:
                reached, outcome = True, "landed"
                break
            if abs(cross_track(state)) > scenario.area_half_width_m:
                outcome = "left_area"
                break
            if d_goal < best_dist - cfg.progress_m or d_goal <= cfg.land_radius_m:
                best_dist, best_t = min(best_dist, d_goal), state.t  # descending over the goal also counts as progress
            elif state.t - best_t >= cfg.lost_s:
                outcome = "lost"
                break
        elif state.x >= scenario.goal_m:
            reached = True
            break
        if state.t >= cfg.t_max or (t_stop is not None and state.t >= t_stop - 1e-6):
            outcome = outcome or "timeout"
            break

    result = EpisodeResult(
        episode_id=episode_id, seed=scenario.seed, split=scenario.split, policy=getattr(policy, "name", "policy"), judge=judge_name,
        crashed=crashed, reached_goal=reached, distance_m=round(distance, 3), mission_time_s=round(state.t, 3), n_frames=len(records),
        n_calls=sum(r.asked for r in records), delay_charged_s=round(delay_total, 3), latencies_s=latencies,
        obstacle_family=",".join(o.family for o in scenario.obstacles) or None,
        extra={"stack": getattr(stack, "name", type(stack).__name__), "backend": getattr(sim, "name", type(sim).__name__),
               "delay_s": cfg.delay_s, "condition": scenario.condition, "collision_source": collision_source,
               "start_index": getattr(sim, "start_index", None), "scenery_truth_frames": scenery_frames})
    if guard_hit is not None:
        result.extra["guard_hit"] = guard_hit
        outcome = "crashed"
    if crash is not None:
        result.extra["crash"] = crash
    if cone:
        result.extra.update({"outcome": outcome or ("crashed" if crashed else "timeout"), "goal_xy": list(scenario.goal_xy),
                             "goal_dist_m": round(goal_dist(state), 2), "end_alt_m": round(state.alt, 2),
                             "n_obstacles": len(scenario.obstacles), "phases": phases, "vlm_directions": directions,
                             "vlm_calls": calls,
                             "avoid_bands": avoid_bands})
    if image_ms:  # wall-clock cost of the gate's image features per frame (PLAN_DETAILED.md §4.2 budget: under 1 ms)
        result.extra["image_features_ms"] = {"median": round(float(np.median(image_ms)), 3), "p95": round(float(np.percentile(image_ms, 95)), 3)}
    return records, result


def _save(d: Path, k: int, obs: Any, est: DepthEstimate) -> tuple[str, str, str, str]:
    from PIL import Image

    stem = d / f"{k:04d}"
    Image.fromarray(obs.rgb).save(f"{stem}_rgb.png")
    Image.fromarray(obs.seg).save(f"{stem}_seg.png")
    np.save(f"{stem}_depth.npy", obs.depth.astype(np.float16))
    np.save(f"{stem}_depth_est.npy", est.depth.astype(np.float16))
    return f"{stem}_rgb.png", f"{stem}_depth.npy", f"{stem}_seg.png", f"{stem}_depth_est.npy"


def main(argv: list[str] | None = None) -> None:
    from control.rules import RuleController
    from eval.factory import make_judge, make_sim, make_stack
    from policies.base import make_policy
    from sim.scenario import generate

    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--policy", default="never", help="policy name, or a JSON spec like '{\"policy\": \"random\", \"p\": 0.1}'")
    p.add_argument("--seed", type=int, default=101_000, help="default: the first val seed of configs/seeds.yaml")
    p.add_argument("--split", default="val")
    p.add_argument("--stack", default="true-depth", choices=["true-depth", "blind", "da2", "clear"])
    p.add_argument("--backend", default="raycast", choices=["raycast", "airsim"])
    p.add_argument("--judge", default="surrogate", choices=["surrogate", "qwen", "none"])
    p.add_argument("--delay", type=float, default=1.0)
    p.add_argument("--log", help="write the flight log (one FrameRecord per line) here")
    a = p.parse_args(argv)
    spec = json.loads(a.policy) if a.policy.strip().startswith("{") else {"policy": a.policy}
    scenario = generate(a.seed, a.split)
    judge, judge_name = make_judge(a.judge)
    records, result = run_episode(scenario, make_sim(a.backend), make_stack(a.stack), RuleController(), make_policy(spec), judge,
                                  LoopConfig(delay_s=a.delay), judge_name=judge_name)
    if a.log:
        Path(a.log).parent.mkdir(parents=True, exist_ok=True)
        Path(a.log).write_text("".join(r.to_json() + "\n" for r in records))
    print(result.to_json())


if __name__ == "__main__":
    main()
