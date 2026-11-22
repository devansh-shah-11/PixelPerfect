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

The flight ends on a collision, at the goal, or at ``t_max``. Wall-clock time
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
from sim.geometry import hits_ground, segment_hits
from sim.raycast import corridor_truth
from sim.scenario import Scenario


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
    d_safe: float = 8.0  # for the L1 label handed to the oracle; match the controller
    d_judge: float = 15.0  # the surrogate's truth: a surface in the window closer than this "crosses" it
    save_dir: str | None = None  # write rgb/depth/seg per frame here (large); off by default

    def __post_init__(self) -> None:
        if self.delay_s <= 0:
            raise ValueError("delay_s must be > 0 (PLAN_DETAILED.md D10)")


def run_episode(scenario: Scenario, sim: Any, stack: Any, controller: Any, policy: Any, judge: Any = None,
                cfg: LoopConfig | None = None, params: DroneParams | None = None, judge_name: str = "none",
                snapshots: list[ForkPoint] | None = None, resume: ForkPoint | None = None,
                t_stop: float | None = None) -> tuple[list[FrameRecord], EpisodeResult]:
    """Fly one episode. For replay labels (M6.1): ``snapshots`` collects a ``ForkPoint`` per step;
    ``resume`` continues from one (the simulator must already hold this scenario; ``controller`` is
    replaced by the snapshot's copy) and ``t_stop`` ends the flight at that simulated time."""
    from vlm.overlay import render

    cfg, params = cfg or LoopConfig(), params or DroneParams(v=scenario.speed_mps)
    if resume is None:
        sim.reset(scenario)
        controller.reset()
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
    if resume is not None:
        avoid_until, distance = resume.avoid_until, resume.distance
    episode_id = f"{scenario.split}-{scenario.seed}-{getattr(policy, 'name', 'policy')}"
    save = Path(cfg.save_dir) / episode_id if cfg.save_dir else None
    if save:
        save.mkdir(parents=True, exist_ok=True)

    from perception.depth_utils import corridor_window

    collision_source = None
    for k in range(0 if resume is None else resume.k, int(math.ceil(cfg.t_max / cfg.dt)) + 1):
        if snapshots is not None:
            snapshots.append(ForkPoint(k, state, copy.deepcopy(controller), copy.deepcopy(temporal), avoid_until, distance))
        obs = sim.observe(state)
        # Scenery guard: our swept check only knows the spawned obstacles. On Project AirSim the map has its own
        # buildings, so a ground-truth surface closer than the drone's radius straight ahead also counts as a crash
        # (one step late at most; the camera sits about 0.3 m ahead of the drone origin, job 33776).
        rows, cols = corridor_window(obs.depth.shape)
        if float(np.min(obs.depth[rows, cols])) < params.radius:
            crashed, collision_source = True, "scenery"
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

        gt_min, family = corridor_truth(state, boxes, sim.cam)
        est_min = float(feats["corridor_min"])
        if hasattr(policy, "set_truth"):
            policy.set_truth(gt_min < cfg.d_safe <= est_min)
        asked = bool(policy.should_ask(state.t, feats, public, history))
        answer = conf = lat = None
        charged = 0.0
        if asked:
            if judge is None:
                raise ValueError(f"policy {getattr(policy, 'name', policy)} asked but no judge was given")
            overlay, crop = render(obs.rgb)
            if hasattr(judge, "set_truth"):
                judge.set_truth(family, gt_min < cfg.d_judge and family not in (None, "ground"))
            a = judge.ask(obs.rgb, overlay, crop)
            answer, conf, lat = bool(a.answer), float(a.confidence), float(a.latency_s)
            latencies.append(lat)
            state = hold(state, cfg.delay_s)  # paused-time delay: brake and hold for d (D10, D11)
            charged = cfg.delay_s
            delay_total += charged
            if answer:
                avoid_until = state.t + cfg.vlm_avoid_s
        action = proposed
        if state.t < avoid_until and proposed == Action.CONTINUE:
            action = controller.avoid_action(est, state.public())

        new = step(state, action, cfg.dt, params)
        p0, p1 = state.position, new.position
        hit_ground = hits_ground(p0, p1, params.radius)
        collision = hit_ground or any(segment_hits(p0, p1, b, params.radius) for b in boxes)
        if collision:
            collision_source = "ground" if hit_ground else "obstacle"
        distance += float(np.linalg.norm(p1 - p0))
        paths = _save(save, k, obs, est) if save else ("", "", "", "")
        records.append(FrameRecord(
            episode_id=episode_id, split=scenario.split, seed=scenario.seed, t=round(state.t, 4),
            rgb_path=paths[0], depth_gt_path=paths[1], seg_gt_path=paths[2], depth_est_path=paths[3],
            pose=(state.x, state.y, state.z, state.yaw), velocity=(state.speed * math.cos(state.yaw), state.speed * math.sin(state.yaw), 0.0),
            corridor_min_gt=gt_min, corridor_min_est=est_min, action_proposed=proposed, asked=asked, vlm_answer=answer,
            vlm_conf=conf, vlm_latency_real_s=lat, delay_charged_s=charged, action_taken=action, collision=collision,
            obstacle_family=family if family not in (None, "ground") else None, gate_score=getattr(policy, "last_score", None)))
        history.append(RuntimeStep(t=state.t, asked=asked, vlm_answer=answer, action_taken=action, gate_score=getattr(policy, "last_score", None)))
        state = new
        if collision:
            crashed = True
            break
        if state.x >= scenario.goal_m:
            reached = True
            break
        if state.t >= cfg.t_max or (t_stop is not None and state.t >= t_stop - 1e-6):
            break

    result = EpisodeResult(
        episode_id=episode_id, seed=scenario.seed, split=scenario.split, policy=getattr(policy, "name", "policy"), judge=judge_name,
        crashed=crashed, reached_goal=reached, distance_m=round(distance, 3), mission_time_s=round(state.t, 3), n_frames=len(records),
        n_calls=sum(r.asked for r in records), delay_charged_s=round(delay_total, 3), latencies_s=latencies,
        obstacle_family=",".join(o.family for o in scenario.obstacles) or None,
        extra={"stack": getattr(stack, "name", type(stack).__name__), "backend": getattr(sim, "name", type(sim).__name__),
               "delay_s": cfg.delay_s, "condition": scenario.condition, "collision_source": collision_source,
               "start_index": getattr(sim, "start_index", None)})
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
