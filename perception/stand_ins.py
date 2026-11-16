"""Fast-stack stand-ins for the flight loop (PLAN_DETAILED.md §6, stand-ins; chunk M4.5).

The loop calls ``perceive_obs(obs)`` so a stand-in can read the observation's
ground truth. Both stand-ins below are references or test tools, never results:

- ``TrueDepthStack``: the controller on ground-truth depth ("true-depth flight", PLAN_DETAILED.md
  §2). A pipeline sanity check and an upper bound: it should crash almost never (M4 gate).
- ``BlindStack``: ground-truth depth rendered *without* thin and see-through obstacles, a crude
  imitation of how Depth Anything V2 misses wires, poles and glass. Lets the gated loop be tested
  on a laptop. Its results must never be reported as fast-stack results.

The real fast stack (``perception.depth_anything.DepthAnythingV2``) has only
``perceive(rgb)``; ``RgbStack`` adapts it.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from interfaces import DepthEstimate
from perception.depth_utils import MAX_RANGE_M, clip_depth, corridor_features


class TrueDepthStack:
    """The simulator's depth plus the see-through obstacles (glass) its depth camera cannot see.

    Both Project AirSim and the raycast backend render glass as transparent to depth, so the camera's
    "ground truth" would fly the drone into every pane. The reference must be the true geometry, so
    see-through boxes are ray-cast from the scenario (``sim.boxes``, ``sim.cam``) and merged in.
    """

    name = "true-depth"

    def perceive_obs(self, obs: Any, state: Any = None, sim: Any = None) -> DepthEstimate:
        d = np.asarray(obs.depth, dtype=np.float32)
        glass = [b for b in getattr(sim, "boxes", []) if b.see_through]
        if glass and state is not None:
            from sim.raycast import first_hits

            t, _ = first_hits(state.position, sim.cam.rays(state.yaw), glass)
            d = np.minimum(d, t.reshape(d.shape).astype(np.float32))
        d = clip_depth(d, MAX_RANGE_M)
        return DepthEstimate(depth=d, features=corridor_features(d))


class BlindStack:
    """Ground truth minus every obstacle whose smallest half-size is below ``thin_m``, and minus glass."""

    name = "blind (synthetic stand-in)"

    def __init__(self, thin_m: float = 0.06) -> None:
        self.thin_m = thin_m

    def perceive_obs(self, obs: Any, state: Any = None, sim: Any = None) -> DepthEstimate:
        d = sim.render_depth(state, keep=lambda b: not b.see_through and min(b.half) >= self.thin_m)
        d = clip_depth(d, MAX_RANGE_M)
        return DepthEstimate(depth=d, features=corridor_features(d))


class RgbStack:
    """Adapter for a fast stack that only sees RGB (the real one)."""

    def __init__(self, stack: Any) -> None:
        self.stack = stack
        self.name = getattr(stack, "checkpoint", type(stack).__name__)

    def perceive_obs(self, obs: Any, state: Any = None, sim: Any = None) -> DepthEstimate:
        return self.stack.perceive(np.ascontiguousarray(obs.rgb))
