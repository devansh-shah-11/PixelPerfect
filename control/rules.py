"""Rule controller: estimated corridor depth to action (PLAN_DETAILED.md §4.2; implements ``interfaces.Controller``).

Reads only the fast stack's estimate and the drone's own state (CLAUDE.md rule 5).

- Blocked when the estimated corridor minimum is below ``d_safe``; stays blocked until it is above
  ``d_safe + hysteresis`` (no flicker between climbing and flying on).
- Blocked: climb in place; at the altitude cap, turn in place towards the side whose estimated
  depth is larger.
- Clear: fly on, turning towards the goal bearing first if the heading is off by more than
  ``heading_tol``.

``avoid_action`` is what the flight loop executes when the VLM says "yes, something is in the
path" although the estimate looked clear.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from interfaces import Action, DepthEstimate, DroneState


class RuleController:
    def __init__(self, d_safe: float = 8.0, hysteresis: float = 2.0, alt_max: float = 25.0, alt_margin: float = 1.0,
                 heading_tol: float = math.radians(10.0)) -> None:
        self.d_safe, self.hysteresis, self.alt_max, self.alt_margin, self.heading_tol = d_safe, hysteresis, alt_max, alt_margin, heading_tol
        self.goal_bearing = 0.0  # path frame: the goal lies straight ahead along +x
        self.blocked = False

    def reset(self) -> None:
        self.blocked = False

    def _blocked(self, corridor_min: float) -> bool:
        limit = self.d_safe + (self.hysteresis if self.blocked else 0.0)
        self.blocked = corridor_min < limit
        return self.blocked

    def avoid_action(self, depth: DepthEstimate, state: DroneState) -> Action:
        alt = -state.position[2]
        if alt < self.alt_max - self.alt_margin:
            return Action.CLIMB
        h, w = depth.depth.shape
        band = depth.depth[int(0.35 * h): int(0.65 * h)]
        left, right = float(np.mean(band[:, : w // 2])), float(np.mean(band[:, w // 2:]))
        return Action.YAW_LEFT if left > right else Action.YAW_RIGHT

    def act(self, depth: DepthEstimate, drone_state: DroneState) -> Action:
        if self._blocked(float(depth.features["corridor_min"])):
            return self.avoid_action(depth, drone_state)
        err = (self.goal_bearing - drone_state.yaw + math.pi) % (2 * math.pi) - math.pi
        if abs(err) > self.heading_tol:
            return Action.YAW_RIGHT if err > 0 else Action.YAW_LEFT
        return Action.CONTINUE

    def config(self) -> dict[str, Any]:
        return {"d_safe": self.d_safe, "hysteresis": self.hysteresis, "alt_max": self.alt_max}
