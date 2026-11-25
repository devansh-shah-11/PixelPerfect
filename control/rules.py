"""Rule controller: estimated corridor depth to action (PLAN_DETAILED.md §4.2; implements ``interfaces.Controller``).

Reads only the fast stack's estimate, the drone's own state and the mission (the goal point given at takeoff),
never ground truth about obstacles (CLAUDE.md rule 5).

- Blocked when the estimated corridor minimum is below ``d_safe``; stays blocked until it is above
  ``d_safe + hysteresis`` (no flicker between avoiding and flying on).
- Clear: fly on, turning towards the goal bearing first if the heading is off by more than ``heading_tol``.

How the drone avoids, ``avoid_mode``:

- ``"climb_first"`` (default; the original rule, so existing configs and results keep their meaning): climb in
  place; at the altitude cap, turn in place towards the side whose estimated depth is larger.
- ``"choose"`` (plan v7, DS 2026-11-25): decided by the situation each time. The free distance to the left, to the
  right and above is read from the estimate (median of a band, clipped at ``MAX_RANGE_M``); the side towards the
  goal gets a bonus of ``goal_bias_m``. The best side is taken if it reads clear (at least ``d_safe +
  hysteresis``); if neither side does, the drone climbs ("over"), and at the altitude cap takes the better side.
  The choice is kept for the whole avoidance episode while it stays free, and is re-scored once that side reads
  blocked (below ``d_safe``). The sky reads far in a depth map, so "over" is the fallback, not a scored rival:
  otherwise the drone would almost always climb.

``avoid_action`` is also what the flight loop executes when the VLM says something is in the path although the
estimate looked clear. Its ``direction`` ("left", "right", "over") is the VLM's answer and overrides the
scoring for that avoidance episode: for a wire or a glass pane, the depth map cannot say where the obstacle is.

Mission (``set_mission``; only on courses with a goal point, so line courses behave as before):

- steer towards the bearing from the drone's position to the goal;
- after an avoidance episode ends, keep the current heading until the drone is past the obstacle it avoided:
  further along the heading it had when avoidance started than the obstacle's estimated distance plus
  ``past_margin_m`` (drone size), or, after a mostly sideways turn, once it has flown twice that far. The distance
  is the estimate's corridor minimum when avoidance started, or ``d_judge`` (the range the VLM is asked about) when
  only the VLM saw something. A new obstacle on the way starts a new avoidance episode at once. Then turn back
  towards the goal and descend in place to the cruise height while the corridor reads clear;
- within ``land_radius_m`` of the goal (horizontally), stop and descend to the landing height, then hold;
- soft geofence: the mission area is ``area_half_width_m`` either side of the start-to-goal line. Within
  ``fence_margin_m`` of its edge, a side that would carry the drone further out counts as blocked (for the
  depth-based choice and for the VLM's direction, which then becomes "over"), and a hold after avoiding ends early
  so the drone turns back. Crossing the edge anyway ends the flight in the loop (outcome "left area").
  The landing spot is kept free of placed obstacles by the course generator (``sim.scenario`` cone layout).
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from interfaces import Action, DepthEstimate, DroneState
from perception.depth_utils import MAX_RANGE_M, clip_depth

DIRECTIONS = ("left", "right", "over")


def free_space(depth: np.ndarray) -> dict[str, float]:
    """Free distance (median, metres) in the left, right and upper bands of a depth estimate.

    Side bands: rows 0.35 to 0.65 of the height, the outer 40% of the width on each side; upper band: rows 0 to
    0.3, columns 0.4 to 0.6. None overlaps the corridor window (rows 0.4 to 0.6, columns 0.45 to 0.55).
    """
    d = clip_depth(depth, MAX_RANGE_M)
    h, w = d.shape[:2]
    mid = d[int(0.35 * h): int(0.65 * h)]
    return {"left": float(np.median(mid[:, : int(0.4 * w)])), "right": float(np.median(mid[:, int(0.6 * w):])),
            "over": float(np.median(d[: int(0.3 * h), int(0.4 * w): int(0.6 * w)]))}


def _wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


class RuleController:
    def __init__(self, d_safe: float = 8.0, hysteresis: float = 2.0, alt_max: float = 25.0, alt_margin: float = 1.0,
                 heading_tol: float = math.radians(10.0), avoid_mode: str = "climb_first", goal_bias_m: float = 5.0,
                 past_margin_m: float = 2.0, d_judge: float = 15.0, land_radius_m: float = 2.0, alt_tol_m: float = 0.2,
                 fence_margin_m: float = 3.0, max_side_switches: int = 1) -> None:
        if avoid_mode not in ("climb_first", "choose"):
            raise ValueError(f"avoid_mode must be 'climb_first' or 'choose', got {avoid_mode!r}")
        self.d_safe, self.hysteresis, self.alt_max, self.alt_margin, self.heading_tol = d_safe, hysteresis, alt_max, alt_margin, heading_tol
        self.avoid_mode, self.goal_bias_m, self.past_margin_m, self.d_judge = avoid_mode, goal_bias_m, past_margin_m, d_judge
        self.land_radius_m, self.alt_tol_m, self.fence_margin_m = land_radius_m, alt_tol_m, fence_margin_m
        self.max_side_switches = max_side_switches
        self.goal_bearing = 0.0  # path frame: with no mission the goal lies straight ahead along +x
        self.reset()

    def reset(self) -> None:
        self.blocked = False
        self.goal: tuple[float, float] | None = None
        self.cruise_alt: float | None = None
        self.land_alt: float | None = None
        self.half_width: float | None = None
        self.side: str | None = None  # direction of the current avoidance episode
        self.avoiding = False
        self.climbed = False  # this avoidance episode went over (at least one CLIMB)
        self.switches = 0  # left/right changes in this avoidance episode
        self.bypass_over = False  # the bypass follows an episode that went over: the fence does not cut it short
        self.avoid_start: tuple[float, float, float, float] | None = None  # x, y, heading, obstacle distance
        self.bypass_from: tuple[float, float, float, float] | None = None
        self.landing = False
        self.phase = "cruise"  # set by act(), for logs: cruise, avoid, bypass, return (down to cruise height), landing

    def set_mission(self, goal_xy: tuple[float, float], cruise_alt: float, land_alt: float, area_half_width_m: float | None = None) -> None:
        """The goal point (path frame), cruise and landing heights and the mission area, given at takeoff (after ``reset``)."""
        self.goal, self.cruise_alt, self.land_alt = (float(goal_xy[0]), float(goal_xy[1])), float(cruise_alt), float(land_alt)
        self.half_width = None if area_half_width_m is None else float(area_half_width_m)

    def _cross_track(self, state: DroneState) -> float:
        """Signed distance from the start-to-goal line, positive to its right (NED)."""
        gx, gy = self.goal
        n = math.hypot(gx, gy)
        return (gx * state.position[1] - gy * state.position[0]) / n

    def _outward(self, side: str, state: DroneState) -> bool:
        """True when ``side`` would carry a drone near the edge of the mission area further out."""
        if self.goal is None or self.half_width is None or side == "over":
            return False
        e = self._cross_track(state)
        if abs(e) < self.half_width - self.fence_margin_m:
            return False
        turn = math.radians(45.0) * (1 if side == "right" else -1)
        drift = math.sin(_wrap(state.yaw + turn - math.atan2(self.goal[1], self.goal[0])))  # > 0: moving right of the line
        return drift * e > 0

    def _blocked(self, corridor_min: float) -> bool:
        limit = self.d_safe + (self.hysteresis if self.blocked else 0.0)
        self.blocked = corridor_min < limit
        return self.blocked

    def _goal_side(self, state: DroneState) -> str | None:
        if self.goal is None:
            return None
        x, y = state.position[0], state.position[1]
        err = _wrap(math.atan2(self.goal[1] - y, self.goal[0] - x) - state.yaw)
        return "right" if err > 0 else "left"  # NED: positive yaw turns right

    def _choose(self, free: dict[str, float], state: DroneState, can_climb: bool) -> str:
        bonus = self._goal_side(state)
        free = {s: (0.0 if self._outward(s, state) else v) for s, v in free.items()}
        score = {s: free[s] + (self.goal_bias_m if s == bonus else 0.0) for s in ("left", "right")}
        best = max(score, key=lambda s: (score[s], s == bonus))
        if free[best] >= self.d_safe + self.hysteresis or not can_climb:
            return best
        return "over"

    def avoid_action(self, depth: DepthEstimate, state: DroneState, direction: str | None = None) -> Action:
        alt = -state.position[2]
        can_climb = alt < self.alt_max - self.alt_margin
        if not self.avoiding:  # first step of an avoidance episode: remember where the obstacle is
            cmin = float(depth.features["corridor_min"])
            dist = cmin if cmin < self.d_safe + self.hysteresis else self.d_judge
            self.avoid_start = (state.position[0], state.position[1], state.yaw, dist)
            self.climbed, self.switches = False, 0
        self.avoiding, self.bypass_from, self.phase = True, None, "avoid"
        action = self._avoid_step(depth, state, direction, can_climb)
        self.climbed = self.climbed or action == Action.CLIMB
        return action

    def _avoid_step(self, depth: DepthEstimate, state: DroneState, direction: str | None, can_climb: bool) -> Action:
        if direction is not None and direction not in DIRECTIONS:
            raise ValueError(f"direction must be one of {DIRECTIONS} or None, got {direction!r}")
        if self.avoid_mode == "climb_first" and direction is None:
            if can_climb:
                return Action.CLIMB
            h, w = depth.depth.shape
            band = depth.depth[int(0.35 * h): int(0.65 * h)]
            left, right = float(np.mean(band[:, : w // 2])), float(np.mean(band[:, w // 2:]))
            return Action.YAW_LEFT if left > right else Action.YAW_RIGHT
        free = free_space(depth.depth)
        if direction is not None:
            self.side = "over" if self._outward(direction, state) else direction  # the VLM's answer holds for this episode
        elif self.side is None or (self.side != "over" and (free[self.side] < self.d_safe or self._outward(self.side, state))):
            new = self._choose(free, state, can_climb)
            if self.side in ("left", "right") and new in ("left", "right") and new != self.side:
                # Turning in place moves the obstacle from one side band to the other, so the scores flip with every
                # turn (Job B1 seed 101068: 7 s of left/right in place). Both sides tried: go over, or at the
                # height cap keep the current side.
                if self.switches >= self.max_side_switches:
                    new = "over" if can_climb else self.side
                else:
                    self.switches += 1
            self.side = new
        if self.side == "over" and not can_climb:
            self.side = self._choose(free, state, can_climb=False)
        return {"left": Action.YAW_LEFT, "right": Action.YAW_RIGHT, "over": Action.CLIMB}[self.side]

    def act(self, depth: DepthEstimate, drone_state: DroneState) -> Action:
        x, y, alt = drone_state.position[0], drone_state.position[1], -drone_state.position[2]
        if self.goal is not None and (self.landing or math.hypot(self.goal[0] - x, self.goal[1] - y) <= self.land_radius_m):
            self.landing, self.phase = True, "landing"
            return Action.DESCEND if alt > self.land_alt + 0.01 else Action.STOP  # DESCEND is floored at the landing height
        if self._blocked(float(depth.features["corridor_min"])):
            self.phase = "avoid"
            return self.avoid_action(depth, drone_state)
        if self.avoiding:  # the avoidance episode just ended
            self.avoiding, self.side = False, None
            self.bypass_from = self.avoid_start if self.goal is not None else None
            self.bypass_over = self.climbed
        if self.bypass_from is not None:
            x0, y0, h0, dist = self.bypass_from
            dx, dy = x - x0, y - y0
            ahead, flown = dx * math.cos(h0) + dy * math.sin(h0), math.hypot(dx, dy)
            need = dist + self.past_margin_m
            # The fence cuts only a sideways bypass short. After going over, ending the hold would descend in front of
            # the obstacle, see it again and climb again, in a loop (Job B1 seeds 101071, 101077: outcome "lost").
            near_edge = (not self.bypass_over and self.half_width is not None
                         and abs(self._cross_track(drone_state)) >= self.half_width - self.fence_margin_m)
            if ahead < need and flown < 2 * need and not near_edge:
                self.phase = "bypass"
                return Action.CONTINUE
            self.bypass_from = None
        if self.cruise_alt is not None and alt > self.cruise_alt + self.alt_tol_m:
            self.phase = "return"
            return Action.DESCEND
        self.phase = "cruise"
        bearing = math.atan2(self.goal[1] - y, self.goal[0] - x) if self.goal is not None else self.goal_bearing
        err = _wrap(bearing - drone_state.yaw)
        if abs(err) > self.heading_tol:
            return Action.YAW_RIGHT if err > 0 else Action.YAW_LEFT
        return Action.CONTINUE

    def config(self) -> dict[str, Any]:
        return {"d_safe": self.d_safe, "hysteresis": self.hysteresis, "alt_max": self.alt_max, "avoid_mode": self.avoid_mode,
                "goal_bias_m": self.goal_bias_m, "past_margin_m": self.past_margin_m, "d_judge": self.d_judge,
                "land_radius_m": self.land_radius_m, "fence_margin_m": self.fence_margin_m, "max_side_switches": self.max_side_switches}
