"""Kinematic drone (PLAN_DETAILED.md D4, chunk M4.3).

Kinematic stepping: the pose is set directly each step, no flight dynamics, so
a seed replays exactly (needed for replay labels) and there is no controller
tuning to sink time into. We study perception, not control.

Actions (``interfaces.Action``), each applied for one step of ``dt`` seconds:

- ``CONTINUE``: fly forward at cruise speed ``v``
- ``STOP``: brake and hold (speed 0); also the waiting behaviour while the VLM is asked (D11 MVP)
- ``YAW_LEFT`` / ``YAW_RIGHT``: turn at ``yaw_rate`` while moving at ``v_turn`` (0 = turn in place)
- ``CLIMB``: rise at ``climb_rate`` while moving at ``v_climb`` (0 = climb in place), capped at ``alt_max``
- ``DESCEND``: sink at ``climb_rate`` while moving at ``v_descend`` (0 = descend in place), floored at ``alt_min``
  (the landing height, so landing is not ground contact)

Hysteresis lives in the controller (``control.rules``), not here.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np

from interfaces import Action, DroneState


@dataclass(frozen=True)
class DroneParams:
    v: float = 5.0
    yaw_rate: float = math.radians(45.0)
    v_turn: float = 0.0
    climb_rate: float = 2.0
    v_climb: float = 0.0
    alt_max: float = 25.0
    radius: float = 0.3  # collision sphere
    v_descend: float = 0.0
    alt_min: float = 1.0  # DESCEND stops here: the landing height above the ground


@dataclass(frozen=True)
class KinematicState:
    t: float
    x: float
    y: float
    z: float  # NED: altitude is -z
    yaw: float
    speed: float

    @property
    def position(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z])

    @property
    def alt(self) -> float:
        return -self.z

    def public(self) -> DroneState:
        """What the drone knows about itself at run time (no obstacle information)."""
        return DroneState(t=self.t, position=(self.x, self.y, self.z), yaw=self.yaw, speed=self.speed)


def step(s: KinematicState, action: Action, dt: float, p: DroneParams) -> KinematicState:
    """Next state after ``action`` for ``dt`` seconds."""
    yaw, z, v = s.yaw, s.z, 0.0
    if action == Action.CONTINUE:
        v = p.v
    elif action == Action.STOP:
        v = 0.0
    elif action in (Action.YAW_LEFT, Action.YAW_RIGHT):
        yaw = s.yaw + (-1.0 if action == Action.YAW_LEFT else 1.0) * p.yaw_rate * dt  # NED: positive yaw turns right
        v = p.v_turn
    elif action == Action.CLIMB:
        z = max(-p.alt_max, s.z - p.climb_rate * dt)
        v = p.v_climb
    elif action == Action.DESCEND:
        z = s.z if s.alt <= p.alt_min else min(-p.alt_min, s.z + p.climb_rate * dt)  # never pushes a low drone up
        v = p.v_descend
    else:
        raise ValueError(action)
    yaw = (yaw + math.pi) % (2 * math.pi) - math.pi
    heading = 0.5 * (s.yaw + yaw) if abs(yaw - s.yaw) < math.pi else yaw  # move along the mean heading of the step
    return replace(s, t=s.t + dt, x=s.x + v * dt * math.cos(heading), y=s.y + v * dt * math.sin(heading), z=z, yaw=yaw, speed=v)


def hold(s: KinematicState, seconds: float) -> KinematicState:
    """Brake and hold for ``seconds`` of simulated time (the VLM wait, D10/D11)."""
    return replace(s, t=s.t + seconds, speed=0.0)
