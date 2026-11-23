"""Project AirSim backend for the flight loop (PLAN_DETAILED.md D3, D4, D10; chunk M4.5). Runs on the HPC only.

Same interface as ``sim.raycast.RaycastSim``: ``reset(scenario)``, ``observe(state)``,
``cam``, ``boxes``, ``name``. Built on the calls the frame bank used successfully
(``hpc/capability/capability_test.py``, jobs 16299/16300): ``set_pose``, then one
100 ms ``continue_for_sim_time``, then ``get_images``; obstacles spawned as glTF
boxes with segmentation ID 100. The world stays paused between steps, so the
VLM call happens with the simulator stopped and the delay is charged by the
flight loop's own clock (D10).

[UNTESTED on a live simulator: written on a laptop against the capability-test
helpers. Run ``hpc/m4_checks.sbatch`` first; it exercises the same calls.]

Start poses (D24). ``starts`` is a list of ``[x, y, yaw_deg, clear_m]`` in world
coordinates; a flight uses pose ``seed % len(starts)``, so the pose is part of
the scenario and every policy flies a seed from the same spot (D15). ``clear_m``
is the open distance ahead of that pose, from the capability scene scan
(``results.json`` key ``400px.scene_scan.rows``, ``corridor_min_m``); a flight
whose goal does not fit inside it is refused. The scan looked straight ahead
from one spot, so each pose is [UNVERIFIED] until
``configs/sweeps/airsim_starts_smoke.yaml`` flies it with no "scenery" crash.

Assumptions, each [UNVERIFIED]:
- the RGB array from ``unpack_image`` is BGR (the bank wrote it with ``cv2.imwrite`` and the PNGs
  look right), so it is flipped to RGB here;
- the ground is at world z = 0 near the start pose, so a scenario altitude of ``alt`` m is world
  z = ``-alt``;
- only the "clear" condition is rendered; weather is not reliable in this build (hpc/instructions.md §3.5).
"""
from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from sim.drone import KinematicState
from sim.raycast import Camera, Observation
from sim.scenario import CATALOG, Scenario

REPO = Path(__file__).resolve().parent.parent


def load_capability_module() -> Any:
    """Import hpc/capability/capability_test.py (needs projectairsim and cv2, i.e. the HPC env)."""
    path = REPO / "hpc" / "capability" / "capability_test.py"
    spec = importlib.util.spec_from_file_location("capability_test", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["capability_test"] = mod
    spec.loader.exec_module(mod)
    return mod


def path_to_world(x: float, y: float, z: float, yaw: float, start: tuple[float, float, float]) -> tuple[float, float, float, float]:
    """Map a pose in the scenario's path frame (start at origin, heading +x) to the world, given start (x0, y0, yaw0)."""
    x0, y0, yaw0 = start
    c, s = math.cos(yaw0), math.sin(yaw0)
    return x0 + x * c - y * s, y0 + x * s + y * c, z, yaw0 + yaw


def pick_start(starts: Sequence[Sequence[float]], seed: int, goal_m: float, margin_m: float = 5.0) -> tuple[int, tuple[float, float, float]]:
    """(index, (x0, y0, yaw0 in radians)) of the start pose for ``seed``; raises if the flight does not fit its clearance."""
    i = int(seed) % len(starts)
    x, y, yaw_deg, clear_m = (float(v) for v in starts[i])
    if goal_m + margin_m > clear_m:
        raise ValueError(f"start pose {i} ({x:g}, {y:g}, {yaw_deg:g} deg) has {clear_m:g} m clear; a {goal_m:g} m flight needs {goal_m + margin_m:g} m")
    return i, (x, y, math.radians(yaw_deg))


class AirSimBackend:
    name = "project-airsim"

    def __init__(self, configs: str, scene_file: str = "scene_cap_400.jsonc", start: tuple[float, float, float] = (0.0, 0.0, 0.0),
                 width: int = 400, height: int = 225, seg_id: int = 100, starts: Sequence[Sequence[float]] | None = None,
                 clear_margin_m: float = 5.0) -> None:
        self.configs, self.scene_file, self.start, self.seg_id = configs, scene_file, start, seg_id
        self.starts, self.clear_margin_m = starts, clear_margin_m
        self.start_index: int | None = None  # which of ``starts`` the current flight uses (logged per flight)
        self.cam = Camera(width, height)
        self.boxes: list = []
        self.scenario: Scenario | None = None
        self._ct = self._client = self._world = self._drone = None
        self._spawned: list[str] = []

    def _connect(self) -> None:
        if self._world is not None:
            return
        from projectairsim import Drone, ProjectAirSimClient, World

        self._ct = load_capability_module()
        self._client = ProjectAirSimClient()
        self._ct.connect_with_retry(self._client)
        self._world = World(self._client, self.scene_file, delay_after_load_sec=2, sim_config_path=self.configs)
        self._drone = Drone(self._client, self._world, "Drone1")
        self._world.pause()

    def _clear(self) -> None:
        for name in self._spawned:
            try:
                self._world.destroy_object(name)
            except Exception:  # noqa: BLE001  (already gone)
                pass
        self._spawned = []

    def reset(self, scenario: Scenario) -> None:
        if self.starts:  # before touching the simulator: a flight that does not fit must not spawn anything
            self.start_index, self.start = pick_start(self.starts, scenario.seed, scenario.goal_m, self.clear_margin_m)
        self._connect()
        self._clear()
        self.scenario, self.boxes = scenario, scenario.obbs()
        ct = self._ct
        for i, (o, box) in enumerate(zip(scenario.obstacles, self.boxes)):
            dx, dy, dz = CATALOG[o.family][0]
            colour = (*CATALOG[o.family][1], 0.2 if box.see_through else 1.0)
            wx, wy, wz, wyaw = path_to_world(*box.center, box.yaw, self.start)
            name = f"ep{scenario.seed}_{i}"
            res = self._world.spawn_object_from_file(name, "gltf", ct.box_gltf(dx, dy, dz, colour, box.see_through, binary=True),
                                                     True, ct.make_pose(wx, wy, wz, yaw=wyaw), [1.0, 1.0, 1.0], False)
            spawned = str(res) if res not in (None, False, "") else name
            self._spawned.append(spawned)
            try:
                self._world.set_segmentation_id_by_name(spawned, self.seg_id, False, True)
            except Exception:  # noqa: BLE001  (the bank also tolerated this)
                pass
        self._world.continue_for_sim_time(3 * ct.STEP_NS)  # let spawns land before the first frame

    def observe(self, state: KinematicState) -> Observation:
        ct = self._ct
        wx, wy, wz, wyaw = path_to_world(state.x, state.y, state.z, state.yaw, self.start)
        self._drone.set_pose(ct.make_pose(wx, wy, wz, yaw=wyaw))
        self._world.continue_for_sim_time(ct.STEP_NS)
        bgr, depth, seg = self._grab(ct.make_pose(wx, wy, wz, yaw=wyaw))
        rgb = np.ascontiguousarray(bgr[..., :3][..., ::-1])
        seg_rgb = np.ascontiguousarray(seg[..., :3][..., ::-1])
        return Observation(rgb=rgb, depth=depth.astype(np.float32), seg=seg_rgb)

    def _grab(self, pose: Any, retries: int = 3) -> tuple[Any, Any, Any]:
        """Capture all three channels; on a dropped capture, set the pose again and let the renderer settle.

        Jobs 40412 (GetImages timeout) and 41022 (missing channel) each lost a run to one bad capture. The
        pose is reset and one frame discarded, so a retry cannot return a stale frame; obstacles are static,
        so the observation still depends only on the scenario and the state.
        """
        error = "missing channel"
        for attempt in range(retries + 1):
            try:
                bgr, depth, seg = self._ct.grab(self._drone) if attempt == 0 else self._ct.settle_grab(self._world, self._drone, pose)
                if bgr is not None and depth is not None and seg is not None:
                    return bgr, depth, seg
                error = "missing channel"
            except RuntimeError as e:  # projectairsim raises it for a GetImages timeout
                error = str(e)
            print(f"capture failed ({error}), attempt {attempt + 1} of {retries + 1}", flush=True)
        raise RuntimeError(f"Project AirSim capture failed {retries + 1} times: {error}")

    def render_depth(self, *_: Any, **__: Any) -> np.ndarray:
        raise NotImplementedError("the blind stand-in is raycast-only; use the real fast stack (stack: da2) on Project AirSim")

    def close(self) -> None:
        if self._world is not None:
            self._clear()
        if self._client is not None:
            try:
                self._client.disconnect()
            except Exception:  # noqa: BLE001
                pass
