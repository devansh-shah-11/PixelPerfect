"""Job A world checks on the raycast backend: map objects found where they are, start and landing verdicts, probe and ground rows."""
from __future__ import annotations

import json

import numpy as np

from eval.world_check import region_report, run_ground, run_probe, run_region, scan_line, seg_extent, write_report
from perception.frame_bank import OBSTACLE_SEG_RGB
from perception.stand_ins import TrueDepthStack
from sim.geometry import OBB
from sim.raycast import Camera, RaycastSim

STARTS = [[0.0, 0.0, 0.0, 100.0], [0.0, 0.0, 0.0, 100.0]]


class MapSim(RaycastSim):
    """Raycast backend with fixed 'map' boxes per start pose (stand-ins for CityEnviron scenery)."""

    def __init__(self, maps):
        super().__init__(Camera(160, 90))
        self.maps = maps

    def empty_map(self, start):
        self.scenario, self.boxes = None, list(self.maps[self._pose])

    def use(self, pose):
        self._pose = pose
        return self


class PoseSim(MapSim):
    """Picks the map by the order in which run_region visits poses."""

    def __init__(self, maps):
        super().__init__(maps)
        self._calls = 0

    def empty_map(self, start):
        self.scenario, self.boxes = None, list(self.maps[self._calls % len(self.maps)])
        self._calls += 1


def test_scan_line_finds_a_map_object_and_skips_open_space():
    sim = MapSim([[OBB(center=(30.0, 0.0, -2.0), half=(0.5, 3.0, 3.0), family="building_block")]]).use(0)
    sim.empty_map(STARTS[0])
    pts = scan_line(sim, 0.0, 2.0, x_max=40.0)
    xs = [p["x"] for p in pts]
    assert len(pts) < 41  # skipped ahead
    assert any(p["depth_min"] < 0.3 for p in pts)  # reached the wall
    assert min(p["x"] for p in pts if p["depth_min"] < 0.3) <= 30.0 and xs == sorted(xs)


def test_region_rejects_only_a_blocked_landing_area_or_start(tmp_path):
    side_tree = OBB(center=(30.0, 15.0, -5.0), half=(1.0, 1.0, 5.0), family="building_block")  # beside the path: reported, not a reason to reject
    landing_car = OBB(center=(55.0, 0.0, -0.8), half=(2.0, 1.0, 0.8), family="box_1m_grey")  # at the landing area, 1.6 m tall
    sim = PoseSim([[side_tree], [landing_car]])
    run_region(sim, STARTS, tmp_path, ys=(0.0, 15.0), alts=(1.0, 2.0, 10.0), passes=2, x_max=60.0)
    lines = [json.loads(x) for x in (tmp_path / "region.jsonl").read_text().splitlines()]
    assert len(lines) == 2 * 2 * 2 * 3
    _, verdicts = region_report(lines, passes=2)
    assert verdicts[0]["keep"] and verdicts[0]["map"] > 0  # the tree is counted
    assert not verdicts[1]["keep"] and verdicts[1]["landing_hits"] > 0
    run_region(sim, STARTS, tmp_path, ys=(0.0, 15.0), alts=(1.0, 2.0, 10.0), passes=2, x_max=60.0)  # resumes: nothing new
    assert len((tmp_path / "region.jsonl").read_text().splitlines()) == len(lines)


def test_seg_extent_tells_vertical_from_horizontal():
    seg = np.zeros((90, 160, 3), np.uint8)
    seg[10:80, 70:72] = OBSTACLE_SEG_RGB
    assert seg_extent(seg)["renders"] == "vertical"
    seg[:] = 0
    seg[44:46, 20:140] = OBSTACLE_SEG_RGB
    assert seg_extent(seg)["renders"] == "horizontal"
    assert seg_extent(np.zeros((9, 9, 3), np.uint8))["renders"] == "invisible"


def test_probe_and_ground_rows_and_report(tmp_path):
    sim = RaycastSim(Camera(400, 225))  # a 10 cm pole at 10 m is about 2 pixels wide at 400 px
    run_probe(sim, STARTS, tmp_path, stack=TrueDepthStack(), families=("box_1m_grey", "pole_10cm_axis_y", "glass_pane_alpha20"),
              dists=(10.0,), yaws_deg=(0.0,))
    rows = {r["family"]: r for r in map(json.loads, (tmp_path / "probe.jsonl").read_text().splitlines())}
    assert abs(rows["box_1m_grey"]["depth_min"] - 9.5) < 0.2 and rows["box_1m_grey"]["est_corridor_min"] < 10.0
    assert rows["pole_10cm_axis_y"]["renders"] == "vertical"  # the y-axis pole stands up in our geometry
    assert rows["glass_pane_alpha20"]["seg_px"] == 0 and rows["glass_pane_alpha20"]["corridor_geometry"] < 10.0  # see-through, solid
    assert abs(rows["box_1m_grey"]["depth_min_1m_in_front"] - 1.0) < 0.2
    run_ground(sim, STARTS, tmp_path, stack=TrueDepthStack(), alts=(1.0, 2.0))
    ground = [json.loads(x) for x in (tmp_path / "ground.jsonl").read_text().splitlines()]
    assert len(ground) == 4 and all(not g["guard_fires"] for g in ground)
    text = write_report(tmp_path, STARTS, passes=2)
    assert "Obstacle probe" in text and "Ground check" in text
