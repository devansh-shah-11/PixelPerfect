"""Video of one flight: camera, fast-stack depth, true depth and a top-down map, frame by frame.

Flies one seed with one policy from a sweep config (same simulator, stack, judge, controller and scenario
settings), saves every frame, then composes a video. Each video frame shows:

- top left: the camera image with the corridor window the controller and the VLM look at;
- top right: the fast stack's depth estimate (what the drone believes); bottom left: the true depth;
- bottom right: the course from above (obstacles, goal, mission area, track so far, heading);
- a status line: time, phase, action, the two corridor distances, and every VLM call with its answer and direction.
  A VLM call freezes the picture for the charged delay, so the cost of asking is visible.

    python -m eval.flight_video --config configs/sweeps/airsim_cone_smoke.yaml --seed 101060 --policy always --out results/videos/101060_always

Writes ``flight.mp4`` with ffmpeg if it is installed, else ``flight.gif``. ``--render-only`` rebuilds the video from
an earlier run's folder. Ground truth is drawn for the viewer only; the flight itself never sees it (rule 5).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image, ImageDraw

from vlm.overlay import corridor_box, draw_overlay

TILE_W, TILE_H, BAR_H = 400, 225, 54
MAX_DEPTH = 40.0


def depth_rgb(depth: np.ndarray, max_depth: float = MAX_DEPTH) -> np.ndarray:
    """Near = warm, far = dark blue; sky and beyond ``max_depth`` = darkest."""
    d = np.clip(np.nan_to_num(np.asarray(depth, dtype=np.float32), nan=max_depth, posinf=max_depth), 0, max_depth) / max_depth
    near = 1.0 - d
    r = np.clip(1.6 * near - 0.2, 0, 1)
    g = np.clip(1.6 * near - 0.6, 0, 1) * 0.9 + 0.1 * near
    b = np.clip(0.35 + 0.65 * d * (1 - d) * 4, 0, 1) * (0.4 + 0.6 * near)
    return (np.stack([r, g, b], -1) * 255).astype(np.uint8)


def _tile(img: np.ndarray, label: str) -> Image.Image:
    t = Image.fromarray(img).resize((TILE_W, TILE_H), Image.NEAREST)
    d = ImageDraw.Draw(t)
    d.rectangle([0, 0, 8 + 6 * len(label), 14], fill=(0, 0, 0))
    d.text((4, 1), label, fill=(255, 255, 255))
    return t


def _map_tile(scenario: dict[str, Any], track: list[tuple[float, float, float]], k: int) -> Image.Image:
    from sim.scenario import Scenario

    sc = Scenario.from_dict(scenario)
    im = Image.new("RGB", (TILE_W, TILE_H), (24, 26, 30))
    d = ImageDraw.Draw(im)
    x_min, x_max, y_half = -3.0, 66.0, 22.0
    s = min((TILE_W - 10) / (x_max - x_min), (TILE_H - 20) / (2 * y_half))

    def p(x: float, y: float) -> tuple[float, float]:
        return 5 + (x - x_min) * s, TILE_H / 2 + y * s  # right of the start heading is down

    if sc.goal_xy is not None and sc.area_half_width_m:
        gx, gy = sc.goal_xy
        n = math.hypot(gx, gy)
        nx, ny = -gy / n * sc.area_half_width_m, gx / n * sc.area_half_width_m
        for sgn in (1, -1):
            d.line([p(sgn * nx, sgn * ny), p(gx * 1.1 + sgn * nx, gy * 1.1 + sgn * ny)], fill=(70, 90, 120), width=1)
        d.ellipse([*p(gx - 2, gy - 2), *p(gx + 2, gy + 2)], outline=(90, 200, 100), width=2)
    for b in sc.obbs():
        c, sn = math.cos(b.yaw), math.sin(b.yaw)
        hx, hy = max(b.half[0], 0.15), max(b.half[1], 0.15)
        pts = [(b.center[0] + c * ax * hx - sn * ay * hy, b.center[1] + sn * ax * hx + c * ay * hy) for ax, ay in ((1, 1), (1, -1), (-1, -1), (-1, 1))]
        d.polygon([p(*q) for q in pts], fill=(120, 170, 220) if b.see_through else (230, 140, 60))
    if len(track) > 1:
        d.line([p(x, y) for x, y, _ in track[: k + 1]], fill=(250, 250, 250), width=2)
    x, y, yaw = track[k]
    d.polygon([p(x + 1.8 * math.cos(yaw), y + 1.8 * math.sin(yaw)), p(x + 0.9 * math.cos(yaw + 2.5), y + 0.9 * math.sin(yaw + 2.5)),
               p(x + 0.9 * math.cos(yaw - 2.5), y + 0.9 * math.sin(yaw - 2.5))], fill=(255, 80, 80))
    d.text((4, 1), "top view (orange obstacle, blue glass, green goal)", fill=(255, 255, 255))
    return im


CHASE_BACK_M, CHASE_UP_M, CHASE_AHEAD_M, CHASE_HFOV_DEG = 9.0, 4.0, 4.0, 80.0  # third-person camera: behind and above the drone
CAM_HFOV_DEG = 90.0  # the flight camera (perception.depth_utils.CAMERA_HFOV_DEG)


def _map_points(recs: list[dict[str, Any]], every: int = 3, stride: int = 6) -> list[tuple[int, np.ndarray]]:
    """Surfaces the true depth saw, as world points (x, y, height), per frame: (frame index, N x 3). Viewer only.

    Pixels become rays of the flight camera (distance along the ray, as both backends give it); points within
    0.4 m of the start's ground level are dropped, so the floor does not fill the picture."""
    out = []
    for k in range(0, len(recs), every):
        path = recs[k].get("depth_gt_path")
        if not path or not os.path.exists(path):
            continue
        d = np.load(path).astype(np.float32)[::stride, ::stride]
        h, w = d.shape
        f = (w / 2) / math.tan(math.radians(CAM_HFOV_DEG / 2))
        xs, ys = (np.arange(w) - w / 2 + 0.5) / f, (np.arange(h) - h / 2 + 0.5) / f
        gx, gy = np.meshgrid(xs, ys)
        n = np.sqrt(1 + gx**2 + gy**2)
        x, y, z, yaw = recs[k]["pose"]
        fwd, right = np.array([math.cos(yaw), math.sin(yaw)]), np.array([-math.sin(yaw), math.cos(yaw)])
        dist = np.where(np.isfinite(d) & (d > 0.3) & (d < 60.0), d, np.nan)
        px = x + dist * (fwd[0] + gx * right[0]) / n
        py = y + dist * (fwd[1] + gx * right[1]) / n
        ph = -z + dist * (-gy) / n
        pts = np.stack([px, py, ph], -1).reshape(-1, 3)
        pts = pts[np.isfinite(pts).all(1) & (pts[:, 2] > 0.4)]
        out.append((k, pts.astype(np.float32)))
    return out


def _chase_camera(recs: list[dict[str, Any]], k: int, yaw: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Camera position and its (right, up, forward) axes for frame k, behind the drone along the smoothed heading."""
    x, y, z, _ = recs[k]["pose"]
    drone = np.array([x, y, -z])
    c = drone + np.array([-CHASE_BACK_M * math.cos(yaw), -CHASE_BACK_M * math.sin(yaw), CHASE_UP_M])
    target = drone + np.array([CHASE_AHEAD_M * math.cos(yaw), CHASE_AHEAD_M * math.sin(yaw), 0.0])
    f = (target - c) / np.linalg.norm(target - c)
    r = np.array([-f[1], f[0], 0.0]) / max(math.hypot(f[0], f[1]), 1e-6)
    u = np.array([0.0, 0.0, 1.0]) - f[2] * f
    return c, r, u / np.linalg.norm(u), f


def _project(pts: np.ndarray, cam: tuple, size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pixel columns, rows and depth along the viewing axis for world points (N x 3); depth <= 0.3 is behind the camera."""
    c, r, u, f = cam
    w, h = size
    v = pts - c
    zc = v @ f
    foc = (w / 2) / math.tan(math.radians(CHASE_HFOV_DEG / 2))
    zs = np.maximum(zc, 1e-3)
    return w / 2 + foc * (v @ r) / zs, h / 2 - foc * (v @ u) / zs, zc


def _chase_tile(scenario: dict[str, Any], recs: list[dict[str, Any]], k: int, yaws: list[float],
                map_pts: list[tuple[int, np.ndarray]], size: tuple[int, int]) -> Image.Image:
    """Third person: the drone from behind and above, its path so far, the obstacles, the goal and what the depth has seen."""
    from sim.scenario import Scenario

    sc = Scenario.from_dict(scenario)
    w, h = size
    cam = _chase_camera(recs, k, yaws[k])
    img = np.zeros((h, w, 3), np.uint8)
    img[:] = (24, 26, 30)
    img[h // 2:] = (30, 33, 38)  # a dim ground half; the horizon is only roughly right when the camera pitches
    seen = [p for i, p in map_pts if i <= k]
    if seen:
        pts = np.concatenate(seen)
        x, y, z, _ = recs[k]["pose"]
        pts = pts[(np.abs(pts[:, 0] - x) < 70) & (np.abs(pts[:, 1] - y) < 70)]
        u, v, zc = _project(pts, cam, size)
        ok = (zc > 0.5) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
        shade = np.clip(215 - 4.0 * zc[ok], 60, 215).astype(np.uint8)
        img[v[ok].astype(int), u[ok].astype(int)] = np.stack([shade, shade, np.clip(shade + 20, 0, 255).astype(np.uint8)], -1)
        for dx, dy in ((1, 0), (0, 1)):  # thicken the dots by one pixel
            img[np.clip(v[ok].astype(int) + dy, 0, h - 1), np.clip(u[ok].astype(int) + dx, 0, w - 1)] = np.stack(
                [shade, shade, np.clip(shade + 20, 0, 255).astype(np.uint8)], -1)
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im)

    def line(a: np.ndarray, b: np.ndarray, fill: tuple, width: int = 1) -> None:
        u, v, zc = _project(np.stack([a, b]), cam, size)
        if zc.min() > 0.5:
            d.line([(u[0], v[0]), (u[1], v[1])], fill=fill, width=width)

    for gx in range(-20, 100, 10):  # ground grid at the start's level, 10 m cells
        line(np.array([gx, -40.0, 0.0]), np.array([gx, 40.0, 0.0]), (45, 50, 58))
    for gy in range(-40, 41, 10):
        line(np.array([-20.0, gy, 0.0]), np.array([100.0, gy, 0.0]), (45, 50, 58))
    for b in sc.obbs():  # obstacles as boxes (a wire is a long thin box)
        c, sn = math.cos(b.yaw), math.sin(b.yaw)
        hx, hy, hz = max(b.half[0], 0.08), max(b.half[1], 0.08), max(b.half[2], 0.08)
        corners = {}
        for sx in (-1, 1):
            for sy in (-1, 1):
                for sz in (-1, 1):
                    corners[(sx, sy, sz)] = np.array([b.center[0] + c * sx * hx - sn * sy * hy, b.center[1] + sn * sx * hx + c * sy * hy,
                                                      -b.center[2] + sz * hz])  # centre z is NED: height = -z
        col = (120, 170, 220) if b.see_through else (240, 150, 60)
        for a_, b_ in ((a, b) for a in corners for b in corners if a < b and sum(i != j for i, j in zip(a, b)) == 1):
            line(corners[a_], corners[b_], col, 2)
    if sc.goal_xy is not None:
        gx, gy = sc.goal_xy
        line(np.array([gx, gy, 0.0]), np.array([gx, gy, 6.0]), (90, 220, 110), 3)
        for a in range(0, 360, 30):
            a0, a1 = math.radians(a), math.radians(a + 30)
            line(np.array([gx + 2 * math.cos(a0), gy + 2 * math.sin(a0), 0.0]), np.array([gx + 2 * math.cos(a1), gy + 2 * math.sin(a1), 0.0]), (90, 220, 110), 2)
    for i in range(max(0, k - 400), k):  # path flown so far
        line(np.array([recs[i]["pose"][0], recs[i]["pose"][1], -recs[i]["pose"][2]]),
             np.array([recs[i + 1]["pose"][0], recs[i + 1]["pose"][1], -recs[i + 1]["pose"][2]]), (250, 250, 250))
    x, y, z, yaw = recs[k]["pose"]  # the drone: a cross along and across its heading, nose in red
    pos = np.array([x, y, -z])
    fwd, side = np.array([math.cos(yaw), math.sin(yaw), 0.0]), np.array([-math.sin(yaw), math.cos(yaw), 0.0])
    line(pos - 0.6 * side, pos + 0.6 * side, (230, 230, 230), 3)
    line(pos - 0.6 * fwd, pos + 0.9 * fwd, (255, 80, 80), 3)
    line(pos, np.array([x, y, 0.0]), (110, 110, 130))  # drop line to the ground: height
    d.rectangle([0, 0, 330, 14], fill=(0, 0, 0))
    d.text((4, 1), "third person (camera behind the drone; dots = what depth has seen)", fill=(255, 255, 255))
    return im


def _smooth_yaws(recs: list[dict[str, Any]], alpha: float = 0.15) -> list[float]:
    """Heading for the chase camera, low-passed so a turn in place does not spin the picture."""
    out, cur = [], recs[0]["pose"][3]
    for r in recs:
        cur += alpha * ((r["pose"][3] - cur + math.pi) % (2 * math.pi) - math.pi)
        out.append(cur)
    return out


def compose(rec: dict[str, Any], scenario: dict[str, Any], track: list, k: int, phase: str, vlm: str, alt: float,
            chase: Image.Image | None = None) -> Image.Image:
    rgb = np.asarray(Image.open(rec["rgb_path"]).convert("RGB"))
    rgb = draw_overlay(rgb, corridor_box(rgb.shape), fill_alpha=0.0)
    est = np.load(rec["depth_est_path"]).astype(np.float32)
    gt = np.load(rec["depth_gt_path"]).astype(np.float32)
    rows = 3 if chase is not None else 2
    frame = Image.new("RGB", (2 * TILE_W, rows * TILE_H + BAR_H), (10, 10, 12))
    frame.paste(_tile(rgb, "camera (red: corridor window)"), (0, 0))
    frame.paste(_tile(depth_rgb(est), "fast stack depth (what the drone believes)"), (TILE_W, 0))
    frame.paste(_tile(depth_rgb(gt), "true depth (viewer only)"), (0, TILE_H))
    frame.paste(_map_tile(scenario, track, k), (TILE_W, TILE_H))
    if chase is not None:
        frame.paste(chase, (0, 2 * TILE_H))
    d = ImageDraw.Draw(frame)
    act = rec["action_taken"]
    line1 = (f"t {rec['t']:5.1f} s   height {alt:4.1f} m   phase {phase:<8}   action {act:<9}   "
             f"corridor: estimate {rec['corridor_min_est']:5.1f} m, true {rec['corridor_min_gt']:5.1f} m")
    d.text((8, rows * TILE_H + 6), line1, fill=(235, 235, 235))
    d.text((8, rows * TILE_H + 26), vlm, fill=(255, 210, 90) if vlm else (150, 150, 150))
    return frame


def render(folder: Path, fps: int = 10) -> Path:
    recs = [json.loads(line) for line in (folder / "flight.jsonl").read_text().splitlines() if line.strip()]
    result = json.loads((folder / "result.json").read_text())
    scenario = json.loads((folder / "scenario.json").read_text())
    extra = result.get("extra", {})
    phases = extra.get("phases") or []
    directions = list(extra.get("vlm_directions") or [])
    track = [(r["pose"][0], r["pose"][1], r["pose"][3]) for r in recs]
    yaws, map_pts = _smooth_yaws(recs), _map_points(recs)
    frames_dir = folder / "video_frames"
    frames_dir.mkdir(exist_ok=True)
    n, ask_i, last_vlm = 0, 0, ""
    for k, rec in enumerate(recs):
        phase = "-"
        for t, ph in phases:
            if t <= rec["t"] + 1e-6:
                phase = ph
        repeat = 1
        if rec["asked"]:
            direction = directions[ask_i] if ask_i < len(directions) else None
            ask_i += 1
            ans = "YES" if rec["vlm_answer"] else "no"
            last_vlm = f"VLM asked at t {rec['t']:.1f} s: {ans}" + (f", go {direction}" if direction else "") + \
                       f"   (drone holds {rec['delay_charged_s']:.1f} s)"
            repeat = 1 + int(round(rec["delay_charged_s"] * fps))  # the pause the call costs
        img = compose(rec, scenario, track, k, phase, last_vlm, -rec["pose"][2],
                      _chase_tile(scenario, recs, k, yaws, map_pts, (2 * TILE_W, TILE_H)))
        for _ in range(repeat):
            img.save(frames_dir / f"{n:05d}.png")
            n += 1
    end = Image.open(frames_dir / f"{n - 1:05d}.png")
    ImageDraw.Draw(end).text((8, 3 * TILE_H + 40), f"END: {extra.get('outcome') or ('crashed' if result['crashed'] else 'reached goal' if result['reached_goal'] else 'stopped')}"
                             f"   VLM calls {result['n_calls']}   time {result['mission_time_s']:.1f} s", fill=(255, 120, 120))
    for _ in range(2 * fps):
        end.save(frames_dir / f"{n:05d}.png")
        n += 1
    if shutil.which("ffmpeg"):
        out = folder / "flight.mp4"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps), "-i", str(frames_dir / "%05d.png"),
                        "-pix_fmt", "yuv420p", "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", str(out)], check=True)
    else:
        out = folder / "flight.gif"
        imgs = [Image.open(frames_dir / f"{i:05d}.png") for i in range(0, n, 2)]
        imgs[0].save(out, save_all=True, append_images=imgs[1:], duration=int(2000 / fps), loop=0)
    return out


def fly(cfg: dict[str, Any], seed: int, policy_name: str, folder: Path) -> None:
    from control.rules import RuleController
    from eval.factory import fork_judge, make_judge, make_sim, make_stack
    from eval.run_episode import LoopConfig, run_episode
    from eval.sweep import resolve_scenario
    from policies.base import make_policy
    from sim.scenario import generate

    scen_cfg, frozen_starts = resolve_scenario(cfg)
    backend_kw = {k: (os.path.expandvars(v) if isinstance(v, str) else v) for k, v in cfg.get("backend_kw", {}).items()}
    if frozen_starts and cfg.get("backend") == "airsim":
        backend_kw["starts"] = frozen_starts
    spec = next((dict(p) for p in cfg["policies"] if p["name"] == policy_name), None)
    if spec is None:
        raise SystemExit(f"policy {policy_name!r} is not in the config; choose one of {[p['name'] for p in cfg['policies']]}")
    spec.pop("name")
    policy = make_policy(spec)
    policy.name = policy_name
    sim = make_sim(cfg.get("backend", "raycast"), **backend_kw)
    stack = make_stack(cfg.get("stack", "true-depth"), **cfg.get("stack_kw", {}))
    judge, judge_name = make_judge(cfg.get("judge", "none"), **cfg.get("judge_kw", {}))
    loop = LoopConfig(delay_s=cfg.get("delay_s", 1.0), **{**cfg.get("loop", {}), "save_dir": str(folder / "raw")})
    scenario = generate(seed, cfg.get("split", "val"), scen_cfg)
    try:
        records, result = run_episode(scenario, sim, stack, RuleController(**{"d_safe": loop.d_safe, **cfg.get("controller", {})}),
                                      policy, fork_judge(judge, seed, 0), loop, judge_name=judge_name)
    finally:
        if hasattr(sim, "close"):
            sim.close()
    (folder / "flight.jsonl").write_text("".join(r.to_json() + "\n" for r in records))
    (folder / "result.json").write_text(result.to_json())
    (folder / "scenario.json").write_text(scenario.to_json())
    print(result.to_json(), flush=True)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--config", help="sweep config (simulator, stack, judge, controller, scenario)")
    p.add_argument("--seed", type=int)
    p.add_argument("--policy", default="never", help="a policy name from the config")
    p.add_argument("--out", required=True, help="folder for frames, logs and the video")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", dest="overrides",
                   help="override a config key, dotted for nested ones, e.g. --set stack=true-depth")
    p.add_argument("--render-only", action="store_true", help="rebuild the video from --out without flying")
    p.add_argument("--fps", type=int, default=10, help="10 = real time at dt 0.1 s")
    a = p.parse_args(argv)
    folder = Path(a.out)
    folder.mkdir(parents=True, exist_ok=True)
    if not a.render_only:
        if a.config is None or a.seed is None:
            raise SystemExit("--config and --seed are needed to fly")
        from eval.sweep import apply_overrides
        fly(apply_overrides(yaml.safe_load(Path(a.config).read_text()), a.overrides), a.seed, a.policy, folder)
    print(f"video: {render(folder, a.fps)}", flush=True)


if __name__ == "__main__":
    main()
