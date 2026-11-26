"""Flight video on the raycast backend: one short cone flight, frames composed, video written."""
from __future__ import annotations

import yaml
from PIL import Image

from eval.flight_video import main


def test_flight_video_flies_and_renders(tmp_path):
    cfg = yaml.safe_load(open("configs/sweeps/raycast_cone_smoke.yaml"))
    cfg["backend_kw"] = {"width": 80, "height": 45}
    cfg["loop"] = {"t_max": 3.0}
    path = tmp_path / "c.yaml"
    path.write_text(yaml.safe_dump(cfg))
    main(["--config", str(path), "--seed", "101000", "--policy", "always", "--out", str(tmp_path / "v")])
    out = list((tmp_path / "v").glob("flight.*"))
    assert any(p.suffix in (".mp4", ".gif") and p.stat().st_size > 0 for p in out)
    n_frames = len(list((tmp_path / "v" / "video_frames").glob("*.png")))
    assert n_frames > 30  # 3 s of flight plus the held frames of each VLM call and the end card
    frame = Image.open(tmp_path / "v" / "video_frames" / "00000.png")
    assert frame.height > 3 * 225  # camera, depth, true depth, top view and the third-person tile (3 rows) plus the status bar
    chase = frame.crop((0, 2 * 225, 800, 3 * 225))
    assert len({chase.getpixel((x, y)) for x in range(0, 800, 7) for y in range(0, 225, 7)}) > 3  # something is drawn in it
