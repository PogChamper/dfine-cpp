#!/usr/bin/env python3
"""Render the README side-by-side throughput demo (gif + mp4), headless.

Each panel plays back the same image stream over the same (slowed-down) wall
clock and advances at ITS backend's measured throughput (--left-fps /
--right-fps, or N repeatable --panel LABEL:FPS specs — taken from the
benchmark table; this script does not measure). The timeline runs at
--slowmo x slow motion: at real speed the backends exceed the display frame
rate and the panels would look identical; slowed 10x, the left panel visibly
lingers on each frame while the right one streams. Boxes come from a single
detection pass with the C++ runtime via the Python bindings, so every panel
shows identical, real detections; only the playback rate differs. The footer
states exactly that.

usage:
  LD_LIBRARY_PATH=<tensorrt_libs> python make_demo_gif.py \
      --engine trt-files/engines/dfine_m_fp16_st.engine \
      --images-dir /mnt/d/datasets/coco/val2017 --limit 400 \
      --left-fps 31 --right-fps 272 \
      --left-label "PyTorch FP32" --right-label "D-FINE-cpp FP16" \
      --seconds 8 --out-dir /tmp/demo
  # 3+ panels: --panel "PyTorch FP32:66" --panel "surgical+slim:533" --panel "fast:598"

Requires: pillow, numpy, ffmpeg on PATH, the dfine Python package importable
(PYTHONPATH=python) and libdfine.so discoverable (DFINE_LIBRARY or build/).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

PANEL_W, PANEL_H = 600, 440
HEADER_H, FOOTER_H = 34, 46
DISPLAY_FPS = 15

# Fixed per-class colors (hash-based, stable across frames).
def class_color(cid: int) -> tuple[int, int, int]:
    rng = np.random.default_rng(cid * 9973 + 7)
    r, g, b = rng.integers(80, 255, 3)
    return int(r), int(g), int(b)


def load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for cand in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        if Path(cand).is_file():
            return ImageFont.truetype(cand, size)
    return ImageFont.load_default()


def detect_all(engine: str, images: list[Path], threshold: float):
    from dfine import Detector

    results = []
    with Detector(engine, threshold=threshold) as det:
        for p in images:
            img = np.asarray(Image.open(p).convert("RGB"))
            results.append((img, det.detect(img)))
    return results


def draw_panel(img: np.ndarray, dets, label: str, fps: float, processed: int,
               font, font_small) -> Image.Image:
    im = Image.fromarray(img).convert("RGB")
    # letterbox into the fixed panel (display only; detection ran on the original)
    scale = min(PANEL_W / im.width, PANEL_H / im.height)
    nw, nh = int(im.width * scale), int(im.height * scale)
    im = im.resize((nw, nh), Image.BILINEAR)
    d = ImageDraw.Draw(im)
    for det in dets:
        x1, y1, x2, y2 = (v * scale for v in det.box.as_tuple())
        color = class_color(det.class_id)
        d.rectangle([x1, y1, x2, y2], outline=color, width=2)
        d.text((x1 + 2, max(0, y1 - 14)), f"{det.class_name} {det.score:.2f}",
               fill=color, font=font_small)
    panel = Image.new("RGB", (PANEL_W, PANEL_H + HEADER_H), (18, 18, 18))
    panel.paste(im, ((PANEL_W - nw) // 2, HEADER_H + (PANEL_H - nh) // 2))
    hd = ImageDraw.Draw(panel)
    hd.text((10, 7), f"{label} — {fps:g} FPS", fill=(240, 240, 240), font=font)
    txt = f"{processed} frames"
    hd.text((PANEL_W - 10 - hd.textlength(txt, font=font), 7), txt,
            fill=(160, 220, 160), font=font)
    return panel


def extract_video_frames(video: str, out_dir: Path, limit: int) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    import os
    env = {k: v for k, v in os.environ.items() if k != "LD_LIBRARY_PATH"}
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", video,
                    "-frames:v", str(limit), str(out_dir / "s%05d.png")], check=True, env=env)
    return sorted(out_dir.glob("s*.png"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True)
    ap.add_argument("--images-dir", help="image-sequence source (slideshow mode)")
    ap.add_argument("--video", help="video source: frames are extracted in order, so the two "
                                    "panels race through the SAME clip at their throughputs")
    ap.add_argument("--credit", default="", help="source credit appended to the caption")
    ap.add_argument("--limit", type=int, default=400)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--left-fps", type=float)
    ap.add_argument("--right-fps", type=float)
    ap.add_argument("--left-label", default="PyTorch FP32")
    ap.add_argument("--right-label", default="D-FINE-cpp FP16")
    ap.add_argument("--panel", action="append", metavar="LABEL:FPS",
                    help="repeatable panel spec, left to right; overrides --left-*/--right-*")
    ap.add_argument("--seconds", type=float, default=8.0, help="output clip duration")
    ap.add_argument("--slowmo", type=float, default=10.0,
                    help="slow-motion factor (simulated wall time = seconds / slowmo)")
    ap.add_argument("--gpu-name", default="RTX 4070 Ti SUPER")
    ap.add_argument("--model-name", default="D-FINE-M", help="model name shown in the caption")
    ap.add_argument("--out-dir", default="demo_out")
    args = ap.parse_args()

    if args.panel:
        panels = []
        for spec in args.panel:
            label, sep, fps_str = spec.rpartition(":")
            try:
                fps = float(fps_str)
            except ValueError:
                sep = ""
            if not sep or not label:
                print(f"bad --panel spec {spec!r} (want LABEL:FPS)", file=sys.stderr)
                return 1
            panels.append((label, fps))
        if len(panels) < 2:
            print("need at least two --panel LABEL:FPS specs", file=sys.stderr)
            return 1
    elif args.left_fps is not None and args.right_fps is not None:
        panels = [(args.left_label, args.left_fps), (args.right_label, args.right_fps)]
    else:
        print("need --left-fps and --right-fps, or 2+ --panel specs", file=sys.stderr)
        return 1

    out = Path(args.out_dir)
    (out / "frames").mkdir(parents=True, exist_ok=True)
    if args.video:
        images = extract_video_frames(args.video, out / "src_frames", args.limit)
    elif args.images_dir:
        images = sorted(Path(args.images_dir).glob("*.jpg"))[: args.limit]
    else:
        print("need --video or --images-dir", file=sys.stderr)
        return 1
    if not images:
        print("no source frames found", file=sys.stderr)
        return 1

    print(f"[demo] detecting on {len(images)} images ...")
    data = detect_all(args.engine, images, args.threshold)

    font = load_font(16)
    font_small = load_font(11)
    n_frames = int(args.seconds * DISPLAY_FPS)
    W = PANEL_W * len(panels) + 12 * (len(panels) - 1)
    H = PANEL_H + HEADER_H + FOOTER_H
    src = "video" if args.video else "COCO val2017"
    caption = (f"{args.slowmo:g}x slow motion · {src} · {args.model_name} · {args.gpu_name} · "
               f"identical detections all panels (C++ runtime); frame counters advance at each "
               f"backend's measured e2e throughput"
               + (f" · {args.credit}" if args.credit else ""))

    print(f"[demo] rendering {n_frames} frames ...")
    for f in range(n_frames):
        t = f / DISPLAY_FPS / args.slowmo  # simulated wall-clock seconds
        canvas = Image.new("RGB", (W, H), (10, 10, 10))
        for side, (label, fps) in enumerate(panels):
            processed = int(t * fps)
            img, dets = data[processed % len(data)]
            panel = draw_panel(img, dets, label, fps, processed, font, font_small)
            canvas.paste(panel, (side * (PANEL_W + 12), 0))
        cd = ImageDraw.Draw(canvas)
        cd.text((10, PANEL_H + HEADER_H + 14), caption, fill=(140, 140, 140), font=font_small)
        canvas.save(out / "frames" / f"f{f:05d}.png")

    mp4 = out / "demo.mp4"
    gif = out / "demo.gif"
    palette = out / "palette.png"
    fr = str(DISPLAY_FPS)
    # System ffmpeg must not inherit LD_LIBRARY_PATH (a conda/TensorRT lib dir
    # shadows its shared libraries and it fails to start).
    import os
    env = {k: v for k, v in os.environ.items() if k != "LD_LIBRARY_PATH"}
    def ff(args_):
        subprocess.run(["ffmpeg", "-y", "-loglevel", "warning", *args_], check=True, env=env)

    gif_w = 420 * len(panels)  # 420 px per panel keeps the 2-panel gif at its historical 840
    seq = ["-framerate", fr, "-i", str(out / "frames" / "f%05d.png")]
    ff([*seq, "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "24", str(mp4)])
    ff([*seq, "-vf", f"fps=12,scale={gif_w}:-1:flags=lanczos,palettegen",
        "-frames:v", "1", "-update", "1", str(palette)])
    ff([*seq, "-i", str(palette),
        "-lavfi", f"fps=12,scale={gif_w}:-1:flags=lanczos[x];[x][1:v]paletteuse", str(gif)])
    for p in (mp4, gif):
        print(f"[demo] {p}  {p.stat().st_size / 1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
