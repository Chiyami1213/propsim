"""README gallery: one screenshot per registered model (developer tool, not a runtime dependency).

    python tools/make_screenshots.py                 # windowed (needs a display), all models
    python tools/make_screenshots.py --offscreen     # headless Qt platform
    python tools/make_screenshots.py --only solid.grain,ep.ion

Opens the main window at 1600 x 900, loads the model's demo case from cases/ when one exists (otherwise the model's
defaults), lets Live models render or runs Run models to completion and seeks to 60 % of the run, then saves
docs/gallery/NN-model-name.png (at most 1600 px wide, optimized to stay under about 600 KB).  The 3D view is checked
after capture: if it came out black, the plotter's own render is composited in its place.  Uses a temporary data
folder, so your saved runs are not touched.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import re
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "gallery"
MAX_W, MAX_BYTES = 1600, 600_000


def slug(text: str) -> str:
    text = text.replace("₂", "2").replace("&", "and").replace("*", "-star")
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def demo_cases() -> dict[str, Path]:
    cases = {}
    for f in sorted((ROOT / "cases").glob("*.psim.json")):
        cases.setdefault(json.loads(f.read_text(encoding="utf-8"))["model"], f)
    return cases


def models() -> list[tuple[str, str]]:
    from propsim.modules import registry
    return [(cls.key, label) for entries in registry.TREE.values() for label, mode, cls in entries
            if cls is not None and mode != "tool"]       # tools (the cycle sandbox) are captured separately


def save_png(qimage, path: Path) -> int:
    from PIL import Image
    from PySide6.QtCore import QBuffer, QIODevice
    buf = QBuffer()
    buf.open(QIODevice.WriteOnly)
    qimage.save(buf, "PNG")
    img = Image.open(io.BytesIO(bytes(buf.data()))).convert("RGB")
    if img.width > MAX_W:
        img = img.resize((MAX_W, round(img.height * MAX_W / img.width)), Image.LANCZOS)
    img.save(path, optimize=True)
    if path.stat().st_size > MAX_BYTES:                      # palette PNG, no dithering: text stays crisp
        img.quantize(256, method=Image.Quantize.FASTOCTREE, dither=Image.Dither.NONE).save(path, optimize=True)
    return path.stat().st_size


def fix_black_3d(win, img):
    """If the 3D region of the window grab is black, paint the plotter's own render over it."""
    import numpy as np
    from PySide6.QtCore import QPoint, QRect
    from PySide6.QtGui import QImage, QPainter
    vp = win.vp
    host = getattr(vp, "host", None) or vp
    tl = host.mapTo(win, QPoint(0, 0))
    dpr = img.devicePixelRatio() or 1.0
    rect = QRect(round(tl.x() * dpr), round(tl.y() * dpr), round(host.width() * dpr), round(host.height() * dpr))
    region = img.copy(rect).convertToFormat(QImage.Format_RGB888)
    arr = np.frombuffer(region.constBits(), np.uint8, region.sizeInBytes())
    if arr.size and arr.mean() > 4.0:
        return img, False
    shot = np.ascontiguousarray(vp.plotter.screenshot(return_img=True, window_size=[rect.width(), rect.height()]))
    h, w = shot.shape[:2]
    qi = QImage(shot.data, w, h, 3 * w, QImage.Format_RGB888).copy()
    out = img.copy()
    out.setDevicePixelRatio(1.0)
    p = QPainter(out)
    p.drawImage(rect, qi)
    p.end()
    return out, True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--offscreen", action="store_true", help="use the offscreen Qt platform")
    ap.add_argument("--only", default="", help="comma-separated model keys")
    ap.add_argument("--seek", type=float, default=0.6, help="Run models: fraction of the run to show")
    args = ap.parse_args()
    if args.offscreen:
        os.environ["QT_QPA_PLATFORM"] = "offscreen"
    sys.path.insert(0, str(ROOT))
    from PySide6.QtWidgets import QApplication
    from propsim.ui import worker as W
    from propsim.ui.main_window import MainWindow

    OUT.mkdir(parents=True, exist_ok=True)
    app = QApplication.instance() or QApplication(sys.argv)
    win = MainWindow(home=Path(tempfile.mkdtemp(prefix="propsim_shots_")), use_settings=False)
    win.resize(1600, 900)
    win.show()
    cases = demo_cases()
    wanted = {k for k in args.only.split(",") if k}

    def pump(seconds: float) -> None:
        t0 = time.time()
        while time.time() - t0 < seconds:
            app.processEvents()

    pump(1.0)
    win.log.clear()                                     # no local paths (user name) in published images
    manual = []
    for i, (key, label) in enumerate(models(), 1):
        if wanted and key not in wanted:
            continue
        name = f"{i:02d}-{slug(label)}.png"
        try:
            if key in cases:
                win.open_case(str(cases[key]))
            else:
                win.select_model(key)
            pump(1.5)
            if win.ctrl.mode == "run":
                win.start_run()
                t0 = time.time()
                while win.worker is not None and time.time() - t0 < 600:
                    app.processEvents()
                if win.result is None:
                    raise RuntimeError("run produced no result")
                win.playback.pause()
                win.playback.seek_time(args.seek * float(win.result.t[-1]))
                pump(2.0)
            img, fixed = fix_black_3d(win, win.grab().toImage())
            size = save_png(img, OUT / name)
            print(f"{name:48s} {size / 1024:6.0f} KB{'  (3D composited from the plotter)' if fixed else ''}")
        except Exception as e:                                           # noqa: BLE001 - report, keep going
            manual.append((name, key, str(e)))
            print(f"{name:48s} FAILED: {e}")
    if not wanted or "liquid.sandbox" in wanted:                           # the cycle sandbox is its own window
        try:
            win.open_cycle_sandbox()
            box = win.sandbox
            box.resize(1500, 900)
            pump(2.5)
            size = save_png(box.grab().toImage(), OUT / "cycle-sandbox.png")
            print(f"{'cycle-sandbox.png':48s} {size / 1024:6.0f} KB")
            box.close()
        except Exception as e:                                           # noqa: BLE001
            manual.append(("cycle-sandbox.png", "liquid.sandbox", str(e)))
            print(f"{'cycle-sandbox.png':48s} FAILED: {e}")
    for mode in ("solid", "hybrid"):                                       # the grain designer window, per motor type
        if wanted and f"grain.{mode}" not in wanted:
            continue
        try:
            win.open_grain_designer(mode)
            gw = win.grain_win
            gw.resize(1500, 900)
            pump(3.0)
            img, _ = fix_black_3d(gw, gw.grab().toImage())
            size = save_png(img, OUT / f"grain-designer-{mode}.png")
            print(f"{'grain-designer-' + mode + '.png':48s} {size / 1024:6.0f} KB")
        except Exception as e:                                           # noqa: BLE001
            manual.append((f"grain-designer-{mode}.png", f"grain.{mode}", str(e)))
            print(f"{'grain-designer-' + mode + '.png':48s} FAILED: {e}")
    win.close()
    W.shutdown()
    if manual:
        print("\nCapture by hand (Win+Shift+S), window 1600 x 900:")
        for name, key, why in manual:
            print(f"  {name}: select '{key}' in the Modules tree ({why})")
    return 1 if manual else 0


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    raise SystemExit(main())
