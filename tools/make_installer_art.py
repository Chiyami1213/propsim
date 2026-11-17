"""Wizard artwork for the Windows installer (Inno Setup, modern wizard style), at every DPI scale Inno Setup picks from.

Writes build/installer/side_<pct>.bmp (welcome / finish banner, 164 x 314 at 100 %) and small_<pct>.bmp (inner-page
header icon, 55 x 55 at 100 %).  Drawn supersampled, then reduced with a Lanczos filter, so edges stay crisp.
Run: python tools/make_installer_art.py
"""
from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "build" / "installer"
ICON = ROOT / "propsim" / "assets" / "icon.png"
SCALES = [100, 125, 150, 175, 200, 225, 250]
SIDE = (164, 314)
SMALL = (55, 55)
SS = 4                                                   # supersampling factor
FONTS = Path("C:/Windows/Fonts")


def _font(names: list[str], size: int):
    for n in names:
        f = FONTS / n
        if f.exists():
            return ImageFont.truetype(str(f), size)
    return ImageFont.load_default()


def _lerp(a, b, t):
    return tuple(int(round(a[i] + (b[i] - a[i]) * t)) for i in range(3))


def side_image(scale: int, version: str) -> Image.Image:
    w, h = round(SIDE[0] * scale / 100), round(SIDE[1] * scale / 100)
    W, H = w * SS, h * SS
    top, bottom = (13, 20, 34), (22, 52, 98)
    img = Image.new("RGB", (W, H))
    px = ImageDraw.Draw(img)
    for y in range(H):
        px.line([(0, y), (W, y)], fill=_lerp(top, bottom, (y / H) ** 1.3))
    # plot-style grid (major + minor), very faint
    grid = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    g = ImageDraw.Draw(grid)
    step = W // 8
    for k in range(0, max(W, H) // step + 2):
        a = 22 if k % 2 == 0 else 10
        g.line([(k * step, 0), (k * step, H)], fill=(160, 190, 230, a), width=SS)
        g.line([(0, k * step), (W, k * step)], fill=(160, 190, 230, a), width=SS)
    img = Image.alpha_composite(img.convert("RGBA"), grid)
    # exhaust plume rising from the bottom: warm glow + core
    plume = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    pd = ImageDraw.Draw(plume)
    cx = W // 2
    for i in range(90):                                  # from the throat down through the bell and beyond
        t = i / 89
        yy = H * 0.665 + t * H * 0.36
        half = W * (0.035 + 0.24 * t ** 1.1)
        col = _lerp((255, 236, 170), (255, 84, 52), t ** 0.8) + (int(235 * (1 - 0.55 * t)),)
        pd.ellipse([cx - half, yy - half * 0.30, cx + half, yy + half * 0.30], fill=col)
    plume = plume.filter(ImageFilter.GaussianBlur(W * 0.035))
    img = Image.alpha_composite(img, plume)
    # nozzle contour strokes (icon motif), vertical: chamber at the top of the plume
    nz = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    nd = ImageDraw.Draw(nz)
    import math
    y0, y1 = H * 0.60, H * 0.98
    for s in (-1, 1):
        pts = []
        for i in range(80):
            t = i / 79
            y = y0 + (y1 - y0) * t
            r = W * (0.13 - 0.07 * math.exp(-((t - 0.22) / 0.12) ** 2) + 0.20 * max(t - 0.22, 0) ** 1.1)
            pts.append((cx + s * r, y))
        nd.line(pts, fill=(78, 161, 255, 255), width=int(W * 0.035), joint="curve")
        nd.line(pts, fill=(190, 222, 255, 200), width=int(W * 0.012), joint="curve")
    img = Image.alpha_composite(img, nz)
    # app icon and title
    ic = Image.open(ICON).convert("RGBA")
    isz = int(W * 0.46)
    ic = ic.resize((isz, isz), Image.LANCZOS)
    shadow = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    ImageDraw.Draw(shadow).rounded_rectangle([cx - isz // 2, int(H * 0.085) + SS * 6, cx + isz // 2,
                                              int(H * 0.085) + isz + SS * 6], radius=isz // 5, fill=(0, 0, 0, 120))
    img = Image.alpha_composite(img, shadow.filter(ImageFilter.GaussianBlur(SS * 6)))
    img.alpha_composite(ic, (cx - isz // 2, int(H * 0.085)))
    d = ImageDraw.Draw(img)
    f_title = _font(["segoeuib.ttf", "seguisb.ttf", "arialbd.ttf"], int(W * 0.155))
    f_sub = _font(["segoeui.ttf", "arial.ttf"], int(W * 0.066))
    ty = int(H * 0.085) + isz + int(H * 0.035)
    d.text((cx, ty), "PropSim", font=f_title, fill=(236, 241, 248), anchor="mt")
    ty += int(W * 0.20)
    for line in ("Rocket Propulsion", "Simulator"):
        d.text((cx, ty), line, font=f_sub, fill=(150, 170, 198), anchor="mt")
        ty += int(W * 0.085)
    d.text((cx, ty + int(W * 0.03)), f"v{version}", font=f_sub, fill=(110, 135, 170), anchor="mt")
    return img.convert("RGB").resize((w, h), Image.LANCZOS)


def small_image(scale: int) -> Image.Image:
    w, h = round(SMALL[0] * scale / 100), round(SMALL[1] * scale / 100)
    bg = Image.new("RGBA", (w * SS, h * SS), (255, 255, 255, 255))
    ic = Image.open(ICON).convert("RGBA").resize((w * SS, h * SS), Image.LANCZOS)
    bg.alpha_composite(ic)
    return bg.convert("RGB").resize((w, h), Image.LANCZOS)


def main() -> int:
    sys.path.insert(0, str(ROOT))
    from propsim import __version__
    OUT.mkdir(parents=True, exist_ok=True)
    for s in SCALES:
        side_image(s, __version__).save(OUT / f"side_{s}.bmp")
        small_image(s).save(OUT / f"small_{s}.bmp")
    side_image(200, __version__).save(OUT / "preview_side.png")
    print(f"wizard art -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
