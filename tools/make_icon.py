"""Generate the PropSim app icon (assets/icon.png, assets/icon.ico). Run: python tools/make_icon.py

Motif: a de Laval nozzle cross-section (converging-diverging bell) with an exhaust plume, drawn with the
theme palette (thrust blue wall, hot-orange -> pressure-purple plume) on a dark rounded tile.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtCore import QPointF, QRectF, Qt  # noqa: E402
from PySide6.QtGui import (QBrush, QColor, QGuiApplication, QImage, QLinearGradient, QPainter,  # noqa: E402
                           QPainterPath, QPen, QRadialGradient)

ASSETS = Path(__file__).resolve().parents[1] / "propsim" / "assets"
S = 512


def draw() -> QImage:
    img = QImage(S, S, QImage.Format_ARGB32)
    img.fill(Qt.transparent)
    p = QPainter(img)
    p.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)

    # tile
    tile = QPainterPath()
    tile.addRoundedRect(QRectF(16, 16, S - 32, S - 32), 96, 96)
    g = QLinearGradient(0, 0, S, S)
    g.setColorAt(0, QColor("#1b2130"))
    g.setColorAt(1, QColor("#0b0e14"))
    p.fillPath(tile, QBrush(g))
    p.setPen(QPen(QColor("#2c3550"), 6))
    p.drawPath(tile)
    p.setClipPath(tile)

    cy = 256.0
    # wall half-radius profile r(x): chamber -> throat -> bell
    xs = [70, 150, 215, 250, 290, 340, 400, 450]
    rs = [96, 96, 66, 34, 40, 66, 96, 112]

    def contour(sign: float) -> QPainterPath:
        path = QPainterPath(QPointF(xs[0], cy + sign * rs[0]))
        # smooth through points using quad midpoints
        for i in range(1, len(xs) - 1):
            mx, my = (xs[i] + xs[i + 1]) / 2, cy + sign * (rs[i] + rs[i + 1]) / 2
            path.quadTo(QPointF(xs[i], cy + sign * rs[i]), QPointF(mx, my))
        path.lineTo(QPointF(xs[-1], cy + sign * rs[-1]))
        return path

    up, lo = contour(-1), contour(+1)

    # plume: hot core (orange/red) fading to purple, expanding past the exit
    plume = QPainterPath(QPointF(250, cy - 30))
    plume.cubicTo(QPointF(330, cy - 36), QPointF(420, cy - 70), QPointF(500, cy - 120))
    plume.lineTo(QPointF(500, cy + 120))
    plume.cubicTo(QPointF(420, cy + 70), QPointF(330, cy + 36), QPointF(250, cy + 30))
    plume.closeSubpath()
    pg = QLinearGradient(250, 0, 500, 0)
    pg.setColorAt(0.0, QColor("#ffd27a"))
    pg.setColorAt(0.25, QColor("#ff9d3c"))
    pg.setColorAt(0.6, QColor("#ff5a3c"))
    pg.setColorAt(1.0, QColor(176, 124, 255, 0))
    p.fillPath(plume, QBrush(pg))

    # glow at the throat
    rg = QRadialGradient(QPointF(250, cy), 120)
    rg.setColorAt(0, QColor(255, 210, 122, 150))
    rg.setColorAt(1, QColor(255, 157, 60, 0))
    p.setPen(Qt.NoPen)
    p.setBrush(QBrush(rg))
    p.drawEllipse(QPointF(250, cy), 120, 120)

    # chamber body
    body = QPainterPath()
    body.addRoundedRect(QRectF(46, cy - 96, 110, 192), 22, 22)
    bg = QLinearGradient(0, cy - 96, 0, cy + 96)
    bg.setColorAt(0, QColor("#5fb0ff"))
    bg.setColorAt(1, QColor("#2a6fc0"))
    p.fillPath(body, QBrush(bg))
    # oxidizer / fuel bands on the chamber
    p.fillRect(QRectF(46, cy - 96, 110, 28), QColor("#ff9d3c"))
    p.fillRect(QRectF(46, cy + 68, 110, 28), QColor("#5cc97a"))

    # nozzle walls
    wall = QPen(QColor("#4ea1ff"), 26, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
    p.setPen(wall)
    p.setBrush(Qt.NoBrush)
    p.drawPath(up)
    p.drawPath(lo)
    hi = QPen(QColor("#a9d3ff"), 6, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
    p.setPen(hi)
    p.drawPath(up)
    p.drawPath(lo)
    p.end()
    return img


def main() -> int:
    QGuiApplication(sys.argv)
    ASSETS.mkdir(parents=True, exist_ok=True)
    img = draw()
    img.save(str(ASSETS / "icon.png"))
    from PIL import Image
    im = Image.open(ASSETS / "icon.png").convert("RGBA")
    im.save(ASSETS / "icon.ico", sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
    print("wrote", ASSETS / "icon.png", ASSETS / "icon.ico")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
