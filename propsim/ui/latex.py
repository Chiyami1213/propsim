"""mathtext -> cached QPixmap, plus LatexLabel / LatexButton that re-render on theme change."""
from __future__ import annotations

import io
from functools import lru_cache

import matplotlib
import numpy as np
from matplotlib import mathtext
from matplotlib.image import imread
from matplotlib.font_manager import FontProperties
from matplotlib.mathtext import MathTextParser
from PySide6.QtCore import QPointF, QSize, Qt
from PySide6.QtGui import (QColor, QFont, QFontMetricsF, QGuiApplication, QIcon, QImage, QPainter, QPixmap,
                           QTransform)
from PySide6.QtWidgets import QLabel, QPushButton

from .text import _MATH_SPLIT, to_math
from .theme import theme

matplotlib.rcParams["mathtext.fontset"] = "cm"


_PARSER: MathTextParser | None = None


def screen_dpr() -> float:
    """Device pixel ratio of the primary screen (render at exactly this ratio: no resampling blur)."""
    app = QGuiApplication.instance()
    scr = app.primaryScreen() if app is not None else None
    return float(scr.devicePixelRatio()) if scr is not None else 2.0


def ui_family() -> str:
    app = QGuiApplication.instance()
    return QGuiApplication.font().family() if app is not None else "Segoe UI"


def latex_pixmap(text: str, color: str, size_pt: float = 10.0, dpr: float | None = None) -> QPixmap:
    r"""Render mixed text and math, e.g. ``r"$\dot{m}_{ox}$ (kg·s⁻¹)"``: plain words in the UI font, ``$...$`` segments
    through mathtext (Computer Modern), aligned on one baseline and rasterized at the screen's device pixel ratio."""
    return _compose(text, color, float(size_pt), float(dpr or screen_dpr()), ui_family())


def _math_raster(seg: str, size_pt: float, dpi: float):
    """Coverage image (uint8, 255 = ink), ascent and depth (px) of one ``$...$`` segment."""
    global _PARSER
    if _PARSER is None:
        _PARSER = MathTextParser("agg")
    r = _PARSER.parse(seg, dpi=dpi, prop=FontProperties(size=size_pt))
    img = np.asarray(r.image, dtype=np.uint8)
    return img, float(r.depth), float(r.width)


@lru_cache(maxsize=8192)
def _compose(text: str, color: str, size_pt: float, dpr: float, family: str) -> QPixmap:
    try:
        s = to_math(text)
    except Exception:                                      # never lose a label to a normalisation corner case
        s = text
    font = QFont(family)
    font.setPixelSize(max(1, round(size_pt * 96.0 / 72.0 * dpr)))
    font.setHintingPreference(QFont.PreferNoHinting)
    fm = QFontMetricsF(font)
    pieces = []                                            # (kind, payload, ascent, descent, width)
    for part in _MATH_SPLIT.split(s or ""):
        if not part:
            continue
        if part.startswith("$") and part.endswith("$") and len(part) >= 2:
            try:
                img, depth, w = _math_raster(part, size_pt, 96.0 * dpr)
                pieces.append(("m", img, img.shape[0] - depth, depth, max(w, img.shape[1])))
                continue
            except (ValueError, RuntimeError):
                part = part.strip("$")
        pieces.append(("t", part, fm.ascent(), fm.descent(), fm.horizontalAdvance(part)))
    if not pieces:
        pieces.append(("t", " ", fm.ascent(), fm.descent(), fm.horizontalAdvance(" ")))
    asc = max(p[2] for p in pieces)
    desc = max(p[3] for p in pieces)
    W = int(np.ceil(sum(p[4] for p in pieces))) + 2
    H = int(np.ceil(asc + desc)) + 1
    out = QImage(W, H, QImage.Format_ARGB32_Premultiplied)
    out.fill(Qt.transparent)
    c = QColor(color)
    pa = QPainter(out)
    pa.setRenderHints(QPainter.Antialiasing | QPainter.TextAntialiasing)
    pa.setFont(font)
    pa.setPen(c)
    x = 1.0
    base = round(asc)
    for kind, payload, a, _, w in pieces:
        if kind == "t":
            pa.drawText(QPointF(x, base), payload)
        else:
            h, wi = payload.shape
            rgba = np.empty((h, wi, 4), np.uint8)
            rgba[..., 0], rgba[..., 1], rgba[..., 2] = c.red(), c.green(), c.blue()
            rgba[..., 3] = payload
            qi = QImage(rgba.data, wi, h, 4 * wi, QImage.Format_RGBA8888).copy()
            pa.drawImage(QPointF(round(x), base - round(a)), qi)     # whole pixels: crisp glyph edges
        x += w
    pa.end()
    pm = QPixmap.fromImage(out)
    pm.setDevicePixelRatio(dpr)
    return pm


@lru_cache(maxsize=256)
def latex_pixmap_legacy(text: str, color: str, size_pt: float = 10.0, dpr: float = 2.0) -> QPixmap:
    """Whole-string mathtext rendering (kept for image export paths that want a single font)."""
    buf = io.BytesIO()
    try:                                                   # all math as LaTeX: units, dims, formulas, loose tokens
        mathtext.math_to_image(to_math(text), buf, prop=FontProperties(size=size_pt), dpi=96 * dpr, format="png",
                               color="black")
    except ValueError:                                     # never lose a label to a normalisation corner case
        buf = io.BytesIO()
        mathtext.math_to_image(text, buf, prop=FontProperties(size=size_pt), dpi=96 * dpr, format="png", color="black")
    buf.seek(0)
    img = imread(buf, format="png")                       # float RGB(A) in [0, 1]
    alpha = 1.0 - img[..., :3].mean(axis=-1)
    c = QColor(color)
    h, w = alpha.shape
    rgba = np.empty((h, w, 4), np.uint8)
    rgba[..., 0], rgba[..., 1], rgba[..., 2] = c.red(), c.green(), c.blue()
    rgba[..., 3] = np.clip(alpha * 255.0 + 0.5, 0, 255).astype(np.uint8)
    qi = QImage(rgba.data, w, h, 4 * w, QImage.Format_RGBA8888).copy()
    pm = QPixmap.fromImage(qi)
    pm.setDevicePixelRatio(dpr)
    return pm


def clear_cache() -> None:
    _compose.cache_clear()
    latex_pixmap_legacy.cache_clear()


class LatexLabel(QLabel):
    """A label whose text is mathtext. ``color`` is a theme key ("text", "dim", "accent", ...) or hex."""

    def __init__(self, text: str = "", color: str = "text", size_pt: float = 10.0, rotate: int = 0,
                 parent=None) -> None:
        super().__init__(parent)
        self._text, self._color, self._size, self._rot = text, color, size_pt, rotate
        self.setToolTip("")
        theme.changed.connect(self._render)
        self._render()

    def setLatex(self, text: str, color: str | None = None) -> None:
        self._text = text
        if color:
            self._color = color
        self._render()

    def latex(self) -> str:
        return self._text

    def _render(self, *_) -> None:
        if not self._text:
            self.clear()
            return
        pm = latex_pixmap(self._text, theme.color(self._color), self._size)
        if self._rot:
            pm_dpr = pm.devicePixelRatio()
            pm = pm.transformed(QTransform().rotate(self._rot), Qt.SmoothTransformation)
            pm.setDevicePixelRatio(pm_dpr)
        self.setPixmap(pm)
        self.setMinimumSize(pm.deviceIndependentSize().toSize())


class LatexButton(QPushButton):
    """Flat button whose caption is mathtext (rendered as an icon)."""

    def __init__(self, text: str, color: str = "dim", size_pt: float = 9.0, parent=None) -> None:
        super().__init__(parent)
        self._text, self._color, self._size = text, color, size_pt
        theme.changed.connect(self._render)
        self._render()

    def setLatex(self, text: str) -> None:
        self._text = text
        self._render()

    def _render(self, *_) -> None:
        pm = latex_pixmap(self._text, theme.color(self._color), self._size)
        self.setIcon(QIcon(pm))
        self.setIconSize(QSize(int(pm.width() / pm.devicePixelRatio()), int(pm.height() / pm.devicePixelRatio())))
