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
from PySide6.QtCore import QEvent, QPointF, QSize, Qt
from PySide6.QtGui import (QColor, QFont, QFontMetricsF, QGuiApplication, QIcon, QImage, QPainter, QPixmap,
                           QTransform)
from PySide6.QtWidgets import (QLabel, QPushButton, QSizePolicy, QStyle, QStyleOptionButton, QStylePainter,
                               QVBoxLayout, QWidget)

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


def _whole_logical(px: int, dpr: float) -> int:
    """Smallest device-pixel size >= ``px`` whose logical size (px / dpr) is a whole number, so a pixmap at a fractional
    scale factor (125 %, 150 %...) is never resampled when a widget (icon, tab, label) shows it at its logical size."""
    for L in range(int(np.ceil(px / dpr)), int(np.ceil(px / dpr)) + 12):
        d = L * dpr
        if abs(d - round(d)) < 1e-6:
            return int(round(d))
    return px


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
    W = _whole_logical(int(np.ceil(sum(p[4] for p in pieces))) + 2, dpr)
    H = _whole_logical(int(np.ceil(asc + desc)) + 1, dpr)
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


def wrap_latex(text: str, width: float, size: float = 10.0, color: str = "text") -> list[str]:
    """Greedy line breaking of mixed text and ``$...$`` math into lines no wider than ``width`` (logical pixels); a
    formula is never split."""
    protected, inside = [], False
    for ch in text:                                       # spaces inside math must not break lines
        if ch == "$":
            inside = not inside
        protected.append("" if (ch == " " and inside) else ch)
    col = theme.color(color)
    lines, cur = [], ""
    for w in "".join(protected).split(" "):
        w = w.replace("", " ")
        cand = f"{cur} {w}" if cur else w
        if cur and latex_pixmap(cand, col, size, 1.0).width() > width:
            lines.append(cur)
            cur = w
        else:
            cur = cand
    if cur:
        lines.append(cur)
    return lines


class LatexLabel(QLabel):
    """A label whose text is mathtext. ``color`` is a theme key ("text", "dim", "accent", ...) or hex.  ``shrink`` lets
    a long line be clipped instead of forcing its panel wide."""

    def __init__(self, text: str = "", color: str = "text", size_pt: float = 10.0, rotate: int = 0,
                 parent=None, shrink: bool = False) -> None:
        super().__init__(parent)
        self._shrink = shrink
        if shrink:
            self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)     # its pixmap width never sizes the panel
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
        pm = latex_pixmap(self._text, theme.color(self._color), self._size, self.devicePixelRatioF())
        if self._rot:
            pm_dpr = pm.devicePixelRatio()
            pm = pm.transformed(QTransform().rotate(self._rot), Qt.SmoothTransformation)
            pm.setDevicePixelRatio(pm_dpr)
        self.setPixmap(pm)
        sz = pm.deviceIndependentSize().toSize()
        self.setMinimumSize(0 if self._shrink else sz.width(), sz.height())


class LatexButton(QPushButton):
    """Button whose caption is mathtext. The text pixmap is rendered at this widget's own device pixel ratio and drawn on
    whole device pixels (never scaled), so the caption stays sharp at any display scaling (100 %, 125 %, 150 %...)."""

    def __init__(self, text: str, color: str = "dim", size_pt: float = 9.0, parent=None) -> None:
        super().__init__(parent)
        self._text, self._color, self._size = text, color, size_pt
        self._pm: QPixmap | None = None
        theme.changed.connect(self._render)
        self._render()

    def setLatex(self, text: str) -> None:
        self._text = text
        self._render()

    def _render(self, *_) -> None:
        self._pm = latex_pixmap(self._text, theme.color(self._color), self._size, self.devicePixelRatioF())
        self.updateGeometry()
        self.update()

    def event(self, e) -> bool:
        t = getattr(QEvent.Type, "DevicePixelRatioChange", None)
        if t is not None and e.type() == t:                   # moved to a screen with another scaling
            self._render()
        return super().event(e)

    def showEvent(self, e) -> None:
        super().showEvent(e)
        if self._pm is not None and abs(self._pm.devicePixelRatio() - self.devicePixelRatioF()) > 1e-6:
            self._render()

    def sizeHint(self) -> QSize:
        d = self._pm.devicePixelRatio()
        content = QSize(int(np.ceil(self._pm.width() / d)), int(np.ceil(self._pm.height() / d)))
        opt = QStyleOptionButton()
        self.initStyleOption(opt)
        opt.text, opt.icon = "", QIcon()
        return self.style().sizeFromContents(QStyle.CT_PushButton, opt, content, self).expandedTo(QSize(0, 0))

    def paintEvent(self, _) -> None:
        p = QStylePainter(self)
        opt = QStyleOptionButton()
        self.initStyleOption(opt)
        opt.text, opt.icon = "", QIcon()
        p.drawControl(QStyle.CE_PushButtonBevel, opt)
        pm, dpr = self._pm, self.devicePixelRatioF()
        w, h = pm.width() / pm.devicePixelRatio(), pm.height() / pm.devicePixelRatio()
        x = round((self.width() - w) / 2 * dpr) / dpr                # snap to the device pixel grid
        y = round((self.height() - h) / 2 * dpr) / dpr
        if not self.isEnabled():
            p.setOpacity(0.4)
        p.drawPixmap(QPointF(x, y), pm)


class LatexParagraph(QWidget):
    """Mixed text and math that wraps to the available width, one :class:`LatexLabel` per line; it never forces its
    panel wider.  Same ``setLatex`` / ``latex`` interface as :class:`LatexLabel`."""

    def __init__(self, text: str = "", color: str = "text", size_pt: float = 10.0, parent=None) -> None:
        super().__init__(parent)
        self._text, self._color, self._size, self._w = text, color, size_pt, 0
        self._lay = QVBoxLayout(self)
        self._lay.setContentsMargins(0, 0, 0, 0)
        self._lay.setSpacing(1)
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        theme.changed.connect(self._rebuild)
        self._rebuild()

    def setLatex(self, text: str, color: str | None = None) -> None:
        self._text = text
        if color:
            self._color = color
        self._rebuild()

    def latex(self) -> str:
        return self._text

    def _rebuild(self, *_) -> None:
        while self._lay.count():
            w = self._lay.takeAt(0).widget()
            if w is not None:
                w.hide()
                w.setParent(None)
                w.deleteLater()
        lines = wrap_latex(self._text, max(self.width() or 360, 60), self._size, self._color) if self._text else []
        h = 0
        for ln in lines:
            lbl = LatexLabel(ln, self._color, self._size, shrink=True)
            self._lay.addWidget(lbl)
            lbl.show()
            h += lbl.minimumHeight() + 1
        self.setFixedHeight(max(h - 1, 0))
        self._w = self.width()

    def resizeEvent(self, e) -> None:
        super().resizeEvent(e)
        if abs(self.width() - self._w) > 3:
            self._rebuild()

    def sizeHint(self) -> QSize:
        return QSize(120, self.height())
