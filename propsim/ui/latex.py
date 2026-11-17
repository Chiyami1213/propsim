"""mathtext -> cached QPixmap, plus LatexLabel / LatexButton that re-render on theme change."""
from __future__ import annotations

import io
from functools import lru_cache

import matplotlib
import numpy as np
from matplotlib import mathtext
from matplotlib.image import imread
from matplotlib.font_manager import FontProperties
from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QColor, QIcon, QImage, QPixmap, QTransform
from PySide6.QtWidgets import QLabel, QPushButton

from .theme import theme

matplotlib.rcParams["mathtext.fontset"] = "cm"


@lru_cache(maxsize=4096)
def latex_pixmap(text: str, color: str, size_pt: float = 10.0, dpr: float = 2.0) -> QPixmap:
    r"""Render mixed text and mathtext, e.g. ``r"$\dot{m}_{ox}$ (kg·s⁻¹)"``. Colour is part of the cache key.

    Sizes follow Qt's 96-dpi point convention. ``math_to_image`` yields an opaque white background in current Matplotlib, so the glyphs are rendered in
    black and the luminance is converted to an alpha mask tinted with ``color`` (transparent background)."""
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
    latex_pixmap.cache_clear()


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
