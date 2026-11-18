"""3D viewport: PyVista scene host + toolbar row, readout card, colour bar and caption (Section 5.3 item 2).

Backends: ``QtInteractor`` on a real platform; under ``QT_QPA_PLATFORM=offscreen`` (no GL context) a headless
``pv.Plotter(off_screen=True)`` whose frames are shown in a QLabel, so tests and screenshots still work.
"""
from __future__ import annotations

import matplotlib
import numpy as np
import pyvista as pv
from PySide6.QtCore import QEvent, Qt, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPixmap
from PySide6.QtWidgets import QApplication, QFrame, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from .text import apa
from .latex import LatexButton, LatexLabel, latex_pixmap
from .theme import theme

CUTAWAYS = [45, 90, 135, 0]


class _ImageHost(QLabel):
    """Headless host: renders the off-screen plotter to a pixmap on demand."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.plotter = pv.Plotter(off_screen=True)
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(100, 100)

    def render(self) -> None:
        w, h = max(self.width(), 50), max(self.height(), 50)
        self.plotter.window_size = [w, h]
        img = np.ascontiguousarray(self.plotter.screenshot(return_img=True))
        q = QImage(img.data, img.shape[1], img.shape[0], img.shape[1] * 3, QImage.Format_RGB888).copy()
        self.setPixmap(QPixmap.fromImage(q))

    def resizeEvent(self, ev) -> None:
        super().resizeEvent(ev)
        self.render()


class ColorBar(QWidget):
    """Gradient bar with a LaTeX title and min/max labels (bottom-left overlay)."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.cmap, self.lo, self.hi, self.title = "inferno", 0.0, 1.0, r"$T$ (K)"
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.setFixedSize(90, 205)
        theme.changed.connect(self.update)

    def set(self, cmap: str, lo: float, hi: float, title: str) -> None:
        self.cmap, self.lo, self.hi, self.title = cmap, lo, hi, title
        self.update()

    def paintEvent(self, _) -> None:
        p = QPainter(self)
        col = theme.color("dim")
        pm = latex_pixmap(self.title, col, 9.0)
        p.drawPixmap(0, 0, pm)
        cm = matplotlib.colormaps[self.cmap]
        x0, y0, w, h = 2, 22, 12, 160
        for i in range(h):
            r, g, b, _a = cm(1 - i / (h - 1))
            p.fillRect(x0, y0 + i, w, 1, QColor.fromRgbF(r, g, b))
        p.setPen(QColor(col))
        p.drawText(x0 + w + 6, y0 + 10, f"{self.hi:.4g}")
        p.drawText(x0 + w + 6, y0 + h, f"{self.lo:.4g}")


class Viewport3D(QWidget):
    resetRequested = Signal()
    cutawayChanged = Signal(int)
    fieldChanged = Signal(int)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.headless = QApplication.platformName() == "offscreen"
        if self.headless:
            self.host = _ImageHost()
        else:
            from pyvistaqt import QtInteractor
            self.host = QtInteractor(self)
        self.plotter = self.host.plotter if self.headless else self.host
        self._cut_i = 0
        self._fields: list[str] = []
        self._field_i = 0
        # toolbar row
        self.b_reset = QPushButton("⟲ Reset")
        self.b_cut = LatexButton(r"Cutaway $45^{\circ}$")
        self.b_field = LatexButton(r"Field: $T$ (K)")
        self.b_bar = QPushButton("◧ Colorbar")
        self.b_snap = QPushButton("Snapshot")
        row = QHBoxLayout()
        row.setContentsMargins(8, 8, 8, 4)
        for b in (self.b_reset, self.b_cut, self.b_field, self.b_bar, self.b_snap):
            b.setMinimumWidth(110)
            row.addWidget(b)
        row.addStretch(1)
        # container with overlays
        self.stage = QWidget()
        sl = QVBoxLayout(self.stage)
        sl.setContentsMargins(0, 0, 0, 0)
        sl.addWidget(self.host)
        self.readout = QFrame(self.stage)
        self.readout.setObjectName("readout")
        rl = QVBoxLayout(self.readout)
        rl.setContentsMargins(12, 8, 12, 8)
        rl.setSpacing(3)
        self.readout_lbls = [LatexLabel("", "text", 10.5) for _ in range(4)]
        for l_ in self.readout_lbls:
            rl.addWidget(l_)
        self.colorbar = ColorBar(self.stage)
        self.caption = LatexLabel("", "dim", 9.0, shrink=True)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        lay.addLayout(row)
        lay.addWidget(self.stage, 1)
        cap = QHBoxLayout()
        cap.setContentsMargins(36, 2, 8, 4)
        cap.addWidget(self.caption)
        cap.addStretch(1)
        lay.addLayout(cap)
        self.stage.installEventFilter(self)
        # wiring
        self.b_reset.clicked.connect(self.resetRequested)
        self.b_cut.clicked.connect(self._next_cut)
        self.b_field.clicked.connect(self._next_field)
        self.b_bar.clicked.connect(lambda: self.colorbar.setVisible(not self.colorbar.isVisible()))
        theme.changed.connect(self.apply_theme)
        self.apply_theme()

    # ---- API for model controllers -----------------------------------------------------------
    @property
    def cutaway(self) -> int:
        return CUTAWAYS[self._cut_i]

    def set_fields(self, latex_labels: list[str]) -> None:
        self._fields, self._field_i = latex_labels, 0
        self._show_field()
        self.b_field.setEnabled(len(latex_labels) > 1)

    @property
    def field_index(self) -> int:
        return self._field_i

    def set_readout(self, lines: list[str]) -> None:
        for lbl, s in zip(self.readout_lbls, lines + [""] * 4):
            lbl.setLatex(s)
        self.readout.adjustSize()
        self._place()

    def set_caption(self, s: str) -> None:
        self.caption.setLatex(s)

    def set_colorbar(self, cmap: str, lo: float, hi: float, title: str) -> None:
        self.colorbar.set(cmap, lo, hi, title)

    def render(self) -> None:
        if self.headless:
            self.host.render()
        else:
            self.plotter.render()

    def reset_view(self, position=None, zoom: float = 1.5) -> None:
        self.plotter.camera_position = position or "iso"
        self.plotter.reset_camera()
        self.plotter.camera.zoom(zoom)
        self.render()

    def snapshot(self, path: str) -> None:
        self.plotter.screenshot(path)

    def apply_theme(self, *_) -> None:
        self.plotter.set_background(theme.t["viewport_bg"])
        c = QColor(theme.t["card"])
        self.readout.setStyleSheet(
            f"QFrame#readout {{ background: rgba({c.red()},{c.green()},{c.blue()},215); "
            f"border: 1px solid {theme.t['edge']}; border-radius: 6px; }}")
        self.render()

    # ---- internals ---------------------------------------------------------------------------
    def _next_cut(self) -> None:
        self._cut_i = (self._cut_i + 1) % len(CUTAWAYS)
        a = self.cutaway
        self.b_cut.setLatex(rf"Cutaway ${a}^{{\circ}}$" if a else "No Cutaway")
        self.cutawayChanged.emit(a)

    def _next_field(self) -> None:
        self._field_i = (self._field_i + 1) % max(len(self._fields), 1)
        self._show_field()
        self.fieldChanged.emit(self._field_i)

    def _show_field(self) -> None:
        if self._fields:
            self.b_field.setLatex(f"Field: {apa(self._fields[self._field_i])}")

    def _place(self) -> None:
        w, h = self.stage.width(), self.stage.height()
        self.readout.move(w - self.readout.width() - 14, 12)
        self.colorbar.move(18, h - self.colorbar.height() - 12)
        self.readout.raise_()
        self.colorbar.raise_()

    def eventFilter(self, obj, ev):
        if obj is self.stage and ev.type() == QEvent.Resize:
            self._place()
        return False

    def resizeEvent(self, ev) -> None:
        super().resizeEvent(ev)
        self._place()

    def showEvent(self, ev) -> None:
        super().showEvent(ev)
        self._place()
