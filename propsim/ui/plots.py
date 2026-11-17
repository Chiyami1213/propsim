"""Gridded, LaTeX-labelled pyqtgraph plots with a playback cursor, and the tabbed plot area."""
from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QGridLayout, QHBoxLayout, QTabWidget, QWidget

from ..core.result import Curve, PlotSpec, TabSpec
from .latex import LatexLabel, latex_pixmap
from .theme import theme

pg.setConfigOptions(antialias=True, background=None)


class GridPlot(QWidget):
    """One plot: LaTeX title / axis labels (pixmaps), major grid always on, optional time cursor."""

    def __init__(self, spec: PlotSpec, parent=None) -> None:
        super().__init__(parent)
        self.spec = spec
        self.pw = pg.PlotWidget()
        self.pi = self.pw.getPlotItem()
        self.pi.showGrid(x=True, y=True, alpha=0.35)
        self.pi.hideButtons()
        self.pi.setMenuEnabled(False)
        self.title = LatexLabel(spec.title, "text", 10.0)
        self.xl = LatexLabel(spec.xlabel, "text", 9.5)
        self.yl = LatexLabel(spec.ylabel, "text", 9.5, rotate=90)
        self.cursor = pg.InfiniteLine(angle=90, movable=False)
        self.dot = pg.ScatterPlotItem(size=8, pxMode=True)
        self.pi.addItem(self.cursor, ignoreBounds=True)
        self.pi.addItem(self.dot, ignoreBounds=True)
        self._items: list[tuple[pg.PlotDataItem, Curve]] = []
        self._markers: list[pg.ScatterPlotItem] = []
        lay = QGridLayout(self)
        lay.setContentsMargins(2, 2, 2, 2)
        lay.setSpacing(1)
        lay.addWidget(self.title, 0, 0, 1, 2, Qt.AlignHCenter)
        lay.addWidget(self.yl, 1, 0, Qt.AlignVCenter)
        lay.addWidget(self.pw, 1, 1)
        lay.addWidget(self.xl, 2, 1, Qt.AlignHCenter)
        theme.changed.connect(self.apply_theme)
        self.set_spec(spec)
        self.apply_theme()

    # ---- content ------------------------------------------------------------------------
    def set_spec(self, spec: PlotSpec) -> None:
        self.spec = spec
        for lab, txt in ((self.title, spec.title), (self.xl, spec.xlabel), (self.yl, spec.ylabel)):
            if lab.latex() != txt:
                lab.setLatex(txt)
        for it, _ in self._items:
            self.pi.removeItem(it)
        for m in self._markers:
            self.pi.removeItem(m)
        self._items, self._markers = [], []
        for c in spec.curves:
            it = pg.PlotDataItem(np.asarray(c.x, float), np.asarray(c.y, float), connect="finite")
            self.pi.addItem(it)
            self._items.append((it, c))
        for x, y, col in spec.markers:
            m = pg.ScatterPlotItem([x], [y], size=9, pxMode=True)
            self.pi.addItem(m)
            self._markers.append(m)
        # no cursor on an empty plot: painting the dashed line over a degenerate view range crashed Qt
        # (intermittent access violation in InfiniteLine.paint)
        show = spec.cursor and bool(spec.curves)
        self.cursor.setVisible(show)
        self.dot.setVisible(show)
        self._style()
        self.pi.enableAutoRange()

    def _style(self) -> None:
        for it, c in self._items:
            col = theme.color(c.color)
            it.setPen(pg.mkPen(col, width=1.8, style=Qt.DashLine if c.style == "dash" else Qt.SolidLine))
        for m, (x, y, col) in zip(self._markers, self.spec.markers):
            c = theme.color(col)
            m.setBrush(pg.mkBrush(c))
            m.setPen(pg.mkPen(theme.color("text"), width=1))

    def apply_theme(self, *_) -> None:
        t = theme.t
        self.pw.setBackground(t["plot_bg"])
        for ax in ("left", "bottom"):
            a = self.pi.getAxis(ax)
            a.setPen(pg.mkPen(t["grid_major"]))
            a.setTextPen(pg.mkPen(t["dim"]))
        self.cursor.setPen(pg.mkPen(t["text"], style=Qt.DashLine, width=1))
        self._style()

    # ---- playback cursor -------------------------------------------------------------------
    def set_cursor(self, t: float) -> None:
        if not self.spec.cursor or not self._items:
            return
        self.cursor.setValue(t)
        c = self._items[0][1]
        x, y = np.asarray(c.x, float), np.asarray(c.y, float)
        yy = np.interp(t, x, y) if len(x) else np.nan
        # NaN neighbours (e.g. O/F while the valve is closed) give NaN -> hide the dot
        if np.isfinite(yy):
            self.dot.setData([t], [yy], brush=pg.mkBrush(theme.color(c.color)), pen=pg.mkPen(theme.color(c.color)))
        else:
            self.dot.setData([], [])

    def data_dict(self, prefix: str = "") -> dict[str, np.ndarray]:
        out: dict[str, np.ndarray] = {}
        for k, c in enumerate(self.spec.curves):
            out[f"{prefix}c{k}_x"] = np.asarray(c.x, float)
            out[f"{prefix}c{k}_y"] = np.asarray(c.y, float)
        return out


class PlotTabs(QTabWidget):
    """Tab bar (LaTeX captions as pixmaps) with plots side by side per tab, sharing the playback cursor."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setDocumentMode(True)
        self.setIconSize(QSize(120, 18))
        self._tabs: list[TabSpec] = []
        self._plots: list[list[GridPlot]] = []
        self._t = 0.0
        self.currentChanged.connect(self._icons)
        theme.changed.connect(self._icons)

    def set_tabs(self, tabs: list[TabSpec]) -> None:
        same = len(tabs) == len(self._tabs) and all(
            a.name == b.name and len(a.plots) == len(b.plots) for a, b in zip(tabs, self._tabs))
        if same:
            for tab, plots in zip(tabs, self._plots):
                for ps, gp in zip(tab.plots, plots):
                    gp.set_spec(ps)
        else:
            cur = self.currentIndex()
            self.blockSignals(True)
            while self.count():
                w = self.widget(0)
                self.removeTab(0)
                w.deleteLater()
            self._plots = []
            for tab in tabs:
                page = QWidget()
                h = QHBoxLayout(page)
                h.setContentsMargins(4, 2, 4, 4)
                row = []
                for ps in tab.plots:
                    gp = GridPlot(ps)
                    row.append(gp)
                    h.addWidget(gp, 1)
                self._plots.append(row)
                self.addTab(page, "")
            self.blockSignals(False)
            self.setCurrentIndex(min(max(cur, 0), max(len(tabs) - 1, 0)))
        self._tabs = tabs
        self._icons()
        self.set_cursor(self._t)

    def _icons(self, *_) -> None:
        for i, tab in enumerate(self._tabs):
            if "$" not in tab.name:                      # plain caption: let Qt draw it (crisper)
                self.setTabText(i, tab.name)
                self.setTabIcon(i, QIcon())
                continue
            col = theme.color("text" if i == self.currentIndex() else "dim")
            pm = latex_pixmap(tab.name, col, 9.5)
            self.setTabText(i, "")
            self.setTabIcon(i, QIcon(pm))
            self.setIconSize(QSize(max(int(pm.width() / pm.devicePixelRatio()), 40), 18))

    def set_cursor(self, t: float) -> None:
        self._t = t
        for row in self._plots:
            for gp in row:
                gp.set_cursor(t)

    def grid_plots(self) -> list[GridPlot]:
        return [gp for row in self._plots for gp in row]

    def export_data(self) -> dict[str, np.ndarray]:
        out: dict[str, np.ndarray] = {}
        for i, row in enumerate(self._plots):
            for j, gp in enumerate(row):
                out.update(gp.data_dict(f"tab{i}_plot{j}_"))
        return out
