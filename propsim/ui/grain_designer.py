"""Motor designer window (solid and hybrid): design the grain cross-section, see the whole grain in 3D with a side-view
cutaway, and read the burning perimeter, port area, remaining propellant and (solid) K_n and pressure curves.  A separate
window like the cycle sandbox; every panel can be dragged.  "Open in Main Window" carries the design into the simulators."""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QComboBox, QDockWidget, QLabel, QMainWindow, QProgressBar, QPushButton, QScrollArea, QToolBar,
                               QVBoxLayout, QWidget)

from ..modules.hybrid_live import PortLiveModel
from ..modules.solid_models import SolidGrainLiveModel
from .panels import PAD, PanelTitle
from .params import ParamForm
from .plots import PlotTabs
from .text import apa
from .viewport3d import Viewport3D

MODES = {"solid": ("Solid", SolidGrainLiveModel), "hybrid": ("Hybrid", PortLiveModel)}


class GrainDesigner(QMainWindow):
    """One window, two grain families: ``mode`` picks the solid or the hybrid designer."""

    def __init__(self, main=None, mode: str = "solid", parent=None) -> None:
        super().__init__(parent or main)
        self.main = main
        self.setWindowTitle("Motor Designer")
        self.resize(1500, 900)
        self.mode = mode
        self.ctrl = None
        self.form: ParamForm | None = None
        self.vp = Viewport3D()
        self.plots = PlotTabs()
        self.params_holder = QScrollArea()
        self.params_holder.setWidgetResizable(True)
        self.hover = QLabel("hover: move over a parameter for its meaning and [dimensions]")
        self.hover.setObjectName("caption")
        self.hover.setWordWrap(True)
        side = QWidget()
        sl = QVBoxLayout(side)
        sl.setContentsMargins(PAD, 0, PAD, PAD)
        sl.addWidget(self.params_holder, 1)
        sl.addWidget(self.hover)
        self.setCentralWidget(self.vp)
        self.dock_params = self._dock("Params", "Parameters", side)
        self.dock_plots = self._dock("Plots", "Plots", self.plots)
        self.docks = [self.dock_params, self.dock_plots]
        self.default_layout()
        tb = QToolBar("Grain")
        tb.setMovable(False)
        self.addToolBar(tb)
        tb.addWidget(QLabel(" Motor type "))
        self.mode_box = QComboBox()
        for key, (name, _) in MODES.items():
            self.mode_box.addItem(name, key)
        self.mode_box.setCurrentIndex(self.mode_box.findData(mode))
        self.mode_box.currentIndexChanged.connect(lambda _: self.set_mode(self.mode_box.currentData()))
        tb.addWidget(self.mode_box)
        tb.addSeparator()
        use = QPushButton("Use in Analysis")
        use.setToolTip("make this the active design: the solid or hybrid analysis modules start from it")
        use.clicked.connect(self.use_in_analysis)
        tb.addWidget(use)
        save = QPushButton("Save Design")
        save.clicked.connect(self.save_design)
        tb.addWidget(save)
        opn = QPushButton("Open Design")
        opn.clicked.connect(self.open_design)
        tb.addWidget(opn)
        send = QPushButton("Open in Main Window")
        send.clicked.connect(self.open_in_main)
        tb.addWidget(send)
        panels = QPushButton("Panels")
        from PySide6.QtWidgets import QMenu
        pm = QMenu(panels)
        for d in self.docks:
            pm.addAction(d.toggleViewAction())
        pm.addSeparator()
        pm.addAction("Restore Default Layout", self.default_layout)
        panels.setMenu(pm)
        tb.addWidget(panels)
        self.busy = QProgressBar()                                  # busy bar while the design is evaluated
        self.busy.setRange(0, 0)
        self.busy.setFixedWidth(180)
        self.busy.hide()
        self.statusBar().addPermanentWidget(self.busy)
        self.vp.cutawayChanged.connect(lambda _: self.refresh())
        self.vp.fieldChanged.connect(lambda _: self.refresh())
        self.set_mode(mode)

    # ------------------------------------------------------------------ layout
    def _dock(self, name: str, title: str, widget: QWidget) -> QDockWidget:
        d = QDockWidget(apa(title), self)
        d.setObjectName("gd" + name)
        d.setFeatures(QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable | QDockWidget.DockWidgetClosable)
        d.setTitleBarWidget(PanelTitle(d, title))
        d.setWidget(widget)
        return d

    def default_layout(self) -> None:
        for d in self.docks:
            d.setFloating(False)
            self.removeDockWidget(d)
        self.addDockWidget(Qt.RightDockWidgetArea, self.dock_params)
        self.addDockWidget(Qt.BottomDockWidgetArea, self.dock_plots)
        for d in self.docks:
            d.show()
        self.resizeDocks([self.dock_params], [430], Qt.Horizontal)
        self.resizeDocks([self.dock_plots], [330], Qt.Vertical)

    # ------------------------------------------------------------------ model
    def set_mode(self, mode: str) -> None:
        self.mode = mode
        _, cls = MODES[mode]
        self.ctrl = cls()                                   # its own controller: the main window's scene is untouched
        self.form = ParamForm(self.ctrl.Params)
        self.form.changed.connect(self.refresh)
        self.form.hovered.connect(self.hover.setText)
        self.params_holder.setWidget(self.form)
        self.vp.plotter.clear()
        self.ctrl.reset_scene()
        self.vp.set_fields(self.ctrl.fields)
        self.vp.legend_labels = self.ctrl.legend_labels
        self.setWindowTitle(f"Motor Designer - {MODES[mode][0]}")
        self.refresh()

    def refresh(self, *_) -> None:
        if self.ctrl is None or self.form is None:
            return
        from PySide6.QtWidgets import QApplication
        self.busy.show()
        QApplication.processEvents()
        try:
            p = self.form.values()
            r = self.ctrl.evaluate(p)
            self.plots.set_tabs(self.ctrl.tabs(p, r))
            self.ctrl.update_scene(self.vp, p, r, None)
            self.vp.set_readout(self.ctrl.readout(p, r, None))
            self.vp.set_caption(self.ctrl.caption(p, r, None))
        finally:
            self.busy.hide()

    # ---- designs ------------------------------------------------------------------------------------------
    def make_design(self):
        from .. import design as D
        return D.Design(self.mode, f"{MODES[self.mode][0]} grain design", self.form.values().model_dump(), "user")

    def load_design(self, d) -> None:
        """Show a saved design (its kind picks solid or hybrid)."""
        if d.kind != self.mode:
            self.mode_box.setCurrentIndex(self.mode_box.findData(d.kind))
        keep = set(self.ctrl.Params.model_fields)
        self.form.set_values({k: v for k, v in d.data.items() if k in keep})
        self.refresh()

    def use_in_analysis(self) -> None:
        if self.main is not None:
            self.main.set_design(self.make_design())

    def save_design(self) -> None:
        from PySide6.QtWidgets import QFileDialog
        home = self.main.home if self.main is not None else None
        start = str((home / "designs") if home else "") + f"/{self.mode}_grain"
        path, _ = QFileDialog.getSaveFileName(self, "Save Design", start, "PropSim design (*.psimdesign.json)")
        if path:
            d = self.make_design()
            from .busy import busy
            with busy(self, "Saving design"):
                d.save(path)
            if self.main is not None:
                self.main.set_design(d)

    def open_design(self) -> None:
        if self.main is not None:
            self.main.open_design()

    def open_in_main(self) -> None:
        """Select the matching Live model in the main window with these parameters."""
        if self.main is None:
            return
        self.main.select_model(self.ctrl.key, self.form.values().model_dump())
        self.main.raise_()
        self.main.activateWindow()
