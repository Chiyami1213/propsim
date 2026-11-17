"""Main window: six dockable panels, bottom playback bar (Section 5.1). Wires every region together."""
from __future__ import annotations

import os
import re
import time
from pathlib import Path

from PySide6.QtCore import QByteArray, QProcess, QSettings, Qt, QUrl
from PySide6.QtGui import QAction, QDesktopServices, QKeySequence, QStandardItem, QStandardItemModel
from PySide6.QtWidgets import (QApplication, QCheckBox, QDockWidget, QFileDialog, QFrame, QLabel, QListWidget,
                               QListWidgetItem, QMainWindow, QMessageBox, QScrollArea, QSplitter,
                               QTreeView, QVBoxLayout, QWidget)

from ..core import io as pio
from ..core.result import Result
from ..modules import registry
from ..modules.base import ModelController
from . import latex
from .export import save_tab_png
from .. import HELP_URL, __version__, storage
from .latex import LatexLabel
from .latex_widgets import LatexItemDelegate, LatexLog
from .text import apa
from .params import ParamForm
from .playback import PlaybackBar, PlaybackModel
from .plots import PlotTabs
from .theme import build_qss, theme
from .viewport3d import Viewport3D
from . import gc_guard
from .worker import RunWorker

MODULE_TITLES = {"Hybrid": "Hybrid Rocket", "Liquid": "Liquid Engine", "Solid": "Solid Rocket Motor",
                 "Electric": "Electric Propulsion", "Vehicle": "Vehicle", "Toy": "Toy"}
KEY_ROLE = Qt.UserRole + 1


def default_home() -> Path:
    """Data folder: the one chosen in the app (File > Data Folder), else PROPSIM_HOME, else %APPDATA%/PropSim."""
    return storage.resolve_home(QSettings("PropSim", "PropSim").value("data_dir") or None)


PAD = 10                                                  # one padding for every panel (title and content)
LAYOUT_VERSION = 2                                        # bump when the dock set changes: old saved states are ignored


def _heading(text: str) -> QLabel:
    lbl = QLabel(apa(text))
    lbl.setObjectName("heading")
    lbl.setContentsMargins(PAD, PAD, PAD, PAD // 2)
    return lbl


def _panel(*widgets: QWidget, stretch: int = 0) -> QWidget:
    """Panel body with the shared padding; the first widget takes the free space."""
    w = QWidget()
    lay = QVBoxLayout(w)
    lay.setContentsMargins(PAD, 0, PAD, PAD)
    lay.setSpacing(PAD // 2 + 1)
    for i, x in enumerate(widgets):
        lay.addWidget(x, 1 if i == 0 else stretch)
    return w


class MainWindow(QMainWindow):
    def __init__(self, home: Path | None = None, use_settings: bool = True) -> None:
        super().__init__()
        self.home = storage.ensure_layout(Path(home) if home else (default_home() if use_settings else
                                                                    storage.resolve_home()))
        self.runs_dir = self.home / "runs"
        migrated = storage.migrate_legacy(self.home) if home is None else 0
        gc_guard.install(parent=self)                      # cyclic GC only between events, never inside a paint
        self.use_settings = use_settings
        self.resize(1600, 1000)
        self.ctrl: ModelController | None = None
        self.form: ParamForm | None = None
        self.result: Result | None = None
        self.result_params = None
        self.worker: RunWorker | None = None
        self.case_name = "(unsaved)"
        self.saved: dict[str, dict] = {}                  # run name -> {path, model, params, result, label}
        self.active_run: str | None = None
        self._stale_noted = False
        self._switching = False
        self.playback = PlaybackModel(self)
        self._build_ui()
        self._build_menus()
        self._apply_theme()
        theme.changed.connect(lambda *_: self._apply_theme())
        self._load_saved_runs()
        self.playback.frameChanged.connect(self._on_frame)
        if use_settings:
            self._restore_layout()
        self._log(f"data folder: {self.home}")
        if migrated:
            self._log(f"copied {migrated} saved files from the previous data folder (~/.propsim)")

    # =================================================================== construction
    def _build_ui(self) -> None:
        """Six dockable panels (drag any title bar; View > Restore Default Layout puts them back), one padding."""
        self.setDockOptions(QMainWindow.AnimatedDocks | QMainWindow.AllowNestedDocks | QMainWindow.AllowTabbedDocks)
        self.setDockNestingEnabled(True)
        # --- modules tree ---
        self.tree = QTreeView()
        self.tree.setHeaderHidden(True)
        self.tree.setEditTriggers(QTreeView.NoEditTriggers)
        self.tree_model = QStandardItemModel()
        for mod, entries in registry.TREE.items():
            top = QStandardItem(apa(mod))
            top.setFlags(Qt.ItemIsEnabled)
            for label, mode, cls in entries:
                it = QStandardItem(f"{apa(label)}  ({'Live' if mode == 'live' else 'Run'})")
                it.setData(cls.key if cls else None, KEY_ROLE)
                it.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable if cls else Qt.NoItemFlags)
                top.appendRow(it)
            self.tree_model.appendRow(top)
        self.tree.setModel(self.tree_model)
        self.tree.setItemDelegate(LatexItemDelegate(9.5, "dim", "accent", self.tree))
        self.tree.setIndentation(18)
        self.tree.expand(self.tree_model.index(list(registry.TREE).index("Hybrid"), 0))
        self.tree.selectionModel().currentChanged.connect(self._on_tree)
        self.left_dock = self._dock("Modules", _heading("Modules"), _panel(self.tree))
        # --- saved runs ---
        self.runs_list = QListWidget()
        self.runs_list.itemClicked.connect(self._on_run_clicked)
        self.compare_cb = QCheckBox("Compare Overlay")
        self.compare_cb.toggled.connect(lambda _: self._refresh_tabs())
        self.runs_dock = self._dock("SavedRuns", _heading("Saved Runs"), _panel(self.runs_list, self.compare_cb))
        # --- 3D view and plots ---
        self.vp = Viewport3D()
        self.plot_tabs = PlotTabs()
        self.vp_dock = self._dock("View3D", _heading("3D View"), _panel(self.vp))
        self.plots_dock = self._dock("Plots", _heading("Plots"), _panel(self.plot_tabs))
        # --- parameters ---
        self.params_title = LatexLabel("", "text", 10.0)            # model name, first line of the panel body
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.hover = LatexLabel("", "dim", 9.0)
        hb = QFrame()
        hb.setObjectName("hoverbox")
        hl = QVBoxLayout(hb)
        hl.setContentsMargins(PAD, PAD // 2, PAD, PAD // 2)
        hl.addWidget(self.hover)
        body = _panel(self.scroll, hb)
        body.layout().insertWidget(0, self.params_title)
        self.right_dock = self._dock("Parameters", _heading("Parameters"), body)
        # --- run log ---
        self.log = LatexLog()
        self.summary = LatexLabel("", "accent", 9.5)
        self.log_dock = self._dock("RunLog", _heading("Run Log"), _panel(self.log, self.summary))
        self.docks = [self.left_dock, self.runs_dock, self.vp_dock, self.plots_dock, self.right_dock, self.log_dock]
        self.default_layout()
        # --- bottom bar ---
        self.bar = PlaybackBar(self.playback)
        self.addToolBar(Qt.BottomToolBarArea, self.bar)
        self.bar.runClicked.connect(self.start_run)
        self.bar.cancelClicked.connect(self.cancel_run)
        self.bar.exportRequested.connect(self.export)
        # --- viewport signals ---
        self.vp.resetRequested.connect(lambda: self.vp.reset_view())
        self.vp.cutawayChanged.connect(lambda _: self._refresh_scene())
        self.vp.fieldChanged.connect(lambda _: self._refresh_scene())

    def _dock(self, name: str, title: QWidget, widget: QWidget) -> QDockWidget:
        d = QDockWidget(name, self)
        d.setObjectName(name + "Dock")
        d.setFeatures(QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable)
        d.setTitleBarWidget(title)                         # drag the panel by its title
        d.setWidget(widget)
        self.addDockWidget(Qt.LeftDockWidgetArea, d)
        return d

    def default_layout(self) -> None:
        """Modules / Saved Runs | 3D View / Plots | Parameters / Run Log, in the mockup proportions."""
        for d in self.docks:
            d.setFloating(False)
            self.removeDockWidget(d)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.left_dock)
        self.splitDockWidget(self.left_dock, self.vp_dock, Qt.Horizontal)
        self.splitDockWidget(self.vp_dock, self.right_dock, Qt.Horizontal)
        self.splitDockWidget(self.left_dock, self.runs_dock, Qt.Vertical)
        self.splitDockWidget(self.vp_dock, self.plots_dock, Qt.Vertical)
        self.splitDockWidget(self.right_dock, self.log_dock, Qt.Vertical)
        for d in self.docks:
            d.show()
        w = max(self.width(), 1200)
        self.resizeDocks([self.left_dock, self.vp_dock, self.right_dock], [265, w - 265 - 450, 450], Qt.Horizontal)
        h = max(self.height() - 40, 600)
        self.resizeDocks([self.left_dock, self.runs_dock], [int(0.62 * h), int(0.38 * h)], Qt.Vertical)
        self.resizeDocks([self.vp_dock, self.plots_dock], [int(0.55 * h), int(0.45 * h)], Qt.Vertical)
        self.resizeDocks([self.right_dock, self.log_dock], [int(0.75 * h), int(0.25 * h)], Qt.Vertical)

    def _build_menus(self) -> None:
        mb = self.menuBar()
        f = mb.addMenu("&File")
        self._act(f, "Open case", self.open_case, QKeySequence.Open)
        self._act(f, "Save case", self.save_case, QKeySequence.Save)
        self._act(f, "Open result", self.open_result)
        f.addSeparator()
        self._act(f, "Restore last session", self.restore_session)
        self._act(f, "Data folder", self.choose_data_folder)
        self._act(f, "Open data folder", lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.home))))
        f.addSeparator()
        x = f.addMenu(apa("Export code"))
        self._act(x, "Python script (.py)", lambda: self.export(".py"))
        self._act(x, "MATLAB script (.m)", lambda: self.export(".m"))
        f.addSeparator()
        self._act(f, "Quit", self.close, QKeySequence.Quit)
        e = mb.addMenu("&Edit")
        self._act(e, "Reset parameters to defaults", self.reset_params)
        v = mb.addMenu("&View")
        self._act(v, "Toggle light/dark theme", self.toggle_theme, "Ctrl+T")
        self._act(v, "Restore default layout", self.reset_layout, "Ctrl+Shift+R")
        s = mb.addMenu("&Simulation")
        self._act(s, "Run", self.start_run, "Ctrl+R")
        self._act(s, "Cancel", self.cancel_run, "Esc")
        va = mb.addMenu("&Validation")
        self._act(va, "Run reference + core tests", self.run_validation)
        h = mb.addMenu("&Help")
        self._act(h, "Online help (GitHub)", lambda: QDesktopServices.openUrl(QUrl(HELP_URL)), "F1")
        self._act(h, "About PropSim", lambda: QMessageBox.about(
            self, "About PropSim", f"PropSim {__version__}\nInteractive 3D Rocket Propulsion Simulator"))

    def _act(self, menu, text, slot, shortcut=None) -> QAction:
        a = QAction(apa(text), self)                       # menus use APA title case
        a.triggered.connect(lambda _=False: slot())
        if shortcut:
            a.setShortcut(shortcut)
        menu.addAction(a)
        return a

    # =================================================================== theme
    def _apply_theme(self) -> None:
        latex.clear_cache()
        QApplication.instance().setStyleSheet(build_qss(theme.t))
        for w in self.findChildren(LatexLabel):
            w._render()

    def toggle_theme(self) -> None:
        theme.set("light" if theme.name == "dark" else "dark")

    # =================================================================== model selection
    def _on_tree(self, cur, _prev=None) -> None:
        key = cur.data(KEY_ROLE) if cur.isValid() else None
        if key and (self.ctrl is None or key != self.ctrl.key):
            self.select_model(key)

    def select_model(self, key: str, params: dict | None = None) -> None:
        self.cancel_run()
        self.backup_session()           # the model being left is the latest session state
        self._switching = True          # playback frames emitted while rebuilding must not touch half-built state
        try:
            self._select_model(key, params)
        finally:
            self._switching = False

    def _select_model(self, key: str, params: dict | None) -> None:
        ctrl = registry.get(key)
        self.ctrl = ctrl
        self.result = None
        self.result_params = None
        self.active_run = None
        self._stale_noted = False
        self.playback.set_range(0, 1.0)
        base = ctrl.Params(**params) if params else None
        self.form = ParamForm(ctrl.Params, base)
        self.form.changed.connect(self._on_params_changed)
        self.form.hovered.connect(self.hover.setLatex)
        self.scroll.setWidget(self.form)
        self.params_title.setLatex(apa(ctrl.title))
        self.vp.plotter.clear()                  # drop the previous model's meshes (their handles are reset below)
        ctrl.reset_scene()
        self.vp.set_fields(ctrl.fields)
        self.bar.set_mode(ctrl.mode)
        self.bar.set_range(0)
        self.summary.setLatex("")
        self.hover.setLatex(r"hover: move over a parameter for its meaning and [dimensions]")
        self._sync_title()
        self._sync_tree_selection(key)
        self._refresh_runs_list()
        if ctrl.mode == "live":
            self.refresh_live()
        else:
            p = self.form.values()
            ctrl.build_scene(self.vp, p, None)
            ctrl.update_scene(self.vp, p, None, None)
            self._set_overlay(p, None, None)
            self.plot_tabs.set_tabs(ctrl.tabs(p, None))
            self._log(f"{ctrl.title}: set parameters and press Run")

    def _sync_tree_selection(self, key: str) -> None:
        sm = self.tree.selectionModel()
        sm.blockSignals(True)
        for r in range(self.tree_model.rowCount()):
            top = self.tree_model.item(r)
            for c in range(top.rowCount()):
                if top.child(c).data(KEY_ROLE) == key:
                    self.tree.expand(top.index())
                    sm.setCurrentIndex(top.child(c).index(), sm.SelectionFlag.ClearAndSelect)
        sm.blockSignals(False)

    def _sync_title(self) -> None:
        mod = MODULE_TITLES.get(self.ctrl.module, self.ctrl.module) if self.ctrl else ""
        self.setWindowTitle(f"PropSim — {apa(mod)} · Case: {self.case_name}")

    # =================================================================== Live mode
    def _on_params_changed(self) -> None:
        if self.ctrl.mode == "live":
            self.refresh_live()
        elif not self._stale_noted and self.result is not None:
            self._stale_noted = True
            self._log("parameters changed — shown results are from the previous run; press Run to update")
        elif self.result is None:
            p = self.form.values()
            self.ctrl.build_scene(self.vp, p, None)
            self.ctrl.update_scene(self.vp, p, None, None)

    def refresh_live(self) -> None:
        t0 = time.perf_counter()
        p = self.form.values()
        r = self.ctrl.evaluate(p)
        self.last_eval_ms = (time.perf_counter() - t0) * 1e3     # physics only: the < 50 ms Live budget
        self.live_state = r
        self.plot_tabs.set_tabs(self.ctrl.tabs(p, r))
        self.ctrl.update_scene(self.vp, p, r, None)
        self._set_overlay(p, r, None)
        self.last_live_ms = (time.perf_counter() - t0) * 1e3     # eval + plots + 3D update + render

    def _refresh_scene(self) -> None:
        if self.ctrl is None:
            return
        if self.ctrl.mode == "live":
            self.refresh_live()
        else:
            self._on_frame(self.playback.frame)

    def _set_overlay(self, p, r, frame) -> None:
        self.vp.set_readout(self.ctrl.readout(p, r, frame))
        self.vp.set_caption(self.ctrl.caption(p, r, frame))

    # =================================================================== Run mode
    def start_run(self) -> None:
        if self.ctrl is None or self.ctrl.mode != "run" or self.worker is not None:
            return
        try:
            p = self.form.values()
            if hasattr(p, "to_si"):
                p.to_si()                     # surface unit-conversion / consistency errors before the thread starts
        except Exception as e:                                          # noqa: BLE001
            self._log(f"invalid parameters: {e}")
            return
        self.result_params = p
        self.worker = RunWorker(self.ctrl.sim, p, self)
        self.worker.progressed.connect(self.bar.set_progress)
        self.worker.done.connect(self._run_done)
        self.worker.failed.connect(self._run_failed)
        self.worker.cancelled.connect(self._run_cancelled)
        self.worker.finished.connect(self._worker_finished)
        self.bar.set_running(True)
        self.bar.set_progress(0)
        self.playback.pause()
        self._log(f"running {self.ctrl.title} …")
        self.worker.start()

    def cancel_run(self) -> None:
        if self.worker is not None:
            self.worker.cancel()               # kills the worker process; ``finished`` clears self.worker

    def _worker_finished(self) -> None:
        w, self.worker = self.worker, None
        if w is not None:
            w.deleteLater()
        self.bar.set_running(False)

    def _run_cancelled(self) -> None:
        self._log("run cancelled")
        self.bar.set_progress(0)

    def _run_failed(self, tb: str) -> None:
        self._log("run FAILED:\n" + tb)

    def _run_done(self, res: Result) -> None:
        self.result = res
        self._stale_noted = False
        for line in res.log:
            self._log(line)
        name = self._save_run(res)
        self.summary.setLatex(self.ctrl.summary_line(self.result_params, res))
        self.playback.set_range(res.n, float(res.t[-1]))
        self.bar.set_range(res.n, str(res.meta.get("t_unit", "s")))
        self._refresh_tabs()
        self.ctrl.build_scene(self.vp, self.result_params, res)
        self.playback.seek(0)
        self._log(f"saved {name}.psim.h5")
        self.playback.play()

    def _on_frame(self, i: int) -> None:
        if self._switching or self.ctrl is None or self.ctrl.mode != "run":
            return
        p = self.result_params or self.form.values()
        r = self.result
        self.ctrl.update_scene(self.vp, p, r, i if r is not None else None)
        self._set_overlay(p, r, i if r is not None else None)
        if r is not None:
            self.plot_tabs.set_cursor(float(r.t[i]))

    def _refresh_tabs(self) -> None:
        if self.ctrl is None or self.ctrl.mode != "run" or self.result is None:
            return
        compare = None
        if self.compare_cb.isChecked():
            compare = [(n, d["result"]) for n, d in self.saved.items() if d["model"] == self.ctrl.key]
        self.plot_tabs.set_tabs(self.ctrl.tabs(self.result_params, self.result, compare))

    # =================================================================== saved runs
    def _save_run(self, res: Result) -> str:
        n = 1
        while f"run_{n:02d}" in self.saved or (self.runs_dir / f"run_{n:02d}.psim.h5").exists():
            n += 1
        name = f"run_{n:02d}"
        path = self.runs_dir / f"{name}.psim.h5"
        pio.save_result_h5(path, res, self.ctrl.key, self.result_params.model_dump())
        self.saved[name] = dict(path=path, model=self.ctrl.key, params=self.result_params.model_dump(),
                                result=res, label=time.strftime("%H:%M"))
        self.active_run = name
        self._refresh_runs_list()
        return name

    def _load_saved_runs(self) -> None:
        for path in sorted(self.runs_dir.glob("run_*.psim.h5")):
            try:
                res, model, params = pio.load_result_h5(path)
                self.saved[path.name.removesuffix(".psim.h5")] = dict(path=path, model=model, params=params,
                                                                      result=res, label="")
            except Exception:                                           # noqa: BLE001 - skip unreadable files
                continue
        self._refresh_runs_list()

    def _refresh_runs_list(self) -> None:
        self.runs_list.clear()
        for name, d in self.saved.items():
            it = QListWidgetItem(f"{'●' if name == self.active_run else '○'} {name}  {d['model'].split('.')[-1]}")
            it.setData(KEY_ROLE, name)
            self.runs_list.addItem(it)

    def _on_run_clicked(self, item: QListWidgetItem) -> None:
        d = self.saved[item.data(KEY_ROLE)]
        self.load_result(d["result"], d["model"], d["params"], item.data(KEY_ROLE))

    def load_result(self, res: Result, model: str, params: dict, name: str | None = None) -> None:
        self.select_model(model, params)
        self.result = res
        self.result_params = self.ctrl.Params(**params)
        self.active_run = name
        self.summary.setLatex(self.ctrl.summary_line(self.result_params, res))
        self.playback.set_range(res.n, float(res.t[-1]))
        self.bar.set_range(res.n, str(res.meta.get("t_unit", "s")))
        self._refresh_tabs()
        self.ctrl.build_scene(self.vp, self.result_params, res)
        self.playback.seek(res.n - 1)
        self._refresh_runs_list()

    # =================================================================== files
    def open_case(self, path: str | None = None) -> None:
        path = path or QFileDialog.getOpenFileName(self, apa("Open case"), str(self.home / "cases"), "PropSim case (*.psim.json)")[0]
        if not path:
            return
        model, params = pio.load_case(path)
        self.case_name = Path(path).name
        self.select_model(model, params)

    def save_case(self, path: str | None = None) -> None:
        if self.ctrl is None:
            return
        path = path or QFileDialog.getSaveFileName(self, apa("Save case"), str(self.home / "cases" / "case.psim.json"),
                                                   "PropSim case (*.psim.json)")[0]
        if path:
            bak = storage.backup_file(path, self.home)              # keep the overwritten version
            if bak:
                self._log(f"previous version backed up to backups/{bak.name}")
            pio.save_case(path, self.ctrl.key, self.form.values().model_dump())
            self.case_name = Path(path).name
            self._sync_title()

    def open_result(self, path: str | None = None) -> None:
        path = path or QFileDialog.getOpenFileName(self, apa("Open result"), str(self.runs_dir), "PropSim result (*.psim.h5)")[0]
        if path:
            res, model, params = pio.load_result_h5(path)
            self.load_result(res, model, params)

    def reset_params(self) -> None:
        if self.form:
            self.form.set_values(self.ctrl.Params())
            self._on_params_changed()

    def export(self, kind: str, path: str | None = None) -> None:
        """PNG (Matplotlib figure of the current tab), CSV, .mat, .h5, and code (.py / .m scripts). ``path`` bypasses the dialog (tests)."""
        if self.ctrl is None:
            return
        live = self.ctrl.mode == "live"
        r = self.live_state if live else self.result
        if r is None:
            self._log("nothing to export yet — run the simulation first")
            return
        p = self.form.values() if live else self.result_params
        ext = {"PNG": ".png", "CSV": ".csv", ".mat": ".mat", ".h5": ".h5", ".py": ".py", ".m": ".m"}[kind]
        default = re.sub(r"[^A-Za-z0-9]+", "_", self.ctrl.title.lower()).strip("_") if kind in (".py", ".m") else "export"
        if not path:
            path = QFileDialog.getSaveFileName(self, f"Export {kind}", str(self.home / "exports" / f"{default}{ext}"),
                                               f"*{ext}")[0]
        if not path:
            return
        if kind in (".py", ".m"):
            from ..export_code import export as export_code
            files = export_code(self.ctrl.key, p.model_dump(), path, tabs=self.ctrl.tabs(p, r))
            self._log("exported code " + ", ".join(f.name for f in files))
            return
        data = self.ctrl.export_data(p, r)
        if kind == "PNG":
            idx = max(self.plot_tabs.currentIndex(), 0)
            tabs = self.ctrl.tabs(p, r)
            t = float(r.t[self.playback.frame]) if not live else None
            save_tab_png(path, tabs[idx], t)
        elif kind == "CSV":
            pio.export_csv(path, data)
        elif kind == ".mat":
            pio.export_mat(path, data)
        else:
            if live:
                pio.export_h5_dict(path, data, {"model": self.ctrl.key, "params": p.model_dump_json()})
            else:
                pio.save_result_h5(path, r, self.ctrl.key, p.model_dump())
        self._log(f"exported {Path(path).name}")

    # =================================================================== misc
    def _log(self, s: str) -> None:
        self.log.append_line(f"{time.strftime('%H:%M:%S')}  {s}")

    def run_validation(self) -> None:
        root = Path(__file__).resolve().parents[2]
        proc = QProcess(self)
        proc.setWorkingDirectory(str(root))
        proc.setProcessChannelMode(QProcess.MergedChannels)
        proc.readyReadStandardOutput.connect(self._proc_output)
        proc.start("python", ["-m", "pytest", "tests/reference", "tests/core", "-q", "--no-header",
                              "-p", "no:cacheprovider"])
        self._log("validation started (pytest tests/reference tests/core) …")
        self._proc = proc

    def _proc_output(self) -> None:
        for line in bytes(self._proc.readAllStandardOutput()).decode(errors="replace").splitlines():
            if line.strip():
                self._log(line.strip())

    # ---- layout persistence ---------------------------------------------------------------------
    def _settings(self) -> QSettings:
        return QSettings("PropSim", "PropSim")

    def _restore_layout(self) -> None:
        s = self._settings()
        g, st = s.value("geometry"), s.value("state")
        if isinstance(g, QByteArray):
            self.restoreGeometry(g)
        if isinstance(st, QByteArray):
            self.restoreState(st, LAYOUT_VERSION)               # False (ignored) for an older dock set

    def reset_layout(self) -> None:
        self.default_layout()

    # ---- data folder and session backup -------------------------------------------------------------
    def set_home(self, path: Path) -> None:
        """Switch the data folder (saved runs are reloaded from the new location)."""
        self.home = storage.ensure_layout(Path(path))
        self.runs_dir = self.home / "runs"
        self.saved, self.active_run = {}, None
        self._load_saved_runs()
        self._log(f"data folder: {self.home}")

    def choose_data_folder(self, path: str | None = None, copy: bool | None = None) -> None:
        path = path or QFileDialog.getExistingDirectory(self, apa("Choose data folder"), str(self.home))
        if not path or Path(path).resolve() == self.home.resolve():
            return
        old = self.home
        if copy is None:
            copy = QMessageBox.question(self, apa("Data folder"), "Copy saved runs and cases to the new folder?",
                                        QMessageBox.Yes | QMessageBox.No) == QMessageBox.Yes
        self.set_home(Path(path))
        if copy:
            n = storage.copy_data(old, self.home)
            self._load_saved_runs()
            self._log(f"copied {n} files from {old}")
        if self.use_settings:
            self._settings().setValue("data_dir", str(self.home))

    def _session_file(self) -> Path:
        return self.home / "backups" / "last_session.psim.json"

    def backup_session(self) -> None:
        if self.ctrl is not None and self.form is not None:
            try:
                pio.save_case(self._session_file(), self.ctrl.key, self.form.values().model_dump())
            except Exception:                                           # noqa: BLE001 - a backup never blocks closing
                pass

    def restore_session(self) -> None:
        f = self._session_file()
        if f.is_file():
            self.open_case(str(f))
        else:
            self._log("no session backup yet")

    def closeEvent(self, ev) -> None:
        self.backup_session()
        self.cancel_run()
        if self.use_settings:
            s = self._settings()
            s.setValue("geometry", self.saveGeometry())
            s.setValue("state", self.saveState(LAYOUT_VERSION))
        super().closeEvent(ev)
