"""Main window: docks, centre splitter, bottom playback bar (Section 5.1). Wires every region together."""
from __future__ import annotations

import os
import time
from pathlib import Path

from PySide6.QtCore import QByteArray, QProcess, QSettings, Qt
from PySide6.QtGui import QAction, QKeySequence, QStandardItem, QStandardItemModel
from PySide6.QtWidgets import (QApplication, QCheckBox, QDockWidget, QFileDialog, QFrame, QLabel, QListWidget,
                               QListWidgetItem, QMainWindow, QMessageBox, QPlainTextEdit, QScrollArea, QSplitter,
                               QTreeView, QVBoxLayout, QWidget)

from ..core import io as pio
from ..core.result import Result
from ..modules import registry
from ..modules.base import ModelController
from . import latex
from .export import save_tab_png
from .latex import LatexLabel
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
    return Path(os.environ.get("PROPSIM_HOME", Path.home() / ".propsim"))


def _heading(text: str) -> QLabel:
    lbl = QLabel(text)
    lbl.setObjectName("heading")
    lbl.setContentsMargins(10, 8, 10, 4)
    return lbl


class MainWindow(QMainWindow):
    def __init__(self, home: Path | None = None, use_settings: bool = True) -> None:
        super().__init__()
        self.home = Path(home) if home else default_home()
        self.runs_dir = self.home / "runs"
        self.runs_dir.mkdir(parents=True, exist_ok=True)
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

    # =================================================================== construction
    def _build_ui(self) -> None:
        # --- viewport + plots (centre) ---
        self.vp = Viewport3D()
        self.plot_tabs = PlotTabs()
        self.centre = QSplitter(Qt.Vertical)
        for w in (self.vp, self.plot_tabs):
            card = QFrame()
            card.setObjectName("card")
            lay = QVBoxLayout(card)
            lay.setContentsMargins(0, 0, 0, 0)
            lay.addWidget(w)
            self.centre.addWidget(card)
        self.centre.setSizes([550, 450])
        self.centre.setChildrenCollapsible(False)
        self.setCentralWidget(self.centre)
        # --- left dock ---
        self.tree = QTreeView()
        self.tree.setHeaderHidden(True)
        self.tree.setEditTriggers(QTreeView.NoEditTriggers)
        self.tree_model = QStandardItemModel()
        for mod, entries in registry.TREE.items():
            top = QStandardItem(mod)
            top.setFlags(Qt.ItemIsEnabled)
            for label, mode, cls in entries:
                it = QStandardItem(f"{label}  ({mode})")
                it.setData(cls.key if cls else None, KEY_ROLE)
                it.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable if cls else Qt.NoItemFlags)
                top.appendRow(it)
            self.tree_model.appendRow(top)
        self.tree.setModel(self.tree_model)
        self.tree.setIndentation(18)
        self.tree.expand(self.tree_model.index(list(registry.TREE).index("Hybrid"), 0))
        self.tree.selectionModel().currentChanged.connect(self._on_tree)
        self.runs_list = QListWidget()
        self.runs_list.itemClicked.connect(self._on_run_clicked)
        self.compare_cb = QCheckBox("compare overlay")
        self.compare_cb.toggled.connect(lambda _: self._refresh_tabs())
        left = QWidget()
        ll = QVBoxLayout(left)
        ll.setContentsMargins(4, 0, 4, 8)
        ll.addWidget(self.tree, 3)
        ll.addWidget(_heading("SAVED RUNS"))
        ll.addWidget(self.runs_list, 2)
        ll.addWidget(self.compare_cb)
        self.left_dock = self._dock("Modules", "MODULES", left, Qt.LeftDockWidgetArea)
        # --- right dock ---
        self.params_title = _heading("PARAMETERS")
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.hover = LatexLabel("", "dim", 9.0)
        hb = QFrame()
        hb.setObjectName("hoverbox")
        hl = QVBoxLayout(hb)
        hl.setContentsMargins(10, 6, 10, 6)
        hl.addWidget(self.hover)
        top = QWidget()
        tl = QVBoxLayout(top)
        tl.setContentsMargins(4, 0, 4, 6)
        tl.addWidget(self.params_title)
        tl.addWidget(self.scroll, 1)
        tl.addWidget(hb)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(500)
        self.summary = LatexLabel("", "accent", 9.5)
        bot = QWidget()
        bl = QVBoxLayout(bot)
        bl.setContentsMargins(4, 0, 4, 6)
        bl.addWidget(_heading("RUN LOG"))
        bl.addWidget(self.log, 1)
        bl.addWidget(self.summary)
        rs = QSplitter(Qt.Vertical)
        rs.addWidget(top)
        rs.addWidget(bot)
        rs.setSizes([750, 250])
        self.right_split = rs
        self.right_dock = self._dock("Parameters", None, rs, Qt.RightDockWidgetArea)
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
        self.resizeDocks([self.left_dock, self.right_dock], [245, 450], Qt.Horizontal)

    def _dock(self, name: str, heading: str | None, widget: QWidget, area) -> QDockWidget:
        d = QDockWidget(name, self)
        d.setObjectName(name + "Dock")
        d.setFeatures(QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable)
        d.setTitleBarWidget(_heading(heading) if heading else QWidget())
        d.setWidget(widget)
        self.addDockWidget(area, d)
        return d

    def _build_menus(self) -> None:
        mb = self.menuBar()
        f = mb.addMenu("&File")
        self._act(f, "Open case…", self.open_case, QKeySequence.Open)
        self._act(f, "Save case…", self.save_case, QKeySequence.Save)
        self._act(f, "Open result…", self.open_result)
        f.addSeparator()
        self._act(f, "Quit", self.close, QKeySequence.Quit)
        e = mb.addMenu("&Edit")
        self._act(e, "Reset parameters to defaults", self.reset_params)
        v = mb.addMenu("&View")
        self._act(v, "Toggle light/dark theme", self.toggle_theme, "Ctrl+T")
        self._act(v, "Reset window layout", self.reset_layout)
        s = mb.addMenu("&Simulation")
        self._act(s, "Run", self.start_run, "Ctrl+R")
        self._act(s, "Cancel", self.cancel_run, "Esc")
        va = mb.addMenu("&Validation")
        self._act(va, "Run reference + core tests", self.run_validation)
        h = mb.addMenu("&Help")
        self._act(h, "About PropSim", lambda: QMessageBox.about(
            self, "PropSim", "PropSim 0.1 (Phase 1 skeleton)\nInteractive 3D rocket propulsion simulator."))

    def _act(self, menu, text, slot, shortcut=None) -> QAction:
        a = QAction(text, self)
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
        self.params_title.setText(f"PARAMETERS — {ctrl.title}")
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
        self.setWindowTitle(f"PropSim — {mod} · case: {self.case_name}")

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
        self.bar.set_range(res.n)
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
        self.bar.set_range(res.n)
        self._refresh_tabs()
        self.ctrl.build_scene(self.vp, self.result_params, res)
        self.playback.seek(res.n - 1)
        self._refresh_runs_list()

    # =================================================================== files
    def open_case(self, path: str | None = None) -> None:
        path = path or QFileDialog.getOpenFileName(self, "Open case", str(self.home), "PropSim case (*.psim.json)")[0]
        if not path:
            return
        model, params = pio.load_case(path)
        self.case_name = Path(path).name
        self.select_model(model, params)

    def save_case(self, path: str | None = None) -> None:
        if self.ctrl is None:
            return
        path = path or QFileDialog.getSaveFileName(self, "Save case", str(self.home / "case.psim.json"),
                                                   "PropSim case (*.psim.json)")[0]
        if path:
            pio.save_case(path, self.ctrl.key, self.form.values().model_dump())
            self.case_name = Path(path).name
            self._sync_title()

    def open_result(self, path: str | None = None) -> None:
        path = path or QFileDialog.getOpenFileName(self, "Open result", str(self.runs_dir), "PropSim result (*.psim.h5)")[0]
        if path:
            res, model, params = pio.load_result_h5(path)
            self.load_result(res, model, params)

    def reset_params(self) -> None:
        if self.form:
            self.form.set_values(self.ctrl.Params())
            self._on_params_changed()

    def export(self, kind: str, path: str | None = None) -> None:
        """PNG (Matplotlib figure of the current tab), CSV, .mat, .h5. ``path`` bypasses the dialog (tests)."""
        if self.ctrl is None:
            return
        live = self.ctrl.mode == "live"
        r = self.live_state if live else self.result
        if r is None:
            self._log("nothing to export yet — run the simulation first")
            return
        p = self.form.values() if live else self.result_params
        ext = {"PNG": ".png", "CSV": ".csv", ".mat": ".mat", ".h5": ".h5"}[kind]
        if not path:
            path = QFileDialog.getSaveFileName(self, f"Export {kind}", str(self.home / f"export{ext}"),
                                               f"*{ext}")[0]
        if not path:
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
        self.log.appendPlainText(f"{time.strftime('%H:%M:%S')}  {s}")

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
            self.restoreState(st)

    def reset_layout(self) -> None:
        self.addDockWidget(Qt.LeftDockWidgetArea, self.left_dock)
        self.addDockWidget(Qt.RightDockWidgetArea, self.right_dock)
        for d in (self.left_dock, self.right_dock):
            d.setFloating(False)
            d.show()
        self.resizeDocks([self.left_dock, self.right_dock], [245, 450], Qt.Horizontal)
        self.centre.setSizes([550, 450])
        self.right_split.setSizes([750, 250])

    def closeEvent(self, ev) -> None:
        self.cancel_run()
        if self.use_settings:
            s = self._settings()
            s.setValue("geometry", self.saveGeometry())
            s.setValue("state", self.saveState())
        super().closeEvent(ev)
