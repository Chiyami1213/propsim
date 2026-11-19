"""Main window: six dockable panels, bottom playback bar (Section 5.1). Wires every region together."""
from __future__ import annotations

import os
import re
import time
from pathlib import Path

from PySide6.QtCore import QByteArray, QProcess, QSettings, Qt, QTimer, QUrl
from PySide6.QtGui import QAction, QDesktopServices, QKeySequence, QStandardItem, QStandardItemModel
from PySide6.QtWidgets import (QApplication, QCheckBox, QDockWidget, QFileDialog, QFrame, QLabel, QListWidget,
                               QListWidgetItem, QMainWindow, QMessageBox, QScrollArea, QSplitter,
                               QTreeView, QVBoxLayout, QWidget)

from ..core import io as pio
from ..core.result import Result
from ..modules import descriptions, registry
from ..modules.base import ModelController
from . import latex
from .export import save_tab_png
from .. import GUIDE_URL, HELP_URL, __version__, storage
from .latex import LatexLabel, LatexParagraph
from .latex_widgets import LatexItemDelegate, LatexLog
from .text import apa
from .busy import busy, busy_action
from .panels import PAD, GuideTree, PanelTitle, add_panel_menu
from .params import ParamForm
from .playback import PlaybackBar, PlaybackModel
from .plots import PlotTabs
from .theme import THEME_NAMES, build_qss, theme
from .viewport3d import Viewport3D
from . import gc_guard
from .worker import RunWorker

MODULE_TITLES = {"Hybrid": "Hybrid Rocket", "Liquid": "Liquid Engine", "Solid": "Solid Rocket Motor",
                 "Electric": "Electric Propulsion", "Nuclear": "Nuclear Thermal Propulsion", "Vehicle": "Vehicle", "Toy": "Toy"}
KEY_ROLE = Qt.UserRole + 1


def color_scheme_for(theme_name: str):
    """The system color scheme (title bars, dialogs) that goes with a theme: light for Light, dark for the others."""
    return Qt.ColorScheme.Light if theme_name == "light" else Qt.ColorScheme.Dark


def app_settings() -> QSettings:
    """The app's settings store: the registry on Windows; tests switch the default format to a temporary INI file."""
    return QSettings(QSettings.defaultFormat(), QSettings.UserScope, "PropSim", "PropSim")


def default_home() -> Path:
    """Data folder: the one chosen in the app (File > Data Folder), else PROPSIM_HOME, else %APPDATA%/PropSim."""
    return storage.resolve_home(app_settings().value("data_dir") or None)


LAYOUT_VERSION = 2                                        # bump when the dock set changes: old saved states are ignored


def _heading(text: str) -> QLabel:
    lbl = QLabel(apa(text))
    lbl.setObjectName("heading")
    lbl.setContentsMargins(PAD, PAD, PAD, PAD // 2)
    return lbl                                            # plain heading (dock titles use PanelTitle)


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
        self.design = None                                 # the active design the analysis modules start from
        self.bindings: dict[str, tuple[QAction, str]] = {}      # command -> (action, default shortcut)
        self.confirm_close = False                         # the app entry point turns this on
        self._layout_restored = False
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
        self._prev_values: dict = {}
        self.playback = PlaybackModel(self)
        self._build_ui()
        self._build_menus()
        self.load_shortcuts()
        self._apply_theme()
        theme.changed.connect(lambda *_: self._apply_theme())
        theme.changed.connect(self._save_theme)
        self._load_saved_runs()
        self.playback.frameChanged.connect(self._on_frame)
        if use_settings:
            self._restore_theme()
            self._restore_layout()
        self.apply_preferences()
        self._layout_on_show = not self._layout_restored    # Qt redistributes dock sizes on first show
        self._tour = None
        if use_settings:
            QTimer.singleShot(900, self._offer_tutorial)
        self._log(f"data folder: {self.home}")
        if migrated:
            self._log(f"copied {migrated} saved files from the previous data folder (~/.propsim)")

    # =================================================================== construction
    def _build_ui(self) -> None:
        """Six dockable panels (drag any title bar; View > Restore Default Layout puts them back), one padding."""
        self.setDockOptions(QMainWindow.AnimatedDocks | QMainWindow.AllowNestedDocks | QMainWindow.AllowTabbedDocks)
        self.setDockNestingEnabled(True)
        # --- modules tree ---
        self.tree = GuideTree()
        self.tree.setHeaderHidden(True)
        self.tree.setEditTriggers(QTreeView.NoEditTriggers)
        self.tree_model = QStandardItemModel()
        for mod, entries in registry.TREE.items():
            top = QStandardItem(apa(mod))
            top.setFlags(Qt.ItemIsEnabled)
            top.setToolTip(descriptions.MODULES.get(mod, ""))
            for label, mode, cls in entries:
                kind = {"live": "Live", "run": "Run", "tool": "Editor"}[mode]
                it = QStandardItem(f"{apa(label)}  ({kind})")
                it.setData(cls.key if cls else None, KEY_ROLE)
                it.setToolTip(descriptions.MODELS.get(cls.key, "") if cls else descriptions.PLANNED)
                it.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable if cls else Qt.NoItemFlags)
                top.appendRow(it)
            self.tree_model.appendRow(top)
        self.tree.setModel(self.tree_model)
        self.tree.setItemDelegate(LatexItemDelegate(9.5, "dim", "accent", self.tree))
        self.tree.setIndentation(18)
        self.tree.expand(self.tree_model.index(list(registry.TREE).index("Hybrid"), 0))
        self.tree.selectionModel().currentChanged.connect(self._on_tree)
        self.left_dock = self._dock("Modules", "Modules", _panel(self.tree))
        # --- saved runs ---
        self.runs_list = QListWidget()
        self.runs_list.itemClicked.connect(self._on_run_clicked)
        self.compare_cb = QCheckBox("Compare Overlay")
        self.compare_cb.toggled.connect(lambda _: self._refresh_tabs())
        self.runs_dock = self._dock("SavedRuns", "Saved Runs", _panel(self.runs_list, self.compare_cb))
        # --- 3D view and plots ---
        self.vp = Viewport3D()
        self.plot_tabs = PlotTabs()
        self.vp_dock = self._dock("View3D", "3D View", _panel(self.vp))
        self.plots_dock = self._dock("Plots", "Plots", _panel(self.plot_tabs))
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
        self.params_title.setContentsMargins(0, PAD // 2, 0, 0)       # air between the panel title bar and the model name
        self.right_dock = self._dock("Parameters", "Parameters", body)
        # --- run log ---
        self.log = LatexLog()
        self.summary = LatexParagraph("", "accent", 9.5)
        self.log_dock = self._dock("RunLog", "Run Log", _panel(self.log, self.summary))
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

    def _dock(self, name: str, text: str, widget: QWidget) -> QDockWidget:
        d = QDockWidget(apa(text), self)
        d.setObjectName(name + "Dock")
        d.setFeatures(QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable | QDockWidget.DockWidgetClosable)
        d.setTitleBarWidget(PanelTitle(d, text))           # drag grip top left, close button top right
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
        self.resizeDocks([self.left_dock, self.vp_dock, self.right_dock], [335, w - 335 - 450, 450], Qt.Horizontal)
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
        self._act(f, "Open design", self.open_design)
        self._act(f, "Screenshot", self.take_screenshot, "F2")
        f.addSeparator()
        self._act(f, "Restore last session", self.restore_session)
        self._act(f, "Data folder", self.choose_data_folder)
        self._act(f, "Open data folder", lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.home))))
        f.addSeparator()
        tx = f.addMenu(apa("Send thrust to vehicle"))
        for k in (1, 2, 3):
            self._act(tx, f"Stage {k}", lambda k=k: self.send_thrust(k))
        rp = f.addMenu(apa("Export report"))
        self._act(rp, "HTML report (.html)", lambda: self.export_report("html"))
        self._act(rp, "PDF report (.pdf)", lambda: self.export_report("pdf"))
        self._act(rp, "LaTeX tables and figures (.tex)", lambda: self.export_report("tex"))
        self._act(f, "Export thrust curve (.eng)", lambda: self.export(".eng"))
        x = f.addMenu(apa("Export code"))
        self._act(x, "Python script (.py)", lambda: self.export(".py"))
        self._act(x, "MATLAB script (.m)", lambda: self.export(".m"))
        f.addSeparator()
        self._act(f, "Quit", self.close, QKeySequence.Quit)
        e = mb.addMenu("&Edit")
        self._act(e, "Reset parameters to defaults", self.reset_params)
        e.addSeparator()
        self._act(e, "Preferences", self.open_preferences, "F9")
        v = mb.addMenu("&View")
        add_panel_menu(v.addMenu(apa("Panels")), self.docks)
        self._act(v, "Color theme", self.choose_theme, "F10")
        self._act(v, "Restore default layout", self.reset_layout, "F12")
        self.full_screen_act = self._act(v, "Full screen", self.toggle_full_screen, "F11")
        d = mb.addMenu("&Design")                          # build the engine or motor first, then analyze it
        self._act(d, "Liquid engine designer", self.open_cycle_sandbox, "Ctrl+E")
        self._act(d, "Solid motor designer", lambda: self.open_grain_designer("solid"), "Ctrl+G")
        self._act(d, "Hybrid motor designer", lambda: self.open_grain_designer("hybrid"))
        for name in ("Nuclear thermal designer", "Electric propulsion designer"):
            self._act(d, name + " (planned)", lambda: None).setEnabled(False)
        s = mb.addMenu("&Analysis")                        # test and analyze a design with the analysis modules
        self._act(s, "Run", self.start_run, "Ctrl+R")
        self._act(s, "Cancel", self.cancel_run, "Esc")
        s.addSeparator()
        self._act(s, "Use active design in this module", self.apply_design)
        self._act(s, "Clear active design", self.clear_design)
        s.addSeparator()
        va = s.addMenu(apa("Validation"))
        self._act(va, "Run reference + core tests", self.run_validation)
        h = mb.addMenu("&Help")
        self._act(h, "Tutorial", self.start_tutorial, "F1")
        self._act(h, "User guide", lambda: QDesktopServices.openUrl(QUrl(GUIDE_URL)))
        self._act(h, "Online help (GitHub)", lambda: QDesktopServices.openUrl(QUrl(HELP_URL)))
        self._act(h, "About PropSim", lambda: QMessageBox.about(
            self, "About PropSim", f"PropSim {__version__}\nInteractive 3D Rocket Propulsion Simulator"))

    def _act(self, menu, text, slot, shortcut=None) -> QAction:
        a = QAction(apa(text), self)                       # menus use APA title case
        a.triggered.connect(lambda _=False: slot())
        if shortcut:
            a.setShortcut(shortcut)
        menu.addAction(a)
        name = f"{menu.title().replace('&', '')}: {a.text()}"
        self.bindings[name] = (a, a.shortcut().toString(QKeySequence.PortableText))
        return a

    # ---- preferences: key bindings and appearance options ---------------------------------------------
    def set_shortcut(self, name: str, seq: str) -> None:
        """Give command ``name`` the shortcut ``seq`` (portable text, empty = none); the choice is remembered when it
        differs from the default."""
        action, default = self.bindings[name]
        action.setShortcut(QKeySequence(seq, QKeySequence.PortableText) if seq else QKeySequence())
        if self.use_settings:
            s = self._settings()
            if seq == default:
                s.remove("shortcuts/" + name)
            else:
                s.setValue("shortcuts/" + name, seq or "none")

    def load_shortcuts(self) -> None:
        if not self.use_settings:
            return
        s = self._settings()
        for name, (action, _) in self.bindings.items():
            v = s.value("shortcuts/" + name)
            if isinstance(v, str):
                action.setShortcut(QKeySequence() if v == "none" else QKeySequence(v, QKeySequence.PortableText))

    def pref_bool(self, name: str, default: bool) -> bool:
        if not self.use_settings:
            return default
        v = self._settings().value("pref/" + name)
        return default if v is None else v in (True, "true", "1", 1)

    def set_pref(self, name: str, value: bool) -> None:
        if self.use_settings:
            self._settings().setValue("pref/" + name, bool(value))

    def apply_preferences(self) -> None:
        self.vp.set_legend_visible(self.pref_bool("show_legend", True))
        self.vp.set_colorbar_visible(self.pref_bool("show_colorbar", True))
        if getattr(self, "confirm_close_managed", False):
            self.confirm_close = self.pref_bool("confirm_quit", True)

    def open_preferences(self) -> None:
        from .preferences import PreferencesDialog
        PreferencesDialog(self).exec()

    # =================================================================== theme
    @staticmethod
    def _sync_color_scheme() -> None:
        """Window frames and title bars (the system's, dark by default on a dark Windows) follow the app theme, so a
        dialog in the light theme does not get a black title bar."""
        from PySide6.QtGui import QGuiApplication
        try:
            QGuiApplication.styleHints().setColorScheme(color_scheme_for(theme.name))
        except (AttributeError, RuntimeError):                  # Qt older than 6.8: the system scheme stays
            pass

    def _apply_theme(self) -> None:
        latex.clear_cache()
        QApplication.instance().setStyleSheet(build_qss(theme.t))
        self._sync_color_scheme()
        for w in self.findChildren(LatexLabel):
            w._render()

    def take_screenshot(self) -> Path | None:
        """F2: save the whole window as a PNG in ``<data folder>/screenshots`` (the screen pixels, so the 3D view is in
        it too); returns the file."""
        folder = self.home / "screenshots"
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / time.strftime("screenshot_%Y%m%d_%H%M%S.png")
        n = 1
        while path.exists():                                  # two shots in one second
            path = folder / time.strftime(f"screenshot_%Y%m%d_%H%M%S_{n}.png")
            n += 1
        screen = self.screen()
        pm = screen.grabWindow(self.winId()) if (screen is not None and not self.vp.headless) else self.grab()
        if pm.isNull() or not pm.save(str(path), "PNG"):
            self._log("screenshot failed")
            return None
        self._log(f"screenshot saved: {path}")
        return path

    # ---- designs ---------------------------------------------------------------------------------------
    def set_design(self, design) -> None:
        """Make ``design`` the active one: analysis modules of its kind start from it when they are opened."""
        self.design = design
        self._log(f"active design: {design.name} ({design.kind}); open an analysis module to use it")
        self._sync_title()

    def clear_design(self) -> None:
        self.design = None
        self._log("no active design: analysis modules use their own defaults")
        self._sync_title()

    def apply_design(self) -> None:
        """Reload the open analysis module from the active design."""
        from .. import design as D
        if self.ctrl is not None and D.applies_to(self.design, self.ctrl.key):
            self.select_model(self.ctrl.key, None)
        else:
            self._log("the active design does not apply to this module")

    @busy_action("Opening design")
    def open_design(self, path: str | None = None) -> None:
        """Open a saved design, make it active and show it in its designer."""
        from .. import design as D
        if not path:
            path, _ = QFileDialog.getOpenFileName(self, "Open Design", str(self.home / "designs"),
                                                  "PropSim design (*.psimdesign.json)")
        if not path:
            return
        try:
            d = D.Design.load(path)
        except (ValueError, OSError, KeyError) as e:
            self._log(f"cannot open design: {e}")
            return
        self.set_design(d)
        if d.kind == "liquid":
            self.open_cycle_sandbox()
            self.sandbox.set_doc(d.data)
        elif d.kind in ("solid", "hybrid"):
            self.open_grain_designer(d.kind)
            self.grain_win.load_design(d)

    def toggle_full_screen(self) -> None:
        """F11: full-screen mode on and off (the window keeps its normal size and place for the way back)."""
        if self.isFullScreen():
            self.showNormal()
        else:
            self.showFullScreen()

    def start_tutorial(self) -> None:
        """Guided tour (arrow keys move between steps, Esc leaves); also offered once on the first launch."""
        from .tutorial import TutorialOverlay
        if getattr(self, "_tour", None) is not None:
            return
        self._tour = TutorialOverlay(self)
        self._tour.finished.connect(self._tour_done)

    def _tour_done(self, done: bool) -> None:
        self._tour = None
        if self.use_settings:
            self._settings().setValue("tour_done", True)

    def _offer_tutorial(self) -> None:
        if not self.use_settings or not self.isVisible() or self._settings().value("tour_done") in (True, "true"):
            return
        self._settings().setValue("tour_done", True)                  # ask once, whatever the answer
        box = QMessageBox(self)
        box.setWindowTitle(apa("Welcome to PropSim"))
        box.setText("Take a one-minute tour of the window? You can move between steps with the left and right arrow "
                    "keys, and replay it any time from Help, Tutorial.")
        go = box.addButton("Start Tour", QMessageBox.AcceptRole)
        box.addButton("Not Now", QMessageBox.RejectRole)
        box.exec()
        if box.clickedButton() is go:
            self.start_tutorial()

    def choose_theme(self) -> None:
        from .theme_picker import ThemeDialog
        dlg = ThemeDialog(self)
        dlg.exec()

    def set_color_theme(self, name: str) -> None:
        theme.set(name)

    # =================================================================== model selection
    def _on_tree(self, cur, _prev=None) -> None:
        key = cur.data(KEY_ROLE) if cur.isValid() else None
        if key in registry.TOOL_KEYS:
            self.open_tool(key)
            return
        if key and (self.ctrl is None or key != self.ctrl.key):
            self.select_model(key)

    def open_tool(self, key: str) -> None:
        """Open the window of a tree "tool" entry (cycle sandbox, solid or hybrid motor designer)."""
        if key == registry.CycleSandboxTool.key:
            self.open_cycle_sandbox()
        else:
            self.open_grain_designer("solid" if key == registry.GrainDesignerSolidTool.key else "hybrid")

    def open_grain_designer(self, mode: str = "solid") -> None:
        """Open (or raise) the solid or hybrid motor designer window, in solid or hybrid mode."""
        from .grain_designer import GrainDesigner
        if getattr(self, "grain_win", None) is None:
            self.grain_win = GrainDesigner(self, mode)
        elif self.grain_win.mode != mode:
            self.grain_win.mode_box.setCurrentIndex(self.grain_win.mode_box.findData(mode))
        self.grain_win.show()
        self.grain_win.raise_()
        self.grain_win.activateWindow()

    def open_cycle_sandbox(self) -> None:
        """Open (or raise) the liquid-engine cycle sandbox window."""
        from .cycle_sandbox import CycleSandbox
        if getattr(self, "sandbox", None) is None:
            self.sandbox = CycleSandbox(self)
        self.sandbox.show()
        self.sandbox.raise_()
        self.sandbox.activateWindow()

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
        if params is None:
            from .. import design as D
            up, clamped = D.fit(D.updates(self.design, key), ctrl.Params)
            if up:
                params = up
                self._log(f"{ctrl.title}: parameters taken from the design '{self.design.name}'")
                if clamped:
                    self._log("outside this module's slider range, limited: " + ", ".join(clamped))
        base = ctrl.Params(**params) if params else None
        self.form = ParamForm(ctrl.Params, base)
        self._prev_values = self.form.values().model_dump()
        self.form.changed.connect(self._on_params_changed)
        self.form.hovered.connect(self.hover.setLatex)
        self.scroll.setWidget(self.form)
        self.params_title.setLatex(apa(ctrl.title))
        self.vp.plotter.clear()                  # drop the previous model's meshes (their handles are reset below)
        ctrl.reset_scene()
        self.vp.set_fields(ctrl.fields)
        self.vp.legend_labels = ctrl.legend_labels
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
        design = f" · Design: {self.design.name}" if getattr(self, "design", None) else ""
        self.setWindowTitle(f"PropSim — {apa(mod)} · Case: {self.case_name}{design}")

    # =================================================================== Live mode
    def _on_params_changed(self) -> None:
        p = self.form.values()
        upd = self.ctrl.preset_updates(p, self._prev_values) if self.ctrl is not None else {}
        if upd:
            note = upd.pop("_note", "")
            self.form.set_values(upd)                                 # silent: written once, refreshed below
            if note:
                self._log(note)
        self._prev_values = self.form.values().model_dump()
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
            self.ctrl.prepare_params(p, self.home)
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
    @busy_action("Opening case")
    def open_case(self, path: str | None = None) -> None:
        path = path or QFileDialog.getOpenFileName(self, apa("Open case"), str(self.home / "cases"), "PropSim case (*.psim.json)")[0]
        if not path:
            return
        model, params = pio.load_case(path)
        self.case_name = Path(path).name
        self.select_model(model, params)

    @busy_action("Saving case")
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

    @busy_action("Opening result")
    def open_result(self, path: str | None = None) -> None:
        path = path or QFileDialog.getOpenFileName(self, apa("Open result"), str(self.runs_dir), "PropSim result (*.psim.h5)")[0]
        if path:
            res, model, params = pio.load_result_h5(path)
            self.load_result(res, model, params)

    def reset_params(self) -> None:
        if self.form:
            self.form.set_values(self.ctrl.Params())
            self._on_params_changed()

    @busy_action("Exporting")
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
        if kind == ".eng" and not hasattr(self.ctrl, "export_eng"):
            self._log("a thrust curve (.eng) can be exported from Solid, Internal ballistics after a run")
            return
        ext = {"PNG": ".png", "CSV": ".csv", ".mat": ".mat", ".h5": ".h5", ".py": ".py", ".m": ".m", ".eng": ".eng"}[kind]
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
        if kind == ".eng":
            Path(path).write_text(self.ctrl.export_eng(p, r), encoding="utf-8")
            self._log(f"exported {Path(path).name}")
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
        return app_settings()

    def _save_theme(self, name: str) -> None:
        if self.use_settings:
            self._settings().setValue("theme", name)

    def _restore_theme(self) -> None:
        name = self._settings().value("theme")
        if isinstance(name, str) and name in THEME_NAMES:
            theme.set(name)

    def _restore_layout(self) -> None:
        s = self._settings()
        g, st = s.value("geometry"), s.value("state")
        if isinstance(g, QByteArray):
            self.restoreGeometry(g)
        if isinstance(st, QByteArray):
            self._layout_restored = self.restoreState(st, LAYOUT_VERSION)   # False for an older dock set

    def showEvent(self, ev) -> None:
        super().showEvent(ev)
        if self._layout_on_show:
            self._layout_on_show = False
            QTimer.singleShot(0, self.default_layout)

    def reset_layout(self) -> None:
        self.default_layout()

    @busy_action("Exporting report")
    def export_report(self, kind: str, path: str | None = None) -> None:
        """HTML / PDF report or LaTeX tables of the current model, parameters, results and plots."""
        if self.ctrl is None:
            return
        live = self.ctrl.mode == "live"
        r = self.live_state if live else self.result
        if r is None:
            self._log("nothing to report yet: run the simulation first")
            return
        p = self.form.values() if live else self.result_params
        ext = {"html": ".html", "pdf": ".pdf", "tex": ".tex"}[kind]
        stem = re.sub(r"[^A-Za-z0-9]+", "_", self.ctrl.title.lower()).strip("_") + "_report"
        if not path:
            path = QFileDialog.getSaveFileName(self, apa("Export report"), str(self.home / "exports" / f"{stem}{ext}"),
                                               f"*{ext}")[0]
        if not path:
            return
        from .. import report
        log = [] if live else list(r.log)
        try:
            files = report.export(kind, path, self.ctrl, p, r, log)
        except Exception as e:                                          # noqa: BLE001 - report, never crash the app
            self._log(f"report failed: {e}")
            return
        self._log("exported report " + ", ".join(f.name for f in files))

    def send_thrust(self, stage: int) -> None:
        """Save the thrust curve of the current hybrid / solid / liquid Run result as the thrust profile of a vehicle stage
        (``<data folder>/profiles/stage<k>.json``); the vehicle model flies it when that stage's source is *profile*."""
        from ..core.trajectory import profiles
        if self.ctrl is None or self.ctrl.mode != "run" or self.result is None:
            self._log("run a hybrid, solid or liquid model first, then send its thrust to the vehicle")
            return
        try:
            prof = profiles.from_result(self.ctrl.key, self.result)
        except ValueError as e:
            self._log(str(e))
            return
        path = profiles.save(prof, self.home / "profiles" / f"stage{stage}.json")
        I, m = profiles.impulse(prof)
        self._log(f"thrust profile of {self.ctrl.title} saved for vehicle stage {stage}: burn {prof['t'][-1]:.2f} (s), "
                  f"peak {max(prof['F']) / 1e3:.2f} (kN), impulse {I / 1e3:.1f} (kN·s), propellant {m:.2f} (kg) -> {path.name}")

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

    def _confirm_quit(self) -> bool:
        """Ask before leaving: True to quit.  Mentions a run that is still going."""
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Question)
        box.setWindowTitle(apa("Quit PropSim"))
        box.setText("Quit PropSim?")
        info = "Your session is backed up automatically and can be restored from File, Restore Last Session."
        if self.worker is not None:
            info = "A run is in progress and will be cancelled. " + info
        box.setInformativeText(info)
        quit_btn = box.addButton("Quit", QMessageBox.AcceptRole)
        cancel = box.addButton("Cancel", QMessageBox.RejectRole)
        box.setDefaultButton(cancel)
        box.setEscapeButton(cancel)
        box.exec()
        return box.clickedButton() is quit_btn

    def closeEvent(self, ev) -> None:
        if self.confirm_close and not self._confirm_quit():
            ev.ignore()
            return
        self.backup_session()
        self.cancel_run()
        if self.use_settings:
            s = self._settings()
            s.setValue("geometry", self.saveGeometry())
            s.setValue("state", self.saveState(LAYOUT_VERSION))
        super().closeEvent(ev)
