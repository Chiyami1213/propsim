"""Acceptance test (PROPSIM_BUILD_PROMPT Section 5.4): offscreen launch, load the demo case, run, seek to t = 6.40 s,
save a screenshot, and check region geometry against the mockup proportions (+-10 %)."""
import time
from pathlib import Path

import pytest
from PySide6.QtGui import QAction
from PySide6.QtWidgets import QApplication, QDockWidget

from propsim.ui.latex import LatexLabel
from propsim.ui.main_window import MainWindow

from propsim.storage import platform_default

ROOT = Path(__file__).resolve().parents[2]
SHOT = platform_default() / "dev" / "phase1_screenshot_offscreen.png"


def _wait(win, limit=60):
    t0 = time.time()
    while win.worker is not None and time.time() - t0 < limit:
        QApplication.processEvents()


@pytest.fixture(scope="module")
def win(tmp_path_factory):
    app = QApplication.instance() or QApplication([])
    w = MainWindow(home=tmp_path_factory.mktemp("home"), use_settings=False)
    w.resize(1600, 1000)
    w.open_case(str(ROOT / "cases" / "hybrid_demo.psim.json"))
    w.show()
    app.processEvents()
    w.start_run()
    _wait(w)
    w.playback.pause()
    w.playback.seek_time(6.4)
    for _ in range(10):
        app.processEvents()
    yield w
    w.close()


def test_run_completed_and_seeked(win):
    assert win.result is not None and win.result.n == 601
    assert win.playback.time == pytest.approx(6.4, abs=0.01)
    assert win.bar.time_lbl.latex().startswith(r"$t$ = 6.40")


def test_regions_exist_and_proportions(win):
    assert (win.width(), win.height()) == (1600, 1000)
    assert win.left_dock.width() == pytest.approx(335, rel=0.10)
    assert win.right_dock.width() == pytest.approx(450, rel=0.10)
    assert win.bar.height() == 40
    h = lambda d: d.height()
    assert h(win.vp_dock) / (h(win.vp_dock) + h(win.plots_dock)) == pytest.approx(0.55, rel=0.10)
    assert h(win.right_dock) / (h(win.right_dock) + h(win.log_dock)) == pytest.approx(0.75, rel=0.10)
    assert win.vp.width() > 0.45 * win.width()                          # Modules is wide enough to show full labels
    plots = win.plot_tabs.grid_plots()[:3]
    xs = [p.mapTo(win, p.rect().topLeft()).x() for p in plots]
    assert len(plots) == 3 and xs == sorted(xs)
    assert all(p.pi.ctrl.xGridCheck.isChecked() and p.pi.ctrl.yGridCheck.isChecked() for p in plots)
    assert {d.objectName() for d in win.findChildren(QDockWidget)} == {
        "ModulesDock", "SavedRunsDock", "View3DDock", "PlotsDock", "ParametersDock", "RunLogDock"}
    for d in win.findChildren(QDockWidget):                       # every panel can be dragged and floated
        assert d.features() & QDockWidget.DockWidgetMovable and d.features() & QDockWidget.DockWidgetFloatable
        assert d.titleBarWidget() is not None and d.titleBarWidget().height() > 0
    assert len(win.vp.readout_lbls) == 4 and all(lbl.latex() for lbl in win.vp.readout_lbls)
    assert win.vp.colorbar.isVisible() and win.vp.caption.latex().startswith(r"$t$ = 6.40")
    for b in (win.vp.b_reset, win.vp.b_cut, win.vp.b_field, win.vp.b_bar, win.vp.b_snap):
        assert b.isVisible()
    assert set(win.bar.export_btns) == {"PNG", "CSV", ".mat", ".h5", ".py", ".m"}
    assert win.bar.badge.text() == "Run"


def test_math_labels_are_latex(win):
    for lbl in win.findChildren(LatexLabel):
        assert not any(a in lbl.latex() for a in ("O/F", "mdot", "r_dot", "Pc ")), lbl.latex()
    for gp in win.plot_tabs.grid_plots():
        assert "$" in gp.xl.latex() and "$" in gp.yl.latex()


def test_screenshot_saved(win):
    SHOT.parent.mkdir(parents=True, exist_ok=True)
    assert win.grab().save(str(SHOT))
    assert SHOT.stat().st_size > 20_000


def test_live_mode_exports_and_theme(win, tmp_path):
    win.open_case(str(ROOT / "cases" / "nozzle_demo.psim.json"))
    QApplication.processEvents()
    assert win.bar.badge.text() == "Live" and not win.bar.run_btn.isEnabled()
    before = win.live_state["cf"]
    row = win.form.rows["eps"]
    row.slider.setValue(row.slider.value() + 100)                       # drag the eps slider
    assert win.live_state["cf"] != before and win.last_eval_ms < 50
    for kind, name in (("CSV", "a.csv"), (".mat", "a.mat"), (".h5", "a.h5"), ("PNG", "a.png")):
        win.export(kind, str(tmp_path / name))
        assert (tmp_path / name).stat().st_size > 0
    from propsim.ui.theme import THEME_NAMES
    for name in THEME_NAMES:                                              # every color theme renders the window
        win.set_color_theme(name)
        QApplication.processEvents()
        assert win.grab().size().width() > 0
    win.set_color_theme("dark")


def test_case_save_load(win, tmp_path):
    win.open_case(str(ROOT / "cases" / "hybrid_demo.psim.json"))
    win.form.rows["n"].set_value(0.7)
    win.save_case(str(tmp_path / "x.psim.json"))
    win.open_case(str(tmp_path / "x.psim.json"))
    assert win.form.values().n == pytest.approx(0.7)
    assert "x.psim.json" in win.windowTitle()


def test_toy_run(win):
    win.select_model("toy.springmass")
    win.start_run()
    _wait(win, 30)
    assert win.result is not None and win.result.meta["E0"] > 0
    E = win.result.channels["E"]
    assert E[-1] < E[0]                                                 # damping dissipates energy


def test_cancel_stops_worker(win):
    win.select_model("hybrid.burn0d")
    win.start_run()
    win.cancel_run()
    _wait(win, 30)
    assert win.worker is None


def test_panels_restore_default_layout(win):
    win.vp_dock.setFloating(True)
    win.removeDockWidget(win.log_dock)
    QApplication.processEvents()
    act = next(a for a in win.findChildren(QAction) if a.text() == "Restore Default Layout")
    assert act.shortcut().toString() == "Ctrl+Shift+R"
    act.trigger()
    for _ in range(5):
        QApplication.processEvents()
    assert not any(d.isFloating() for d in win.docks) and all(d.isVisible() for d in win.docks)
    assert win.vp_dock.x() > win.left_dock.x() and win.right_dock.x() > win.vp_dock.x()
    assert win.plots_dock.y() > win.vp_dock.y() and win.log_dock.y() > win.right_dock.y()
