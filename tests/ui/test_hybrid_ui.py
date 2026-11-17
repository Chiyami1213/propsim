"""Hybrid tools in the main window: selection, Live budget, scene reset on model switch, Run models via the worker."""
import time

import pytest
from PySide6.QtWidgets import QApplication

from propsim.modules import registry
from propsim.ui import worker as W
from propsim.ui.latex import LatexLabel
from propsim.ui.main_window import MainWindow

LIVE = ["hybrid.regression", "hybrid.port", "hybrid.flow", "hybrid.fuel", "hybrid.stability"]


@pytest.fixture(scope="module")
def win(tmp_path_factory):
    app = QApplication.instance() or QApplication([])
    w = MainWindow(home=tmp_path_factory.mktemp("home"), use_settings=False)
    w.resize(1600, 1000)
    w.show()
    app.processEvents()
    yield w
    w.close()
    W.shutdown()


def _wait(win, limit=90):
    t0 = time.time()
    while win.worker is not None and time.time() - t0 < limit:
        QApplication.processEvents()


def test_all_hybrid_models_are_in_the_tree():
    live = {c.key for entries in registry.TREE.values() for _, m, c in entries if c is not None and m == "live"}
    run = {c.key for entries in registry.TREE.values() for _, m, c in entries if c is not None and m == "run"}
    assert set(LIVE) <= live
    assert {"hybrid.burn0d", "hybrid.burn1d", "hybrid.n2o"} <= run


@pytest.mark.parametrize("key", LIVE)
def test_live_tool_renders_within_budget(win, key):
    win.select_model(key)
    for _ in range(5):
        QApplication.processEvents()
    assert win.ctrl.key == key and win.bar.badge.text() == "LIVE"
    assert win.last_eval_ms < 50.0, win.last_eval_ms
    plots = win.plot_tabs.grid_plots()
    assert len(plots) >= 6
    assert all(p.pi.ctrl.xGridCheck.isChecked() and p.pi.ctrl.yGridCheck.isChecked() for p in plots)
    for gp in plots:
        assert "$" in gp.xl.latex() and "$" in gp.yl.latex()
    for lbl in win.findChildren(LatexLabel):
        assert "O/F" not in lbl.latex(), lbl.latex()
    assert len(win.vp.readout_lbls) == 4 and all(lbl.latex() for lbl in win.vp.readout_lbls)


def test_switching_models_clears_the_previous_scene(win):
    win.select_model("hybrid.regression")
    QApplication.processEvents()
    names = set(win.vp.plotter.renderer.actors)
    assert "port" in names
    win.select_model("hybrid.port")
    QApplication.processEvents()
    names = set(win.vp.plotter.renderer.actors)
    assert "grain" in names and "port" not in names and "case" not in names
    win.select_model("hybrid.regression")                       # and back again: scene is rebuilt, not stale
    QApplication.processEvents()
    assert {"port", "case"} <= set(win.vp.plotter.renderer.actors) and "grain" not in set(win.vp.plotter.renderer.actors)


def test_choice_change_updates_live_tool(win):
    win.select_model("hybrid.port")
    before = win.ctrl.evaluate(win.form.values())["A_now"]
    row = win.form.rows["layout"]
    row.slider.setCurrentIndex(row.meta["choices"].index("center6"))
    QApplication.processEvents()
    assert win.ctrl.evaluate(win.form.values())["A_now"] > before


@pytest.mark.parametrize("key,seek", [("hybrid.burn1d", 6.0), ("hybrid.n2o", 6.0)])
def test_run_models_through_the_worker(win, key, seek):
    win.select_model(key)
    win.start_run()
    _wait(win)
    r = win.result
    assert r is not None and r.n == 601 and win.worker is None
    win.playback.pause()
    win.playback.seek_time(seek)
    for _ in range(10):
        QApplication.processEvents()
    assert win.vp.caption.latex().startswith(r"$t$ = ")
    assert [win.plot_tabs.tabText(i) for i in range(win.plot_tabs.count())][0] in ("Time histories", "Tank & feed")
    assert win.summary.latex()
    if key == "hybrid.n2o":
        assert {"tank", "tank_liquid"} <= set(win.vp.plotter.renderer.actors)
