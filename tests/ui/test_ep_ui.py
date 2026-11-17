"""Electric tools in the main window: Live budget and standards, Run models through the process worker, 3D scenes."""
import time

import pytest
from PySide6.QtWidgets import QApplication

from propsim.ui import worker as W
from propsim.ui.latex import LatexLabel
from propsim.ui.main_window import MainWindow

LIVE = ["ep.trades", "ep.thermal", "ep.ion", "ep.hall"]


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


def _wait(win, limit=120):
    t0 = time.time()
    while win.worker is not None and time.time() - t0 < limit:
        QApplication.processEvents()


def _standards(win, n_plots=2):
    plots = win.plot_tabs.grid_plots()
    assert len(plots) >= n_plots
    assert all(p.pi.ctrl.xGridCheck.isChecked() and p.pi.ctrl.yGridCheck.isChecked() for p in plots)
    for gp in plots:
        assert "$" in gp.xl.latex() and "$" in gp.yl.latex()
    assert len(win.vp.readout_lbls) == 4 and all(lbl.latex() for lbl in win.vp.readout_lbls)
    for lbl in win.findChildren(LatexLabel):
        assert "E x B" not in lbl.latex()


@pytest.mark.parametrize("key", LIVE)
def test_live_tools(win, key):
    win.select_model(key)
    for _ in range(5):
        QApplication.processEvents()
    assert win.ctrl.key == key and win.bar.badge.text() == "Live"
    assert win.last_eval_ms < 50.0, win.last_eval_ms
    _standards(win)
    assert len(win.vp.plotter.renderer.actors) >= 2


def test_live_parameter_change_redraws(win):
    win.select_model("ep.ion")
    row = win.form.rows["Vb"]
    before = win.ctrl.evaluate(win.form.values())["F"]
    row.slider.setValue(row.slider.maximum())
    for _ in range(3):
        QApplication.processEvents()
    assert win.ctrl.evaluate(win.form.values())["F"] != pytest.approx(before)
    assert win.last_eval_ms < 50.0


@pytest.mark.parametrize("key, actor", [("ep.spiral", "craft"), ("ep.hall_run", "exb"), ("ep.optics", "ions")])
def test_run_models(win, key, actor):
    win.select_model(key)
    win.start_run()
    _wait(win)
    r = win.result
    assert r is not None and win.worker is None
    win.playback.pause()
    win.playback.seek_time(0.5 * float(r.t[-1]))
    for _ in range(10):
        QApplication.processEvents()
    _standards(win)
    assert win.vp.caption.latex() and win.summary.latex()
    assert actor in set(win.vp.plotter.renderer.actors)
