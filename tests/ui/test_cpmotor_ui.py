"""CP motor models in the main window: standards (grid, LaTeX), run through the worker, 3D scene, sweep, US units."""
import time

import pytest
from PySide6.QtWidgets import QApplication

from propsim.ui import worker as W
from propsim.ui.main_window import MainWindow


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


@pytest.mark.parametrize("key, unit", [("solid.cpmotor", "MPa"), ("solid.cpmotor_us", "psi")])
def test_run_and_standards(win, key, unit):
    win.select_model(key)
    win.form.set_values({**win.form.values().model_dump(), "ends": "burning", "erosion": "pressure"})
    win.start_run()
    _wait(win)
    r = win.result
    assert r is not None and win.worker is None and r.meta["n_cases"] == 3
    win.playback.pause()
    win.playback.seek_time(0.5 * float(r.t[-1]))
    for _ in range(10):
        QApplication.processEvents()
    plots = win.plot_tabs.grid_plots()
    assert len(plots) >= 3 and all(p.pi.ctrl.xGridCheck.isChecked() and p.pi.ctrl.yGridCheck.isChecked() for p in plots)
    assert any(unit in p.yl.latex() for p in plots)
    assert {"bore", "end_head", "end_aft", "nozzle", "case"} <= set(win.vp.plotter.renderer.actors)
    assert unit in win.vp.readout_lbls[0].latex() and win.summary.latex()


def test_sweep_overlays_values_and_end_faces_recede(win):
    win.select_model("solid.cpmotor")
    win.form.set_values({**win.form.values().model_dump(), "sweep": "r_i", "ends": "burning", "erosion": "none"})
    win.start_run()
    _wait(win)
    r = win.result
    assert r.meta["sweep"] == "r_i" and len(win.plot_tabs.grid_plots()) >= 3
    g = win.ctrl.tabs(win.result_params, r)[0].plots[0]
    assert len(g.curves) == 3 and "r_i" in g.curves[0].label
    c = r.channels
    assert c["w"][0] == 0.0 and c["w"][-1] > 0.04 and c["Ab"].max() > c["Ab"][0]           # the burning area grows with the web
