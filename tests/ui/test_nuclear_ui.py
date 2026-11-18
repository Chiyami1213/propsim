"""Nuclear tools in the main window: Live budget and standards, the kinetics Run model through the process worker,
3D scenes, and every cycle / geometry / heat-transfer choice."""
import time

import pytest
from PySide6.QtWidgets import QApplication

from propsim.ui import worker as W
from propsim.ui.main_window import MainWindow

LIVE = ["nuc.perf", "nuc.crit", "nuc.element", "nuc.cycles", "nuc.poisons"]


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


def _pump(n=5):
    for _ in range(n):
        QApplication.processEvents()


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


def _set_choice(win, name, value):
    row = win.form.rows[name]
    row.slider.setCurrentIndex(row.meta["choices"].index(value))
    _pump()


@pytest.mark.parametrize("key", LIVE)
def test_live_tools(win, key):
    win.select_model(key)
    _pump()
    assert win.ctrl.key == key and win.bar.badge.text() == "Live"
    assert win.last_eval_ms < 50.0, win.last_eval_ms
    _standards(win)
    assert len(win.vp.plotter.renderer.actors) >= 2


@pytest.mark.parametrize("name, values", [("kind", ["cold", "expander", "hot"])])
def test_cycle_choices(win, name, values):
    win.select_model("nuc.cycles")
    _pump()
    for v in values:
        _set_choice(win, name, v)
        assert win.form.values().kind == v
        assert win.vp.caption.latex()


def test_geometry_and_heat_transfer_choices(win):
    win.select_model("nuc.crit")
    _pump()
    for g in ("sphere", "slab", "cylinder"):
        _set_choice(win, "geom", g)
        assert "core" in set(win.vp.plotter.renderer.actors)
    win.select_model("nuc.element")
    _pump()
    _set_choice(win, "hc_mode", "db")
    r = win.ctrl.evaluate(win.form.values())
    assert r["hc"][0] != pytest.approx(r["hc"][-1])                  # Dittus-Boelter varies along the channel


@pytest.mark.parametrize("scenario", ["startup", "shutdown"])
def test_kinetics_run(win, scenario):
    win.select_model("nuc.kinetics")
    _pump()
    _set_choice(win, "scenario", scenario)
    win.start_run()
    _wait(win)
    r = win.result
    assert r is not None and win.worker is None
    win.playback.pause()
    win.playback.seek_time(0.5 * float(r.t[-1]))
    _pump(10)
    _standards(win)
    assert win.vp.caption.latex() and win.summary.latex()
    assert {"core", "reflector", "drum0"} <= set(win.vp.plotter.renderer.actors)
