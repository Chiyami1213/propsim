"""Vehicle tools in the main window: Live budget and standards, the ascent Run model through the worker, and the
hybrid-to-vehicle thrust-profile hand-off (File > Send Thrust to Vehicle)."""
import json
import time

import pytest
from PySide6.QtWidgets import QApplication

from propsim.ui import worker as W
from propsim.ui.latex import LatexLabel
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


def _standards(win):
    plots = win.plot_tabs.grid_plots()
    assert len(plots) >= 2
    assert all(p.pi.ctrl.xGridCheck.isChecked() and p.pi.ctrl.yGridCheck.isChecked() for p in plots)
    for gp in plots:
        assert "$" in gp.xl.latex() and "$" in gp.yl.latex()
    assert len(win.vp.readout_lbls) == 4 and all(lbl.latex() for lbl in win.vp.readout_lbls)
    for lbl in win.findChildren(LatexLabel):
        assert "Δv" not in lbl.latex()


def test_live_staging_tool(win):
    win.select_model("vehicle.staging")
    for _ in range(5):
        QApplication.processEvents()
    assert win.bar.badge.text() == "Live" and win.last_eval_ms < 50.0, win.last_eval_ms
    _standards(win)
    assert {"stage0", "payload"} <= set(win.vp.plotter.renderer.actors)
    row = win.form.rows["s1_prop"]
    before = win.ctrl.evaluate(win.form.values())["dv"][0]
    row.slider.setValue(row.slider.value() + 200)
    QApplication.processEvents()
    assert win.ctrl.evaluate(win.form.values())["dv"][0] > before and win.last_eval_ms < 50.0


def test_ascent_run_orbit(win):
    win.select_model("vehicle.ascent")
    win.start_run()
    _wait(win)
    r = win.result
    assert r is not None and win.worker is None and r.meta["orbit"] and not r.meta["impact"]
    win.playback.pause()
    win.playback.seek_time(0.3 * float(r.t[-1]))
    for _ in range(10):
        QApplication.processEvents()
    _standards(win)
    assert {"craft", "path", "flown"} <= set(win.vp.plotter.renderer.actors)
    assert win.vp.caption.latex() and win.summary.latex()


def test_thrust_profile_hand_off(win):
    win.send_thrust(1)                                                   # nothing to send yet: a log message, no crash
    win.select_model("hybrid.burn0d")
    win.start_run()
    _wait(win)
    win.send_thrust(1)
    path = win.home / "profiles" / "stage1.json"
    prof = json.loads(path.read_text(encoding="utf-8"))
    assert prof["source"] == "hybrid.burn0d" and max(prof["F"]) > 1000.0 and prof["t"][0] == 0.0
    win.select_model("vehicle.ascent")
    win.form.set_values({**win.form.values().model_dump(), "n_stages": "1", "s1_src": "profile", "s1_dry": 20.0,
                         "s1_prop": 30.0, "s1_d": 0.2, "m_payload": 3.0, "m_fair": 0.0, "elevation": 88.0,
                         "rail": 4.0, "h_kick": 0.0, "kick": 0.0, "guid": "gt", "perigee_cut": 0.0, "t_max": 600.0})
    win.start_run()
    _wait(win)
    r = win.result
    assert r is not None and r.meta["impact"]
    burnout = dict(r.meta["events"])["Stage 1 burnout"]
    assert burnout == pytest.approx(prof["t"][-1], abs=0.05)             # flew the imported curve
    assert any("imported thrust profile" in line for line in r.log)


def test_missing_profile_is_reported(win):
    (win.home / "profiles" / "stage1.json").unlink()
    win.select_model("vehicle.ascent")
    win.form.set_values({**win.form.values().model_dump(), "s1_src": "profile"})
    win.start_run()
    for _ in range(5):
        QApplication.processEvents()
    assert win.worker is None                                          # refused before any process starts
    assert "none was sent" in win.log.toPlainText()


def test_engine_presets_fill_the_stage_from_the_library(win):
    from propsim import data
    win.select_model("vehicle.ascent")
    row = win.form.rows["s1_engine"]
    row.slider.setCurrentIndex(row.meta["choices"].index("RS25"))
    for _ in range(4):
        QApplication.processEvents()
    v = win.form.values()
    assert v.s1_F == pytest.approx(data.engine_value("RS25", "engine", "thrust", "vac") / 1e3, abs=0.06)
    assert v.s1_isp == pytest.approx(452.3, abs=0.06)
    Ae = (data.engine_value("RS25", "engine", "thrust", "vac") - data.engine_value("RS25", "engine", "thrust", "sl")) / 101325.0
    assert v.s1_Ae == pytest.approx(Ae * 1e4, abs=10.0)                      # the slider steps in 10 cm²
    log = win.log.toPlainText()
    assert "component library" in log and "RS-25" in log and "derived from the sea-level and vacuum thrust" in log
    row.slider.setCurrentIndex(row.meta["choices"].index("custom"))
    QApplication.processEvents()
    assert win.form.values().s1_F == v.s1_F                                    # choosing Custom leaves the values alone
    win.form.rows["s2_engine"].slider.setCurrentIndex(win.form.rows["s2_engine"].meta["choices"].index("RL10C_1"))
    for _ in range(4):
        QApplication.processEvents()
    v = win.form.values()
    assert v.s2_isp == pytest.approx(449.7, abs=0.06)
    assert "treated as vacuum" in win.log.toPlainText()
