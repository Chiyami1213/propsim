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
    assert set(LIVE) <= live | set(registry.EXTRA)                # Port & grain lives in the grain designer window
    assert {"hybrid.burn0d", "hybrid.burn1d", "hybrid.n2o"} <= run


@pytest.mark.parametrize("key", LIVE)
def test_live_tool_renders_within_budget(win, key):
    win.select_model(key)
    for _ in range(5):
        QApplication.processEvents()
    assert win.ctrl.key == key and win.bar.badge.text() == "Live"
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
    assert [win.plot_tabs.tabText(i) for i in range(win.plot_tabs.count())][0] in ("Time Histories", "Tank && Feed")
    assert win.summary.latex()
    if key == "hybrid.n2o":
        assert {"tank", "tank_liquid"} <= set(win.vp.plotter.renderer.actors)


def test_custom_port_and_fuel_stack_editor(win):
    """Custom port and fuel-grain stack reach the quasi-1D hybrid model through the form."""
    import json

    from PySide6.QtWidgets import QDialogButtonBox

    from propsim.core.grain import stack as gs
    from propsim.ui import grain_editor as ge

    win.select_model("hybrid.burn1d")
    form = win.form
    form.rows["kind"].slider.setCurrentIndex(form.rows["kind"].meta["choices"].index("custom"))
    ctx = {k: r.value() for k, r in form.rows.items()}
    sec = ge.SectionEditor("", ctx)
    assert sec.R_out == pytest.approx(ctx["R_p0"] + ctx["web"])         # hybrid: outer radius = port radius + web
    sec.table.item(0, 3).setText("r=20")
    form.rows["section_json"].set_value(sec.text())
    st = ge.HybridStackEditor("", ctx)
    assert st.table.isColumnHidden(1) and st.table.isColumnHidden(2)     # no end faces for hybrid grains
    st._add(gs.GrainDef(0.3))
    st.table.item(1, 4).setText("rho_f=900; a=0.1171; n=0.62; law=power")
    st.refresh()
    assert st.buttons.button(QDialogButtonBox.Ok).isEnabled()
    st.table.item(1, 4).setText("zz=1")
    st.refresh()
    assert not st.buttons.button(QDialogButtonBox.Ok).isEnabled()
    st.table.item(1, 4).setText("rho_f=900; a=0.1171; n=0.62; law=power")
    text = st.text()
    form.rows["stack_json"].set_value(text)
    QApplication.processEvents()
    m = form.values().to_si()
    assert len(m.grains) == 2 and m.grains[1].rho_f == 900.0 and m.grains[1].law == "power"
    assert json.loads(text)[1]["prop"]["law"] == "power"


def test_grain_designer_window_for_both_motor_types(qapp, tmp_path):
    from propsim.ui.main_window import MainWindow
    win = MainWindow(home=tmp_path, use_settings=False)
    win.open_grain_designer("solid")
    gw = win.grain_win
    assert gw.mode == "solid" and gw.ctrl.key == "solid.grain" and gw.plots.count() >= 1
    win.open_grain_designer("hybrid")
    assert gw.mode == "hybrid" and gw.ctrl.key == "hybrid.port"
    gw.open_in_main()
    assert win.ctrl.key == "hybrid.port"
    keys = [c.key for entries in registry.TREE.values() for _, m, c in entries if m == "tool"]
    assert {"grain.solid", "grain.hybrid", "liquid.sandbox"} <= set(keys)


def test_nuclear_designer_keeps_each_study_and_feeds_the_analysis(qapp, tmp_path):
    from propsim import design as D
    from propsim.ui.main_window import MainWindow
    win = MainWindow(home=tmp_path, use_settings=False)
    win.open_nuclear_designer()
    nd = win.nuclear_win
    assert nd.windowTitle().startswith("Nuclear Thermal Designer") and nd.ctrl.key == "nuc.perf"
    nd.form.set_values({"T0": 2600.0, "pc": 6.0})
    nd.set_mode("nuc.crit")                                    # move to another study ...
    assert nd.ctrl.key == "nuc.crit" and nd.saved["nuc.perf"]["T0"] == 2600.0
    nd.set_mode("nuc.perf")                                    # ... and come back: the values are still there
    assert nd.form.values().T0 == 2600.0
    d = nd.make_design()
    assert d.kind == "nuclear" and d.data["nuc.perf"]["pc"] == 6.0
    nd.use_in_analysis()
    assert win.design is not None and win.design.kind == "nuclear"
    win.select_model("nuc.kinetics")
    v = win.form.values()
    assert v.T_set == 2600.0 and v.P == 6.0                    # core temperature and pressure from the performance study
    p = d.save(tmp_path / "n")
    assert D.Design.load(p).data["nuc.perf"]["T0"] == 2600.0
    names = list(win.bindings)
    assert "Design: Nuclear Thermal Designer" in names


def test_electric_designer_feeds_the_electric_runs(qapp, tmp_path):
    import math
    from propsim import design as D
    from propsim.ui.main_window import MainWindow
    win = MainWindow(home=tmp_path, use_settings=False)
    win.open_electric_designer()
    ed = win.electric_win
    assert ed.windowTitle().startswith("Electric Propulsion Designer") and ed.ctrl.key == "ep.trades"
    ed.form.set_values({"isp": 3200.0, "m0": 800.0})
    ed.set_mode("ep.hall")
    ed.form.set_values({"Vd": 350.0, "R_out": 45.0, "R_in": 30.0})
    ed.set_mode("ep.trades")
    assert ed.form.values().isp == 3200.0                      # the mission trade kept its values
    ed.use_in_analysis()
    assert win.design.kind == "electric"
    win.select_model("ep.spiral")
    assert win.form.values().isp == 3200.0 and win.form.values().m0 == 800.0
    win.select_model("ep.hall_run")
    v = win.form.values()
    assert v.Vd == 350.0 and abs(v.A - math.pi * (45.0**2 - 30.0**2) / 100.0) < 0.6                  # on the slider grid
    assert "Design: Electric Propulsion Designer" in list(win.bindings)


def test_designer_hover_text_is_typeset_not_raw_latex(qapp, tmp_path):
    from propsim.ui.latex import LatexLabel
    from propsim.ui.main_window import MainWindow
    win = MainWindow(home=tmp_path, use_settings=False)
    win.open_nuclear_designer()
    nd = win.nuclear_win
    assert isinstance(nd.hover, LatexLabel)
    nd.form.hovered.emit(r"$\mathrm{regression\ law}$ — local regression law  $[1]$")
    assert nd.hover.pixmap() is not None and not nd.hover.pixmap().isNull()
