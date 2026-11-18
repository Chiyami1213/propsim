"""Solid-motor tools in the main window: Live budget, standards (grid + LaTeX), Run models through the worker."""
import time

import pytest
from PySide6.QtWidgets import QApplication, QDialogButtonBox

from propsim.ui import worker as W
from propsim.ui.latex import LatexLabel
from propsim.ui.main_window import MainWindow

LIVE = ["solid.grain", "solid.burnrate"]


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
    assert len(plots) >= 3
    assert all(p.pi.ctrl.xGridCheck.isChecked() and p.pi.ctrl.yGridCheck.isChecked() for p in plots)
    for gp in plots:
        assert "$" in gp.xl.latex() and "$" in gp.yl.latex()
    for lbl in win.findChildren(LatexLabel):
        assert "O/F" not in lbl.latex(), lbl.latex()
    assert len(win.vp.readout_lbls) == 4 and all(lbl.latex() for lbl in win.vp.readout_lbls)


@pytest.mark.parametrize("key", LIVE)
def test_live_tools(win, key):
    win.select_model(key)
    for _ in range(5):
        QApplication.processEvents()
    assert win.ctrl.key == key and win.bar.badge.text() == "Live"
    assert win.last_eval_ms < 50.0, win.last_eval_ms
    _standards(win)


def test_grain_designer_section_choice_updates(win):
    win.select_model("solid.grain")
    row = win.form.rows["kind"]
    before = win.ctrl.evaluate(win.form.values())["b"].perim[0]
    row.slider.setCurrentIndex(row.meta["choices"].index("star"))
    QApplication.processEvents()
    assert win.ctrl.evaluate(win.form.values())["b"].perim[0] != pytest.approx(before)
    assert win.last_eval_ms < 50.0


@pytest.mark.parametrize("key", ["solid.ballistics", "solid.burnback3d"])
def test_run_models(win, key):
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
    if key == "solid.burnback3d":
        assert "surface" in set(win.vp.plotter.renderer.actors)                     # marching-cubes mesh drawn
        assert "phi_snaps" in r.arrays


def test_custom_section_and_stack_editors(win):
    """The grain-builder dialogs build valid JSON, the form carries it, and the Run model accepts it."""
    import json

    from propsim.core.grain import stack as gs
    from propsim.ui import grain_editor as ge

    win.select_model("solid.ballistics")
    form = win.form
    form.rows["kind"].slider.setCurrentIndex(form.rows["kind"].meta["choices"].index("custom"))
    QApplication.processEvents()
    sec = ge.SectionEditor("", {"R_out": 60.0})
    assert sec.spec().shapes[0].kind == "circle" and sec.buttons.button(QDialogButtonBox.Ok).isEnabled()
    sec.table.item(0, 3).setText("r=18")
    sec._add("circle", True, 1, {"r": 5.0, "cx": 35.0})                # an inhibited off-centre bore
    QApplication.processEvents()
    spec = json.loads(sec.text())
    assert spec["shapes"][-1]["inhibited"] and "inhibited" in ge.summary("section", sec.text())
    sec.table.item(0, 3).setText("r=oops")
    sec.refresh()
    assert not sec.buttons.button(QDialogButtonBox.Ok).isEnabled()           # bad input disables OK, nothing is thrown
    form.rows["section_json"].set_value(json.dumps(spec))
    st = ge.StackEditor("", {"L_g": 0.3, "R_out": 60.0})
    st._add(gs.GrainDef(0.3, None, (("a_mm", 9.0), ("cstar", 1400.0)), True, True))
    st.table.item(0, 4).setText("")
    text = st.text()
    assert len(gs.parse_stack(text)) == 2 and ge.summary("stack", text).startswith("2 grain")
    form.rows["stack_json"].set_value(text)
    QApplication.processEvents()
    p = form.values()
    assert p.section_json == json.dumps(spec) and p.stack_json == text
    m = p.to_si()
    assert len(m.stack) == 2 and m.stack[1].prop.cstar == 1400.0 and m.stack[1].burn_head
