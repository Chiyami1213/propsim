"""UI acceptance for the Common 'Thermochemistry & nozzle' Live model (offscreen)."""

import pytest
from PySide6.QtWidgets import QApplication, QComboBox, QSlider

from propsim.core.io import load_case, save_case
from propsim.modules.engine_live import EngineLiveParams
from propsim.ui.latex import LatexLabel
from propsim.ui.main_window import MainWindow

KEY = "common.engine"


@pytest.fixture(scope="module")
def win(tmp_path_factory):
    app = QApplication.instance() or QApplication([])
    w = MainWindow(home=tmp_path_factory.mktemp("home"), use_settings=False)
    w.resize(1600, 1000)
    w.show()
    w.select_model(KEY, {"pair": "LOX/HTPB", "mr": 1.9, "p_c": 3.45, "eps": 8.0})
    app.processEvents()
    yield w
    w.close()


def test_listed_under_common_and_live(win):
    assert win.ctrl.key == KEY and win.ctrl.mode == "live"
    assert win.bar.badge.text() == "Live"
    tabs = win.plot_tabs
    assert [tabs.tabText(i) for i in range(tabs.count())] == ["Performance", "Nozzle Flow"]


def test_form_has_choice_and_slider_rows(win):
    combos = win.form.findChildren(QComboBox)
    assert len(combos) >= 4                                    # pair, flow, source, contour
    assert len(win.form.findChildren(QSlider)) >= 6
    pair_row = win.form.rows["pair"]
    assert pair_row.slider.count() == 8 and pair_row.value() == "LOX/HTPB"


def test_live_update_on_choice_and_slider(win):
    app = QApplication.instance()
    before = win.ctrl.evaluate(win.form.values())["Isp"]
    win.form.rows["pair"].slider.setCurrentIndex(win.form.rows["pair"].meta["choices"].index("LOX/LH2"))
    win.form.rows["mr"].set_value(4.5)                         # near the LH2 optimum (LH2 table range is 2..10)
    app.processEvents()
    lh2 = win.ctrl.evaluate(win.form.values())["Isp"]
    assert lh2 > before + 80                                   # hydrogen is far better than HTPB
    win.form.rows["flow"].slider.setCurrentIndex(1)            # frozen
    app.processEvents()
    assert win.ctrl.evaluate(win.form.values())["Isp"] < lh2
    assert win.last_eval_ms < 50.0, win.last_eval_ms           # Live budget (physics only, tables)
    win.form.rows["pair"].slider.setCurrentIndex(win.form.rows["pair"].meta["choices"].index("LOX/HTPB"))
    win.form.rows["flow"].slider.setCurrentIndex(0)
    win.form.rows["mr"].set_value(1.9)
    app.processEvents()


def test_plots_gridded_latex_and_regions(win):
    plots = win.plot_tabs.grid_plots()
    assert len(plots) == 6                                     # 3 + 3 over the two tabs
    assert all(p.pi.ctrl.xGridCheck.isChecked() and p.pi.ctrl.yGridCheck.isChecked() for p in plots)
    for gp in plots:
        assert "$" in gp.xl.latex() and "$" in gp.yl.latex()
    for lbl in win.findChildren(LatexLabel):
        assert "O/F" not in lbl.latex()
    assert len(win.vp.readout_lbls) == 4 and all(lbl.latex() for lbl in win.vp.readout_lbls)


def test_out_of_range_of_is_clamped_and_noted(win):
    r = win.ctrl.evaluate(EngineLiveParams(pair="LOX/LH2", mr=0.9))
    assert r["clamped"] and r["mr"] == 2.0
    assert "clamped" in win.ctrl.caption(EngineLiveParams(pair="LOX/LH2", mr=0.9), r, None)


def test_case_roundtrip(win, tmp_path):
    path = str(tmp_path / "engine.psim.json")
    p = EngineLiveParams(pair="N2O4/MMH", mr=1.6, p_c=5.0, eps=20.0, kind="cone", alpha=12.0)
    save_case(path, KEY, p.model_dump())
    key, params = load_case(path)[:2]
    assert key == KEY and EngineLiveParams(**params) == p
    win.open_case(path)
    QApplication.instance().processEvents()
    assert win.form.rows["pair"].value() == "N2O4/MMH" and win.form.rows["kind"].value() == "cone"
    assert win.form.rows["alpha"].value() == pytest.approx(12.0)


def test_choice_validation_rejects_unknown():
    with pytest.raises(ValueError):
        EngineLiveParams(pair="LOX/UNOBTAINIUM")
    with pytest.raises(ValueError):
        EngineLiveParams(flow="plasma")
