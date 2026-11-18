"""Text standards for every model: all math in LaTeX (no math left outside $...$ after normalisation), APA title case
for titles / tabs / sections / headers / menus, and lower-case hover helper descriptions."""
import re
import threading

import pytest
from PySide6.QtWidgets import QApplication, QLabel

from propsim.modules import registry
from propsim.ui import worker as W
from propsim.ui.latex import LatexLabel
from propsim.ui.main_window import MainWindow
from propsim.ui.text import _MATH_SPLIT, apa, has_plain_math, hover_message, to_math

KEYS = [c.key for entries in registry.TREE.values() for _, m, c in entries if c is not None and m != "tool"]


def _outside_math(s: str) -> str:
    return "".join(p for p in _MATH_SPLIT.split(s) if not (p.startswith("$") and p.endswith("$") and len(p) >= 2))


def _clean(s: str, where: str) -> None:
    assert not has_plain_math(to_math(s)), f"{where}: math left outside LaTeX in {s!r} -> {to_math(s)!r}"


def _is_apa(s: str) -> bool:
    return apa(s) == s


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


def test_text_helpers():
    assert apa("burn rate & K_n") == "Burn Rate & K_n" and apa("quasi-1D burn") == "Quasi-1D Burn"
    assert apa("heat load (kg·s⁻¹)") == "Heat Load (kg·s⁻¹)"                   # units are never capitalized
    assert apa("the regression of a port") == "The Regression of a Port"
    assert to_math("(kg·m⁻³)") == r"($\mathrm{kg{\cdot}m^{-3}}$)" and to_math(to_math("N₂O")) == to_math("N₂O")
    msg = hover_message(r"\dot{m}", "Oxidizer Mass Flow from NASA CEA", "M T⁻¹")
    assert _outside_math(msg) == _outside_math(msg).lower() and r"\mathrm{M{\cdot}T^{-1}}" in msg


def test_chrome_is_apa_and_latex(win):
    heads = [lbl.text() for lbl in win.findChildren(QLabel) if lbl.objectName() == "heading"]
    assert heads and all(_is_apa(h) for h in heads), heads
    for menu in win.menuBar().findChildren(type(win.menuBar().addMenu("x"))):
        for a in menu.actions():
            if a.text():
                assert _is_apa(a.text()), a.text()
    win.menuBar().clear()
    win._build_menus()
    model = win.tree_model
    for r in range(model.rowCount()):
        top = model.item(r)
        assert _is_apa(top.text())
        for c in range(top.rowCount()):
            t = top.child(c).text()
            label = re.sub(r"\s+\((Live|Run)\)$", "", t)
            assert _is_apa(label), t
            _clean(t, "tree")
    assert win.compare_cb.text() == "Compare Overlay"


@pytest.mark.parametrize("key", KEYS)
def test_model_text_standards(win, key):
    win.select_model(key)
    for _ in range(4):
        QApplication.processEvents()
    ctrl, p = win.ctrl, win.form.values()
    # parameters: section headers APA, hover descriptions lower case, everything math in LaTeX
    for m in ctrl.Params.fields_meta():
        assert _is_apa(apa(m["group"]))
        msg = hover_message(m["symbol"], m["desc"], m["dims"])
        _clean(msg, f"{key} hover {m['name']}")
        assert _outside_math(msg) == _outside_math(msg).lower(), msg
        _clean(rf"${m['symbol']}$ ({m['units']})", f"{key} label {m['name']}")
        for lab in m.get("labels", []):
            _clean(lab, f"{key} choice {m['name']}")
    for g in win.form.findChildren(QLabel):
        if g.objectName() == "groupTitle":
            txt = g.latex() if isinstance(g, LatexLabel) else g.text()
            assert _is_apa(txt), txt
    # plots, readouts, caption, summary: live state, or a real run for Run models
    if ctrl.mode == "live":
        r = ctrl.evaluate(p)
        frame = None
    else:
        r = ctrl.sim.run(p, lambda f: None, threading.Event())
        frame = r.n // 2
        _clean(ctrl.summary_line(p, r), f"{key} summary")
        for line in r.log:
            _clean(line, f"{key} log")
    for tab in ctrl.tabs(p, r):
        _clean(apa(tab.name), f"{key} tab")
        for ps in tab.plots:
            _clean(apa(ps.title), f"{key} title")
            _clean(ps.xlabel, f"{key} xlabel {ps.title}")
            _clean(ps.ylabel, f"{key} ylabel {ps.title}")
            for c in ps.curves:
                if c.label:
                    _clean(c.label, f"{key} legend {ps.title}")
    for s in ctrl.readout(p, r, frame):
        _clean(s, f"{key} readout")
    _clean(ctrl.caption(p, r, frame), f"{key} caption")
    _clean(ctrl.title, f"{key} model title")
    for gp in win.plot_tabs.grid_plots():
        assert _is_apa(gp.title.latex()), gp.title.latex()
