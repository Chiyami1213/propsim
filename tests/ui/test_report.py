"""Report export: HTML (self-contained, math typeset), PDF, LaTeX tables; through the window for a Live and a Run model."""
import re
import time

import pytest
from PySide6.QtWidgets import QApplication

from propsim import report
from propsim.modules import registry
from propsim.ui import worker as W
from propsim.ui.main_window import MainWindow
from propsim.ui.text import has_plain_math, to_math


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


def test_html_report_for_a_live_model(win, tmp_path):
    win.select_model("common.engine")
    for _ in range(3):
        QApplication.processEvents()
    out = tmp_path / "engine.html"
    win.export_report("html", str(out))
    h = out.read_text(encoding="utf-8")
    n_par = len(win.ctrl.Params.fields_meta())
    assert h.startswith("<!doctype html>") and "http://" not in h and "https://" not in h        # fully self-contained
    assert h.count("<tr><td><img") >= n_par                                                         # symbol column is typeset math
    assert h.count("data:image/png;base64,") > n_par and "<h2>Parameters</h2>" in h and "<h2>Plots</h2>" in h
    assert "Thermochemistry" in h and "PropSim 1.0.0" in h


def test_pdf_and_latex_for_a_run_model(win, tmp_path):
    win.select_model("vehicle.ascent")
    win.start_run()
    _wait(win)
    assert win.result is not None
    pdf, tex = tmp_path / "asc.pdf", tmp_path / "asc.tex"
    win.export_report("pdf", str(pdf))
    win.export_report("tex", str(tex))
    raw = pdf.read_bytes()
    assert raw.startswith(b"%PDF") and len(re.findall(rb"/Type\s*/Page\b", raw)) >= 4
    t = tex.read_text(encoding="utf-8")
    n_par = len(win.ctrl.Params.fields_meta())
    assert r"\begin{longtable}" in t and t.count(r"\\") >= n_par and r"\includegraphics" in t
    body = t.split(r"\begin{document}")[1]
    assert "$" in body and "(MPa)" not in body                                   # math stays LaTeX in the source
    assert r"\end{document}" in t
    figs = sorted(tmp_path.glob("asc_fig*.png"))
    assert len(figs) >= 4 and all(f.stat().st_size > 5000 for f in figs)
    html_out = tmp_path / "asc.html"
    win.export_report("html", str(html_out))
    h = html_out.read_text(encoding="utf-8")
    assert "<h2>Events</h2>" in h and "Max-Q" in h and "<h2>Run Log</h2>" in h


def test_rows_are_clean_latex():
    c = registry.get("liquid.injector")
    rows = report.parameter_rows(c, c.Params())
    assert len(rows) == len(c.Params.fields_meta())
    for r in rows:
        assert not has_plain_math(to_math(r["symbol"])) and not has_plain_math(to_math(r["units"]))
    assert any(r["units"] == "(MPa)" for r in rows)
    res, summary, caption = report.result_rows(c, c.Params(), c.evaluate(c.Params()))
    assert len(res) == 4 and all(not has_plain_math(to_math(q)) for q, _ in res)


def test_nothing_to_report_is_a_message_not_a_crash(win, tmp_path):
    win.select_model("hybrid.burn0d")                                 # a Run model with no result yet
    win.result = None
    win.export_report("html", str(tmp_path / "x.html"))
    assert not (tmp_path / "x.html").exists() and "run the simulation first" in win.log.toPlainText()
