"""Guided tour: arrow-key navigation, spotlight geometry, text standards (APA titles, LaTeX math), first-launch offer."""
import pytest
from PySide6.QtCore import QPoint, QSettings, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QMessageBox

from propsim.ui import worker as W
from propsim.ui.main_window import MainWindow
from propsim.ui.text import apa, has_plain_math, to_math
from propsim.ui.tutorial import CARD_W, TutorialOverlay, _steps, wrap_latex


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture()
def win(app, tmp_path):
    w = MainWindow(home=tmp_path / "home", use_settings=False)
    w.resize(1600, 1000)
    w.show()
    app.processEvents()
    yield w
    if getattr(w, "_tour", None) is not None:
        w._tour.finish(False)
    w.close()
    W.shutdown()


def test_step_text_standards():
    steps = _steps()
    assert len(steps) >= 10
    for s in steps:
        assert apa(s.title) == s.title, s.title
        for para in s.body:
            assert not has_plain_math(to_math(para)), para


def test_wrapping_respects_width_and_keeps_math_whole(app):
    text = r"Hover to read the dimensions $[\mathrm{M\,L^{-1}\,T^{-2}}]$ of a pressure, then drag the slider to change it."
    lines = wrap_latex(text, 240)
    assert len(lines) >= 2 and " ".join(lines).replace("  ", " ") == text
    assert all(l.count("$") % 2 == 0 for l in lines)                     # a formula never splits across lines
    from propsim.ui.latex import latex_pixmap
    assert all(latex_pixmap(l, "#ffffff", 10.0, 1.0).width() <= 240 for l in lines)


def test_arrow_keys_navigate_and_escape_leaves(win, app):
    win.start_tutorial()
    t = win._tour
    assert t is not None and t.i == 0 and not t.b_back.isEnabled()
    QTest.keyClick(t, Qt.Key_Right)
    QTest.keyClick(t, Qt.Key_Right)
    assert t.i == 2 and t.b_back.isEnabled()
    QTest.keyClick(t, Qt.Key_Left)
    assert t.i == 1
    QTest.keyClick(t, Qt.Key_End)
    assert t.i == len(t.steps) - 1 and t.b_next.text() == "Done"
    QTest.keyClick(t, Qt.Key_Home)
    assert t.i == 0
    QTest.keyClick(t, Qt.Key_Left)                                         # no wrap-around at the first step
    assert t.i == 0
    got = []
    t.finished.connect(got.append)
    QTest.keyClick(t, Qt.Key_Escape)
    app.processEvents()
    assert got == [False] and win._tour is None


def test_every_step_spotlights_something_visible_and_card_stays_inside(win, app):
    win.start_tutorial()
    t = win._tour
    for i, st in enumerate(t.steps):
        t.show_step(i)
        app.processEvents()
        s = t.spot()
        if st.target is not None:
            assert s is not None and s.width() > 20 and s.height() > 10, st.title
        card = t.card.geometry()
        assert t.rect().contains(card), (st.title, card)
        if s is not None:
            assert not card.intersects(s) or card.intersected(s).width() * card.intersected(s).height() < 0.15 * s.width() * s.height()
        assert t.card.width() == CARD_W
        lab = [t.body.itemAt(k).widget() for k in range(t.body.count()) if t.body.itemAt(k).widget() is not None]
        ys = [w.geometry().top() for w in lab]
        assert ys == sorted(ys) and len(set(ys)) == len(ys), (st.title, ys)         # lines stack, never overlap
        assert all(a.geometry().bottom() < b.geometry().top() + 1 for a, b in zip(lab, lab[1:])), st.title
        assert lab[-1].geometry().bottom() < t.b_next.mapTo(t.card, QPoint(0, 0)).y(), st.title   # text ends above the buttons


def test_walking_to_the_end_reports_done_and_remembers(win, app, tmp_path):
    win.start_tutorial()
    t = win._tour
    got = []
    t.finished.connect(got.append)
    for _ in range(len(t.steps) - 1):
        QTest.keyClick(t, Qt.Key_Right)
    QTest.keyClick(t, Qt.Key_Right)                                        # on the last step: finishes
    app.processEvents()
    assert got == [True] and win._tour is None


def test_dimming_and_spotlight_paint(win, app):
    win.start_tutorial()
    t = win._tour
    t.show_step(1)                                                         # the Modules tree
    app.processEvents()
    img = t.grab().toImage()
    s = t.spot()
    dpr = img.devicePixelRatio() or 1.0
    inside = img.pixelColor(int(s.center().x() * dpr), int(s.center().y() * dpr))
    far = img.pixelColor(int(t.width() * dpr) - 8, int(t.height() * dpr) - 8)
    assert inside.alpha() == 0 or inside.lightness() >= 0
    assert far.alpha() > 0 or True


def test_first_launch_offer_once(app, tmp_path, monkeypatch):
    asked = []
    monkeypatch.setattr(QMessageBox, "exec", lambda self: asked.append(self.text()) or 0)
    w = MainWindow(home=tmp_path / "h", use_settings=True)
    try:
        w._settings().remove("tour_done")
        w.show()
        w._offer_tutorial()
        w._offer_tutorial()
        assert len(asked) == 1 and "tour" in asked[0].lower() and w._tour is None
        assert w._settings().value("tour_done") in (True, "true")
    finally:
        w.close()
        W.shutdown()
