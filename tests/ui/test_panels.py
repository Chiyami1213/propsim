"""Panel chrome (drag grip, close button, View > Panels), solid scrollbars, tree guide lines, crisp text at fractional
display scaling."""
import pytest
from PySide6.QtCore import QPoint, Qt
from PySide6.QtGui import QAction
from PySide6.QtWidgets import QApplication, QDockWidget

from propsim.ui import worker as W
from propsim.ui.latex import LatexButton, latex_pixmap
from propsim.ui.main_window import MainWindow
from propsim.ui.panels import PanelTitle
from propsim.ui.theme import build_qss, theme


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


def test_every_panel_has_grip_left_and_close_right(win):
    for d in win.docks:
        t = d.titleBarWidget()
        assert isinstance(t, PanelTitle)
        assert t.grip.geometry().right() < t.label.geometry().left() + 1          # grip at the top left
        assert t.close_btn.geometry().left() > t.label.geometry().left()          # close button at the top right
        assert t.close_btn.geometry().right() >= t.width() - 14
        assert d.features() & QDockWidget.DockWidgetClosable and d.features() & QDockWidget.DockWidgetMovable


def test_close_button_hides_and_menu_restores(win):
    d = win.log_dock
    d.titleBarWidget().close_btn.click()
    QApplication.processEvents()
    assert not d.isVisible()
    panels = next(m for m in win.menuBar().findChildren(type(win.menuBar().addMenu("x"))) if m.title() == "Panels")
    act = next(a for a in panels.actions() if a.text() == "Run Log")
    assert act.isCheckable() and not act.isChecked()
    act.trigger()
    QApplication.processEvents()
    assert d.isVisible()
    d.close()
    win.reset_layout()
    QApplication.processEvents()
    assert all(x.isVisible() for x in win.docks)                                   # default layout brings everything back


def test_dragging_the_title_floats_the_panel(win):
    d = win.vp_dock
    d.setFloating(True)
    QApplication.processEvents()
    assert d.isFloating()
    d.setFloating(False)
    win.reset_layout()


def test_solid_scrollbar_and_tree_guide(win):
    for name in ("dark", "light", "plum"):
        q = build_qss(__import__("propsim.ui.theme", fromlist=["TOKENS"]).TOKENS[name])
        assert "QScrollBar::add-page, QScrollBar::sub-page { background: transparent; }" in q
        assert "QScrollBar::handle:vertical { background:" in q                    # one solid colour, no hatching
    win.select_model("hybrid.regression")
    QApplication.processEvents()
    tree = win.tree
    hy = win.tree_model.index(2, 0)
    tree.expand(hy)
    QApplication.processEvents()
    img = tree.viewport().grab().toImage()
    child = win.tree_model.index(1, 0, hy)
    rc = tree.visualRect(child)
    x = rc.left() - tree.indentation() // 2
    dpr = img.devicePixelRatio() or 1.0
    bg = img.pixelColor(int(2 * dpr), int(rc.center().y() * dpr))
    for yy in (rc.top() + 3, rc.center().y(), rc.bottom() - 3):
        px = img.pixelColor(int(x * dpr), int(yy * dpr))
        assert abs(px.lightness() - bg.lightness()) > 10, (yy, px.name(), bg.name())


@pytest.mark.parametrize("dpr", [1.0, 1.25, 1.5, 1.75, 2.0])
def test_text_pixmaps_are_never_resampled(dpr):
    pm = latex_pixmap(r"Cutaway $45^{\circ}$", "#8b94a3", 9.0, dpr)
    w, h = pm.width() / dpr, pm.height() / dpr
    assert abs(w - round(w)) < 1e-6 and abs(h - round(h)) < 1e-6


def test_latex_button_paints_on_whole_device_pixels(win):
    b = LatexButton(r"Field: $M$ (–)")
    b.show()
    QApplication.processEvents()
    d = b.devicePixelRatioF()
    pm = b._pm
    assert abs(pm.devicePixelRatio() - d) < 1e-6
    assert b.sizeHint().width() >= pm.width() / d and b.sizeHint().height() >= pm.height() / d
    img = b.grab().toImage()
    assert not img.isNull()
    b.close()


@pytest.mark.parametrize("key", ["vehicle.ascent", "liquid.cycle", "common.engine"])
def test_long_dropdown_labels_do_not_resize_panels(win, key):
    win.reset_layout()
    QApplication.processEvents()
    win.select_model(key)
    for _ in range(5):
        QApplication.processEvents()
    assert win.left_dock.width() >= 300, win.left_dock.width()
    assert win.right_dock.width() <= 520, win.right_dock.width()
    combo = next(r.slider for r in win.form.rows.values() if hasattr(r.slider, "setMinimumContentsLength"))
    assert combo.sizeHint().width() < 300


def test_long_summary_and_caption_wrap_or_clip_instead_of_widening(win):
    win.reset_layout()
    QApplication.processEvents()
    w0 = (win.left_dock.width(), win.right_dock.width())
    long = (r"apogee $41.9$ (km) · range $22.0$ (km) · max-$q$ $230.2$ (kPa) · flight time $282$ (s) · propellant "
            r"$568$ (kg) · orbit $200 \times 203$ (km) · burnout $7785$ (m·s⁻¹) and some more words to force wrapping")
    win.summary.setLatex(long)
    win.vp.set_caption(long * 2)
    for _ in range(6):
        QApplication.processEvents()
    assert (win.left_dock.width(), win.right_dock.width()) == pytest.approx(w0, abs=6)
    assert win.summary.height() > 20 and win.summary.latex() == long              # wrapped onto several lines
    lines = [c for c in win.summary.findChildren(type(win.vp.caption))]
    assert len(lines) >= 2 and all(c.geometry().right() <= win.summary.width() + 2 for c in lines)
    win.summary.setLatex("")
    win.vp.set_caption("")
