"""Color themes: complete token sets, readable contrast, the picker (palettes when wide, names only when narrow),
persistence of the choice."""
import pytest
from PySide6.QtCore import QSettings
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QApplication

from propsim.ui import worker as W
from propsim.ui.main_window import MainWindow
from propsim.ui.theme import THEME_NAMES, TOKENS, build_qss, swatch, theme
from propsim.ui.theme_picker import ThemeDialog, ThemePicker


def _lum(hex_):
    c = QColor(hex_)
    f = lambda v: v / 12.92 if v <= 0.03928 else ((v + 0.055) / 1.055) ** 2.4
    return 0.2126 * f(c.redF()) + 0.7152 * f(c.greenF()) + 0.0722 * f(c.blueF())


def contrast(a, b):
    la, lb = sorted((_lum(a), _lum(b)), reverse=True)
    return (la + 0.05) / (lb + 0.05)


def test_all_themes_have_identical_token_sets():
    assert list(THEME_NAMES) == ["dark", "ocean", "forest", "plum", "light"]
    keys = set(TOKENS["dark"])
    for name, t in TOKENS.items():
        assert set(t) == keys, name
        assert all(QColor(v).isValid() for v in t.values()), name
        assert len(swatch(name)) == 4 and len(set(swatch(name))) == 4
        assert build_qss(t)


@pytest.mark.parametrize("name", list(THEME_NAMES))
def test_contrast_is_readable(name):
    t = TOKENS[name]
    assert contrast(t["text"], t["bg"]) >= 7 and contrast(t["text"], t["panel"]) >= 7
    assert contrast(t["dim"], t["panel"]) >= 4.5 and contrast(t["dim"], t["plot_bg"]) >= 4.5
    assert contrast(t["accent"], t["panel"]) >= 3 and contrast(t["accent"], t["select_bg"]) >= 3
    assert contrast("#ffffff", t["run_btn"]) >= 3.5                       # run button caption


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def test_picker_palettes_collapse_to_names(app):
    pk = ThemePicker()
    pk.resize(800, 90)
    pk.show()
    app.processEvents()
    assert not pk.compact and all(not c.compact for c in pk.cards.values())
    pk.resize(380, 40)
    app.processEvents()
    assert pk.compact and all(c.compact for c in pk.cards.values())
    pk.resize(800, 90)
    app.processEvents()
    assert not pk.compact
    got = []
    pk.changed.connect(got.append)
    pk.cards["forest"].chosen.emit("forest")
    assert theme.name == "forest" and got == ["forest"]
    theme.set("dark")


def test_dialog_and_persistence(app, tmp_path):
    w = MainWindow(home=tmp_path / "h", use_settings=True)
    try:
        w.show()
        app.processEvents()
        dlg = ThemeDialog(w)
        dlg.show()
        app.processEvents()
        dlg.picker.cards["plum"].chosen.emit("plum")
        assert theme.name == "plum" and w._settings().value("theme") == "plum"
        dlg.close()
        w.close()
        theme.name = "dark"                                               # a fresh process starts on the default
        w2 = MainWindow(home=tmp_path / "h", use_settings=True)
        assert theme.name == "plum"                                       # restored at startup
        w2.close()
    finally:
        theme.set("dark")
        W.shutdown()


def test_window_frames_follow_the_theme(qapp, tmp_path):
    """The light theme must not leave black system title bars on dialogs: the scheme follows the theme (the offscreen
    platform of the tests does not report it back, so the mapping and the call are checked)."""
    from PySide6.QtCore import Qt
    from propsim.ui.main_window import MainWindow, color_scheme_for
    from propsim.ui.theme import THEME_NAMES, theme
    assert color_scheme_for("light") == Qt.ColorScheme.Light
    assert all(color_scheme_for(n) == Qt.ColorScheme.Dark for n in THEME_NAMES if n != "light")
    w = MainWindow(home=tmp_path, use_settings=False)
    try:
        theme.set("light")
        theme.set("dark")                                       # switching themes runs the sync without error
    finally:
        theme.set("dark")
    del w
