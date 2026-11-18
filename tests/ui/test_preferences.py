"""Edit > Preferences (appearance, key bindings) and the double-click reset on parameter sliders."""
from __future__ import annotations

from PySide6.QtCore import QEvent, QPoint, Qt
from PySide6.QtGui import QKeySequence, QMouseEvent
from PySide6.QtWidgets import QApplication

from propsim.ui.main_window import MainWindow
from propsim.ui.preferences import PreferencesDialog


def _win(tmp_path):
    return MainWindow(home=tmp_path, use_settings=False)


def test_every_command_is_listed_with_its_default_shortcut(qapp, tmp_path):
    w = _win(tmp_path)
    assert w.bindings["File: Screenshot"][1] == "F2"
    assert w.bindings["View: Full Screen"][1] == "F11"
    assert w.bindings["View: Restore Default Layout"][1] == "F12"
    assert w.bindings["Help: Tutorial"][1] == "F1"
    assert "Edit: Preferences" in w.bindings


def test_changing_a_shortcut_moves_the_action(qapp, tmp_path):
    w = _win(tmp_path)
    w.set_shortcut("File: Screenshot", "F3")
    assert w.bindings["File: Screenshot"][0].shortcut().toString() == "F3"
    w.set_shortcut("File: Screenshot", "")
    assert w.bindings["File: Screenshot"][0].shortcut().isEmpty()


def test_shortcut_choice_is_remembered(qapp, tmp_path):
    w = MainWindow(home=tmp_path, use_settings=True)
    w.set_shortcut("File: Screenshot", "F3")
    w2 = MainWindow(home=tmp_path, use_settings=True)
    assert w2.bindings["File: Screenshot"][0].shortcut().toString() == "F3"
    w2.set_shortcut("File: Screenshot", "F2")                    # back to the default removes the stored override
    w3 = MainWindow(home=tmp_path, use_settings=True)
    assert w3.bindings["File: Screenshot"][0].shortcut().toString() == "F2"


def test_dialog_finds_conflicts_and_applies_unique_bindings(qapp, tmp_path):
    w = _win(tmp_path)
    dlg = PreferencesDialog(w)
    keys = dlg.keys
    keys.edits["File: Screenshot"].setKeySequence(QKeySequence("F11"))        # taken by Full Screen
    bad = keys.conflicts()
    assert bad and set(bad[0][1]) == {"File: Screenshot", "View: Full Screen"}
    keys.edits["File: Screenshot"].setKeySequence(QKeySequence("F3"))
    assert keys.conflicts() == []
    keys.apply()
    assert w.bindings["File: Screenshot"][0].shortcut().toString() == "F3"


def test_restore_defaults_on_the_bindings_page(qapp, tmp_path):
    w = _win(tmp_path)
    dlg = PreferencesDialog(w)
    dlg.keys.edits["File: Screenshot"].setKeySequence(QKeySequence("F5"))
    dlg.keys.restore_defaults()
    assert dlg.keys.edits["File: Screenshot"].keySequence().toString() == "F2"


def test_appearance_options_reach_the_3d_view(qapp, tmp_path):
    w = _win(tmp_path)
    dlg = PreferencesDialog(w)
    assert dlg.appearance.theme.count() == 5
    w.vp.set_legend_visible(False)
    w.vp.set_colorbar_visible(False)
    assert not w.vp._legend_on and not w.vp.colorbar.isVisibleTo(w.vp)
    w.vp.set_legend_visible(True)
    w.vp.set_colorbar_visible(True)
    assert w.vp._legend_on and w.vp.colorbar.isVisibleTo(w.vp)


def test_double_click_resets_a_slider_to_its_default(qapp, tmp_path):
    w = _win(tmp_path)
    w.select_model("hybrid.burn0d")
    row = next(r for r in w.form.rows.values() if hasattr(r, "slider") and hasattr(r, "step"))
    default = row.meta["default"]
    row.set_value(row.lo + 3 * row.step)
    assert abs(row.value() - default) > row.step / 2
    seen = []
    row.changed.connect(lambda: seen.append(1))
    e = QMouseEvent(QEvent.MouseButtonDblClick, QPoint(5, 5), Qt.LeftButton, Qt.LeftButton, Qt.NoModifier)
    QApplication.sendEvent(row.slider, e)
    assert abs(row.value() - default) <= row.step / 2 + 1e-9 and seen


def test_3d_legend_lists_the_solid_colour_blocks(qapp, tmp_path):
    w = _win(tmp_path)
    w.show()
    w.select_model("hybrid.burn0d")
    w.vp.render()
    labels = [lab for _, lab in w.vp.legend._rows]
    assert "Motor case" in labels and "Fuel grain (cut face)" in labels
    assert len(labels) == len(set(labels))
    w.vp._toggle_legend()
    assert not w.vp._legend_on and not w.vp.legend.isVisible()
