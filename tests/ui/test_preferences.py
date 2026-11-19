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


def test_3d_legend_lists_the_solid_color_blocks(qapp, tmp_path):
    w = _win(tmp_path)
    w.show()
    w.select_model("hybrid.burn0d")
    w.vp.render()
    labels = [lab for _, lab in w.vp.legend._rows]
    assert "Motor case" in labels and "Fuel grain (cut face)" in labels
    assert len(labels) == len(set(labels))
    w.vp._toggle_legend()
    assert not w.vp._legend_on and not w.vp.legend.isVisible()


def test_f10_opens_the_color_theme_picker(qapp, tmp_path):
    w = _win(tmp_path)
    assert w.bindings["View: Color Theme"][1] == "F10"


def test_preferences_is_f9_and_targets_show_units(qapp, tmp_path):
    from propsim.ui.cycle_sandbox import output_unit
    w = _win(tmp_path)
    assert w.bindings["Edit: Preferences"][1] == "F9"
    assert output_unit("p_c") == "Pa" and output_unit("thrust") == "N" and output_unit("mixture_ratio") == "–"


def test_active_design_feeds_the_analysis_modules(qapp, tmp_path):
    from propsim import design as D
    w = _win(tmp_path)
    w.set_design(D.Design("solid", "my grain", {"R_out": 55.0, "L_g": 0.9}))
    w.select_model("solid.ballistics")
    v = w.form.values()
    assert v.R_out == 55.0 and v.L_g == 0.9
    w.select_model("hybrid.burn0d")                            # a solid design does not touch a hybrid module
    assert w.form.values().R_p0 == 25.0
    w.clear_design()
    assert w.design is None
    # the Design and Analysis menus exist and Validation sits under Analysis
    names = list(w.bindings)
    assert any(n.startswith("Design:") for n in names) and "Analysis: Run" in names
    assert any(n.startswith("Validation:") for n in names)


def test_every_menu_command_has_a_distinct_default_shortcut(qapp, tmp_path):
    w = _win(tmp_path)
    missing = [k for k, (a, d) in w.bindings.items() if not d and a.isEnabled()]
    assert missing == []
    keys = [d for _, d in w.bindings.values() if d]
    assert len(keys) == len(set(keys))                           # no two commands share a shortcut
    assert w.bindings["Design: Hybrid Motor Designer"][1] == "Ctrl+H"
