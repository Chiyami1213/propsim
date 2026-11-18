"""Loading screen (at least one second, progress bar, icon) and the confirmation before quitting."""
from __future__ import annotations

import time

from PySide6.QtGui import QCloseEvent

from propsim.app import ICON
from propsim.ui.splash import MIN_SECONDS, SplashScreen, run_startup


def test_splash_has_icon_bar_and_progress(qapp):
    s = SplashScreen(str(ICON.with_suffix(".png")), "1.0.0")
    assert not s.icon.pixmap().isNull()
    s.set_progress(0.5)
    assert s.bar.value() == 500
    s.set_progress(2.0)
    assert s.bar.value() == 1000
    s.close()


def test_minimum_display_time_is_one_second():
    assert MIN_SECONDS >= 1.0


def test_startup_runs_steps_in_order_and_waits_for_the_minimum(qapp):
    s = SplashScreen(str(ICON.with_suffix(".png")))
    seen: list[str] = []
    t0 = time.monotonic()
    run_startup(s, [("One", 1.0, lambda: seen.append("a")), ("Two", 3.0, lambda: seen.append("b"))],
                min_seconds=0.4, on_ready=lambda: seen.append("show"))
    assert seen == ["a", "b", "show"]
    assert time.monotonic() - t0 >= 0.4
    assert s.bar.value() == 1000
    assert not s.isVisible()


def test_close_asks_first_and_can_be_cancelled(qapp, tmp_path, monkeypatch):
    from propsim.ui.main_window import MainWindow
    w = MainWindow(home=tmp_path, use_settings=False)
    asked = []
    monkeypatch.setattr(MainWindow, "_confirm_quit", lambda self: asked.append(1) or False)
    ev = QCloseEvent()
    w.confirm_close = True
    w.closeEvent(ev)
    assert asked == [1] and not ev.isAccepted()
    w.confirm_close = False                                   # tests and scripts close silently
    ev2 = QCloseEvent()
    w.closeEvent(ev2)
    assert asked == [1] and ev2.isAccepted()


def test_toolbar_has_a_name_for_save_state(qapp, tmp_path):
    from propsim.ui.main_window import MainWindow
    w = MainWindow(home=tmp_path, use_settings=False)
    assert w.bar.objectName() == "Playback"


def test_f2_screenshot_goes_to_the_data_folder(qapp, tmp_path):
    from propsim.ui.main_window import MainWindow
    w = MainWindow(home=tmp_path, use_settings=False)
    w.show()
    f = w.take_screenshot()
    assert f is not None and f.parent == tmp_path / "screenshots" and f.stat().st_size > 1000
    assert w.take_screenshot() != f                                  # a second shot never overwrites the first
