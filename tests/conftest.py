import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
if sys.platform == "win32":                       # Qt's offscreen plugin finds no fonts on Windows otherwise
    os.environ.setdefault("QT_QPA_FONTDIR", r"C:\Windows\Fonts")


import gc  # noqa: E402

import pytest  # noqa: E402

gc.disable()      # the GUI collects from a timer (propsim.ui.gc_guard); tests collect between tests, never mid-paint


@pytest.fixture(autouse=True)
def _collect_between_tests():
    yield
    gc.collect()


@pytest.fixture(scope="session", autouse=True)
def _isolated_settings(tmp_path_factory):
    """Tests never touch the user's real PropSim settings (on Windows QSettings lives in the registry): the default
    format becomes an INI file in a temporary folder for the whole session."""
    from PySide6.QtCore import QSettings
    root = tmp_path_factory.mktemp("settings")
    QSettings.setDefaultFormat(QSettings.IniFormat)
    QSettings.setPath(QSettings.IniFormat, QSettings.UserScope, str(root))
    yield
    QSettings.setDefaultFormat(QSettings.NativeFormat)
