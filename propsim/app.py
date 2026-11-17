"""Entry point: ``python -m propsim.app`` or the ``propsim`` console script."""
from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication

from .ui.main_window import MainWindow

CASES = Path(__file__).resolve().parents[1] / "cases"
ICON = Path(__file__).resolve().parent / "assets" / "icon.ico"
APP_ID = "propsim.PropSim.1"


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv if argv is None else argv
    if sys.platform == "win32":  # own taskbar identity so Windows shows our icon, not python.exe's
        try:
            import ctypes
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(APP_ID)
        except Exception:
            pass
    app = QApplication(argv)
    app.setApplicationName("PropSim")
    icon = QIcon(str(ICON))
    app.setWindowIcon(icon)
    win = MainWindow()
    win.setWindowIcon(icon)
    case = next((a for a in argv[1:] if a.endswith(".psim.json")), None)
    win.open_case(case or str(CASES / "hybrid_demo.psim.json"))
    win.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
