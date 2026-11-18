"""Entry point: ``python -m propsim.app`` or the ``propsim`` console script."""
from __future__ import annotations

import multiprocessing
import sys
from pathlib import Path


CASES = Path(__file__).resolve().parents[1] / "cases"
ICON = Path(__file__).resolve().parent / "assets" / "icon.ico"
APP_ID = "propsim.PropSim.1"


def main(argv: list[str] | None = None) -> int:
    multiprocessing.freeze_support()          # required for worker processes in the PyInstaller build
    # GUI imports live here, not at module level: worker processes re-import this module and must not load Qt
    from PySide6.QtCore import QSettings, QTimer
    from PySide6.QtGui import QIcon
    from PySide6.QtWidgets import QApplication

    from . import __version__

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

    from .ui.splash import SplashScreen, run_startup

    saved = QSettings(QSettings.defaultFormat(), QSettings.UserScope, "PropSim", "PropSim").value("theme")
    splash = SplashScreen(str(ICON.with_suffix(".png")), __version__, saved if isinstance(saved, str) else "dark")
    ctx: dict = {}

    def load_modules() -> None:                # the heavy imports (matplotlib, PyVista, the solvers) happen here
        from .ui.main_window import MainWindow
        from .ui.worker import prewarm, shutdown
        ctx.update(MainWindow=MainWindow, prewarm=prewarm, shutdown=shutdown)

    def build_window() -> None:
        win = ctx["MainWindow"]()
        win.setWindowIcon(icon)
        win.confirm_close_managed = True       # tests and scripts close windows silently; the real app asks
        win.apply_preferences()
        ctx["win"] = win

    def open_case() -> None:
        case = next((a for a in argv[1:] if a.endswith(".psim.json")), None)
        ctx["win"].open_case(case or str(CASES / "hybrid_demo.psim.json"))

    def start_workers() -> None:
        ctx["prewarm"]()                       # warm worker process: first Run starts without the import delay

    def show_window() -> None:
        ctx["win"].show()
        QTimer.singleShot(900, ctx["win"]._offer_tutorial)      # the splash screen delays the first show

    run_startup(splash, [("Loading Modules", 5.0, load_modules), ("Building Window", 4.0, build_window),
                         ("Opening Example Case", 2.0, open_case), ("Starting Worker", 1.0, start_workers)],
                on_ready=show_window)
    code = app.exec()
    ctx["shutdown"]()
    return code


if __name__ == "__main__":
    raise SystemExit(main())
