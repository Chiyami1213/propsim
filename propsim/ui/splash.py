"""Start-up loading screen: the application icon in the middle, the name, a progress bar and the current step.

It is shown for at least :data:`MIN_SECONDS`, so it never flashes by; the bar follows the real start-up steps and then runs
on to full over the time that is left.  Only ``QtWidgets`` and the theme tokens are imported here, so the screen appears
before the heavy modules (matplotlib, PyVista) are loaded.
"""
from __future__ import annotations

import time
from collections.abc import Callable, Sequence

from PySide6.QtCore import Qt
from PySide6.QtGui import QGuiApplication, QPixmap
from PySide6.QtWidgets import QApplication, QFrame, QLabel, QProgressBar, QVBoxLayout, QWidget

from .theme import TOKENS

MIN_SECONDS = 1.0
ICON_PX = 112


class SplashScreen(QWidget):
    """Frameless card centred on the screen: ``[icon] PropSim, subtitle, progress bar, step``."""

    def __init__(self, icon_png: str, version: str = "", theme_name: str = "dark", parent=None) -> None:
        super().__init__(parent, Qt.SplashScreen | Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        t = TOKENS.get(theme_name, TOKENS["dark"])
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setFixedSize(460, 340)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        card = QFrame()
        card.setObjectName("splashcard")
        outer.addWidget(card)
        lay = QVBoxLayout(card)
        lay.setContentsMargins(40, 28, 40, 26)
        lay.setSpacing(0)
        dpr = self.devicePixelRatioF()
        pm = QPixmap(icon_png)
        if not pm.isNull():
            pm = pm.scaled(round(ICON_PX * dpr), round(ICON_PX * dpr), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            pm.setDevicePixelRatio(dpr)
        self.icon = QLabel()
        self.icon.setPixmap(pm)
        self.icon.setAlignment(Qt.AlignCenter)
        self.name = QLabel("PropSim")
        self.name.setObjectName("splashname")
        self.name.setAlignment(Qt.AlignCenter)
        self.subtitle = QLabel("Rocket Propulsion Simulator" + (f"  ·  {version}" if version else ""))
        self.subtitle.setObjectName("splashsub")
        self.subtitle.setAlignment(Qt.AlignCenter)
        self.bar = QProgressBar()
        self.bar.setRange(0, 1000)
        self.bar.setTextVisible(False)
        self.bar.setFixedHeight(6)
        self.status = QLabel("Starting")
        self.status.setObjectName("splashstatus")
        self.status.setAlignment(Qt.AlignCenter)
        lay.addStretch(1)
        lay.addWidget(self.icon)
        lay.addSpacing(10)
        lay.addWidget(self.name)
        lay.addWidget(self.subtitle)
        lay.addStretch(1)
        lay.addWidget(self.bar)
        lay.addSpacing(8)
        lay.addWidget(self.status)
        self.setStyleSheet(f"""
            QFrame#splashcard {{ background: {t['panel']}; border: 1px solid {t['edge']}; border-radius: 14px; }}
            QLabel {{ background: transparent; border: none; font-family: "Segoe UI", "Inter", sans-serif; }}
            QLabel#splashname {{ color: {t['text']}; font-size: 22pt; font-weight: 600; }}
            QLabel#splashsub {{ color: {t['dim']}; font-size: 9.5pt; }}
            QLabel#splashstatus {{ color: {t['faint']}; font-size: 9pt; }}
            QProgressBar {{ background: {t['btn']}; border: none; border-radius: 3px; }}
            QProgressBar::chunk {{ background: {t['accent']}; border-radius: 3px; }}
        """)

    def set_progress(self, fraction: float) -> None:
        self.bar.setValue(round(1000 * max(0.0, min(1.0, fraction))))

    def set_status(self, text: str) -> None:
        self.status.setText(text)

    def centre(self) -> None:
        scr = QGuiApplication.primaryScreen()
        if scr is not None:
            g = scr.availableGeometry()
            self.move(g.center().x() - self.width() // 2, g.center().y() - self.height() // 2)


def run_startup(splash: SplashScreen, steps: Sequence[tuple[str, float, Callable[[], None]]],
                min_seconds: float = MIN_SECONDS, on_ready: Callable[[], None] | None = None) -> None:
    """Run ``(status text, weight, function)`` steps behind the splash screen, then keep it up until ``min_seconds`` have
    passed since it appeared (the bar runs on to full meanwhile); ``on_ready`` (show the main window) runs before the
    splash closes."""
    splash.centre()
    splash.show()
    t0 = time.monotonic()
    app = QApplication.instance()
    total = sum(w for _, w, _ in steps) or 1.0
    done = 0.0
    for text, weight, fn in steps:
        splash.set_status(text)
        splash.set_progress(done / total)
        app.processEvents()
        fn()
        done += weight
    splash.set_status("Ready")
    t_steps = time.monotonic()
    remaining = max(0.0, min_seconds - (t_steps - t0))
    start = done / total
    while remaining > 0.0:
        now = time.monotonic()
        if now - t0 >= min_seconds:
            break
        splash.set_progress(start + (1.0 - start) * min(1.0, (now - t_steps) / remaining))
        app.processEvents()
        time.sleep(0.01)
    splash.set_progress(1.0)
    app.processEvents()
    if on_ready is not None:
        on_ready()
    splash.close()
