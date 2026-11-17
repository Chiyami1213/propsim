"""Playback: one model emits ``frameChanged``; scrubber, plot cursors, 3D scene and readouts subscribe."""
from __future__ import annotations

from PySide6.QtCore import QElapsedTimer, QObject, Qt, QTimer, Signal
from PySide6.QtWidgets import (QLabel, QProgressBar, QPushButton, QSizePolicy, QSlider, QToolBar,
                               QWidget)

from .latex_widgets import LatexComboBox
from .latex import LatexLabel

SPEEDS = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]


class PlaybackModel(QObject):
    frameChanged = Signal(int)
    playingChanged = Signal(bool)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.n = 0
        self.duration = 1.0
        self.frame = 0
        self.speed = 1.0
        self._pos = 0.0
        self._timer = QTimer(self)
        self._timer.setInterval(16)       # ~60 Hz
        self._timer.timeout.connect(self._tick)
        self._clock = QElapsedTimer()

    @property
    def playing(self) -> bool:
        return self._timer.isActive()

    def set_range(self, n: int, duration: float) -> None:
        self.pause()
        self.n, self.duration = n, duration
        self.seek(0)

    @property
    def time(self) -> float:
        return self.frame * self.duration / max(self.n - 1, 1)

    def seek(self, i: int) -> None:
        i = max(0, min(int(i), max(self.n - 1, 0)))
        self._pos = float(i)
        self.frame = i
        self.frameChanged.emit(i)       # always emitted: callers rely on a refresh after set_range()

    def seek_time(self, t: float) -> None:
        self.seek(round(t / self.duration * (self.n - 1))) if self.n > 1 else self.seek(0)

    def play(self) -> None:
        if self.n < 2:
            return
        if self.frame >= self.n - 1:
            self.seek(0)
        self._clock.start()
        self._timer.start()
        self.playingChanged.emit(True)

    def pause(self) -> None:
        if self._timer.isActive():
            self._timer.stop()
            self.playingChanged.emit(False)

    def toggle(self) -> None:
        self.pause() if self.playing else self.play()

    def set_speed(self, s: float) -> None:
        self.speed = s

    def _tick(self) -> None:
        dt = self._clock.restart() / 1000.0
        self._pos += self.speed * dt * (self.n - 1) / self.duration
        if self._pos >= self.n - 1:
            self.seek(self.n - 1)
            self.pause()
            return
        i = int(self._pos)
        if i != self.frame:
            self.frame = i
            self.frameChanged.emit(i)


class PlaybackBar(QToolBar):
    """Fixed-height 40 px bar: mode badge, Run, Cancel, progress, transport, speed, scrubber, time, exports."""

    runClicked = Signal()
    cancelClicked = Signal()
    exportRequested = Signal(str)

    def __init__(self, playback: PlaybackModel, parent=None) -> None:
        super().__init__("Playback", parent)
        self.pb = playback
        self.setMovable(False)
        self.setFloatable(False)
        self.setFixedHeight(40)
        self.badge = QLabel("Live")
        self.badge.setObjectName("badge")
        self.badge.setProperty("mode", "live")
        self.run_btn = QPushButton("▶ Run")
        self.run_btn.setObjectName("run")
        self.cancel_btn = QPushButton("Cancel")
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setFixedWidth(190)
        self.pct = LatexLabel("0 %", "dim", 9.0)
        self.b_start = QPushButton("◀◀")
        self.b_play = QPushButton("‖")
        self.b_end = QPushButton("▶▶")
        for b in (self.b_start, self.b_play, self.b_end):
            b.setFlat(True)
            b.setFixedWidth(34)
        self.speed = LatexComboBox(9.0)
        for s in SPEEDS:
            self.speed.addItem(f"{s:g}×", s)
        self.speed.setCurrentIndex(SPEEDS.index(1.0))
        self.scrub = QSlider(Qt.Horizontal)
        self.scrub.setMinimumWidth(300)
        self.scrub.setRange(0, 0)
        self.time_lbl = LatexLabel(r"$t$ = 0.00 (s) / 0.0", "text", 9.5)
        for w in (self.badge, self.run_btn, self.cancel_btn, self.progress, self.pct, self.b_start, self.b_play,
                  self.b_end, self.speed):
            self.addWidget(w)
            self.addWidget(_gap())
        self.addWidget(self.scrub)
        self.addWidget(_gap())
        self.addWidget(self.time_lbl)
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.addWidget(spacer)
        self.export_btns = {}
        for name in ("PNG", "CSV", ".mat", ".h5", ".py", ".m"):
            b = QPushButton(name)
            b.setObjectName("export")
            b.clicked.connect(lambda _=False, n=name: self.exportRequested.emit(n))
            self.addWidget(b)
            self.export_btns[name] = b
        self.cancel_btn.setEnabled(False)
        # wiring
        self.run_btn.clicked.connect(self.runClicked)
        self.cancel_btn.clicked.connect(self.cancelClicked)
        self.b_start.clicked.connect(lambda: playback.seek(0))
        self.b_end.clicked.connect(lambda: playback.seek(playback.n - 1))
        self.b_play.clicked.connect(playback.toggle)
        self.speed.currentIndexChanged.connect(lambda _: playback.set_speed(self.speed.currentData()))
        self.scrub.valueChanged.connect(self._scrubbed)
        playback.frameChanged.connect(self._on_frame)
        playback.playingChanged.connect(lambda p: self.b_play.setText("‖" if p else "▶"))
        self.b_play.setText("▶")

    def _scrubbed(self, v: int) -> None:
        if v != self.pb.frame:
            self.pb.seek(v)

    def _on_frame(self, i: int) -> None:
        self.scrub.blockSignals(True)
        self.scrub.setValue(i)
        self.scrub.blockSignals(False)
        self.time_lbl.setLatex(rf"$t$ = {self.pb.time:.2f} ({self.unit}) / {self.pb.duration:.1f}")

    unit = "s"

    def set_range(self, n: int, unit: str = "s") -> None:
        """``unit`` is the display unit of the result's time axis (``Result.meta['t_unit']``, default seconds)."""
        self.unit = unit
        self.scrub.blockSignals(True)
        self.scrub.setRange(0, max(n - 1, 0))
        self.scrub.blockSignals(False)

    def set_mode(self, mode: str) -> None:
        run = mode == "run"
        self.badge.setText("Run" if run else "Live")
        self.badge.setProperty("mode", "run" if run else "live")
        self.badge.style().unpolish(self.badge)
        self.badge.style().polish(self.badge)
        for w in (self.run_btn, self.b_start, self.b_play, self.b_end, self.speed, self.scrub, self.progress):
            w.setEnabled(run)
        if not run:
            self.progress.setValue(0)
            self.pct.setLatex("– %")

    def set_progress(self, f: float) -> None:
        self.progress.setValue(int(round(100 * f)))
        self.pct.setLatex(f"{int(round(100 * f))} %")

    def set_running(self, running: bool) -> None:
        self.run_btn.setEnabled(not running)
        self.cancel_btn.setEnabled(running)


def _gap() -> QWidget:
    w = QWidget()
    w.setFixedWidth(6)
    return w
