"""Animated cycle schematic: the engine cycle that is in the sandbox, drawn as a P&ID with the flow moving along the
lines (arrows in the line color: oxidizer, fuel, hot gas), rotating pumps and turbines and moving shaft links.
Works for every cycle the sandbox can hold (pressure-fed, gas generator, expander, staged combustion, ...).
Exports a looping GIF, or a PNG of the current frame."""
from __future__ import annotations

from io import BytesIO
from pathlib import Path

from PySide6.QtCore import QBuffer, QIODevice, QPointF, QRectF, Qt, QTimer
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPen
from PySide6.QtWidgets import QFileDialog, QHBoxLayout, QLabel, QPushButton, QSlider, QVBoxLayout, QWidget

from .theme import theme

LOOP_FRAMES = 48              # one loop = the flow moves one arrow spacing, rotors turn once


class CycleAnimation(QWidget):
    def __init__(self, canvas, title=lambda: "", parent=None) -> None:
        super().__init__(parent)
        self.canvas, self.title = canvas, title
        self.phase = 0.0
        self.speed = 1.0
        self.timer = QTimer(self)
        self.timer.setInterval(40)
        self.timer.timeout.connect(self._tick)
        self.view = _View(self)
        bar = QHBoxLayout()
        self.play = QPushButton("Pause")
        self.play.clicked.connect(self.toggle)
        gif = QPushButton("Export GIF")
        gif.clicked.connect(lambda: self.export_gif())
        png = QPushButton("Save Image")
        png.clicked.connect(lambda: self.save_image())
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(1, 40)
        self.slider.setValue(10)
        self.slider.valueChanged.connect(lambda v: setattr(self, "speed", v / 10.0))
        for w in (self.play, QLabel("Speed"), self.slider, gif, png):
            bar.addWidget(w)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(6, 0, 6, 6)
        lay.addWidget(self.view, 1)
        lay.addLayout(bar)
        theme.changed.connect(lambda *_: self.view.update())
        self.timer.start()

    # ------------------------------------------------------------------ playback
    def toggle(self) -> None:
        if self.timer.isActive():
            self.timer.stop()
            self.play.setText("Play")
        else:
            self.timer.start()
            self.play.setText("Pause")

    def _tick(self) -> None:
        if self.isVisible():
            self.phase = (self.phase + self.speed / LOOP_FRAMES) % 1.0
            self.view.update()

    # ------------------------------------------------------------------ drawing
    def draw(self, p: QPainter, size: QRectF, phase: float) -> None:
        """Paint the cycle into ``size`` at animation phase ``phase`` (0..1, one loop)."""
        from .cycle_schematic import Schematic
        sch = Schematic(self.canvas.graph)
        if sch.kind != "custom":                                # a known cycle: the textbook-style schematic
            sch.draw(p, size, phase)
            return
        from .cycle_sandbox import NODE_W, EdgeItem, NodeItem
        t = theme.t
        items = self.canvas.scene().items()
        nodes = [i for i in items if isinstance(i, NodeItem)]
        edges = [i for i in items if isinstance(i, EdgeItem)]
        p.setRenderHint(QPainter.Antialiasing)
        p.fillRect(size, QColor(t["viewport_bg"]))
        if not nodes:
            return
        box = QRectF()
        for n in nodes:
            box = box.united(n.mapRectToScene(QRectF(0, 0, NODE_W, n.h)))
        margin, title_h = 30.0, 26.0
        k = min((size.width() - 2 * margin) / box.width(), (size.height() - 2 * margin - title_h) / box.height())
        p.save()
        p.translate(size.left() + (size.width() - k * box.width()) / 2 - k * box.left(),
                    size.top() + title_h + margin / 2 + (size.height() - title_h - margin - k * box.height()) / 2
                    - k * box.top())
        p.scale(k, k)
        for e in edges:
            path = e.path()
            pen = QPen(e.pen())
            pen.setWidthF(3.6 if e.edge["kind"] == "fluid" else 2.4)
            if e.edge["kind"] != "fluid":
                pen.setDashOffset(-phase * 12.0)               # the mechanical link and heat path move too
            p.setPen(pen)
            p.setBrush(Qt.NoBrush)
            p.drawPath(path)
            if e.edge["kind"] == "fluid":
                n = max(1, int(path.length() // 80))
                p.setPen(Qt.NoPen)
                p.setBrush(QColor("#ffffff"))
                for i in range(n):
                    f = (i + phase) / n
                    p.save()
                    p.translate(path.pointAtPercent(f))
                    p.rotate(-path.angleAtPercent(f))
                    p.drawPolygon([QPointF(7, 0), QPointF(-5, -5), QPointF(-5, 5)])
                    p.restore()
        for n in nodes:
            n.phase = phase
            was = n.isSelected()
            p.save()
            p.translate(n.pos())
            if was:
                n.setSelected(False)
            n.paint(p)
            if was:
                n.setSelected(True)
            p.restore()
        p.restore()
        f = QFont(p.font())
        f.setPointSizeF(11.0)
        f.setBold(True)
        p.setFont(f)
        p.setPen(QColor(t["text"]))
        p.drawText(QRectF(size.left() + 10, size.top() + 4, size.width() - 20, title_h), Qt.AlignVCenter | Qt.AlignLeft,
                   self.title())

    def frame(self, phase: float, w: int = 1100, h: int = 700) -> QImage:
        img = QImage(w, h, QImage.Format_RGB32)
        p = QPainter(img)
        try:
            self.draw(p, QRectF(0, 0, w, h), phase)
        finally:
            p.end()
        return img

    # ------------------------------------------------------------------ export
    def export_gif(self, path: str | None = None, frames: int = LOOP_FRAMES) -> Path | None:
        """Looping GIF of one flow cycle."""
        if path is None:
            path, _ = QFileDialog.getSaveFileName(self, "Export Cycle Animation", "cycle.gif", "GIF (*.gif)")
        if not path:
            return None
        from PIL import Image
        from PySide6.QtWidgets import QApplication, QProgressDialog
        prog = QProgressDialog("Exporting the cycle animation", None, 0, frames + 1, self)
        prog.setWindowTitle("Export Cycle Animation")
        prog.setMinimumDuration(0)
        prog.setWindowModality(Qt.WindowModal)
        imgs = []
        try:
            for i in range(frames):
                prog.setValue(i)
                QApplication.processEvents()
                buf = QBuffer()
                buf.open(QIODevice.WriteOnly)
                self.frame(i / frames).save(buf, "PNG")
                imgs.append(Image.open(BytesIO(bytes(buf.data()))).convert("P", palette=Image.ADAPTIVE))
            prog.setLabelText("Writing the file")
            prog.setValue(frames)
            QApplication.processEvents()
            imgs[0].save(path, save_all=True, append_images=imgs[1:], duration=40, loop=0)
        finally:
            prog.setValue(frames + 1)
            prog.close()
        return Path(path)

    def save_image(self, path: str | None = None) -> Path | None:
        if path is None:
            path, _ = QFileDialog.getSaveFileName(self, "Save Cycle Image", "cycle.png", "PNG (*.png)")
        if not path:
            return None
        self.frame(self.phase).save(path, "PNG")
        return Path(path)


class _View(QWidget):
    def __init__(self, owner: CycleAnimation) -> None:
        super().__init__(owner)
        self.owner = owner
        self.setMinimumSize(320, 220)

    def paintEvent(self, _) -> None:
        p = QPainter(self)
        try:
            self.owner.draw(p, QRectF(self.rect()), self.owner.phase)
        finally:
            p.end()
