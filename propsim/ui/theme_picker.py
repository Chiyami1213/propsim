"""Colour-theme picker: one card per theme showing its four-colour palette and name; when the row is too narrow for the
palettes the cards shrink to the theme name only."""
from __future__ import annotations

from PySide6.QtCore import QRectF, QSize, Qt, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QDialog, QHBoxLayout, QLabel, QSizePolicy, QVBoxLayout, QWidget

from .text import apa
from .theme import THEME_NAMES, TOKENS, swatch, theme

MIN_SWATCH_CARD = 112          # px per card below which the palettes are dropped
SWATCH_H = 40


class ThemeCard(QWidget):
    chosen = Signal(str)

    def __init__(self, key: str, parent=None) -> None:
        super().__init__(parent)
        self.key, self.name, self.compact, self._hover = key, THEME_NAMES[key], False, False
        self.setCursor(Qt.PointingHandCursor)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setAccessibleName(f"{self.name} color theme")
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        theme.changed.connect(self._on_theme)
        self.set_compact(False)

    def _on_theme(self, *_) -> None:
        self.update()

    def set_compact(self, on: bool) -> None:
        self.compact = on
        self.setFixedHeight(32 if on else SWATCH_H + 30)
        self.update()

    def sizeHint(self) -> QSize:
        return QSize(120, self.height())

    def enterEvent(self, e) -> None:
        self._hover = True
        self.update()

    def leaveEvent(self, e) -> None:
        self._hover = False
        self.update()

    def focusInEvent(self, e) -> None:
        super().focusInEvent(e)
        self.update()

    def focusOutEvent(self, e) -> None:
        super().focusOutEvent(e)
        self.update()

    def mouseReleaseEvent(self, e) -> None:
        if e.button() == Qt.LeftButton and self.rect().contains(e.position().toPoint()):
            self.chosen.emit(self.key)

    def keyPressEvent(self, e) -> None:
        if e.key() in (Qt.Key_Return, Qt.Key_Enter, Qt.Key_Space):
            self.chosen.emit(self.key)
        else:
            super().keyPressEvent(e)

    def paintEvent(self, _) -> None:
        cur = theme.t
        selected = theme.name == self.key
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w = self.width() - 6.0
        edge = cur["accent"] if (self.hasFocus() or self._hover) else cur["edge"]
        ring = QColor(TOKENS[self.key]["accent"] if selected else edge)
        if self.compact:
            r = QRectF(3, 3, w, self.height() - 6.0)
            path = QPainterPath()
            path.addRoundedRect(r, 7, 7)
            p.fillPath(path, QColor(TOKENS[self.key]["select_bg"] if selected else cur["btn"]))
            p.setPen(QPen(ring, 2.0 if selected else 1.0))
            p.drawPath(path)
            p.setPen(QColor(cur["text"]))
            text_rect, align = r, Qt.AlignCenter
        else:
            r = QRectF(3, 3, w, SWATCH_H)
            clip = QPainterPath()
            clip.addRoundedRect(r, 9, 9)
            p.save()
            p.setClipPath(clip)
            cols = swatch(self.key)
            seg = r.width() / len(cols)
            for i, c in enumerate(cols):
                p.fillRect(QRectF(r.left() + i * seg, r.top(), seg + 1, r.height()), QColor(c))
            p.restore()
            p.setPen(QPen(ring, 3.0 if selected else 1.0))
            p.drawPath(clip)
            p.setPen(QColor(cur["text"]))
            text_rect, align = QRectF(3, SWATCH_H + 6, w, 22), Qt.AlignLeft | Qt.AlignVCenter
        f = QFont(self.font())
        f.setBold(True)
        p.setFont(f)
        p.drawText(text_rect, int(align), self.name)


class ThemePicker(QWidget):
    changed = Signal(str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(12)
        self.cards: dict[str, ThemeCard] = {}
        for key in THEME_NAMES:
            c = ThemeCard(key)
            c.chosen.connect(self._pick)
            lay.addWidget(c, 1, Qt.AlignTop)
            self.cards[key] = c

    def _pick(self, key: str) -> None:
        theme.set(key)
        self.changed.emit(key)
        for c in self.cards.values():
            c.update()

    @property
    def compact(self) -> bool:
        return next(iter(self.cards.values())).compact

    def resizeEvent(self, e) -> None:
        n = len(self.cards)
        per = (self.width() - 12 * (n - 1)) / n
        compact = per < MIN_SWATCH_CARD
        for c in self.cards.values():
            if c.compact != compact:
                c.set_compact(compact)
        super().resizeEvent(e)


class ThemeDialog(QDialog):
    """Dialog hosting the picker; make it narrow and the palettes collapse to names only."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle(apa("Color theme"))
        self.setMinimumWidth(300)
        self.resize(780, 130)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(16, 12, 16, 16)
        head = QLabel(apa("Color theme"))
        head.setObjectName("heading")
        lay.addWidget(head)
        self.picker = ThemePicker()
        lay.addWidget(self.picker)
        lay.addStretch(1)
