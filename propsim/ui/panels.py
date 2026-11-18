"""Panel chrome: a title bar with a drag grip (top left) and a close button (top right), and the module tree with a grey
guide line down the left of every expanded group."""
from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, QSize, Qt
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import QDockWidget, QHBoxLayout, QLabel, QToolButton, QTreeView, QWidget

from .text import apa
from .theme import theme

PAD = 10                                                  # one padding for every panel (title and content)


class _Grip(QWidget):
    """Six dots (2 x 3): the handle to drag the panel by; events pass through to the dock, which moves it."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setFixedSize(16, 18)
        self.setCursor(Qt.SizeAllCursor)
        self.setAccessibleName("Drag panel")
        theme.changed.connect(self._on_theme)

    def _on_theme(self, *_) -> None:
        self.update()

    def paintEvent(self, _) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(theme.t["dim"]))
        for col in (0, 1):
            for row in range(3):
                p.drawEllipse(QPointF(4.5 + col * 5.0, 4.0 + row * 5.0), 1.3, 1.3)


class _Close(QToolButton):
    def __init__(self, dock: QDockWidget, parent=None) -> None:
        super().__init__(parent)
        self.setFixedSize(20, 20)
        self.setCursor(Qt.PointingHandCursor)
        self.setAutoRaise(True)
        self.setAccessibleName("Close panel")
        self.setFocusPolicy(Qt.NoFocus)
        self.clicked.connect(dock.close)
        theme.changed.connect(self._on_theme)

    def _on_theme(self, *_) -> None:
        self.update()

    def paintEvent(self, _) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        hot = self.underMouse()
        if hot:
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(theme.t["select_bg"]))
            p.drawRoundedRect(QRectF(self.rect()).adjusted(1, 1, -1, -1), 5, 5)
        p.setPen(QPen(QColor(theme.t["accent"] if hot else theme.t["dim"]), 1.6, Qt.SolidLine, Qt.RoundCap))
        c, d = self.width() / 2.0, 4.0
        p.drawLine(QPointF(c - d, c - d), QPointF(c + d, c + d))
        p.drawLine(QPointF(c - d, c + d), QPointF(c + d, c - d))


class PanelTitle(QWidget):
    """Title bar of a dock: ``[grip] Title ............ [x]``.  Dragging anywhere on it moves or floats the panel; the
    close button hides it (View > Panels or View > Restore Default Layout brings it back)."""

    def __init__(self, dock: QDockWidget, text: str, parent=None) -> None:
        super().__init__(parent)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(PAD, PAD - 2, PAD - 4, PAD // 2 - 1)
        lay.setSpacing(8)
        self.grip = _Grip()
        self.label = QLabel(apa(text))
        self.label.setObjectName("heading")
        self.close_btn = _Close(dock)
        lay.addWidget(self.grip)
        lay.addWidget(self.label, 1)
        lay.addWidget(self.close_btn)

    def sizeHint(self) -> QSize:
        return QSize(120, 32)


def draw_guide(tree, painter, option, index) -> None:
    """Grey vertical guide down the left of an expanded group's children (drawn after the row, so a selection never hides it)."""
    depth, p = 0, index.parent()
    while p.isValid():
        depth, p = depth + 1, p.parent()
    if depth >= 1:
        x = tree.indentation() * (depth + 1) - tree.indentation() // 2
        painter.save()
        c = QColor(theme.t["faint"])
        c.setAlpha(170)
        painter.setPen(QPen(c, 1))
        painter.drawLine(x, option.rect.top(), x, option.rect.bottom())
        painter.restore()


class GuideTree(QTreeView):
    """Module tree: a grey vertical guide runs down the left of each expanded group's children."""

    def drawRow(self, painter, option, index) -> None:
        super().drawRow(painter, option, index)
        draw_guide(self, painter, option, index)


def add_panel_menu(menu, docks) -> None:
    """Checkable show/hide entries for every panel (the way back after the close button)."""
    for d in docks:
        menu.addAction(d.toggleViewAction())
