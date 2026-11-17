"""Qt widgets that show LaTeX wherever plain-text widgets would otherwise print math: item views (module tree, combo
popups), the combo box face, and the run log (math segments become inline images)."""
from __future__ import annotations

import base64
import html

from PySide6.QtCore import QBuffer, QByteArray, QIODevice, QPoint, QSize, Qt
from PySide6.QtWidgets import (QApplication, QComboBox, QStyle, QStyledItemDelegate, QStyleOptionComboBox,
                               QStyleOptionViewItem, QStylePainter, QTextBrowser)

from .latex import latex_pixmap
from .text import _MATH_SPLIT, to_math
from .theme import theme


def _has_math(text: str) -> bool:
    return "$" in to_math(text or "")


class LatexItemDelegate(QStyledItemDelegate):
    """Draws item text through mathtext when it contains math (after normalisation); plain text otherwise."""

    def __init__(self, size_pt: float = 9.5, normal: str = "dim", selected: str = "accent", parent=None) -> None:
        super().__init__(parent)
        self.size_pt, self.normal, self.selected = size_pt, normal, selected

    def _color(self, option) -> str:
        if not (option.state & QStyle.State_Enabled):
            return theme.color("faint")
        if option.state & QStyle.State_Selected:
            return theme.color(self.selected)
        return theme.color(self.normal)

    def paint(self, painter, option, index) -> None:
        text = index.data(Qt.DisplayRole) or ""
        if not _has_math(text):
            super().paint(painter, option, index)
            return
        opt = QStyleOptionViewItem(option)
        self.initStyleOption(opt, index)
        opt.text = ""
        style = opt.widget.style() if opt.widget else QApplication.style()
        style.drawControl(QStyle.CE_ItemViewItem, opt, painter, opt.widget)
        pm = latex_pixmap(text, self._color(option), self.size_pt)
        r = style.subElementRect(QStyle.SE_ItemViewItemText, opt, opt.widget)
        h = pm.height() / pm.devicePixelRatio()
        painter.drawPixmap(QPoint(r.left() + 2, int(r.top() + (r.height() - h) / 2)), pm)

    def sizeHint(self, option, index) -> QSize:
        base = super().sizeHint(option, index)
        text = index.data(Qt.DisplayRole) or ""
        if not _has_math(text):
            return base
        pm = latex_pixmap(text, self._color(option), self.size_pt)
        d = pm.devicePixelRatio()
        return QSize(max(base.width(), int(pm.width() / d) + 24), max(base.height(), int(pm.height() / d) + 6))


class LatexComboBox(QComboBox):
    """Combo box whose items and face render LaTeX (e.g. propellant pairs with chemical subscripts)."""

    def __init__(self, size_pt: float = 9.5, parent=None) -> None:
        super().__init__(parent)
        self.size_pt = size_pt
        self.view().setItemDelegate(LatexItemDelegate(size_pt, "text", "accent", self))
        theme.changed.connect(self.update)

    def paintEvent(self, e) -> None:
        text = self.currentText()
        if not _has_math(text):
            super().paintEvent(e)
            return
        p = QStylePainter(self)
        opt = QStyleOptionComboBox()
        self.initStyleOption(opt)
        opt.currentText = ""
        p.drawComplexControl(QStyle.CC_ComboBox, opt)
        p.drawControl(QStyle.CE_ComboBoxLabel, opt)
        r = self.style().subControlRect(QStyle.CC_ComboBox, opt, QStyle.SC_ComboBoxEditField, self)
        pm = latex_pixmap(text, theme.color("text"), self.size_pt)
        h = pm.height() / pm.devicePixelRatio()
        p.drawPixmap(QPoint(r.left() + 4, int(r.top() + (r.height() - h) / 2)), pm)


class LatexLog(QTextBrowser):
    """Run log: each line keeps its text, and its math segments (after normalisation) are inline LaTeX images."""

    def __init__(self, size_pt: float = 9.5, parent=None) -> None:
        super().__init__(parent)
        self.size_pt = size_pt
        self.setOpenLinks(False)
        self._plain: list[str] = []

    def _img(self, seg: str) -> str:
        pm = latex_pixmap(seg, theme.color("dim"), self.size_pt)
        ba = QByteArray()
        buf = QBuffer(ba)
        buf.open(QIODevice.WriteOnly)
        pm.save(buf, "PNG")
        d = pm.devicePixelRatio()
        w, h = int(pm.width() / d), int(pm.height() / d)
        b64 = base64.b64encode(bytes(ba)).decode("ascii")
        return f'<img src="data:image/png;base64,{b64}" width="{w}" height="{h}" style="vertical-align:middle">'

    def append_line(self, line: str) -> None:
        self._plain.append(line)
        out = []
        for row in line.split("\n"):
            parts = _MATH_SPLIT.split(to_math(row))
            out.append("".join(self._img(p) if (p.startswith("$") and p.endswith("$") and len(p) >= 2)
                               else html.escape(p).replace("  ", "&nbsp; ") for p in parts))
        self.append("<div>" + "<br>".join(out) + "</div>")       # explicit HTML: always parsed as rich text
        self.verticalScrollBar().setValue(self.verticalScrollBar().maximum())

    def plain_text(self) -> str:
        return "\n".join(self._plain)
