"""Edit > Preferences: an Appearance page (colour theme, 3D view and quit options) and a Key Bindings page (every menu
command with an editable shortcut, conflicts refused, per-row and global reset)."""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QKeySequence
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QDialogButtonBox, QFormLayout, QHBoxLayout, QHeaderView,
                               QKeySequenceEdit, QLabel, QLineEdit, QMessageBox, QPushButton, QTableWidget,
                               QTableWidgetItem, QTabWidget, QVBoxLayout, QWidget)

from .text import apa
from .theme import THEME_NAMES, theme


def _seq(text: str) -> QKeySequence:
    return QKeySequence(text, QKeySequence.PortableText) if text else QKeySequence()


class AppearancePage(QWidget):
    def __init__(self, win, parent=None) -> None:
        super().__init__(parent)
        self.win = win
        form = QFormLayout(self)
        form.setContentsMargins(16, 16, 16, 16)
        form.setVerticalSpacing(12)
        self.theme = QComboBox()
        for key, name in THEME_NAMES.items():
            self.theme.addItem(name, key)
        self.theme.setCurrentIndex(max(self.theme.findData(theme.name), 0))
        self.legend = QCheckBox("Show the legend in the 3D view")
        self.colorbar = QCheckBox("Show the colorbar in the 3D view")
        self.confirm = QCheckBox("Ask for confirmation before quitting")
        self.legend.setChecked(win.pref_bool("show_legend", True))
        self.colorbar.setChecked(win.pref_bool("show_colorbar", True))
        self.confirm.setChecked(win.pref_bool("confirm_quit", True))
        form.addRow(QLabel(apa("Color theme")), self.theme)
        form.addRow(QLabel(apa("3D view")), self.legend)
        form.addRow("", self.colorbar)
        form.addRow(QLabel(apa("Quitting")), self.confirm)

    def restore_defaults(self) -> None:
        self.theme.setCurrentIndex(max(self.theme.findData("dark"), 0))
        self.legend.setChecked(True)
        self.colorbar.setChecked(True)
        self.confirm.setChecked(True)

    def apply(self) -> None:
        self.win.set_color_theme(self.theme.currentData())
        self.win.set_pref("show_legend", self.legend.isChecked())
        self.win.set_pref("show_colorbar", self.colorbar.isChecked())
        self.win.set_pref("confirm_quit", self.confirm.isChecked())
        self.win.apply_preferences()


class KeyBindingsPage(QWidget):
    COLS = ("Command", "Shortcut", "Default", "")

    def __init__(self, win, parent=None) -> None:
        super().__init__(parent)
        self.win = win
        self.keys = list(win.bindings)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(12, 12, 12, 12)
        top = QHBoxLayout()
        self.filter = QLineEdit()
        self.filter.setPlaceholderText("Filter commands")
        self.filter.textChanged.connect(self._filter)
        reset = QPushButton("Restore All Defaults")
        reset.clicked.connect(self.restore_defaults)
        top.addWidget(self.filter, 1)
        top.addWidget(reset)
        lay.addLayout(top)
        self.table = QTableWidget(len(self.keys), len(self.COLS))
        self.table.setHorizontalHeaderLabels([apa(c) for c in self.COLS])
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionMode(QTableWidget.NoSelection)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setShowGrid(False)
        h = self.table.horizontalHeader()
        h.setSectionResizeMode(0, QHeaderView.Stretch)
        for c in (1, 2, 3):
            h.setSectionResizeMode(c, QHeaderView.ResizeToContents)
        self.edits: dict[str, QKeySequenceEdit] = {}
        for r, key in enumerate(self.keys):
            action, default = win.bindings[key]
            it = QTableWidgetItem(key)
            it.setFlags(Qt.ItemIsEnabled)
            self.table.setItem(r, 0, it)
            ed = QKeySequenceEdit(action.shortcut())
            ed.setMaximumSequenceLength(1)
            ed.setMinimumWidth(150)
            self.edits[key] = ed
            self.table.setCellWidget(r, 1, ed)
            d = QTableWidgetItem(_seq(default).toString(QKeySequence.NativeText) or "–")
            d.setFlags(Qt.ItemIsEnabled)
            self.table.setItem(r, 2, d)
            b = QPushButton("Reset")
            b.clicked.connect(lambda _=False, k=key: self.edits[k].setKeySequence(_seq(win.bindings[k][1])))
            self.table.setCellWidget(r, 3, b)
            self.table.setRowHeight(r, 32)
        lay.addWidget(self.table, 1)
        hint = QLabel("Click a shortcut field and press the new keys. Clear a field to leave the command without a "
                      "shortcut. Two commands cannot share a shortcut.")
        hint.setWordWrap(True)
        hint.setObjectName("caption")
        lay.addWidget(hint)

    def _filter(self, text: str) -> None:
        t = text.strip().lower()
        for r, key in enumerate(self.keys):
            self.table.setRowHidden(r, bool(t) and t not in key.lower())

    def restore_defaults(self) -> None:
        for key, ed in self.edits.items():
            ed.setKeySequence(_seq(self.win.bindings[key][1]))

    def conflicts(self) -> list[tuple[str, list[str]]]:
        seen: dict[str, list[str]] = {}
        for key, ed in self.edits.items():
            s = ed.keySequence().toString(QKeySequence.PortableText)
            if s:
                seen.setdefault(s, []).append(key)
        return [(s, ks) for s, ks in seen.items() if len(ks) > 1]

    def apply(self) -> None:
        for key, ed in self.edits.items():
            self.win.set_shortcut(key, ed.keySequence().toString(QKeySequence.PortableText))


class PreferencesDialog(QDialog):
    def __init__(self, win, parent=None) -> None:
        super().__init__(parent or win)
        self.setWindowTitle(apa("Preferences"))
        self.resize(720, 560)
        lay = QVBoxLayout(self)
        self.tabs = QTabWidget()
        self.appearance = AppearancePage(win)
        self.keys = KeyBindingsPage(win)
        self.tabs.addTab(self.appearance, apa("Appearance"))
        self.tabs.addTab(self.keys, apa("Key bindings"))
        lay.addWidget(self.tabs)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel | QDialogButtonBox.RestoreDefaults)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        bb.button(QDialogButtonBox.RestoreDefaults).clicked.connect(self._restore_page)
        lay.addWidget(bb)

    def _restore_page(self) -> None:
        (self.appearance if self.tabs.currentWidget() is self.appearance else self.keys).restore_defaults()

    def accept(self) -> None:
        bad = self.keys.conflicts()
        if bad:
            lines = "\n".join(f"{QKeySequence(s).toString(QKeySequence.NativeText)}: {', '.join(ks)}" for s, ks in bad)
            QMessageBox.warning(self, apa("Shortcut conflict"), "These shortcuts are used by more than one command:\n\n"
                                + lines)
            self.tabs.setCurrentWidget(self.keys)
            return
        self.appearance.apply()
        self.keys.apply()
        super().accept()
