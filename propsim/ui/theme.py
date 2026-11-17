"""Theme tokens (PROPSIM_BUILD_PROMPT Section 5.2), quantity colours, and the QSS builder."""
from __future__ import annotations

from PySide6.QtCore import QObject, Signal

TOKENS = {
    "dark": dict(bg="#14171c", panel="#1c2027", edge="#2c323c", grid_minor="#262b33", plot_bg="#11141a",
                 viewport_bg="#0d1014", text="#e6e9ee", dim="#8b94a3", accent="#4ea1ff", select_bg="#22324a",
                 run_btn="#2a6fd6", ok="#5cc97a",
                 # derived (not in the token table): button fill, groove, hover-box, title bar
                 btn="#232833", groove="#343b47", card="#151a22", hover_bg="#1d2633", hover_edge="#2d3a4d",
                 titlebar="#0f1115", grid_major="#3a414d", faint="#5d6676"),
    "light": dict(bg="#f3f5f8", panel="#ffffff", edge="#d6dce5", grid_minor="#e9eef5", plot_bg="#ffffff",
                  viewport_bg="#e9eef5", text="#132033", dim="#5b6779", accent="#1d6cb0", select_bg="#dbe9f8",
                  run_btn="#2a6fd6", ok="#2e8b57",
                  btn="#e6ebf2", groove="#c9d2de", card="#f7f9fc", hover_bg="#eaf1fa", hover_edge="#bcd0e8",
                  titlebar="#e9edf3", grid_major="#c4ccd8", faint="#9aa5b5"),
}
# quantity colours (both themes)
QUANTITY = dict(oxidizer="#ff9d3c", fuel="#5cc97a", hot="#ff5a3c", pressure="#b07cff", thrust="#4ea1ff")
CMAPS = dict(temperature="inferno", mach="viridis", signed="coolwarm")


class _Theme(QObject):
    changed = Signal(str)

    def __init__(self) -> None:
        super().__init__()
        self.name = "dark"

    @property
    def t(self) -> dict[str, str]:
        return TOKENS[self.name]

    def color(self, key: str) -> str:
        """Token or quantity colour name -> hex; hex strings pass through."""
        if key.startswith("#"):
            return key
        return QUANTITY.get(key) or self.t[key]

    def set(self, name: str) -> None:
        if name != self.name:
            self.name = name
            self.changed.emit(name)


theme = _Theme()


def build_qss(t: dict[str, str]) -> str:
    return f"""
* {{ font-family: "Segoe UI", "Inter", sans-serif; font-size: 10pt; color: {t['text']}; }}
QMainWindow, QDockWidget, QToolBar {{ background: {t['bg']}; }}
QMainWindow::separator {{ background: {t['bg']}; width: 4px; height: 4px; }}
QMenuBar {{ background: {t['bg']}; color: {t['dim']}; }}
QMenuBar::item:selected {{ background: {t['select_bg']}; }}
QMenu {{ background: {t['panel']}; border: 1px solid {t['edge']}; }}
QMenu::item:selected {{ background: {t['select_bg']}; color: {t['accent']}; }}
QDockWidget > QWidget {{ background: {t['panel']}; }}
QFrame#card, QWidget#card {{ background: {t['panel']}; border: 1px solid {t['edge']}; border-radius: 6px; }}
QLabel {{ background: transparent; border: none; }}
QLabel#heading {{ color: {t['dim']}; font-weight: bold; font-size: 9pt; letter-spacing: 1px; }}
QLabel#groupTitle {{ color: {t['accent']}; font-weight: bold; }}
QLabel#value {{ color: {t['dim']}; font-size: 9pt; }}
QLabel#caption {{ color: {t['dim']}; font-size: 8.5pt; }}
QFrame#hoverbox {{ background: {t['hover_bg']}; border: 1px solid {t['hover_edge']}; border-radius: 5px; }}
QFrame#readout {{ background: {t['card']}; border: 1px solid {t['edge']}; border-radius: 6px; }}
QPlainTextEdit {{ background: {t['panel']}; border: none; color: {t['dim']};
                 font-family: "Cascadia Mono", "Consolas", monospace; font-size: 9pt; }}
QTreeView, QListWidget {{ background: {t['panel']}; border: none; color: {t['dim']}; outline: 0; }}
QTreeView::item, QListWidget::item {{ padding: 3px 2px; }}
QTreeView::item:selected, QListWidget::item:selected {{ background: {t['select_bg']}; color: {t['accent']};
                                                        border-radius: 4px; }}
QTreeView::item:disabled {{ color: {t['faint']}; }}
QCheckBox {{ color: {t['dim']}; }}
QPushButton {{ background: {t['btn']}; border: 1px solid {t['edge']}; border-radius: 5px; padding: 3px 10px;
              color: {t['dim']}; }}
QPushButton:hover {{ border-color: {t['accent']}; }}
QPushButton:disabled {{ color: {t['edge']}; }}
QPushButton#run {{ background: {t['run_btn']}; border-color: {t['run_btn']}; color: white; font-weight: bold; }}
QPushButton#run:disabled {{ background: {t['btn']}; border-color: {t['edge']}; color: {t['edge']}; }}
QPushButton#export {{ padding: 3px 8px; font-size: 8.5pt; }}
QLabel#badge {{ border: 1.5px solid {t['accent']}; border-radius: 6px; padding: 3px 12px; font-weight: bold;
               color: {t['accent']}; background: transparent; }}
QLabel#badge[mode="run"] {{ border-color: {t['ok']}; color: {t['ok']}; background: rgba(92,201,122,0.12); }}
QComboBox {{ background: {t['btn']}; border: 1px solid {t['edge']}; border-radius: 5px; padding: 2px 6px;
            color: {t['dim']}; }}
QComboBox QAbstractItemView {{ background: {t['panel']}; selection-background-color: {t['select_bg']}; }}
QProgressBar {{ background: {t['btn']}; border: none; border-radius: 3px; height: 8px; text-align: center;
               color: transparent; max-height: 8px; }}
QProgressBar::chunk {{ background: {t['ok']}; border-radius: 3px; }}
QSlider::groove:horizontal {{ height: 4px; background: {t['groove']}; border-radius: 2px; }}
QSlider::sub-page:horizontal {{ background: {t['accent']}; border-radius: 2px; }}
QSlider::handle:horizontal {{ width: 12px; height: 12px; margin: -6px 0; border-radius: 7px;
                              background: white; border: 2px solid {t['accent']}; }}
QTabBar::tab {{ background: {t['btn']}; color: {t['dim']}; border: 1px solid {t['edge']}; border-radius: 5px;
               padding: 3px 14px; margin: 3px 4px 3px 0; min-width: 90px; }}
QTabBar::tab:selected {{ background: {t['select_bg']}; color: {t['text']}; }}
QTabWidget::pane {{ border: none; }}
QScrollArea {{ border: none; background: transparent; }}
QScrollArea > QWidget > QWidget {{ background: transparent; }}
QScrollBar:vertical {{ background: {t['panel']}; width: 8px; }}
QScrollBar::handle:vertical {{ background: {t['groove']}; border-radius: 4px; min-height: 20px; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; width: 0; }}
QSplitter::handle {{ background: {t['bg']}; }}
QToolTip {{ background: {t['panel']}; color: {t['text']}; border: 1px solid {t['edge']}; }}
"""
