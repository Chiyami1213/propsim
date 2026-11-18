"""Theme tokens (PROPSIM_BUILD_PROMPT Section 5.2), quantity colors, and the QSS builder."""
from __future__ import annotations

import tempfile
from pathlib import Path

from PySide6.QtCore import QObject, QPointF, Qt, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPen

TOKENS = {
    "dark": dict(bg="#14171c", panel="#1c2027", edge="#2c323c", grid_minor="#262b33", plot_bg="#11141a",
                 viewport_bg="#0d1014", text="#e6e9ee", dim="#8b94a3", accent="#4ea1ff", select_bg="#22324a",
                 run_btn="#2a6fd6", ok="#5cc97a",
                 # derived (not in the token table): button fill, groove, hover-box, title bar
                 btn="#232833", groove="#343b47", card="#151a22", hover_bg="#1d2633", hover_edge="#2d3a4d",
                 titlebar="#0f1115", grid_major="#3a414d", faint="#5d6676", accent2="#ff9d3c"),
    "ocean": dict(bg="#0c1824", panel="#112233", edge="#1f3b57", grid_minor="#182b3d", plot_bg="#0a1520",
                  viewport_bg="#08121c", text="#e6eff8", dim="#8fa9c2", accent="#38a8ff", select_bg="#1c3f63",
                  run_btn="#2a7fe0", ok="#4cd0a4", btn="#15293d", groove="#2c4b6c", card="#0f1e2d", hover_bg="#142a3f",
                  hover_edge="#27465f", titlebar="#09131d", grid_major="#2e4d6c", faint="#58728c", accent2="#2e8bff"),
    "forest": dict(bg="#0d1813", panel="#13231c", edge="#224236", grid_minor="#182b22", plot_bg="#0b1510",
                   viewport_bg="#091210", text="#e5f1ea", dim="#8fb0a0", accent="#3ec48a", select_bg="#1f4535",
                   run_btn="#23885a", ok="#5cc97a", btn="#16281f", groove="#2f5442", card="#101e17", hover_bg="#142a21",
                   hover_edge="#26463a", titlebar="#09110d", grid_major="#2f5242", faint="#587566", accent2="#2fae78"),
    "plum": dict(bg="#15111d", panel="#1d1728", edge="#392d4e", grid_minor="#241d32", plot_bg="#120e19",
                 viewport_bg="#0e0b14", text="#eee9f6", dim="#a496bb", accent="#a97bff", select_bg="#3a2d5e",
                 run_btn="#8b52e6", ok="#6fcf97", btn="#241c33", groove="#473a63", card="#181223", hover_bg="#271e3a",
                 hover_edge="#45365f", titlebar="#0f0b16", grid_major="#46395f", faint="#6d6088", accent2="#9b5de5"),
    "light": dict(bg="#f3f5f8", panel="#ffffff", edge="#d6dce5", grid_minor="#e9eef5", plot_bg="#ffffff",
                  viewport_bg="#e9eef5", text="#132033", dim="#5b6779", accent="#1d6cb0", select_bg="#dbe9f8",
                  run_btn="#2a6fd6", ok="#2e8b57",
                  btn="#e6ebf2", groove="#c9d2de", card="#f7f9fc", hover_bg="#eaf1fa", hover_edge="#bcd0e8",
                  titlebar="#e9edf3", grid_major="#c4ccd8", faint="#9aa5b5", accent2="#e6761a"),
}
# display names in picker order ("dark" stays the stored key of the original dark theme, shown as Graphite)
THEME_NAMES = {"dark": "Graphite", "ocean": "Ocean", "forest": "Forest", "plum": "Plum", "light": "Light"}


def swatch(name: str) -> tuple[str, str, str, str]:
    """Four representative colors of a theme: background, selection, accent, secondary accent."""
    t = TOKENS[name]
    return t["viewport_bg"], t["select_bg"], t["accent"], t["accent2"]
# quantity colors (both themes)
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
        """Token or quantity color name -> hex; hex strings pass through."""
        if key.startswith("#"):
            return key
        return QUANTITY.get(key) or self.t[key]

    def set(self, name: str) -> None:
        if name not in TOKENS:
            raise KeyError(f"unknown theme {name!r}")
        if name != self.name:
            self.name = name
            self.changed.emit(name)


theme = _Theme()


def arrow_png(color: str, up: bool = False) -> str:
    """Path (forward slashes, for ``url()``) of a plain down-pointing chevron in ``color``: the combo box arrow, drawn
    flat instead of the native raised button.  Written once per color to the temporary folder."""
    folder = Path(tempfile.gettempdir()) / "propsim_arrows"
    folder.mkdir(parents=True, exist_ok=True)
    f = folder / f"{'up' if up else 'down'}_{color.lstrip('#')}.png"
    if not f.is_file():
        img = QImage(20, 20, QImage.Format_ARGB32)
        img.fill(Qt.transparent)
        p = QPainter(img)
        p.setRenderHint(QPainter.Antialiasing)
        p.setPen(QPen(QColor(color), 2.0, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
        pts = [QPointF(4.5, 12.5), QPointF(10.0, 7.0), QPointF(15.5, 12.5)] if up else [QPointF(4.5, 7.0), QPointF(10.0, 12.5), QPointF(15.5, 7.0)]
        p.drawPolyline(pts)
        p.end()
        img.save(str(f))
    return f.as_posix()


def build_qss(t: dict[str, str]) -> str:
    return f"""
* {{ font-family: "Segoe UI", "Inter", sans-serif; font-size: 10pt; color: {t['text']}; }}
QMainWindow, QDockWidget, QToolBar, QDialog {{ background: {t['bg']}; }}
QMainWindow::separator {{ background: {t['bg']}; width: 4px; height: 4px; }}
QMenuBar {{ background: {t['bg']}; color: {t['dim']}; }}
QMenuBar::item {{ padding: 4px 8px; margin: 0 1px; background: transparent; border-radius: 4px; }}
QMenuBar::item:selected {{ background: {t['select_bg']}; }}
QMenu {{ background: {t['panel']}; border: 1px solid {t['edge']}; padding: 4px 0; }}
QMenu::item {{ padding: 5px 40px 5px 14px; }}
QMenu::item:selected {{ background: {t['select_bg']}; color: {t['accent']}; }}
QMenu::separator {{ height: 1px; background: {t['edge']}; margin: 4px 10px; }}
QMenu::right-arrow {{ right: 10px; }}
QDockWidget > QWidget {{ background: {t['panel']}; }}
QFrame#card, QWidget#card {{ background: {t['panel']}; border: 1px solid {t['edge']}; border-radius: 6px; }}
QLabel {{ background: transparent; border: none; }}
QLabel#heading {{ color: {t['dim']}; font-weight: bold; font-size: 9pt; letter-spacing: 1px; }}
QLabel#groupTitle {{ color: {t['accent']}; font-weight: bold; }}
QLabel#value {{ color: {t['dim']}; font-size: 9pt; }}
QLabel#caption {{ color: {t['dim']}; font-size: 8.5pt; }}
QFrame#tourcard {{ background: {t['panel']}; border: 1.5px solid {t['accent']}; border-radius: 12px; }}
QFrame#hoverbox {{ background: {t['hover_bg']}; border: 1px solid {t['hover_edge']}; border-radius: 5px; }}
QFrame#readout {{ background: {t['card']}; border: 1px solid {t['edge']}; border-radius: 6px; }}
QPlainTextEdit, QTextBrowser {{ background: {t['panel']}; border: none; color: {t['dim']};
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
QSpinBox, QDoubleSpinBox {{ background: {t['btn']}; border: 1px solid {t['edge']}; border-radius: 5px; padding: 2px 6px;
                           color: {t['text']}; }}
QSpinBox::up-button, QDoubleSpinBox::up-button, QSpinBox::down-button, QDoubleSpinBox::down-button {{
    subcontrol-origin: border; width: 20px; border: none; background: transparent; }}
QSpinBox::up-button, QDoubleSpinBox::up-button {{ subcontrol-position: top right; }}
QSpinBox::down-button, QDoubleSpinBox::down-button {{ subcontrol-position: bottom right; }}
QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {{ image: url({arrow_png(t['dim'], True)}); width: 10px; height: 10px; }}
QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {{ image: url({arrow_png(t['dim'])}); width: 10px; height: 10px; }}
QComboBox::drop-down {{ border: none; background: transparent; width: 24px; }}
QComboBox::down-arrow {{ image: url({arrow_png(t['dim'])}); width: 10px; height: 10px; }}
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
QTabWidget, QTabBar {{ background: {t['panel']}; }}
QTabBar::tear {{ width: 0; }}
QScrollArea {{ border: none; background: transparent; }}
QScrollArea > QWidget > QWidget {{ background: transparent; }}
QScrollBar:vertical {{ background: {t['panel']}; width: 10px; margin: 0; border: none; }}
QScrollBar:horizontal {{ background: {t['panel']}; height: 10px; margin: 0; border: none; }}
QScrollBar::handle:vertical {{ background: {t['groove']}; border-radius: 5px; min-height: 28px; margin: 1px 1px; }}
QScrollBar::handle:horizontal {{ background: {t['groove']}; border-radius: 5px; min-width: 28px; margin: 1px 1px; }}
QScrollBar::handle:hover {{ background: {t['faint']}; }}
QScrollBar::handle:pressed {{ background: {t['dim']}; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; width: 0; background: transparent; border: none; }}
QSplitter::handle {{ background: {t['bg']}; }}
QToolTip {{ background: {t['panel']}; color: {t['text']}; border: 1px solid {t['edge']}; }}
"""
