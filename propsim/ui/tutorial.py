"""Guided tour: a dimmed overlay with a spotlight on one part of the window and a card that explains it.
Keys: Left / Right (or Home / End) move between steps, Esc leaves.  Titles use APA case; all math is LaTeX (the card
wraps mixed text and mathtext line by line)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from PySide6.QtCore import QEvent, QPoint, QRect, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QApplication, QFrame, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from .latex import LatexLabel, wrap_latex
from .text import apa
from .theme import theme

CARD_W = 420
MARGIN = 14


@dataclass
class Step:
    title: str
    body: list[str]                                       # paragraphs (plain words, units in parentheses, math in $...$)
    target: Callable | None = None                        # window -> widget | [widgets] | None
    prep: Callable | None = None


def _steps() -> list[Step]:
    return [
        Step("Welcome to PropSim",
             ["A one-minute tour of the window. Use the left and right arrow keys to move between steps, and Esc to "
              "leave at any time."],
             prep=lambda w: (w.select_model("common.engine") if w.ctrl is None or w.ctrl.key != "common.engine" else None)),
        Step("Modules",
             ["Pick a model here. A model marked (Live) recomputes in under $50$ (ms) while you drag a slider; a model "
              "marked (Run) simulates in a background process with progress and Cancel, then plays back."],
             lambda w: w.tree),
        Step("3D View",
             ["Rotate with the left mouse button, pan with the right, zoom with the wheel. The toolbar sets the cutaway "
              "angle, picks the field painted on the model, toggles the colorbar and saves a snapshot."],
             lambda w: w.vp),
        Step("Plots",
             ["Every plot has a grid and LaTeX labels. Run models draw a cursor at the playback time, and the tabs "
              "group related plots."],
             lambda w: w.plot_tabs),
        Step("Parameters",
             ["Each slider changes one input. Hover over a slider to read its meaning and its dimensions, for example "
              "$[\\mathrm{M\\,L^{-1}\\,T^{-2}}]$ for a pressure, in the box below."],
             lambda w: [w.scroll, w.hover]),
        Step("Run and Playback",
             ["Run starts a simulation of the selected Run model; Cancel stops it. Afterwards the scrubber, the play "
              "button and the speed box replay the result in the plots, the 3D view and the readouts together."],
             lambda w: w.bar),
        Step("Saved Runs and Compare",
             ["Every run is saved automatically. Click one to reload it, or tick Compare Overlay to draw saved runs "
              "of the same model on top of each other."],
             lambda w: [w.runs_list, w.compare_cb]),
        Step("Export",
             ["PNG saves the current plots, CSV, .mat and .h5 save the data, and .py and .m write a runnable script "
              "that rebuilds the plots with your parameters (File, Export Code)."],
             lambda w: list(w.bar.export_btns.values())),
        Step("Panels",
             ["Drag a panel by the grip at its top left to move it or float it; the cross at its top right closes it. "
              "View, Panels brings it back, and F12 restores the default arrangement."],
             lambda w: [d.titleBarWidget() for d in (w.vp_dock, w.plots_dock)]),
        Step("Color Themes",
             ["View, Color Theme (F10) opens the picker with Graphite, Ocean, Forest, Plum and Light. "
              "Your choice is remembered."],
             lambda w: w.menuBar()),
        Step("Engine to Vehicle",
             ["Run a hybrid, solid or liquid model, then choose File, Send Thrust to Vehicle and a stage. The Vehicle "
              "ascent model flies that thrust curve when the stage's thrust source is set to the imported profile."],
             lambda w: w.menuBar()),
        Step("Your Data",
             ["Runs, cases, exports, thrust profiles and backups live in one data folder (File, Data Folder changes "
              "it). Restore Last Session reopens your previous model and parameters."],
             None),
        Step("You Are Ready",
             ["Help, Tutorial replays this tour and Help, Online Help opens the user guide. Press Esc or the "
              "Done button to start."],
             None),
    ]


class TutorialOverlay(QWidget):
    finished = Signal(bool)                               # True when the last step was reached

    def __init__(self, window, steps: list[Step] | None = None) -> None:
        super().__init__(window)
        self.win, self.steps, self.i = window, steps or _steps(), 0
        self.setFocusPolicy(Qt.StrongFocus)
        self.setObjectName("tour")
        self.card = QFrame(self)
        self.card.setObjectName("tourcard")
        self.card.setFixedWidth(CARD_W)
        cl = QVBoxLayout(self.card)
        cl.setContentsMargins(18, 14, 18, 14)
        cl.setSpacing(8)
        self.counter = QLabel("")
        self.counter.setObjectName("caption")
        self.title = LatexLabel("", "accent", 13.0)
        self.body = QVBoxLayout()
        self.body.setSpacing(2)
        cl.addWidget(self.counter)
        cl.addWidget(self.title)
        cl.addLayout(self.body)
        nav = QHBoxLayout()
        nav.setSpacing(8)
        self.b_skip, self.b_back, self.b_next = QPushButton("Skip"), QPushButton("Back"), QPushButton("Next")
        self.b_next.setObjectName("run")
        for b in (self.b_skip, self.b_back, self.b_next):
            b.setFocusPolicy(Qt.NoFocus)                  # arrow keys stay with the tour, never move button focus
        self.b_skip.clicked.connect(lambda: self.finish(False))
        self.b_back.clicked.connect(self.prev)
        self.b_next.clicked.connect(self.next)
        nav.addWidget(self.b_skip)
        nav.addStretch(1)
        nav.addWidget(self.b_back)
        nav.addWidget(self.b_next)
        cl.addLayout(nav)
        theme.changed.connect(self._on_theme)
        window.installEventFilter(self)
        QApplication.instance().installEventFilter(self)     # keys work whichever widget has the focus
        self.setGeometry(window.rect())
        self.raise_()
        self.show()
        self.setFocus()
        self.show_step(0)

    # ------------------------------------------------------------------------------------------ navigation
    def show_step(self, i: int) -> None:
        self.i = max(0, min(i, len(self.steps) - 1))
        st = self.steps[self.i]
        if st.prep:
            st.prep(self.win)
        self.counter.setText(f"{self.i + 1} / {len(self.steps)}")
        self.title.setLatex(apa(st.title))
        while self.body.count():
            w = self.body.takeAt(0).widget()
            if w:
                w.hide()
                w.setParent(None)                         # gone at once, so it cannot overlap the new lines
                w.deleteLater()
        for para in st.body:
            for line in wrap_latex(para, CARD_W - 40):
                lbl = LatexLabel(line, "text", 10.0)
                self.body.addWidget(lbl)
                lbl.show()                                # children added later start hidden, and a layout skips hidden ones
            self.body.addSpacing(6)
        last = self.i == len(self.steps) - 1
        self.b_next.setText("Done" if last else "Next")
        self.b_back.setEnabled(self.i > 0)
        self.b_skip.setVisible(not last)
        self.card.layout().activate()                     # size the card from the new lines before placing it
        self.card.setFixedHeight(self.card.layout().sizeHint().height())
        self.card.layout().setGeometry(self.card.rect())   # lay the lines out now, not on the next event-loop pass
        self.card.layout().activate()
        self._place()
        self.update()

    def next(self) -> None:
        if self.i >= len(self.steps) - 1:
            self.finish(True)
        else:
            self.show_step(self.i + 1)

    def prev(self) -> None:
        self.show_step(self.i - 1)

    def finish(self, done: bool) -> None:
        self.win.removeEventFilter(self)
        QApplication.instance().removeEventFilter(self)
        self.hide()
        self.finished.emit(done)
        self.deleteLater()

    # ------------------------------------------------------------------------------------------ geometry
    def _targets(self) -> list[QWidget]:
        t = self.steps[self.i].target
        if t is None:
            return []
        r = t(self.win)
        r = r if isinstance(r, (list, tuple)) else [r]
        return [w for w in r if w is not None and w.isVisible()]

    def spot(self) -> QRect | None:
        rects = []
        for w in self._targets():
            tl = w.mapTo(self.win, QPoint(0, 0))
            rects.append(QRect(tl, w.size()))
        if not rects:
            return None
        u = rects[0]
        for r in rects[1:]:
            u = u.united(r)
        return u.adjusted(-6, -6, 6, 6).intersected(self.rect())

    def _place(self) -> None:
        cw, ch = self.card.width(), self.card.height()
        W, H = self.width(), self.height()
        s = self.spot()
        if s is None:
            self.card.move((W - cw) // 2, (H - ch) // 2)
            return
        options = [QPoint(s.center().x() - cw // 2, s.bottom() + MARGIN), QPoint(s.center().x() - cw // 2, s.top() - ch - MARGIN),
                   QPoint(s.right() + MARGIN, s.center().y() - ch // 2), QPoint(s.left() - cw - MARGIN, s.center().y() - ch // 2)]
        best, best_cost = None, None
        for p in options:
            x = min(max(p.x(), MARGIN), W - cw - MARGIN)
            y = min(max(p.y(), MARGIN), H - ch - MARGIN)
            r = QRect(x, y, cw, ch)
            inter = r.intersected(s)
            cost = inter.width() * inter.height() + 0.001 * (abs(x - p.x()) + abs(y - p.y()))
            if best_cost is None or cost < best_cost:
                best, best_cost = QPoint(x, y), cost
        if best_cost is not None and best_cost > 0:       # nothing fits clear of the target: dock to the nearer corner
            best = QPoint(W - cw - MARGIN if s.center().x() < W / 2 else MARGIN, H - ch - MARGIN)
        self.card.move(best)

    def _on_theme(self, *_) -> None:
        self.show_step(self.i)

    # ------------------------------------------------------------------------------------------ events
    KEYS = (Qt.Key_Right, Qt.Key_Left, Qt.Key_Home, Qt.Key_End, Qt.Key_Escape)

    def eventFilter(self, obj, ev) -> bool:
        if obj is self.win and ev.type() in (QEvent.Resize, QEvent.Move, QEvent.WindowStateChange):
            self.setGeometry(self.win.rect())
            self._place()
            self.update()
        elif ev.type() == QEvent.ShortcutOverride and ev.key() in self.KEYS:
            ev.accept()                                   # Esc is also the Cancel shortcut: let the tour have the key
        elif ev.type() == QEvent.KeyPress and ev.key() in self.KEYS:
            self.keyPressEvent(ev)
            return True
        return False

    def keyPressEvent(self, e) -> None:
        k = e.key()
        if k == Qt.Key_Right:
            self.next()
        elif k == Qt.Key_Left:
            self.prev()
        elif k == Qt.Key_Home:
            self.show_step(0)
        elif k == Qt.Key_End:
            self.show_step(len(self.steps) - 1)
        elif k == Qt.Key_Escape:
            self.finish(False)
        else:
            e.accept()                                    # nothing else reaches the window behind the tour

    def mousePressEvent(self, e) -> None:
        e.accept()                                        # clicks on the dimmed area do nothing

    def paintEvent(self, _) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        full = QPainterPath()
        full.addRect(QRectF(self.rect()))
        s = self.spot()
        if s is not None:
            hole = QPainterPath()
            hole.addRoundedRect(QRectF(s), 8, 8)
            full = full.subtracted(hole)
        p.fillPath(full, QColor(0, 0, 0, 150))
        if s is not None:
            p.setPen(QPen(QColor(theme.t["accent"]), 2.0))
            p.setBrush(Qt.NoBrush)
            p.drawRoundedRect(QRectF(s), 8, 8)
