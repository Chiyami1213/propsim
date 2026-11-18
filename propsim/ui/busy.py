"""A progress bar for work the user starts from a menu or button (open, save, export, import).

``busy(window, text)`` shows a busy bar and a message in the window's status bar (and the busy cursor) while its block
runs; the yielded bar can be switched to a real progress range (``bar.setRange(0, n)`` then ``setValue``).
``busy_action(text)`` decorates a method that first asks for a file: the bar appears once the file dialog has returned a
path (or at once when the caller passed the path), so it never sits behind an open dialog.
"""
from __future__ import annotations

import functools
import inspect
from contextlib import contextmanager

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QLabel, QProgressBar


@contextmanager
def busy(window, text: str = "Working"):
    sb = window.statusBar()
    label = QLabel(text)
    bar = QProgressBar()
    bar.setRange(0, 0)
    bar.setFixedWidth(220)
    was_visible = sb.isVisible()
    sb.addWidget(label, 1)
    sb.addPermanentWidget(bar)
    sb.show()
    QApplication.setOverrideCursor(Qt.BusyCursor)
    QApplication.processEvents()
    try:
        yield bar
    finally:
        QApplication.restoreOverrideCursor()
        sb.removeWidget(label)
        sb.removeWidget(bar)
        label.deleteLater()
        bar.deleteLater()
        if not was_visible and not sb.currentMessage():
            sb.hide()


def busy_action(text: str, path_arg: str = "path"):
    """Decorator for ``method(self, ..., path=None)``: busy bar from the moment a path is known."""

    def deco(fn):
        sig = inspect.signature(fn)

        @functools.wraps(fn)
        def wrapper(self, *args, **kwargs):
            real = fn.__globals__.get("QFileDialog")
            given = sig.bind_partial(self, *args, **kwargs).arguments.get(path_arg)
            state = {"cm": None}

            def begin():
                if state["cm"] is None:
                    state["cm"] = busy(self, text)
                    state["cm"].__enter__()

            class Proxy:
                """The real file dialog, starting the busy bar when it returns a path."""

                @staticmethod
                def getSaveFileName(*a, **k):
                    r = real.getSaveFileName(*a, **k)
                    if r and r[0]:
                        begin()
                    return r

                @staticmethod
                def getOpenFileName(*a, **k):
                    r = real.getOpenFileName(*a, **k)
                    if r and r[0]:
                        begin()
                    return r

                def __getattr__(self, name):
                    return getattr(real, name)

            if real is not None:
                fn.__globals__["QFileDialog"] = Proxy
            try:
                if given:
                    begin()
                return fn(self, *args, **kwargs)
            finally:
                if real is not None:
                    fn.__globals__["QFileDialog"] = real
                if state["cm"] is not None:
                    state["cm"].__exit__(None, None, None)

        return wrapper

    return deco
