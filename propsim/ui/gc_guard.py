"""Run Python's cyclic garbage collector only at safe points in the GUI process.

PySide/Shiboken wrappers and pyqtgraph hold reference cycles.  When the automatic collector fires *inside* a paint
event (any allocation can trigger it), finalizing a dead cycle can delete a Qt object that the running paint still
uses, which crashes natively (observed as an access violation in ``AxisItem.paint`` / ``QPicture.play``, and only
after allocation-heavy code had run).  Disabling the automatic collector and collecting from a timer slot runs the
same cleanup between events, never mid-paint.
"""
from __future__ import annotations

import gc

from PySide6.QtCore import QObject, QTimer

_timer: QTimer | None = None


def collect() -> None:
    gc.collect()


def install(interval_ms: int = 1500, parent: QObject | None = None) -> None:
    """Disable automatic GC and collect every ``interval_ms`` from the Qt event loop (idempotent)."""
    global _timer
    gc.disable()
    if _timer is None:
        _timer = QTimer(parent)
        _timer.setInterval(interval_ms)
        _timer.timeout.connect(collect)
        _timer.start()
