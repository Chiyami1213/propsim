"""Background Run worker: the UI thread never blocks; progress and cancel are serviced between solver chunks."""
from __future__ import annotations

import threading
import traceback

from PySide6.QtCore import QThread, Signal

from ..core.result import Result


class RunWorker(QThread):
    progressed = Signal(float)
    done = Signal(object)        # Result
    failed = Signal(str)
    cancelled = Signal()

    def __init__(self, sim, params, parent=None) -> None:
        super().__init__(parent)
        self.sim, self.params = sim, params
        self.cancel_event = threading.Event()

    def cancel(self) -> None:
        self.cancel_event.set()

    def run(self) -> None:
        try:
            res: Result = self.sim.run(self.params, self.progressed.emit, self.cancel_event)
            self.done.emit(res)
        except InterruptedError:
            self.cancelled.emit()
        except Exception:                                    # noqa: BLE001 - reported to the UI log
            self.failed.emit(traceback.format_exc())
