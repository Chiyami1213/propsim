"""Background Run worker in a separate *process*.

The numerics (SciPy's LSODA, NumPy, CoolProp, ...) run in a child process, so they can never share memory with the
Qt/VTK/pyqtgraph GUI process: a crash in a solver cannot take the application down, the UI thread never blocks, and
Cancel is a real kill.  A warm spare process is kept ready, so a Run starts without the import delay.

Protocol (multiprocessing ``spawn`` + ``Pipe``): parent sends one job ``(sim_cls, params_cls, params_dict)`` as dotted
paths; the child replies with ``("progress", f)`` messages and a final ``("done", Result)``, ``("cancelled", None)`` or
``("failed", traceback)``.  The child imports no Qt.
"""
from __future__ import annotations

import atexit
import importlib
import multiprocessing as mp
import threading
import traceback

from PySide6.QtCore import QObject, QTimer, Signal

_CTX = mp.get_context("spawn")
_spare: tuple | None = None            # (Process, parent_conn)
_lock = threading.Lock()


def _dotted(obj_or_cls) -> str:
    cls = obj_or_cls if isinstance(obj_or_cls, type) else type(obj_or_cls)
    return f"{cls.__module__}:{cls.__qualname__}"


def _load(path: str):
    mod, name = path.split(":")
    return getattr(importlib.import_module(mod), name)


def _child_main(conn, cancel_ev) -> None:
    """Runs in the worker process: wait for one job, run it, report, exit."""
    try:
        sim_path, params_path, params = conn.recv()
    except EOFError:
        return
    try:
        sim = _load(sim_path)()
        pars = _load(params_path)(**params)
        res = sim.run(pars, lambda f: conn.send(("progress", float(f))), cancel_ev)
        conn.send(("done", res))
    except InterruptedError:
        conn.send(("cancelled", None))
    except BaseException:                                   # noqa: BLE001 - everything is reported to the UI
        conn.send(("failed", traceback.format_exc()))
    finally:
        conn.close()


def _spawn() -> tuple:
    parent, child = _CTX.Pipe()
    ev = _CTX.Event()
    proc = _CTX.Process(target=_child_main, args=(child, ev), daemon=True)
    proc.start()
    child.close()
    return proc, parent, ev


def prewarm() -> None:
    """Start the spare worker process (imports numpy/scipy in the background) if none is waiting."""
    global _spare
    with _lock:
        if _spare is None or not _spare[0].is_alive():
            _spare = _spawn()


def _take() -> tuple:
    global _spare
    with _lock:
        s, _spare = _spare, None
    if s is None or not s[0].is_alive():
        s = _spawn()
    return s


def shutdown() -> None:
    """Terminate the spare process (called at exit)."""
    global _spare
    with _lock:
        s, _spare = _spare, None
    if s is not None and s[0].is_alive():
        s[0].terminate()


atexit.register(shutdown)


class RunWorker(QObject):
    """Runs ``sim.run(params, ...)`` in a worker process; polls the pipe from the GUI thread (no helper threads)."""

    progressed = Signal(float)
    done = Signal(object)        # Result
    failed = Signal(str)
    cancelled = Signal()
    finished = Signal()          # always emitted last, once

    def __init__(self, sim, params, parent=None) -> None:
        super().__init__(parent)
        self.sim, self.params = sim, params
        self._proc = self._conn = self._ev = None
        self._over = False
        self._timer = QTimer(self)
        self._timer.setInterval(10)
        self._timer.timeout.connect(self._poll)

    # ------------------------------------------------------------------------------------------ control
    def start(self) -> None:
        self._proc, self._conn, self._ev = _take()
        self._conn.send((_dotted(self.sim), _dotted(self.params), self.params.model_dump()))
        self._timer.start()
        QTimer.singleShot(0, prewarm)                      # next run finds a warm process waiting

    def cancel(self) -> None:
        """Kill the worker process and report ``cancelled`` immediately."""
        if self._over:
            return
        if self._ev is not None:
            self._ev.set()
        self._kill()
        self._finish("cancelled")

    def wait(self, ms: int = 0) -> bool:                    # API compatibility: completion is signalled, not joined
        return self._over

    # ------------------------------------------------------------------------------------------ internals
    def _kill(self) -> None:
        if self._proc is not None and self._proc.is_alive():
            self._proc.terminate()
            self._proc.join(2.0)

    def _finish(self, kind: str, payload=None) -> None:
        if self._over:
            return
        self._over = True
        self._timer.stop()
        if self._conn is not None:
            self._conn.close()
        if kind == "done":
            self.done.emit(payload)
        elif kind == "failed":
            self.failed.emit(payload)
        elif kind == "cancelled":
            self.cancelled.emit()
        self.finished.emit()

    def _poll(self) -> None:
        if self._over:
            return
        try:
            while self._conn.poll():
                kind, payload = self._conn.recv()
                if kind == "progress":
                    self.progressed.emit(payload)
                else:
                    if self._proc is not None:
                        self._proc.join(2.0)
                    self._finish(kind, payload)
                    return
        except (EOFError, OSError):
            pass
        if not self._proc.is_alive():                       # died without reporting: crash / killed
            code = self._proc.exitcode
            self._finish("failed", f"worker process exited unexpectedly (exit code {code}); the application "
                                   "itself is unaffected")
