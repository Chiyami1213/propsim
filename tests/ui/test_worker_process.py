"""Run worker process: results, progress, Cancel, error reporting and crash isolation."""
import os
import time

import pytest
from PySide6.QtWidgets import QApplication

import sims_for_tests as T
from propsim.ui import worker as W


@pytest.fixture(scope="module", autouse=True)
def app():
    a = QApplication.instance() or QApplication([])
    yield a
    W.shutdown()


def _run(sim, n=5.0, limit=60.0, cancel_after=None):
    ev = {"progress": [], "done": None, "failed": None, "cancelled": False, "finished": 0}
    w = W.RunWorker(sim, T.Par(n=n))
    w.progressed.connect(ev["progress"].append)
    w.done.connect(lambda r: ev.__setitem__("done", r))
    w.failed.connect(lambda s: ev.__setitem__("failed", s))
    w.cancelled.connect(lambda: ev.__setitem__("cancelled", True))
    w.finished.connect(lambda: ev.__setitem__("finished", ev["finished"] + 1))
    t0 = time.time()
    w.start()
    cancelled_at = None
    while ev["finished"] == 0 and time.time() - t0 < limit:
        QApplication.processEvents()
        if cancel_after is not None and cancelled_at is None and time.time() - t0 > cancel_after:
            cancelled_at = time.time()
            w.cancel()
    ev["cancel_latency"] = None if cancelled_at is None else time.time() - cancelled_at
    return w, ev


def test_runs_in_a_different_process_and_returns_result():
    w, ev = _run(T.Counting(), 20)
    assert ev["failed"] is None and ev["done"] is not None and ev["finished"] == 1
    assert ev["done"].meta["pid"] != float(os.getpid())              # not in the GUI process
    assert list(ev["done"].channels["y"]) == pytest.approx(list(ev["done"].t**2))
    assert ev["progress"] and ev["progress"] == sorted(ev["progress"]) and ev["progress"][-1] == pytest.approx(1.0)


def test_exception_in_child_is_reported_with_traceback():
    _, ev = _run(T.Boom())
    assert ev["done"] is None and "deliberate failure" in ev["failed"] and ev["finished"] == 1


def test_cancel_kills_the_worker_quickly():
    w, ev = _run(T.Slow(), cancel_after=0.5)
    assert ev["cancelled"] and ev["done"] is None and ev["finished"] == 1
    assert ev["cancel_latency"] < 1.0
    assert not w._proc.is_alive()


def test_child_crash_does_not_take_the_gui_down():
    """A hard death of the worker (os._exit) becomes a 'failed' message; this test process keeps running."""
    _, ev = _run(T.Crash())
    assert ev["done"] is None and ev["finished"] == 1
    assert "exited unexpectedly" in ev["failed"] and "3" in ev["failed"]
    # and a following run still works
    _, ev2 = _run(T.Counting(), 5)
    assert ev2["done"] is not None


def test_warm_spare_is_ready_after_prewarm():
    W.prewarm()
    proc = W._spare[0]
    assert proc.is_alive()
    t0 = time.time()
    _, ev = _run(T.Counting(), 5)
    assert ev["done"] is not None and time.time() - t0 < 5.0
