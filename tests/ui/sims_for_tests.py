"""Helper simulations for the worker-process tests (imported by name inside the child process)."""
import os
import time

import numpy as np

from propsim.core.result import Result
from propsim.core.schema import P, ParamModel
from propsim.sims.base import Simulation


class Par(ParamModel):
    n: float = P(5.0, symbol=r"n", desc="steps", units="", dims="1", lo=1.0, hi=1000.0, step=1.0)


class Counting(Simulation):
    def run(self, params, progress, cancel):
        n = int(params.n)
        for i in range(n):
            if cancel.is_set():
                raise InterruptedError
            progress((i + 1) / n)
        t = np.linspace(0.0, 1.0, 11)
        return Result(t, {"y": t**2}, {"pid": float(os.getpid())}, ["ok"])


class Slow(Simulation):
    def run(self, params, progress, cancel):
        for i in range(600):
            time.sleep(0.05)
            progress(i / 600)
            if cancel.is_set():
                raise InterruptedError
        return Result(np.zeros(2), {}, {}, [])


class Crash(Simulation):
    """Dies without a Python exception, like a segfault inside a numerical library."""

    def run(self, params, progress, cancel):
        progress(0.1)
        os._exit(3)


class Boom(Simulation):
    def run(self, params, progress, cancel):
        raise ValueError("deliberate failure")
