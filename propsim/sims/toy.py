"""Toy Run model (damped spring-mass) used to prove the Run pipeline independently of any propulsion physics."""
from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp

from ..core.result import Result
from ..core.schema import P, ParamModel
from .base import Simulation


class SpringMassParams(ParamModel):
    m: float = P(1.0, symbol="m", desc="mass", units="kg", dims="M", lo=0.1, hi=10, step=0.1, group="System")
    k: float = P(40.0, symbol="k", desc="spring stiffness", units="N·m⁻¹", dims="M T⁻²", lo=1, hi=200, step=1,
                 group="System")
    c: float = P(0.8, symbol="c", desc="damping coefficient", units="N·s·m⁻¹", dims="M T⁻¹", lo=0, hi=10, step=0.05,
                 group="System")
    x0: float = P(0.5, symbol="x_0", desc="initial displacement", units="m", dims="L", lo=-1, hi=1, step=0.05,
                  group="Initial conditions")
    t_end: float = P(8.0, symbol="t_{end}", desc="simulation time", units="s", dims="T", lo=1, hi=30, step=0.5,
                     group="Initial conditions")


class SpringMass(Simulation):
    N_FRAMES = 401

    def run(self, p: SpringMassParams, progress, cancel) -> Result:
        t = np.linspace(0, p.t_end, self.N_FRAMES)
        y = np.empty((2, self.N_FRAMES))
        y[:, 0] = (p.x0, 0.0)
        for k in range(20):                                    # chunked so Cancel/progress are serviced
            if cancel.is_set():
                raise InterruptedError
            i0, i1 = k * 20, (k + 1) * 20
            s = solve_ivp(lambda _t, q: [q[1], (-p.k * q[0] - p.c * q[1]) / p.m], (t[i0], t[i1]), y[:, i0],
                          t_eval=t[i0:i1 + 1], rtol=1e-9, atol=1e-12)
            y[:, i0:i1 + 1] = s.y
            progress((k + 1) / 20)
        E = 0.5 * p.m * y[1] ** 2 + 0.5 * p.k * y[0] ** 2
        return Result(t, {"x": y[0], "v": y[1], "E": E}, {"E0": float(E[0])}, ["spring-mass toy run"])
