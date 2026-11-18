"""Run driver for the quasi-steady cylindrical-perforated (CP) solid motor: end-face burning, three throat-erosion
treatments, one-parameter sweeps, in SI or US customary units."""
from __future__ import annotations

from typing import ClassVar

import numpy as np

from ..core.ballistics import cpmotor as cp
from ..core.result import Result
from ..core.schema import C, P, ParamModel
from .base import Simulation

IN, FT, PSI, LBM = 0.0254, 0.3048, 6894.757293168, 0.45359237
SWEEP = ["none", "r_o", "r_i", "L", "R_t"]
SWEEP_LABELS = ["No sweep", "Outer radius", "Bore radius", "Grain length", "Throat radius"]
ARRAY_KEYS = ("t", "Pc", "F_vac", "F_sl", "Ab", "rdot", "m_p", "dt", "eps", "CFv", "Rt", "w")


class _CPCommon(ParamModel):
    """Conversion of the display-unit parameters to SI (set by the unit-system subclasses)."""

    # to-SI factors: length, density, velocity (c*), burn rate, pressure, erosion rate
    K: ClassVar[dict] = {}

    def motor(self, **override) -> cp.CPMotor:
        k = self.K
        d = dict(r_o=self.r_o * k["L"], r_i=self.r_i * k["L"], L=self.L * k["L"], rho_p=self.rho_p * k["rho"],
                 cstar=self.cstar * k["V"], n=self.n, r_ref=self.r_ref * k["R"], p_ref=self.p_ref * k["P"],
                 R_t=self.R_t * k["L"], R_e=self.R_e * k["L"], gamma=self.gamma, ends=self.ends == "burning",
                 erosion=self.erosion, edot_ref=self.edot_ref * k["R"], dw=self.dw * k["L"])
        d.update(override)
        return cp.CPMotor(**d)

    def sweep_values(self) -> list[tuple[str, float]]:
        """``(label, value in SI)`` of the swept parameter; a single entry when not sweeping."""
        if self.sweep == "none":
            return [("", getattr(self.motor(), "r_o"))]
        base = getattr(self.motor(), self.sweep)
        f = self.span / 100.0
        return [(f"{base * m / self.K['L']:.3g}", base * m) for m in (1.0 - f, 1.0, 1.0 + f)]


def _fields(unit: str):
    si = unit == "SI"
    L_, rho_, V_, R_, P_ = ("mm", "kg·m⁻³", "m·s⁻¹", "mm·s⁻¹", "MPa") if si else ("in", "lbm·in⁻³", "ft·s⁻¹", "in·s⁻¹", "psi")
    sc = (lambda a, b: a) if si else (lambda a, b: b)

    class Params(_CPCommon):
        K: ClassVar[dict] = (dict(L=1e-3, rho=1.0, V=1.0, R=1e-3, P=1e6) if si
                             else dict(L=IN, rho=LBM / IN**3, V=FT, R=IN, P=PSI))
        r_o: float = P(sc(76.2, 3.0), symbol=r"r_o", desc="grain outer radius (outer surface inhibited)", units=L_, dims="L",
                       lo=sc(20.0, 1.0), hi=sc(300.0, 12.0), step=sc(0.1, 0.01), group="Grain")
        r_i: float = P(sc(31.75, 1.25), symbol=r"r_i", desc="initial bore radius", units=L_, dims="L", lo=sc(5.0, 0.2),
                       hi=sc(250.0, 10.0), step=sc(0.1, 0.01), group="Grain")
        L: float = P(sc(305.0, 12.0), symbol=r"L", desc="grain length", units=L_, dims="L", lo=sc(50.0, 2.0),
                     hi=sc(1500.0, 60.0), step=sc(0.5, 0.05), group="Grain")
        ends: str = C("inhibited", choices=["inhibited", "burning"], labels=["End faces inhibited", "Both end faces burn"],
                      symbol=r"\mathrm{ends}", desc="whether the two end faces of the grain burn with the bore", group="Grain")
        rho_p: float = P(sc(1771.0, 0.064), symbol=r"\rho_p", desc="propellant density", units=rho_, dims="M L⁻³",
                         lo=sc(1000.0, 0.036), hi=sc(2200.0, 0.08), step=sc(5.0, 0.0005), group="Propellant")
        cstar: float = P(sc(1524.0, 5000.0), symbol=r"c^*", desc="characteristic velocity", units=V_, dims="L T⁻¹",
                         lo=sc(800.0, 2600.0), hi=sc(2000.0, 6600.0), step=sc(5.0, 20.0), group="Propellant")
        n: float = P(0.4, symbol=r"n", desc="burn-rate pressure exponent (below 1)", units="", dims="1", lo=0.1, hi=0.9,
                     step=0.01, group="Propellant")
        r_ref: float = P(sc(8.9, 0.35), symbol=r"\dot r_{ref}", desc="reference burn rate", units=R_, dims="L T⁻¹",
                         lo=sc(1.0, 0.04), hi=sc(40.0, 1.6), step=sc(0.1, 0.005), group="Propellant")
        p_ref: float = P(sc(6.9, 1000.0), symbol=r"p_{ref}", desc="reference chamber pressure of the burn rate", units=P_,
                         dims="M L⁻¹ T⁻²", lo=sc(1.0, 150.0), hi=sc(20.0, 3000.0), step=sc(0.1, 10.0), group="Propellant")
        R_t: float = P(sc(10.2, 0.4), symbol=r"R_t", desc="initial throat radius", units=L_, dims="L", lo=sc(3.0, 0.12),
                       hi=sc(80.0, 3.2), step=sc(0.1, 0.01), group="Nozzle")
        R_e: float = P(sc(40.6, 1.6), symbol=r"R_e", desc="nozzle exit radius", units=L_, dims="L", lo=sc(8.0, 0.3),
                       hi=sc(200.0, 8.0), step=sc(0.1, 0.01), group="Nozzle")
        gamma: float = P(1.2, symbol=r"\gamma", desc="ratio of specific heats of the exhaust (ideal nozzle)", units="",
                         dims="1", lo=1.1, hi=1.4, step=0.01, group="Nozzle")
        erosion: str = C("none", choices=cp.EROSION, labels=list(cp.EROSION_LABELS), symbol=r"\mathrm{erosion}",
                         desc="throat erosion treatment of the playback case (all three are drawn when not sweeping)",
                         group="Throat erosion")
        edot_ref: float = P(sc(0.15, 0.006), symbol=r"\dot e_{ref}", desc="throat erosion rate at the reference pressure",
                            units=R_, dims="L T⁻¹", lo=0.0, hi=sc(2.0, 0.08), step=sc(0.01, 0.0005), group="Throat erosion")
        sweep: str = C("none", choices=SWEEP, labels=SWEEP_LABELS, symbol=r"\mathrm{sweep}",
                       desc="draw three values of one parameter (nominal and plus or minus the span) on the plots",
                       group="Sweep and steps")
        span: float = P(20.0, symbol=r"\Delta", desc="sweep span around the nominal value", units="%", dims="1", lo=5.0,
                        hi=50.0, step=1.0, group="Sweep and steps")
        dw: float = P(sc(0.1, 0.004), symbol=r"\Delta w", desc="web step of the march", units=L_, dims="L",
                      lo=sc(0.02, 0.001), hi=sc(2.0, 0.08), step=sc(0.01, 0.0005), group="Sweep and steps")

    Params.__name__ = Params.__qualname__ = "CPParams" + unit
    return Params


CPParamsSI = _fields("SI")
CPParamsUS = _fields("US")


class CPMotorRun(Simulation):
    N_FRAMES = 601

    def run(self, p, progress, cancel) -> Result:
        base = p.motor()
        base.check()
        modes = list(cp.EROSION) if p.sweep == "none" else [p.erosion]
        values = p.sweep_values()
        arrays, summ, n_done = {}, {}, 0
        for ci, (_, v) in enumerate(values):
            mot = base if p.sweep == "none" else p.motor(**{p.sweep: v})
            for mi, mode in enumerate(modes):
                if cancel.is_set():
                    raise InterruptedError("cancelled")
                try:
                    mm = cp.CPMotor(**{**mot.__dict__, "erosion": mode})
                    d = cp.march(mm)
                except ValueError:                       # e.g. a swept bore radius that meets the outer radius
                    continue
                for k in ARRAY_KEYS:
                    arrays[f"c{ci}_m{mi}_{k}"] = d[k]
                summ[(ci, mi)] = cp.summary(mm, d)
                n_done += 1
                progress(n_done / (len(values) * len(modes)))
        if not summ:
            raise ValueError("no valid case: check the sweep range against the geometry")
        ci = 0 if p.sweep == "none" else 1                           # the nominal value (middle of the sweep)
        mi = cp.EROSION.index(p.erosion) if p.sweep == "none" else 0
        sel = (ci, mi) if (ci, mi) in summ else next(iter(summ))
        ci, mi = sel
        d = {k: arrays[f"c{ci}_m{mi}_{k}"] for k in ARRAY_KEYS}
        s = summ[sel]
        t_end = s["t_burn"]
        tg = np.linspace(0.0, t_end, self.N_FRAMES)
        ch = {k: np.interp(tg, d["t"], d[k]) for k in ("Pc", "F_vac", "F_sl", "Ab", "rdot", "m_p", "eps", "CFv", "Rt", "w")}
        meta = dict(s, modes=modes, sweep=p.sweep, sweep_labels=[lab for lab, _ in values],
                    cases=sorted([list(k) for k in summ]), sel=list(sel), n_cases=len(summ))
        meta["sweep_values_si"] = [v for _, v in values]
        k = p.K
        log = [rf"setup CP motor: {'both end faces burn' if base.ends else 'end faces inhibited'}, web {base.web / k['L']:.3g} "
               rf"({'mm' if k['L'] == 1e-3 else 'in'}), {len(d['w'])} web steps of {p.dw:g}",
               rf"equilibrium: $K_n$ = {s['Kn_max']:.0f} at most, peak $p_c$ {s['Pc_max'] / k['P']:.1f} "
               rf"({'MPa' if k['P'] == 1e6 else 'psi'}), burn time {s['t_burn']:.2f} (s), total impulse (vacuum) "
               rf"{s['I_vac'] / (1.0 if k['L'] == 1e-3 else 4.4482216152605):.0f} ({'N·s' if k['L'] == 1e-3 else 'lbf·s'}), $I_{{sp}}$ {s['Isp_vac']:.1f} (s) vacuum and {s['Isp_sl']:.1f} (s) sea level",
               rf"propellant {s['m_propellant']:.3f} (kg), all burnt: mass flow in equals mass flow out at every step"]
        if base.erosion != "none" or p.sweep == "none":
            e = {m: summ[(ci, i)] for i, m in enumerate(modes) if (ci, i) in summ}
            if len(e) == 3:
                log.append(rf"throat growth over the burn: constant rate {e['const']['Rt_end'] / k['L']:.3g}, pressure dependent "
                           rf"{e['pressure']['Rt_end'] / k['L']:.3g} ({'mm' if k['L'] == 1e-3 else 'in'}) from "
                           rf"{base.R_t / k['L']:.3g}; peak pressure falls to {e['pressure']['Pc_max'] / k['P']:.1f} "
                           rf"({'MPa' if k['P'] == 1e6 else 'psi'})")
        log.append("quasi-steady: no ignition transient and no tail-off; ideal-gas nozzle")
        return Result(tg, ch, meta, log, arrays)
