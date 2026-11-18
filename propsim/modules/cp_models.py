"""Solid module: quasi-steady CP motor (SI and US customary units) with end-face burning, throat erosion and sweeps."""
from __future__ import annotations

import numpy as np

from ..core.ballistics import cpmotor as cp
from ..core.result import Curve, PlotSpec, TabSpec
from ..sims.cp_run import ARRAY_KEYS, IN, LBM, PSI, CPMotorRun, CPParamsSI, CPParamsUS
from .base import RUN_COLORS, ModelController
from .geometry import broadcast_scalar, cut_face, rev_grid, thetas

LBF = 4.4482216152605
TT = r"$t$ (s)"
MODE_COLORS = ["thrust", "oxidizer", "fuel"]
SWEEP_SYMBOL = {"r_o": r"r_o", "r_i": r"r_i", "L": r"L", "R_t": r"R_t"}
# display factor from SI and unit label, per unit system
UNITS = {
    "SI": dict(P=(1e-6, "MPa"), F=(1e-3, "kN"), A=(1e4, "cm²"), M=(1.0, "kg"), R=(1e3, "mm·s⁻¹"), L=(1e3, "mm"),
               I=(1e-3, "kN·s")),
    "US": dict(P=(1 / PSI, "psi"), F=(1 / LBF, "lbf"), A=(1 / IN**2, "in²"), M=(1 / LBM, "lbm"), R=(1 / IN, "in·s⁻¹"),
               L=(1 / IN, "in"), I=(1 / LBF, "lbf·s")),
}


class CPModelBase(ModelController):
    legend_labels = {
        "cut": "Propellant grain (cut face)", "end_head": "Inhibited end face", "end_aft": "Inhibited end face"}
    module = "Solid"
    mode = "run"
    system = "SI"

    def __init__(self) -> None:
        self.sim = CPMotorRun()
        self._m: dict = {}

    def _u(self, q: str) -> tuple[float, str]:
        return UNITS[self.system][q]

    def _lab(self, sym: str, q: str) -> str:
        return rf"${sym}$ ({self._u(q)[1]})"

    def reset_scene(self) -> None:
        self._m = {}

    # ---------------------------------------------------------------- plots
    def _series(self, r, key: str, scale: float) -> list[Curve]:
        a, m = r.arrays, r.meta
        out = []
        if m["sweep"] == "none":
            for mi, mode in enumerate(m["modes"]):
                if f"c0_m{mi}_{key}" in a:
                    out.append(Curve(a[f"c0_m{mi}_t"], a[f"c0_m{mi}_{key}"] * scale, MODE_COLORS[mi], cp.EROSION_LABELS[mi]))
        else:
            for ci, lab in enumerate(m["sweep_labels"]):
                if f"c{ci}_m0_{key}" in a:
                    out.append(Curve(a[f"c{ci}_m0_t"], a[f"c{ci}_m0_{key}"] * scale, RUN_COLORS[ci % len(RUN_COLORS)],
                                     rf"${SWEEP_SYMBOL[m['sweep']]}$ = {lab} ({self._u('L')[1]})"))
        return out

    def _plot(self, r, title, key, q, sym, unit_label=None) -> PlotSpec:
        scale, _ = self._u(q) if q else (1.0, "")
        yl = self._lab(sym, q) if q else rf"${sym}$ (–)"
        return PlotSpec(title, TT, yl, self._series(r, key, scale), cursor=True)

    def tabs(self, p, r, compare=None) -> list[TabSpec]:
        if r is None:
            e = lambda title, sym, q: PlotSpec(title, TT, self._lab(sym, q) if q else rf"${sym}$ (–)", [], cursor=True)
            return [TabSpec("Ballistics", [e("Chamber Pressure", "P_c", "P"), e("Thrust (Vacuum)", "F", "F"),
                                            e("Thrust (Sea Level)", "F", "F")]),
                    TabSpec("Grain", []), TabSpec("Nozzle", []), TabSpec("Compare", [])]
        bal = [self._plot(r, "Chamber Pressure", "Pc", "P", "P_c"), self._plot(r, "Thrust (Vacuum)", "F_vac", "F", "F"),
               self._plot(r, "Thrust (Sea Level)", "F_sl", "F", "F")]
        gr = [self._plot(r, "Burning Area", "Ab", "A", "A_b"), self._plot(r, "Propellant Mass", "m_p", "M", "m_p"),
              self._plot(r, "Burn Rate", "rdot", "R", r"\dot r")]
        noz = [self._plot(r, "Throat Radius", "Rt", "L", "R_t"), self._plot(r, "Expansion Ratio", "eps", None, r"\varepsilon"),
               self._plot(r, "Thrust Coefficient", "CFv", None, "C_F")]
        runs = compare or [("current", r)]
        cmp_ = []
        for title, ch, q, sym in (("Thrust (Vacuum)", "F_vac", "F", "F"), ("Chamber Pressure", "Pc", "P", "P_c"),
                                  ("Burning Area", "Ab", "A", "A_b")):
            sc = UNITS[self.system][q][0]
            cmp_.append(PlotSpec(title, TT, self._lab(sym, q),
                                 [Curve(rr.t, rr.channels[ch] * sc, RUN_COLORS[i % len(RUN_COLORS)], name)
                                  for i, (name, rr) in enumerate(runs)], cursor=False))
        return [TabSpec("Ballistics", bal), TabSpec("Grain", gr), TabSpec("Nozzle", noz), TabSpec("Compare", cmp_)]

    def readout(self, p, r, frame) -> list[str]:
        if r is None or frame is None:
            return [rf"$P_c$ = – ({self._u('P')[1]})",
                    rf"$F$ = – ({self._u('F')[1]})", rf"$A_b$ = – ({self._u('A')[1]})", rf"$m_p$ = – ({self._u('M')[1]})"]
        c = r.channels
        return [rf"$P_c$ = {c['Pc'][frame] * self._u('P')[0]:.2f} ({self._u('P')[1]})",
                rf"$F$ = {c['F_vac'][frame] * self._u('F')[0]:.2f} ({self._u('F')[1]})",
                rf"$A_b$ = {c['Ab'][frame] * self._u('A')[0]:.1f} ({self._u('A')[1]})",
                rf"$m_p$ = {c['m_p'][frame] * self._u('M')[0]:.3f} ({self._u('M')[1]})"]

    def caption(self, p, r, frame) -> str:
        if r is None or frame is None:
            return r"press Run: quasi-steady march of the web with end faces, throat erosion and optional sweeps"
        c = r.channels
        Kn = c["Ab"][frame] / (np.pi * c["Rt"][frame] ** 2)
        mode = cp.EROSION_LABELS[cp.EROSION.index(p.erosion)].lower()
        return (rf"$t$ = {r.t[frame]:.2f} (s) · web burnt {c['w'][frame] * self._u('L')[0]:.2f} ({self._u('L')[1]}) · "
                rf"$K_n$ = {Kn:.0f} · $\varepsilon$ = {c['eps'][frame]:.1f} · {mode}")

    def summary_line(self, p, r) -> str:
        m = r.meta
        f, un = self._u("I")
        return (rf"burn time {m['t_burn']:.2f} (s) · peak $P_c$ {m['Pc_max'] * self._u('P')[0]:.1f} ({self._u('P')[1]}) · "
                rf"$I_t$ = {m['I_vac'] * f:.1f} ({un}) · $I_{{sp}}$ = {m['Isp_vac']:.1f} (s)")

    def export_data(self, p, r):
        return {"t_s": r.t, **{k: v for k, v in r.channels.items()}, **r.arrays}

    # ---------------------------------------------------------------- 3D: cutaway grain, bore and end faces regressing
    def _nozzle(self, m: cp.CPMotor):
        s = np.linspace(0.0, 1.0, 24)
        x = m.L + s * 1.6 * m.r_o
        sm = lambda z: z * z * (3 - 2 * z)
        r = np.where(s < 0.35, m.r_o + (m.R_t - m.r_o) * sm(np.clip(s / 0.35, 0, 1)),
                     m.R_t + (m.R_e - m.R_t) * sm(np.clip((s - 0.35) / 0.65, 0, 1)))
        return x, r

    def build_scene(self, vp, p, r=None) -> None:
        pl = vp.plotter
        pl.clear()
        try:
            m = p.motor()
        except Exception:                                    # noqa: BLE001 - an invalid slider combination: empty view
            return
        th = thetas(vp.cutaway)
        pl.add_mesh(rev_grid(np.array([0.0, m.L]), np.full(2, m.r_o * 1.05), th), color="#8a94a6", opacity=0.22, name="case")
        xn, rn = self._nozzle(m)
        pl.add_mesh(rev_grid(xn, rn, th), color="#6b768a", name="nozzle", smooth_shading=True)
        self._m = dict(m=m, th=th, cut=vp.cutaway, pmax=float(r.channels["Pc"].max()) if r is not None else 1.0)
        vp.reset_view(zoom=1.1)

    def update_scene(self, vp, p, r, frame) -> None:
        if not self._m or self._m["cut"] != vp.cutaway:
            self.build_scene(vp, p, r)
        if not self._m:
            return
        pl, m, th = vp.plotter, self._m["m"], self._m["th"]
        if r is None or frame is None:
            w, pc = 0.0, 0.0
        else:
            w, pc = float(r.channels["w"][frame]), float(r.channels["Pc"][frame])
        e = 1.0 if m.ends else 0.0
        x0, x1 = e * w, m.L - e * w
        ri = m.r_i + w
        clim = (0.0, max(self._m["pmax"], 1.0))
        bore = rev_grid(np.linspace(x0, x1, 10), np.full(10, ri), th)
        bore["p"] = broadcast_scalar(np.full(10, pc), len(th))
        pl.add_mesh(bore, scalars="p", cmap="plasma", clim=clim, show_scalar_bar=False, name="bore", smooth_shading=True)
        for nm, x, burning in (("end_head", x0, m.ends), ("end_aft", x1, m.ends)):
            face = rev_grid(np.full(6, x), np.linspace(ri, m.r_o, 6), th)
            if burning:
                face["p"] = broadcast_scalar(np.full(6, pc), len(th))
                pl.add_mesh(face, scalars="p", cmap="plasma", clim=clim, show_scalar_bar=False, name=nm)
            else:
                pl.add_mesh(face, color="#6b768a", name=nm)
        if vp.cutaway > 0:
            x = np.linspace(x0, x1, 8)
            for nm, ang in (("cut0", th[0]), ("cut1", th[-1])):
                pl.add_mesh(cut_face(x, np.full(8, ri), np.full(8, m.r_o), ang, 4), color="#c9a66b", name=nm)
        vp.set_colorbar("plasma", clim[0] * self._u("P")[0], clim[1] * self._u("P")[0], rf"$P_c$ ({self._u('P')[1]})")
        vp.render()


class CPMotorModel(CPModelBase):
    key = "solid.cpmotor"
    title = "CP motor (quasi-steady)"
    Params = CPParamsSI
    system = "SI"
    fields = [r"$P_c$ (MPa)"]


class CPMotorUSModel(CPModelBase):
    key = "solid.cpmotor_us"
    title = "CP motor (US units)"
    Params = CPParamsUS
    system = "US"
    fields = [r"$P_c$ (psi)"]
