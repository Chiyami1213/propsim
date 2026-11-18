"""Hybrid module, Run models: quasi-1D burn (axial port evolution, throttling, throat erosion) and N₂O blowdown.

3D playback shows the port radius varying along the axis (aft stations burn faster because fuel adds to the local
mass flux), colored by the chosen field; the blowdown model adds a tank whose liquid level follows the run.
"""
from __future__ import annotations

import numpy as np

from ..core.feed import rho_l
from ..core.result import Curve, PlotSpec, Result, TabSpec
from ..sims.hybrid_burn1d import HybridBurn1D, HybridBurn1DParams, N2OBlowdown, N2OBlowdownParams
from .base import RUN_COLORS
from .geometry import rev_grid, rev_points
from .hybrid_burn import FIELDS, HybridBurnModel

TT = r"$t$ (s)"
FRAC_COLORS = ["#4ea1ff", "#5cc97a", "#e6c34a", "#ff9d3c", "#ff5a3c"]       # t = 0, 25, 50, 75, 100 % of the run


class HybridBurn1DModel(HybridBurnModel):
    key = "hybrid.burn1d"
    title = "Quasi-1D burn"
    module = "Hybrid"
    mode = "run"
    Params = HybridBurn1DParams
    fields = [f[2] for f in FIELDS]

    def __init__(self) -> None:
        super().__init__()
        self.sim = HybridBurn1D()

    # ---------------------------------------------------------------- 3D hooks
    def _stations(self, r) -> tuple[np.ndarray, int]:
        xc = np.asarray(r.meta["x_cells"], float)
        return xc, len(xc)

    def _radii(self, si, r, frame) -> np.ndarray:
        if r is None or frame is None:
            return np.full_like(self._xg, si.R0)
        xc, n = self._stations(r)
        Rc = np.array([r.channels[f"Rp_{i:02d}"][frame] for i in range(n)]) * 1e-3
        return np.interp(self._xg, xc, Rc)

    def _field_x(self, si, r, frame, ch: str) -> np.ndarray:
        if r is None or frame is None:
            return np.zeros_like(self._xg)
        if ch == "rdot":
            xc, n = self._stations(r)
            v = np.array([r.channels[f"rd_{i:02d}"][frame] for i in range(n)])
            return np.interp(self._xg, xc, np.nan_to_num(v))
        return np.full_like(self._xg, float(np.nan_to_num(r.channels[ch][frame])))

    def _range(self, r, ch, default):
        if r is not None and ch == "rdot":
            n = len(r.meta["x_cells"])
            v = np.concatenate([r.channels[f"rd_{i:02d}"] for i in range(n)])
            v = v[np.isfinite(v)]
            if v.size and v.max() > v.min():
                return float(v.min()), float(v.max())
        return super()._range(r, ch, default)

    # ---------------------------------------------------------------- text
    def caption(self, p, r, frame) -> str:
        if r is None or frame is None:
            return r"press Run to simulate · aft stations burn faster as fuel adds to the local $G$"
        c = r.channels
        web = (c["R"][frame] * 1e-3 - p.R_p0 * 1e-3) / (p.web * 1e-3)
        return (rf"$t$ = {r.t[frame]:.2f} (s) · mean web burned {100 * web:.0f} % · "
                rf"$\dot r$: head {c['rdot_head'][frame]:.2f}, aft {c['rdot_aft'][frame]:.2f} (mm·s⁻¹)")

    def summary_line(self, p, r) -> str:
        s = r.meta
        return (rf"$I_t$ = {s['I_t_kNs']:.1f} (kN·s),  $\bar{{I}}_{{sp}}$ = {s['Isp_avg']:.1f} (s),  "
                rf"$\bar{{\frac{{O}}{{F}}}}$ = {s['OF_avg']:.2f}")

    # ---------------------------------------------------------------- plots
    def _empty_hist(self):
        e = lambda title, yl: PlotSpec(title, TT, yl, [], cursor=True)
        return [e("Chamber pressure", r"$P_c$ (MPa)"), e("Thrust", r"$F$ (kN)"),
                e("Mixture-ratio shift", r"$\frac{O}{F}$ (–)")]

    def _profiles(self, r) -> list[PlotSpec]:
        """Axial profiles of port radius, regression rate and mass flux at 0/25/50/75/100 % of the run."""
        n = len(r.meta["x_cells"])
        x = np.array(r.meta["x_cells"]) * 1e3
        idx = [int(round(f * (r.n - 1))) for f in (0.0, 0.25, 0.5, 0.75, 1.0)]
        pick = lambda pre, i: np.array([r.channels[f"{pre}_{k:02d}"][i] for k in range(n)])
        curves = lambda pre, skip0=False: [
            Curve(x, pick(pre, i), col, f"{100 * k // 4} %") for k, (i, col) in enumerate(zip(idx, FRAC_COLORS))
            if not (skip0 and i == 0)]
        xl = r"$x$ (mm)"
        return [PlotSpec("Port radius along the grain", xl, r"$R_p$ (mm)", curves("Rp")),
                PlotSpec("Regression rate along the grain", xl, r"$\dot r$ (mm·s⁻¹)", curves("rd", True)),
                PlotSpec("Local mass flux", xl, r"$G$ (kg·m⁻²·s⁻¹)", curves("G", True))]

    def tabs(self, p, r, compare=None) -> list[TabSpec]:
        if r is None:
            return [TabSpec("Time histories", self._empty_hist()), TabSpec("Port evolution", []),
                    TabSpec(r"Regression $\dot r(t)$", []), TabSpec("Nozzle", []), TabSpec("Compare", [])]
        c, t = r.channels, r.t
        n = len(r.meta["x_cells"])
        cur = lambda y, col, label="", style="solid": Curve(t, y, col, label, style)
        hist = [PlotSpec("Chamber pressure", TT, r"$P_c$ (MPa)", [cur(c["Pc"], "pressure")], cursor=True),
                PlotSpec("Thrust", TT, r"$F$ (kN)", [cur(c["F"], "thrust")], cursor=True),
                PlotSpec("Mixture-ratio shift", TT, r"$\frac{O}{F}$ (–)", [cur(c["OF"], "oxidizer")], cursor=True)]
        reg = [PlotSpec("Regression rate", TT, r"$\dot r$ (mm·s⁻¹)",
                        [cur(c["rdot_head"], "oxidizer", "head"), cur(c["rdot_aft"], "fuel", "aft"),
                         cur(c["rdot"], "text", "mean", "dash")], cursor=True),
               PlotSpec("Port radius", TT, r"$R_p$ (mm)",
                        [cur(c["Rp_00"], "oxidizer", "head"), cur(c[f"Rp_{n - 1:02d}"], "fuel", "aft")], cursor=True),
               PlotSpec("Head-end mass flux", TT, r"$G$ (kg·m⁻²·s⁻¹)", [cur(c["G_head"], "oxidizer")], cursor=True)]
        noz = [PlotSpec("Throat radius", TT, r"$R_t$ (mm)", [cur(c["Rt"], "hot")], cursor=True),
               PlotSpec("Specific impulse", TT, r"$I_{sp}$ (s)", [cur(c["Isp"], "thrust")], cursor=True),
               PlotSpec("Characteristic velocity", TT, r"$c^*$ (m·s⁻¹)", [cur(c["cstar"], "pressure")], cursor=True)]
        runs = compare or [("current", r)]

        def overlay(ch, title, yl):
            cs = [Curve(rr.t, rr.channels[ch], RUN_COLORS[i % len(RUN_COLORS)], name) for i, (name, rr) in
                  enumerate(runs)]
            return PlotSpec(title, TT, yl, cs, cursor=False)
        cmp_ = [overlay("Pc", "Chamber pressure", r"$P_c$ (MPa)"), overlay("F", "Thrust", r"$F$ (kN)"),
                overlay("OF", "Mixture-ratio shift", r"$\frac{O}{F}$ (–)")]
        return [TabSpec("Time histories", hist), TabSpec("Port evolution", self._profiles(r)),
                TabSpec(r"Regression $\dot r(t)$", reg), TabSpec("Nozzle", noz), TabSpec("Compare", cmp_)]

    def export_data(self, p, r: Result) -> dict[str, np.ndarray]:
        names = {"Pc": "Pc_MPa", "F": "F_kN", "OF": "O_over_F", "mdot_ox": "mdot_ox_kgps", "mdot_f": "mdot_f_kgps",
                 "rdot": "rdot_mean_mmps", "rdot_head": "rdot_head_mmps", "rdot_aft": "rdot_aft_mmps",
                 "R": "Rp_mean_mm", "Rt": "Rt_mm", "Isp": "Isp_s", "Tc": "Tc_K", "cstar": "cstar_mps"}
        out = {"t_s": r.t, **{v: r.channels[k] for k, v in names.items()}}
        n = len(r.meta["x_cells"])
        out["x_cells_mm"] = np.array(r.meta["x_cells"]) * 1e3
        out["Rp_x_t_mm"] = np.array([r.channels[f"Rp_{i:02d}"] for i in range(n)])
        return out


class N2OBlowdownModel(HybridBurn1DModel):
    key = "hybrid.n2o"
    title = "N₂O blowdown"
    module = "Hybrid"
    mode = "run"
    Params = N2OBlowdownParams

    TANK_GAP, TANK_L = 0.08, 0.6

    def __init__(self) -> None:
        super().__init__()
        self.sim = N2OBlowdown()

    # ---------------------------------------------------------------- tank mesh
    def _tank_geom(self, p) -> tuple[float, float, float]:
        V = p.V_tank * 1e-3
        Rt = float(np.sqrt(V / (np.pi * self.TANK_L)))
        x1 = -self.TANK_GAP
        return Rt, x1 - self.TANK_L, x1

    def _extra_build(self, pl, si, p, th, m) -> None:
        Rt, x0, x1 = self._tank_geom(p)
        shell = rev_grid(np.array([x0, x1]), np.full(2, Rt), th)
        pl.add_mesh(shell, color="#6b768a", opacity=0.25, name="tank", smooth_shading=True)
        liq = rev_grid(np.array([x0, x0 + p.fill * (x1 - x0)]), np.full(2, 0.96 * Rt), th)
        pl.add_mesh(liq, color="#ff9d3c", opacity=0.9, name="tank_liquid", smooth_shading=True)
        m["tank_liq"] = liq

    def _extra_update(self, si, p, r, frame, m, th) -> None:
        Rt, x0, x1 = self._tank_geom(p)
        if r is None or frame is None or "m_liq" not in r.channels:
            f = p.fill
        else:
            V = p.V_tank * 1e-3
            f = float(np.clip(r.channels["m_liq"][frame] / float(rho_l(r.channels["T_tank"][frame])) / V, 0.0, 1.0))
        m["tank_liq"].points = rev_points(np.array([x0, x0 + max(f, 1e-4) * (x1 - x0)]), np.full(2, 0.96 * Rt), th)

    # ---------------------------------------------------------------- text and plots
    def caption(self, p, r, frame) -> str:
        if r is None or frame is None:
            return r"press Run to simulate · self-pressurized N₂O tank feeding the motor"
        c = r.channels
        return (rf"$t$ = {r.t[frame]:.2f} (s) · $P_{{tank}}$ = {c['p_tank'][frame] / 1e6:.2f} (MPa) · "
                rf"$T_{{tank}}$ = {c['T_tank'][frame]:.1f} (K) · liquid {c['m_liq'][frame]:.2f} (kg)")

    def summary_line(self, p, r) -> str:
        s = r.meta
        return (rf"$I_t$ = {s['I_t_kNs']:.1f} (kN·s),  $\bar{{I}}_{{sp}}$ = {s['Isp_avg']:.1f} (s),  "
                rf"N₂O used {s['m_ox_used']:.2f} of {s['m_ox_tank0']:.2f} (kg)")

    def tabs(self, p, r, compare=None) -> list[TabSpec]:
        base = super().tabs(p, r, compare)
        if r is None:
            return [TabSpec("Tank & feed", self._empty_hist())] + base
        c, t = r.channels, r.t
        cur = lambda y, col, label="", style="solid": Curve(t, y, col, label, style)
        tank = [PlotSpec("Feed pressures", TT, r"$P$ (MPa)",
                         [cur(c["p_tank"] / 1e6, "oxidizer", "tank"), cur(c["Pc"], "pressure", "chamber")],
                         cursor=True),
                PlotSpec("Tank temperature", TT, r"$T$ (K)", [cur(c["T_tank"], "hot")], cursor=True),
                PlotSpec("N₂O inventory", TT, r"$m$ (kg)",
                         [cur(c["m_tank"], "oxidizer", "total"), cur(c["m_liq"], "thrust", "liquid", "dash")],
                         cursor=True)]
        return [TabSpec("Tank & feed", tank)] + base

    def export_data(self, p, r: Result) -> dict[str, np.ndarray]:
        out = super().export_data(p, r)
        out.update({"p_tank_Pa": r.channels["p_tank"], "T_tank_K": r.channels["T_tank"],
                    "m_liquid_kg": r.channels["m_liq"], "m_tank_kg": r.channels["m_tank"]})
        return out

