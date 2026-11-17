r"""Common module, Live model: propellant pair + O/F + P_c + area ratio -> CEA performance and nozzle.

Thermochemistry comes from ``core.thermo`` (RocketCEA tables or direct CEA, equilibrium or frozen); the nozzle
flow uses constant-gamma isentropic relations with the throat gamma from CEA.

.. math:: C_F = \frac{I_{sp} g_0}{c^*}, \qquad I_{sp}(P_a) = I_{vac} - \frac{P_a\,\varepsilon\, c^*}{P_c\, g_0}
"""
from __future__ import annotations

import numpy as np

from ..core.constants import G0
from ..core.nozzle import (T_ratio, contour, divergence_factor, eps_from_pratio, exit_state, mach_along_contour,
                           mach_from_area_array, p_ratio)
from ..core.nozzle.contour import rao_angles
from ..core.result import Curve, PlotSpec, TabSpec
from ..core.schema import C, P, ParamModel
from ..core.thermo.cea import PAIRS, Thermo, cea_available
from .nozzle_live import NozzleLiveModel

_KEYS = list(PAIRS)


class EngineLiveParams(ParamModel):
    pair: str = C("LOX/RP-1", choices=_KEYS, labels=[PAIRS[k].label for k in _KEYS], symbol=r"\mathrm{propellants}",
                  desc="oxidizer / fuel pair (NASA CEA)", group="Propellants")
    flow: str = C("eq", choices=["eq", "frozen"], labels=["Equilibrium flow", "Frozen flow"],
                  symbol=r"\mathrm{nozzle\ flow}", desc="chemistry model through the nozzle", group="Propellants")
    source: str = C("tables", choices=["tables", "cea"], labels=["CEA tables (fast)", "CEA direct (slower)"],
                    symbol=r"\mathrm{thermo\ source}", desc="pre-tabulated CEA grid or direct RocketCEA call",
                    group="Propellants")
    mr: float = P(2.3, symbol=r"\frac{O}{F}", desc="oxidizer-to-fuel mass ratio (clamped to the pair's table range)",
                  units="", dims="1", lo=0.8, hi=12.0, step=0.05, group="Chamber")
    p_c: float = P(3.5, symbol=r"P_c", desc="chamber pressure", units="MPa", dims="M L⁻¹ T⁻²", lo=0.5, hi=20.0,
                   step=0.05, group="Chamber")
    D_t: float = P(32.0, symbol=r"D_t", desc="throat diameter", units="mm", dims="L", lo=10.0, hi=100.0, step=0.5,
                   group="Nozzle")
    eps: float = P(6.0, symbol=r"\varepsilon=\frac{A_e}{A_t}", desc="nozzle expansion ratio", units="", dims="1",
                   lo=1.5, hi=100.0, step=0.1, group="Nozzle")
    kind: str = C("bell", choices=["bell", "cone"], labels=["Bell (parabolic)", "Conical"],
                  symbol=r"\mathrm{contour}", desc="divergent-section shape", group="Nozzle")
    L_frac: float = P(0.80, symbol=r"f_L", desc="bell length as a fraction of the 15° cone length", units="", dims="1",
                      lo=0.6, hi=1.0, step=0.01, group="Nozzle")
    alpha: float = P(15.0, symbol=r"\alpha", desc="conical-nozzle half angle", units="°", dims="1", lo=8.0, hi=25.0,
                     step=0.5, group="Nozzle")
    p_a: float = P(101.325, symbol=r"P_a", desc="ambient pressure (0 = vacuum)", units="kPa", dims="M L⁻¹ T⁻²",
                   lo=0.0, hi=101.325, step=0.5, group="Ambient")


_THERMO: dict[tuple, Thermo] = {}


def get_thermo(pair: str, source: str, frozen: bool) -> tuple[Thermo, str]:
    """Cached ``Thermo``; falls back to the tables when direct CEA is requested but not installed."""
    note = ""
    if source == "cea" and not cea_available():
        source, note = "tables", " (rocketcea not installed: using tables)"
    k = (pair, source, frozen)
    if k not in _THERMO:
        _THERMO[k] = Thermo(pair, source, frozen)
    return _THERMO[k], note


class EngineLiveModel(NozzleLiveModel):
    key = "common.engine"
    title = "Thermochemistry & nozzle"
    module = "Common"
    mode = "live"
    Params = EngineLiveParams

    # ------------------------------------------------------------------ physics (pure, Live)
    @staticmethod
    def _perf(th: Thermo, mr, Pc: float, eps, pa: float, g: float, lam):
        r"""Return ``(isp_vac, isp_amb, c*)`` (s, s, m/s); arrays broadcast over ``mr`` and ``eps``.

        CEA gives :math:`I_{vac}`; the momentum part :math:`I_{vac}-\frac{p_e\varepsilon c^*}{P_c g_0}` is scaled by the
        divergence factor ``lam``; the pressure term uses the isentropic :math:`p_e/P_c` at the point's ``g``.
        Flow separation (Summerfield, :math:`p_e<0.35\,p_a`) truncates the nozzle at :math:`\varepsilon_{sep}` for the
        ambient value only (vacuum flow never separates)."""
        eps = np.asarray(eps, float)

        def one(e, pa_):
            r = th.backend.evaluate(th.pair.key, mr, Pc, e, th.frozen)
            pe = p_ratio(mach_from_area_array(e.ravel(), g, True), g).reshape(e.shape)
            press = pe * e * r["cstar"] / G0
            return lam * (r["ivac"] - press) + press - pa_ * e * r["cstar"] / (Pc * G0), r["cstar"]

        vac, cs = one(eps, 0.0)
        if pa <= 0:
            return vac, vac, cs
        eps_eff = np.minimum(eps, max(float(eps_from_pratio(0.35 * pa / Pc, g)[0]), 1.5))
        amb = one(eps_eff, pa)[0] if np.any(eps_eff != eps) else vac - pa * eps * cs / (Pc * G0)
        return vac, amb, cs

    def evaluate(self, p: EngineLiveParams) -> dict:
        th, note = get_thermo(p.pair, p.source, p.flow == "frozen")
        pr = PAIRS[p.pair]
        Pc, pa = p.p_c * 1e6, p.p_a * 1e3
        mr = float(np.clip(p.mr, pr.mr_lo, pr.mr_hi))
        pt = th.point(mr, Pc, p.eps, pa)
        g, cs = pt["gamma"], pt["cstar"]
        Rt = p.D_t * 1e-3 / 2
        At = np.pi * Rt**2
        Me, pe_pc = exit_state(p.eps, g)

        def lam_of(eps):          # cone: half angle; bell: exit angle (estimate)
            if p.kind == "cone":
                return float(divergence_factor(np.radians(p.alpha)))
            return float(divergence_factor(rao_angles(float(eps), p.L_frac, g)[1]))

        lam = lam_of(p.eps)
        iv, isl, _ = self._perf(th, mr, Pc, p.eps, pa, g, lam)
        isp_vac, isp_sl = float(iv), float(isl)
        cf = isp_sl * G0 / cs
        eps_sep = float(eps_from_pratio(0.35 * pa / Pc, g)[0]) if pa > 0 else np.inf
        # curves vs O/F and vs eps use the same thrust model; coarser in direct-CEA mode to stay near the budget
        fast = th.source.endswith("tables")
        mr_c = np.linspace(pr.mr_lo, pr.mr_hi, 41 if fast else 9)
        isp_v_c, isp_s_c, cs_c = self._perf(th, mr_c, Pc, p.eps, pa, g, lam)
        eps_c = np.geomspace(1.5, max(40.0, 1.3 * p.eps), 40 if fast else 10)
        lam_c = np.array([lam_of(e) for e in eps_c])
        iv_e, ia_e, _ = self._perf(th, mr, Pc, eps_c, pa, g, lam_c)
        cfv_c, cfa_c = iv_e * G0 / cs, ia_e * G0 / cs
        eps_opt = float(eps_from_pratio(pa / Pc, g)[0]) if pa > 0 else np.nan
        x, r = contour(Rt, p.eps, f=p.L_frac, g=g, kind=p.kind, cone_deg=p.alpha)
        M = mach_along_contour(x, r, Rt, g)
        mdot = Pc * At / cs
        return dict(cs=cs, Tc=pt["Tc"], g=g, Mw=pt["Mw"], Me=float(Me), pe_pc=float(pe_pc), cf=cf,
                    cf_vac=isp_vac * G0 / cs, F=cf * Pc * At, mdot=mdot, Isp=isp_sl, Isp_vac=isp_vac, lam=lam,
                    mr=mr, mr_c=mr_c, isp_v_c=isp_v_c, isp_s_c=isp_s_c, cs_c=cs_c, eps_c=eps_c, cfv_c=cfv_c,
                    cfa_c=cfa_c, eps_opt=eps_opt, eps_sep=eps_sep, x=x, r=r, M=M, T=pt["Tc"] * T_ratio(M, g),
                    p=Pc * p_ratio(M, g), Pc=Pc, pa=pa, Rt=Rt, src=th.source + note, clamped=(mr != p.mr),
                    separated=bool(p.eps > eps_sep))

    # ------------------------------------------------------------------ plots
    def tabs(self, p, r, compare=None) -> list[TabSpec]:
        isp = PlotSpec("Specific impulse", r"$\frac{O}{F}$ (–)", r"$I_{sp}$ (s)",
                       [Curve(r["mr_c"], r["isp_v_c"], "thrust", "vacuum"),
                        Curve(r["mr_c"], r["isp_s_c"], "dim", "at P_a", "dash")],
                       markers=[(r["mr"], float(np.interp(r["mr"], r["mr_c"], r["isp_v_c"])), "hot")])
        cst = PlotSpec("Char. velocity", r"$\frac{O}{F}$ (–)", r"$c^*$ (m·s⁻¹)",
                       [Curve(r["mr_c"], r["cs_c"], "oxidizer")], markers=[(r["mr"], r["cs"], "hot")])
        e, cfa, cfv = r["eps_c"], r["cfa_c"], r["cfv_c"]
        if np.isfinite(r["eps_opt"]):
            under, over = e <= r["eps_opt"], e >= r["eps_opt"]
            # include the crossing point in both segments so they join
            ko = float(np.interp(r["eps_opt"], e, cfa))
            segs = [Curve(np.append(e[under], r["eps_opt"]), np.append(cfa[under], ko), "thrust", "under-expanded"),
                    Curve(np.insert(e[over], 0, r["eps_opt"]), np.insert(cfa[over], 0, ko), "hot", "over-expanded")]
            mk = [(r["eps_opt"], ko, "fuel")]
        else:
            segs, mk = [Curve(e, cfa, "thrust", "at P_a")], []
        cf = PlotSpec("Thrust coefficient", r"$\varepsilon=\frac{A_e}{A_t}$ (–)", r"$C_F$ (–)",
                      segs + [Curve(e, cfv, "dim", "vacuum", "dash")], markers=[(p.eps, r["cf"], "thrust")] + mk)
        xmm = r["x"] * 1e3
        pa_line = np.full_like(xmm, r["pa"] / r["Pc"])
        mach = PlotSpec("Mach number", r"$x$ (mm)", r"$M$ (–)", [Curve(xmm, r["M"], "#21918c")],
                        markers=[(xmm[-1], r["Me"], "#21918c")])
        pres = PlotSpec("Static pressure", r"$x$ (mm)", r"$\frac{p}{P_c}$ (–)",
                        [Curve(xmm, r["p"] / r["Pc"], "pressure"), Curve(xmm, pa_line, "dim", "ambient", "dash")],
                        markers=[(xmm[-1], r["pe_pc"], "pressure")])
        temp = PlotSpec("Static temperature", r"$x$ (mm)", r"$T$ (K)", [Curve(xmm, r["T"], "hot")])
        return [TabSpec("Performance", [isp, cst, cf]), TabSpec("Nozzle flow", [mach, pres, temp])]

    def readout(self, p, r, frame) -> list[str]:
        return [rf"$P_c$ = {p.p_c:.2f} (MPa)", rf"$F$ = {r['F'] / 1e3:.2f} (kN)", rf"$C_F$ = {r['cf']:.3f}",
                rf"$I_{{sp}}$ = {r['Isp']:.1f} (s)"]

    def caption(self, p, r, frame) -> str:
        state = ("over-expanded" if r["pe_pc"] * r["Pc"] < r["pa"] * 0.98 else
                 "under-expanded" if r["pe_pc"] * r["Pc"] > r["pa"] * 1.02 else "optimally expanded")
        clamp = r" · $\frac{O}{F}$ clamped to range" if r["clamped"] else ""
        if r["separated"]:        # Summerfield criterion (Sutton, Ch. 3): thrust computed for the truncated nozzle
            clamp += rf" · flow separates at $\varepsilon_{{sep}}$ = {r['eps_sep']:.1f}"
        return (rf"$c^*$ = {r['cs']:.0f} (m·s⁻¹) · $\gamma$ = {r['g']:.3f} · $\bar{{M}}$ = {r['Mw'] * 1e3:.1f} (g·mol⁻¹)"
                rf" · $T_c$ = {r['Tc']:.0f} (K) · $I_{{sp,vac}}$ = {r['Isp_vac']:.1f} (s) · $M_e$ = {r['Me']:.2f} · {state}{clamp} · {r['src']}")

    def export_data(self, p, r) -> dict[str, np.ndarray]:
        return {"OF": r["mr_c"], "Isp_vac_s": r["isp_v_c"], "Isp_amb_s": r["isp_s_c"], "cstar_mps": r["cs_c"],
                "eps": r["eps_c"], "CF_vac": r["cfv_c"], "CF_amb": r["cfa_c"], "x_m": r["x"], "r_m": r["r"],
                "Mach": r["M"], "T_K": r["T"], "p_Pa": r["p"]}
