"""Hybrid module, Live tools (continued): core flow fields, fuel & oxidizer, ILFI instability and scaling."""
from __future__ import annotations

import numpy as np
import pyvista as pv

from ..core.feed import rho_l as n2o_rho_l
from ..core.flowfields import bidirectional_mantle, mantle_asymptotic, streamline, taylor_culick
from ..core.result import Curve, PlotSpec, TabSpec
from ..core.schema import C, P, ParamModel
from ..core.stability import freq_estimate, ilfi_freq, ilfi_tau_bl2, similarity_scale
from ..core.thermo.cea import PAIRS, Thermo
from .hybrid_live import TubeLive

IN, LBM, PSI = 0.0254, 0.45359237, 6894.757293168


# ====================================================================================== core flow fields
class FlowLiveParams(ParamModel):
    mode: str = C("tc", choices=["tc", "headwall"], labels=["Taylor–Culick (no headwall flow)",
                                                          "Taylor–Culick with headwall injection"],
                  symbol=r"\mathrm{core\ flow}", desc="Majdalani core flow for a porous (burning) port",
                  group="Taylor–Culick")
    u_h: float = P(0.5, symbol=r"u_h", desc="headwall injection speed (nondimensional, headwall mode)", units="",
                   dims="1", lo=0.0, hi=2.0, step=0.01, group="Taylor–Culick")
    z: float = P(3.0, symbol=r"z", desc="axial station in port radii", units="", dims="1", lo=0.2, hi=10.0,
                 step=0.1, group="Taylor–Culick")
    kappa: float = P(0.01, symbol=r"\kappa", desc="vortex swirl parameter of the bidirectional-vortex mantle",
                     units="", dims="1", lo=0.001, hi=0.05, step=0.001, group="Bidirectional vortex mantle")
    eps: float = P(1.0, symbol=r"\varepsilon", desc="fuel blowing parameter of the mantle (x 1e-3)", units="10⁻³",
                   dims="1", lo=0.01, hi=10.0, step=0.01, group="Bidirectional vortex mantle")


class FlowLiveModel(TubeLive):
    key = "hybrid.flow"
    title = "Core flow fields"
    Params = FlowLiveParams
    fields = [r"$u_z$ on the axis (–)"]
    cmap = "viridis"
    port_opacity = 0.35                      # see the streamlines through the port wall
    R_in, L_tube = 0.025, 0.25

    def evaluate(self, p: FlowLiveParams) -> dict:
        uh = p.u_h if p.mode == "headwall" else 0.0
        r = np.linspace(0.0, 1.0, 101)
        ur, uz = taylor_culick(r, p.z, uh)
        seeds = np.linspace(0.15, 0.95, 8)
        lines = [streamline(r0, 0.3, 10.0, uh) for r0 in seeds]
        eps_g = np.geomspace(1e-4, 1e-2, 40)
        kap_g = np.linspace(0.001, 0.05, 40)
        eps = p.eps * 1e-3
        beta = bidirectional_mantle(p.kappa, eps)
        zz = np.linspace(0.0, 10.0, 40)
        return dict(r=r, ur=ur, uz=uz, uh=uh, lines=lines, seeds=seeds, eps_g=eps_g,
                    beta_e=np.array([bidirectional_mantle(p.kappa, e) for e in eps_g]),
                    beta_e_asym=np.array([mantle_asymptotic(p.kappa, e) for e in eps_g]),
                    beta_k=np.array([bidirectional_mantle(k, eps) for k in kap_g]), kap_g=kap_g, beta=beta,
                    beta_asym=mantle_asymptotic(p.kappa, eps), zz=zz, u_axis=np.pi * (zz + uh))

    def _profile(self, p, r):
        x = np.linspace(0.0, self.L_tube, 40)
        z = x / self.R_in
        return x, np.full_like(x, self.R_in), np.pi * (z + r["uh"]), r"$u_z$ on the axis (–)"

    def _extra(self, vp, p, r) -> None:
        """Streamlines (meridional planes) coloured by speed |u|."""
        pl = vp.plotter
        th = [0.0, np.pi / 2, np.pi, 3 * np.pi / 2]
        pts, conn, speed = [], [], []
        for (z, rr) in r["lines"]:
            ur, uz = taylor_culick(rr, z, r["uh"])
            for a in (th if vp.cutaway == 0 else th[:3]):
                base = len(pts)
                x = z * self.R_in
                R = rr * self.R_in
                pts.extend(np.c_[x, R * np.cos(a), R * np.sin(a)].tolist())
                conn.append([len(z)] + list(range(base, base + len(z))))
                speed.extend(np.hypot(ur, uz).tolist())
        if not pts:
            return
        poly = pv.PolyData(np.array(pts))
        poly.lines = np.hstack(conn)
        poly["speed"] = np.array(speed)
        pl.add_mesh(poly, scalars="speed", cmap="turbo", line_width=2, show_scalar_bar=False, name="stream")

    def tabs(self, p, r, compare=None) -> list[TabSpec]:
        rl = r"$r$ (–)"
        flow = [PlotSpec("Radial velocity", rl, r"$u_r$ (–)", [Curve(r["r"], r["ur"], "oxidizer")],
                         markers=[(1.0, -1.0, "hot")]),
                PlotSpec("Axial velocity", rl, r"$u_z$ (–)", [Curve(r["r"], r["uz"], "thrust")],
                         markers=[(1.0, 0.0, "hot")]),
                PlotSpec("Streamlines", r"$z$ (–)", r"$r$ (–)",
                         [Curve(z, rr, "fuel") for (z, rr) in r["lines"]])]
        mant = [PlotSpec("Mantle radius vs blowing", r"$\varepsilon$ (–)", r"$\beta^*$ (–)",
                         [Curve(r["eps_g"], r["beta_e"], "thrust", "Eq. 78"),
                          Curve(r["eps_g"], r["beta_e_asym"], "dim", "Eqs. 79–80", "dash")],
                         markers=[(p.eps * 1e-3, r["beta"], "hot")]),
                PlotSpec("Mantle radius vs swirl", r"$\kappa$ (–)", r"$\beta^*$ (–)",
                         [Curve(r["kap_g"], r["beta_k"], "pressure")], markers=[(p.kappa, r["beta"], "hot")]),
                PlotSpec("Axial velocity on the axis", r"$z$ (–)", r"$u_z$ (–)", [Curve(r["zz"], r["u_axis"], "fuel")],
                         markers=[(p.z, np.pi * (p.z + r["uh"]), "hot")])]
        return [TabSpec("Core flow", flow), TabSpec("Vortex mantle", mant)]

    def readout(self, p, r, frame) -> list[str]:
        return [rf"$z$ = {p.z:.1f} (–)", rf"$u_z(0)$ = {np.pi * (p.z + r['uh']):.2f} (–)",
                rf"$\beta^*$ = {r['beta']:.4f}", rf"$\beta^*_{{asym}}$ = {r['beta_asym']:.4f}"]

    def caption(self, p, r, frame) -> str:
        return (r"$u_r=-\frac{\sin(\frac{\pi}{2}r^2)}{r}$ , $u_z=\pi(z+u_h)\cos(\frac{\pi}{2}r^2)$ · "
                r"wall: $u_r$ = −1, $u_z$ = 0")

    def export_data(self, p, r) -> dict[str, np.ndarray]:
        return {"r": r["r"], "u_r": r["ur"], "u_z": r["uz"], "eps": r["eps_g"], "beta_star": r["beta_e"]}


# ====================================================================================== fuel & oxidizer
ADDITIVES = ["none", "Al", "Mg", "B"]
# Approximate reference constants: (heat of combustion to the stable oxide, MJ/kg; density, kg/m3)
ADD_PROPS = {"Al": (31.0, 2700.0), "Mg": (24.7, 1740.0), "B": (58.0, 2340.0)}
HTPB = (42.0, 920.0)
# Handbook bulk densities (kg/m3): storable / cryogenic liquids at normal storage state; GOX as ideal gas at 5 MPa
RHO_OX = {"LOX": 1141.0, "N2O4": 1443.0, "GOX": 65.7}
RHO_F = {"RP-1": 810.0, "H2": 70.8, "CH4": 423.0, "Ethanol": 789.0, "MMH": 874.0, "HTPB": 920.0}


def _rho_ox(name: str) -> float:
    return float(n2o_rho_l(293.15)) if name == "N2O" else RHO_OX[name]


class FuelLiveParams(ParamModel):
    additive: str = C("Al", choices=ADDITIVES, labels=["None", "Aluminium", "Magnesium", "Boron"],
                      symbol=r"\mathrm{additive}", desc="energetic additive mixed into HTPB", group="Additive energy")
    loading: float = P(15.0, symbol=r"w_a", desc="additive mass fraction", units="%", dims="1", lo=0.0, hi=60.0,
                       step=0.5, group="Additive energy")
    pair: str = C("N2O/HTPB", choices=list(PAIRS), labels=[PAIRS[k].label for k in PAIRS],
                  symbol=r"\mathrm{propellants}", desc="oxidizer / fuel pair for the density impulse",
                  group="Density impulse")
    p_c: float = P(3.5, symbol=r"P_c", desc="chamber pressure", units="MPa", dims="M L⁻¹ T⁻²", lo=0.5, hi=20.0,
                   step=0.05, group="Density impulse")
    eps: float = P(6.0, symbol=r"\varepsilon", desc="nozzle expansion ratio", units="", dims="1", lo=2.0, hi=60.0,
                   step=0.1, group="Density impulse")
    mr: float = P(5.0, symbol=r"\frac{O}{F}", desc="oxidizer-to-fuel ratio (clamped to the table range)", units="",
                  dims="1", lo=0.8, hi=12.0, step=0.05, group="Density impulse")


def _mix(w_add: float, add: str) -> tuple[float, float]:
    """Mass-weighted heat of combustion (MJ/kg) and volume-weighted density (kg/m3) of HTPB + additive."""
    if add == "none":
        return HTPB
    h_a, rho_a = ADD_PROPS[add]
    h = (1 - w_add) * HTPB[0] + w_add * h_a
    rho = 1.0 / ((1 - w_add) / HTPB[1] + w_add / rho_a)
    return h, rho


class FuelLiveModel(TubeLive):
    key = "hybrid.fuel"
    title = "Fuel & oxidizer"
    Params = FuelLiveParams
    fields = [r"energy density (GJ·m⁻³)"]
    cmap = "inferno"

    def evaluate(self, p: FuelLiveParams) -> dict:
        w = p.loading / 100.0
        h_now, rho_now = _mix(w, p.additive)
        wg = np.linspace(0.0, 0.6, 40)
        curves = {a: np.array([_mix(x, a) for x in wg]) for a in ("Al", "Mg", "B")}
        pr = PAIRS[p.pair]
        th = Thermo(p.pair, "tables")
        ox, fu = pr.ox, pr.fuel
        rho_o, rho_f = _rho_ox(ox), (rho_now if (fu == "HTPB" and p.additive != "none") else RHO_F[fu])
        mr_g = np.linspace(pr.mr_lo, pr.mr_hi, 60)
        mr = float(np.clip(p.mr, pr.mr_lo, pr.mr_hi))
        isp = th.isp_vac(mr_g, p.p_c * 1e6, p.eps)
        rho_b = (1 + mr_g) / (mr_g / rho_o + 1.0 / rho_f)
        return dict(h=h_now, rho=rho_now, e=h_now * rho_now / 1e3, wg=wg, curves=curves, mr_g=mr_g, mr=mr, isp=isp,
                    rho_b=rho_b, di=rho_b * isp, rho_ox=rho_o, rho_f=rho_f,
                    di_now=float(np.interp(mr, mr_g, rho_b * isp)), isp_now=float(np.interp(mr, mr_g, isp)),
                    rho_b_now=float(np.interp(mr, mr_g, rho_b)), clamped=(mr != p.mr), i_opt=int(np.argmax(rho_b * isp)))

    def _profile(self, p, r):
        x = np.linspace(0.0, self.L_tube, 40)
        return x, np.full_like(x, self.R_in), np.full_like(x, r["e"]), r"energy density (GJ·m⁻³)"

    def tabs(self, p, r, compare=None) -> list[TabSpec]:
        W = r"$w_a$ (%)"
        wg = 100 * r["wg"]
        cols = {"Al": "oxidizer", "Mg": "fuel", "B": "pressure"}
        add = [PlotSpec("Heat of combustion", W, r"$\Delta h_c$ (MJ·kg⁻¹)",
                        [Curve(wg, r["curves"][a][:, 0], cols[a], a) for a in ADD_PROPS],
                        markers=[(p.loading, r["h"], "hot")]),
               PlotSpec("Fuel density", W, r"$\rho_f$ (kg·m⁻³)",
                        [Curve(wg, r["curves"][a][:, 1], cols[a], a) for a in ADD_PROPS],
                        markers=[(p.loading, r["rho"], "hot")]),
               PlotSpec("Volumetric energy", W, r"$\rho_f \Delta h_c$ (GJ·m⁻³)",
                        [Curve(wg, r["curves"][a][:, 0] * r["curves"][a][:, 1] / 1e3, cols[a], a) for a in ADD_PROPS],
                        markers=[(p.loading, r["e"], "hot")])]
        OF = r"$\frac{O}{F}$ (–)"
        di = [PlotSpec("Bulk density", OF, r"$\rho_b$ (kg·m⁻³)", [Curve(r["mr_g"], r["rho_b"], "oxidizer")],
                       markers=[(r["mr"], r["rho_b_now"], "hot")]),
              PlotSpec("Vacuum specific impulse", OF, r"$I_{sp,vac}$ (s)", [Curve(r["mr_g"], r["isp"], "thrust")],
                       markers=[(r["mr"], r["isp_now"], "hot")]),
              PlotSpec("Density impulse", OF, r"$\rho_b I_{sp}$ (kg·s·m⁻³)", [Curve(r["mr_g"], r["di"], "hot")],
                       markers=[(r["mr"], r["di_now"], "thrust")])]
        return [TabSpec("Additive energy", add), TabSpec("Density impulse", di)]

    def readout(self, p, r, frame) -> list[str]:
        return [rf"$\Delta h_c$ = {r['h']:.1f} (MJ·kg⁻¹)", rf"$\rho_f$ = {r['rho']:.0f} (kg·m⁻³)",
                rf"$\rho_b$ = {r['rho_b_now']:.0f} (kg·m⁻³)", rf"$\rho_b I_{{sp}}$ = {r['di_now'] / 1e3:.0f} (10³ kg·s·m⁻³)"]

    def caption(self, p, r, frame) -> str:
        clamp = r" · $\frac{O}{F}$ clamped to range" if r["clamped"] else ""
        return (rf"{PAIRS[p.pair].label}: $\rho_{{ox}}$ = {r['rho_ox']:.0f}, $\rho_f$ = {r['rho_f']:.0f} (kg·m⁻³) · "
                rf"densest-impulse $\frac{{O}}{{F}}$ = {r['mr_g'][r['i_opt']]:.2f}{clamp} · approximate handbook constants")

    def export_data(self, p, r) -> dict[str, np.ndarray]:
        return {"loading": r["wg"], "OF": r["mr_g"], "Isp_vac_s": r["isp"], "rho_bulk": r["rho_b"],
                "density_impulse": r["di"]}


# ====================================================================================== instability & scaling
class StabilityLiveParams(ParamModel):
    preset: str = C("custom", choices=["custom", "ames"], labels=["Custom values", "NASA Ames 4NF-04 (Table 8)"],
                    symbol=r"\mathrm{case}", desc="reference test case (overrides the sliders when selected)",
                    group="Motor")
    L: float = P(1.0, symbol=r"L", desc="grain length", units="m", dims="L", lo=0.2, hi=3.0, step=0.01, group="Motor")
    p_c: float = P(3.5, symbol=r"P_c", desc="chamber pressure", units="MPa", dims="M L⁻¹ T⁻²", lo=0.5, hi=10.0,
                   step=0.05, group="Motor")
    Go: float = P(120.0, symbol=r"G_o", desc="oxidizer mass flux", units="kg·m⁻²·s⁻¹", dims="M L⁻² T⁻¹", lo=20.0,
                  hi=600.0, step=1.0, group="Motor")
    of: float = P(3.0, symbol=r"\frac{O}{F}", desc="oxidizer-to-fuel ratio", units="", dims="1", lo=0.8, hi=12.0,
                  step=0.05, group="Motor")
    RT: float = P(6.4, symbol=r"RT_{av}", desc="average gas R·T in the port (x 1e5)", units="10⁵ m²·s⁻²",
                  dims="L² T⁻²", lo=2.0, hi=12.0, step=0.05, group="Motor")
    s: float = P(1.0, symbol=r"s", desc="geometric scale factor for the similarity calculator", units="", dims="1",
                 lo=0.25, hi=4.0, step=0.05, group="Scaling")


class StabilityLiveModel(TubeLive):
    key = "hybrid.stability"
    title = "Instability & scaling"
    Params = StabilityLiveParams
    fields = [r"$f_{ILFI}$ (Hz)"]
    cmap = "magma"

    @staticmethod
    def _vals(p) -> tuple[float, float, float, float, float]:
        if p.preset == "ames":
            return 45 * IN, 568.5 * PSI, 0.48 * LBM / IN**2, 3.01, 6.38e5
        return p.L, p.p_c * 1e6, p.Go, p.of, p.RT * 1e5

    def evaluate(self, p: StabilityLiveParams) -> dict:
        L, Pc, Go, OF, RT = self._vals(p)
        tau = float(ilfi_tau_bl2(L, Pc, Go, OF, RT))
        f_il, f_est = float(ilfi_freq(tau)), float(freq_estimate(L, Pc, Go, OF, RT))
        Lg = np.linspace(0.2, 3.0, 50)
        Pg = np.linspace(0.5e6, 10e6, 50)
        Gg = np.linspace(20.0, 600.0, 50)
        sg = np.linspace(0.25, 4.0, 40)
        base = dict(mdot_ox=1.0, thrust=1.0, t_burn=1.0, freq=f_il, L=L)
        return dict(L=L, Pc=Pc, Go=Go, OF=OF, RT=RT, tau=tau, f=f_il, f_est=f_est, Lg=Lg, Pg=Pg, Gg=Gg, sg=sg,
                    tau_L=ilfi_tau_bl2(Lg, Pc, Go, OF, RT) * 1e3, f_P=ilfi_freq(ilfi_tau_bl2(L, Pg, Go, OF, RT)),
                    f_G=ilfi_freq(ilfi_tau_bl2(L, Pc, Gg, OF, RT)), f_s=similarity_scale(base, 1.0)["freq"] / sg,
                    scaled=similarity_scale(base, p.s))

    def _profile(self, p, r):
        x = np.linspace(0.0, self.L_tube, 40)
        return x, np.full_like(x, self.R_in), np.full_like(x, r["f"]), r"$f_{ILFI}$ (Hz)"

    def tabs(self, p, r, compare=None) -> list[TabSpec]:
        sweeps = [PlotSpec("Boundary-layer delay", r"$L$ (m)", r"$\tau_{bl2}$ (ms)", [Curve(r["Lg"], r["tau_L"], "oxidizer")],
                           markers=[(r["L"], r["tau"] * 1e3, "hot")]),
                  PlotSpec("Frequency vs chamber pressure", r"$P_c$ (MPa)", r"$f$ (Hz)",
                           [Curve(r["Pg"] / 1e6, r["f_P"], "pressure")], markers=[(r["Pc"] / 1e6, r["f"], "hot")]),
                  PlotSpec("Frequency vs oxidizer flux", r"$G_o$ (kg·m⁻²·s⁻¹)", r"$f$ (Hz)",
                           [Curve(r["Gg"], r["f_G"], "thrust")], markers=[(r["Go"], r["f"], "hot")])]
        scale = [PlotSpec("Frequency under scaling", r"$s$ (–)", r"$f$ (Hz)", [Curve(r["sg"], r["f_s"], "fuel")],
                          markers=[(p.s, r["scaled"]["freq"], "hot")]),
                 PlotSpec("Flow and thrust under scaling", r"$s$ (–)", r"$\frac{\dot m}{\dot m_0}$, $\frac{F}{F_0}$ (–)",
                          [Curve(r["sg"], r["sg"] ** 2, "oxidizer", "ṁ, F")], markers=[(p.s, p.s**2, "hot")]),
                 PlotSpec("Burn time under scaling", r"$s$ (–)", r"$\frac{t_b}{t_{b,0}}$ (–)",
                          [Curve(r["sg"], r["sg"], "thrust")], markers=[(p.s, p.s, "hot")])]
        return [TabSpec("ILFI", sweeps), TabSpec("Scaling", scale)]

    def readout(self, p, r, frame) -> list[str]:
        return [rf"$\tau_{{bl2}}$ = {r['tau'] * 1e3:.2f} (ms)", rf"$f_{{ILFI}}$ = {r['f']:.2f} (Hz)",
                rf"$f_{{est}}$ = {r['f_est']:.2f} (Hz)", rf"$f$ at $s$ = {p.s:.2f}: {r['scaled']['freq']:.2f} (Hz)"]

    def caption(self, p, r, frame) -> str:
        tag = " · NASA Ames 4NF-04: book 18.2 (ms), 26.2 (Hz), measured 32.75 (Hz)" if p.preset == "ames" else ""
        return (rf"$\tau_{{bl2}}=\frac{{2.05\,L\,P_c}}{{(2+\frac{{\dot m_f}}{{\dot m_{{ox}}}})G_o\,RT_{{av}}}}$ , "
                rf"$f=\frac{{0.48}}{{\tau_{{bl2}}}}${tag}")

    def export_data(self, p, r) -> dict[str, np.ndarray]:
        return {"L_m": r["Lg"], "tau_ms": r["tau_L"], "Pc_MPa": r["Pg"] / 1e6, "f_vs_Pc_Hz": r["f_P"]}

