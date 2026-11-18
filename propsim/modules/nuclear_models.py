"""Nuclear module: NTR performance, reactor criticality, fuel-element thermal hydraulics, engine cycles and
fission-product poisons (Live); reactor kinetics and startup / shutdown (Run).

Physics in :mod:`propsim.core.nuclear` (Emrich, *Principles of Nuclear Rocket Propulsion*, 2nd ed.)."""
from __future__ import annotations

import numpy as np
import pyvista as pv

from ..core.nozzle import T_ratio, contour, mach_along_contour
from ..core.nuclear import criticality as cr
from ..core.nuclear import cycles as cy
from ..core.nuclear import kinetics as kn
from ..core.nuclear import performance as pf
from ..core.nuclear import poisons as po
from ..core.nuclear import thermal as th
from ..core.result import Curve, PlotSpec, TabSpec
from ..core.schema import C, P, ParamModel
from ..sims.nuclear_run import KineticsParams, ReactorKinetics
from ..ui.theme import CMAPS
from .base import ModelController
from .geometry import broadcast_scalar, cut_face, rev_grid, thetas

PROP_KEYS = list(pf.PROPELLANTS)
PROP_LABELS = [v[2] for v in pf.PROPELLANTS.values()]
REFLECTOR = "#6b7a8f"
REACTOR_LEGEND = {"reflector": "Beryllium reflector", "reflector_cut": "Reflector (cut face)",
                  "drum": "Control drum (beryllium)", "absorber": "Drum absorber plate", "plenum": "Plenum and vessel"}
DRUM = "#9aa6b8"
ABSORBER = "#2b2f38"


def _warm_coolprop() -> None:
    """The first CoolProp hydrogen call loads the fluid (about 2 s); do it when the model is created, not in the
    first Live evaluation."""
    from CoolProp.CoolProp import PropsSI
    PropsSI("H", "T", 300.0, "P", 1e6, "Hydrogen")


def reactor_meshes(vp, R: float, H: float, t_refl: float, cut: float, x0: float = 0.0, drum_turn: float = 0.0,
                   n_drums: int = 12, core_scalars=None, cmap: str = "inferno", clim=None, name: str = "core") -> None:
    """Core cylinder (axis along x from ``x0`` to ``x0 + H``), beryllium reflector annulus and control drums with
    their absorber plates turned by ``drum_turn`` (rad, 0 = absorber facing the core)."""
    pl = vp.plotter
    th_ = thetas(cut, 72)
    x = np.linspace(x0, x0 + H, 24)
    core = rev_grid(x, np.full_like(x, R), th_)
    if core_scalars is not None:
        core["field"] = broadcast_scalar(core_scalars(x), len(th_))
        pl.add_mesh(core, scalars="field", cmap=cmap, clim=clim, show_scalar_bar=False, name=name, smooth_shading=True)
    else:
        pl.add_mesh(core, color="#ff7a3c", name=name, smooth_shading=True)
    Ro = R + t_refl
    pl.add_mesh(rev_grid(x, np.full_like(x, Ro), th_), color=REFLECTOR, opacity=0.35, name="reflector",
                smooth_shading=True)
    if cut > 0:
        for k, ang in enumerate((0.0, np.radians(360.0 - cut))):
            pl.add_mesh(cut_face(x, np.full_like(x, R), np.full_like(x, Ro), ang), color=REFLECTOR, opacity=0.8,
                        name=f"reflector_cut{k}")
    rd = 0.35 * t_refl
    rc = R + 0.5 * t_refl
    keep = np.radians(360.0 - cut) if cut > 0 else 2 * np.pi
    for k in range(n_drums):
        a = 2 * np.pi * (k + 0.5) / n_drums
        if a > keep:
            continue
        c = np.array([x0 + 0.5 * H, rc * np.cos(a), rc * np.sin(a)])
        pl.add_mesh(pv.Cylinder(center=c, direction=(1, 0, 0), radius=rd, height=0.96 * H, resolution=24),
                    color=DRUM, name=f"drum{k}", smooth_shading=True)
        face = a + np.pi + drum_turn                          # absorber plate direction (toward the core at 0)
        pc = c + 0.7 * rd * np.array([0.0, np.cos(face), np.sin(face)])
        pl.add_mesh(pv.Cylinder(center=pc, direction=(1, 0, 0), radius=0.35 * rd, height=0.97 * H, resolution=12),
                    color=ABSORBER, name=f"absorber{k}")


# ====================================================================================== NTR performance (Live)
class PerfParams(ParamModel):
    prop: str = C("H2", choices=PROP_KEYS, labels=PROP_LABELS, symbol=r"\mathrm{propellant}",
                  desc="working fluid heated by the reactor (frozen flow)", group="Propellant")
    T0: float = P(2750.0, symbol=r"T_0", desc="chamber (reactor outlet) temperature", units="K", dims="Θ", lo=1000.0,
                  hi=3500.0, step=10.0, group="Chamber")
    pc: float = P(7.0, symbol=r"P_c", desc="chamber pressure", units="MPa", dims="M L⁻¹ T⁻²", lo=0.5, hi=20.0,
                  step=0.1, group="Chamber")
    T_in: float = P(300.0, symbol=r"T_{in}", desc="core inlet temperature after regenerative cooling", units="K",
                    dims="Θ", lo=25.0, hi=1000.0, step=5.0, group="Chamber")
    eps: float = P(200.0, symbol=r"\varepsilon=\frac{A_e}{A_t}", desc="nozzle expansion ratio", units="", dims="1",
                   lo=10.0, hi=500.0, step=1.0, group="Nozzle")
    eta_n: float = P(0.98, symbol=r"\eta_{C_F}", desc="thrust coefficient efficiency (divergence and friction)",
                     units="", dims="1", lo=0.85, hi=1.0, step=0.005, group="Nozzle")
    pa: float = P(0.0, symbol=r"P_a", desc="ambient pressure", units="kPa", dims="M L⁻¹ T⁻²", lo=0.0, hi=101.325,
                  step=0.5, group="Nozzle")
    F: float = P(111.0, symbol=r"F", desc="vacuum thrust", units="kN", dims="M L T⁻²", lo=5.0, hi=1500.0, step=1.0,
                 group="Reactor")
    q_core: float = P(1.3, symbol=r"\bar q'''", desc="average core power density", units="MW·L⁻¹", dims="M L⁻¹ T⁻³",
                      lo=0.2, hi=10.0, step=0.05, group="Reactor")
    LD: float = P(1.0, symbol=r"\frac{L}{D}", desc="core length over diameter", units="", dims="1", lo=0.3, hi=3.0,
                  step=0.05, group="Reactor")
    t_refl: float = P(15.0, symbol=r"t_r", desc="beryllium reflector thickness (drawing only)", units="cm", dims="L",
                      lo=5.0, hi=40.0, step=0.5, group="Reactor")
    mr_chem: float = P(6.0, symbol=r"\frac{O}{F}", desc="mixture ratio of the LOX / LH₂ engine compared with",
                       units="", dims="1", lo=3.0, hi=8.0, step=0.1, group="Chemical Comparison")
    dv: float = P(4.0, symbol=r"\Delta v", desc="stage velocity increment for the payload comparison", units="km·s⁻¹",
                  dims="L T⁻¹", lo=0.5, hi=12.0, step=0.1, group="Mission")
    f_ntr: float = P(0.30, symbol=r"f_{inert,N}", desc="inert mass fraction of the nuclear stage (tanks, engine, shield)",
                     units="", dims="1", lo=0.05, hi=0.6, step=0.01, group="Mission")
    f_chem: float = P(0.12, symbol=r"f_{inert,C}", desc="inert mass fraction of the chemical stage", units="", dims="1",
                      lo=0.03, hi=0.5, step=0.01, group="Mission")


class NTRPerformanceModel(ModelController):
    key = "nuc.perf"
    title = "NTR performance"
    module = "Nuclear"
    mode = "live"
    Params = PerfParams
    fields = [r"$T$ (K)"]
    legend_labels = {**REACTOR_LEGEND, "core": "Reactor core", "nozzle": "Nozzle (temperature)"}

    def __init__(self) -> None:
        self._m: dict = {}
        self._built = None
        _warm_coolprop()

    def evaluate(self, p: PerfParams) -> dict:
        from ..core.thermo.cea import Thermo
        mw, g, _ = pf.PROPELLANTS[p.prop]
        o = pf.performance(p.T0, p.pc * 1e6, p.eps, mw, g, p.F * 1e3, p.pa * 1e3, p.T_in, p.eta_n, p.q_core * 1e9,
                           p.LD, fluid="Hydrogen" if p.prop == "H2" else None)
        T_g = np.linspace(1000.0, 3500.0, 60)
        isp_T = {k: pf.isp_vs_eps(T_g[:, None], v[0], v[1], np.array([p.eps]))[:, 0] * p.eta_n
                 for k, v in pf.PROPELLANTS.items()}
        eps_g = np.geomspace(10.0, 500.0, 60)
        isp_e = pf.isp_vs_eps(p.T0, mw, g, eps_g) * p.eta_n
        chem = Thermo("LOX/LH2")
        eps_c = np.minimum(eps_g, 100.0)
        isp_c_g = chem.isp_vac(p.mr_chem, p.pc * 1e6, eps_c)
        isp_c = float(chem.isp_vac(p.mr_chem, p.pc * 1e6, min(p.eps, 100.0)))
        dv_g = np.linspace(0.5, 12.0, 80) * 1e3
        pl_n = pf.payload_fraction(dv_g, o["isp_v"], p.f_ntr)
        pl_c = pf.payload_fraction(dv_g, isp_c, p.f_chem)
        q_g = np.linspace(0.2, 10.0, 60)
        D_g = (4.0 * o["P_R"] / (q_g * 1e9) / (np.pi * p.LD)) ** (1 / 3)
        x, rr = contour(o["Rt"], p.eps, Rc=2.0 * o["Rt"], f=0.8, g=g)
        M = mach_along_contour(x, rr, o["Rt"], g)
        return dict(**o, T_g=T_g, isp_T=isp_T, eps_g=eps_g, isp_e=isp_e, isp_c_g=isp_c_g, isp_c=isp_c, dv_g=dv_g,
                    pl_n=pl_n, pl_c=pl_c, q_g=q_g, D_g=D_g, x=x, r=rr, Tn=p.T0 * T_ratio(M, g),
                    pl_now=(float(pf.payload_fraction(p.dv * 1e3, o["isp_v"], p.f_ntr)),
                            float(pf.payload_fraction(p.dv * 1e3, isp_c, p.f_chem))))

    def tabs(self, p, r, compare=None) -> list[TabSpec]:
        cols = {"H2": "thrust", "NH3": "fuel", "He": "pressure"}
        tc = PlotSpec("Specific impulse vs chamber temperature", r"$T_0$ (K)", r"$I_{sp}$ (s)",
                      [Curve(r["T_g"], r["isp_T"][k], cols[k], PROP_LABELS[i].split(" (")[0].lower())
                       for i, k in enumerate(PROP_KEYS)]
                      + [Curve(r["T_g"], np.full_like(r["T_g"], r["isp_c"]), "oxidizer", "LOX / LH₂ (CEA)", "dash")],
                      markers=[(p.T0, r["isp_v"], "text")])
        ie = PlotSpec("Specific impulse vs expansion ratio", r"$\varepsilon$ (–)", r"$I_{sp}$ (s)",
                      [Curve(r["eps_g"], r["isp_e"], "thrust", "nuclear thermal"),
                       Curve(r["eps_g"], r["isp_c_g"], "oxidizer", "LOX / LH₂ (CEA)", "dash")],
                      markers=[(p.eps, r["isp_v"], "thrust")])
        core = PlotSpec("Core diameter vs power density", r"$\bar q'''$ (MW·L⁻¹)", r"$D_{core}$ (m)",
                        [Curve(r["q_g"], r["D_g"], "hot")], markers=[(p.q_core, r["D_core"], "hot")])
        pay = PlotSpec("Payload fraction vs velocity increment", r"$\Delta v$ (km·s⁻¹)", r"$\frac{m_{pl}}{m_0}$ (–)",
                       [Curve(r["dv_g"] / 1e3, np.clip(r["pl_n"], -0.2, 1), "thrust", "nuclear thermal"),
                        Curve(r["dv_g"] / 1e3, np.clip(r["pl_c"], -0.2, 1), "oxidizer", "LOX / LH₂", "dash")],
                       markers=[(p.dv, r["pl_now"][0], "thrust"), (p.dv, r["pl_now"][1], "oxidizer")])
        noz = PlotSpec("Nozzle gas temperature", r"$x$ (m)", r"$T$ (K)", [Curve(r["x"], r["Tn"], "hot")])
        return [TabSpec("Specific impulse", [tc, ie]), TabSpec("Reactor & mission", [core, pay, noz])]

    def readout(self, p, r, frame) -> list[str]:
        return [rf"$I_{{sp}}$ = {r['isp_v']:.0f} (s)", rf"$\dot m$ = {r['mdot']:.2f} (kg·s⁻¹)",
                rf"$P_R$ = {r['P_R'] / 1e6:.0f} (MW)", rf"$D_{{core}}$ = {r['D_core']:.2f} (m)"]

    def caption(self, p, r, frame) -> str:
        amb = rf" · $I_{{sp}}$ at $P_a$ = {r['isp_a']:.0f} (s)" if p.pa > 0 else ""
        return (rf"$c^*$ = {r['cstar']:.0f} (m·s⁻¹) · $C_F$ = {r['cf_v']:.3f} · "
                rf"$\frac{{P_R}}{{F}}$ = {r['P_per_F'] / 1e3:.2f} (MW·kN⁻¹) · "
                rf"{r['isp_v'] / r['isp_c']:.2f} times the LOX / LH₂ $I_{{sp}}${amb}")

    def export_data(self, p, r):
        return {"T0_K": r["T_g"], **{f"Isp_{k}_s": v for k, v in r["isp_T"].items()}, "eps": r["eps_g"],
                "Isp_eps_s": r["isp_e"], "Isp_LOX_LH2_s": r["isp_c_g"], "dv_mps": r["dv_g"],
                "payload_ntr": r["pl_n"], "payload_chem": r["pl_c"]}

    def build_scene(self, vp, p, r=None) -> None:
        r = r or self.evaluate(p)
        pl = vp.plotter
        pl.clear()
        R = 0.5 * r["D_core"]
        H = r["L_core"]
        tr = p.t_refl / 100.0
        # reactor upstream of the nozzle; nozzle contour starts at the chamber radius right after the core
        x_noz0 = float(r["x"][0])
        gap = 0.3 * R
        x0 = x_noz0 - gap - H
        clim = self._clim(p, r)
        reactor_meshes(vp, R, H, tr, vp.cutaway, x0=x0, core_scalars=lambda x: np.full_like(x, p.T0), clim=clim)
        th_ = thetas(vp.cutaway)
        xp = np.linspace(x0 + H, x_noz0, 8)
        rp = np.interp(xp, [x0 + H, x_noz0], [R + tr, float(r["r"][0])])
        pl.add_mesh(rev_grid(xp, rp, th_), color=REFLECTOR, opacity=0.6, name="plenum")
        noz = rev_grid(r["x"], r["r"], th_)
        noz["field"] = broadcast_scalar(r["Tn"], len(th_))
        pl.add_mesh(noz, scalars="field", cmap=CMAPS["temperature"], clim=clim, show_scalar_bar=False, name="nozzle",
                    smooth_shading=True)
        self._built = (vp.cutaway, round(R, 6), round(H, 6), round(float(r["Rt"]), 7), p.eps, p.t_refl)
        self._m = {"built": True}
        vp.reset_view(zoom=1.6)

    def update_scene(self, vp, p, r, frame) -> None:
        R = 0.5 * r["D_core"]
        key = (vp.cutaway, round(R, 6), round(r["L_core"], 6), round(float(r["Rt"]), 7), p.eps, p.t_refl)
        if not self._m or self._built != key:
            self.build_scene(vp, p, r)
        vp.set_colorbar(CMAPS["temperature"], *self._clim(p, r), r"$T$ (K)")
        vp.render()

    @staticmethod
    def _clim(p, r) -> tuple[float, float]:
        """Colour range with the cold nozzle exit a quarter up the map, so it stays visible on the dark viewport."""
        lo = float(r["Tn"].min())
        return lo - 0.33 * (p.T0 - lo), p.T0 + 1e-6


# ====================================================================================== reactor criticality (Live)
class CritParams(ParamModel):
    nuSf: float = P(0.13, symbol=r"\nu\Sigma_f", desc="one-group production cross section", units="cm⁻¹", dims="L⁻¹",
                    lo=0.01, hi=0.5, step=0.001, group="Core Material")
    Sa: float = P(0.10, symbol=r"\Sigma_a", desc="one-group absorption cross section", units="cm⁻¹", dims="L⁻¹",
                  lo=0.005, hi=0.5, step=0.001, group="Core Material")
    D: float = P(1.0, symbol=r"D", desc="core diffusion coefficient", units="cm", dims="L", lo=0.2, hi=5.0, step=0.01,
                 group="Core Material")
    rho_f: float = P(10.0, symbol=r"\rho_f", desc="core density (for the critical mass)", units="g·cm⁻³",
                     dims="M L⁻³", lo=1.0, hi=20.0, step=0.1, group="Core Material")
    geom: str = C("cylinder", choices=["cylinder", "sphere", "slab"],
                  labels=["Finite cylinder", "Sphere", "Infinite slab"], symbol=r"\mathrm{geometry}",
                  desc="core shape (geometric buckling of Emrich Table 8.2)", group="Geometry")
    R: float = P(20.0, symbol=r"R", desc="core radius (half-thickness for a slab)", units="cm", dims="L", lo=2.0,
                 hi=200.0, step=0.5, group="Geometry")
    HD: float = P(1.0, symbol=r"\frac{H}{D}", desc="cylinder height over diameter", units="", dims="1", lo=0.3,
                  hi=3.0, step=0.05, group="Geometry")
    b: float = P(0.0, symbol=r"b", desc="reflector thickness (0 for a bare core)", units="cm", dims="L", lo=0.0,
                 hi=60.0, step=0.5, group="Reflector")
    Dr: float = P(0.5, symbol=r"D_r", desc="reflector diffusion coefficient", units="cm", dims="L", lo=0.1, hi=3.0,
                  step=0.01, group="Reflector")
    Lr: float = P(5.0, symbol=r"L_r", desc="reflector diffusion length", units="cm", dims="L", lo=0.5, hi=60.0,
                  step=0.1, group="Reflector")


class CriticalityModel(ModelController):
    key = "nuc.crit"
    title = "Reactor criticality"
    module = "Nuclear"
    mode = "live"
    Params = CritParams
    fields = [r"$\frac{\phi}{\phi_0}$ (–)"]
    legend_labels = {"core": "Core (flux)", "reflector": "Reflector", "core_cut": "Core (cut face)"}

    def __init__(self) -> None:
        self._m: dict = {}
        self._built = None

    def evaluate(self, p: CritParams) -> dict:
        cm = 1e-2
        a = cr.analyse(p.nuSf / cm, p.Sa / cm, p.D * cm, p.geom, p.R * cm, p.HD, p.b * cm, p.Dr * cm, p.Lr * cm,
                       p.rho_f * 1e3)
        R_g = np.linspace(2.0, 200.0, 200) * cm
        k_g = np.array([cr.k_eff(p.nuSf / cm, p.Sa / cm, p.D * cm,
                                 cr.geometric_buckling(p.geom, R, 2 * p.HD * R, a["ext"])) for R in R_g])
        kb_g = np.array([cr.k_eff(p.nuSf / cm, p.Sa / cm, p.D * cm,
                                  cr.geometric_buckling(p.geom, R, 2 * p.HD * R, 2 * p.D * cm)) for R in R_g])
        b_g = np.linspace(0.0, 60.0, 61) * cm
        sav = np.array([cr.reflector_savings(a["Bm2"], p.D * cm, p.Dr * cm, p.Lr * cm, b) if a["Bm2"] > 0 else np.nan
                        for b in b_g])
        Rc_g = np.array([cr.critical_size(p.geom, a["Bm2"], s, p.HD) for s in sav])
        # radial flux profile through core and reflector (cylinder / sphere radial, slab across)
        Rr = p.R * cm
        r_core = np.linspace(0, Rr, 60)
        f_core = cr.flux_shape(p.geom, r_core, np.zeros_like(r_core), Rr, a["H"], a["ext"])
        r_ref = np.linspace(Rr, Rr + p.b * cm, 30) if p.b > 0 else np.array([Rr])
        f_ref = f_core[-1] * cr.reflector_flux(r_ref, Rr, p.b * cm, p.Dr * cm, p.Lr * cm) if p.b > 0 else f_core[-1:]
        return dict(**a, R_g=R_g, k_g=k_g, kb_g=kb_g, b_g=b_g, sav=sav, Rc_g=Rc_g, r_core=r_core, f_core=f_core,
                    r_ref=r_ref, f_ref=f_ref)

    def tabs(self, p, r, compare=None) -> list[TabSpec]:
        xl = r"$R$ (cm)"
        k = PlotSpec("Multiplication factor vs core size", xl, r"$k_{eff}$ (–)",
                     [Curve(r["R_g"] * 100, r["k_g"], "thrust", "this core"),
                      Curve(r["R_g"] * 100, r["kb_g"], "dim", "bare", "dash"),
                      Curve(r["R_g"] * 100, np.ones_like(r["R_g"]), "text", "critical", "dash")],
                     markers=[(p.R, r["keff"], "thrust")])
        refl = PlotSpec("Critical size vs reflector thickness", r"$b$ (cm)", r"$R_c$ (cm)",
                        [Curve(r["b_g"] * 100, r["Rc_g"] * 100, "fuel")], markers=[(p.b, r["Rc"] * 100, "fuel")])
        sav = PlotSpec("Reflector savings", r"$b$ (cm)", r"$\delta$ (cm)", [Curve(r["b_g"] * 100, r["sav"] * 100, "pressure")],
                       markers=[(p.b, r["ext"] * 100, "pressure")])
        flux = PlotSpec("Flux through core and reflector", r"$r$ (cm)", r"$\frac{\phi}{\phi_0}$ (–)",
                        [Curve(r["r_core"] * 100, r["f_core"], "hot", "core"),
                         Curve(r["r_ref"] * 100, r["f_ref"], "thrust", "reflector")])
        return [TabSpec("Criticality", [k, refl]), TabSpec("Flux & reflector", [flux, sav])]

    def readout(self, p, r, frame) -> list[str]:
        Rc = r["Rc"] * 100 if np.isfinite(r["Rc"]) else float("nan")
        return [rf"$k_\infty$ = {r['kinf']:.3f}", rf"$k_{{eff}}$ = {r['keff']:.4f}", rf"$R_c$ = {Rc:.2f} (cm)",
                rf"$M_c$ = {r['Mc']:.1f} (kg)"]

    def caption(self, p, r, frame) -> str:
        if r["Bm2"] <= 0:
            return r"$k_\infty \le 1$: no size is critical; raise $\nu\Sigma_f$ or lower $\Sigma_a$"
        state = "supercritical" if r["keff"] > 1.0 else "subcritical"
        return (rf"$B_m^2$ = {r['Bm2'] * 1e-4:.4f} (cm⁻²) · $L^2$ = {r['L2'] * 1e4:.2f} (cm²) · "
                rf"$\delta$ = {r['ext'] * 100:.2f} (cm) · $\rho$ = {r['rho'] * 100:.2f} (%) ({state})")

    def export_data(self, p, r):
        return {"R_m": r["R_g"], "k_eff": r["k_g"], "k_eff_bare": r["kb_g"], "b_m": r["b_g"], "savings_m": r["sav"],
                "Rc_m": r["Rc_g"]}

    def build_scene(self, vp, p, r=None) -> None:
        r = r or self.evaluate(p)
        pl = vp.plotter
        pl.clear()
        cm = 1e-2
        R = p.R * cm
        if p.geom == "sphere":
            sph = pv.Sphere(radius=R, theta_resolution=60, phi_resolution=40,
                            end_theta=360.0 - vp.cutaway if vp.cutaway else 360.0)
            rr = np.linalg.norm(sph.points, axis=1)
            sph["field"] = cr.flux_shape("sphere", rr, 0 * rr, R, R, r["ext"])
            pl.add_mesh(sph, scalars="field", cmap=CMAPS["temperature"], clim=(0, 1), show_scalar_bar=False,
                        name="core", smooth_shading=True)
            if p.b > 0:
                pl.add_mesh(pv.Sphere(radius=R + p.b * cm, theta_resolution=60, phi_resolution=40,
                                      end_theta=360.0 - vp.cutaway if vp.cutaway else 360.0),
                            color=REFLECTOR, opacity=0.3, name="reflector")
        else:
            H = r["H"] if p.geom == "cylinder" else 2.0 * R
            Rs = R if p.geom == "cylinder" else 1.5 * R              # a slab is drawn as a thick disc
            nz, nr = 40, 20
            zc = np.linspace(-0.5 * H, 0.5 * H, nz)
            th_ = thetas(vp.cutaway, 72)
            if p.geom == "cylinder":
                f_surf = cr.flux_shape("cylinder", np.full(nz, Rs), zc, R, H, r["ext"])
            else:
                f_surf = cr.flux_shape("slab", zc, 0 * zc, 0.5 * H, H, r["ext"])
            side = rev_grid(zc, np.full(nz, Rs), th_)
            side["field"] = broadcast_scalar(f_surf, len(th_))
            pl.add_mesh(side, scalars="field", cmap=CMAPS["temperature"], clim=(0, 1), show_scalar_bar=False,
                        name="core", smooth_shading=True, opacity=0.6 if vp.cutaway > 0 else 1.0)
            if vp.cutaway > 0:
                for k, ang in enumerate((0.0, np.radians(360.0 - vp.cutaway))):
                    face = cut_face(zc, np.zeros(nz), np.full(nz, Rs), ang, nr)
                    pts = face.points
                    rad = np.hypot(pts[:, 1], pts[:, 2])
                    if p.geom == "cylinder":
                        f = cr.flux_shape("cylinder", rad, pts[:, 0], R, H, r["ext"])
                    else:
                        f = cr.flux_shape("slab", pts[:, 0], 0 * rad, 0.5 * H, H, r["ext"])
                    face["field"] = f
                    pl.add_mesh(face, scalars="field", cmap=CMAPS["temperature"], clim=(0, 1), show_scalar_bar=False,
                                name=f"core_cut{k}")
            if p.b > 0:
                Ro = Rs + p.b * cm
                pl.add_mesh(rev_grid(zc, np.full(nz, Ro), th_), color=REFLECTOR, opacity=0.3, name="reflector")
        self._built = (vp.cutaway, p.geom, p.R, p.HD, p.b, round(float(r["ext"]), 6))
        self._m = {"built": True}
        vp.reset_view(zoom=1.2)

    def update_scene(self, vp, p, r, frame) -> None:
        if not self._m or self._built != (vp.cutaway, p.geom, p.R, p.HD, p.b, round(float(r["ext"]), 6)):
            self.build_scene(vp, p, r)
        vp.set_colorbar(CMAPS["temperature"], 0.0, 1.0, r"$\frac{\phi}{\phi_0}$ (–)")
        vp.render()


# ====================================================================================== fuel element (Live)
class ElementParams(ParamModel):
    ri: float = P(1.5, symbol=r"r_i", desc="coolant channel radius", units="mm", dims="L", lo=0.5, hi=5.0, step=0.05,
                  group="Fuel Cell")
    ro: float = P(3.5, symbol=r"r_o", desc="equivalent fuel cell outer radius", units="mm", dims="L", lo=1.0, hi=10.0,
                  step=0.05, group="Fuel Cell")
    L: float = P(100.0, symbol=r"L", desc="active core length", units="cm", dims="L", lo=20.0, hi=200.0, step=1.0,
                 group="Fuel Cell")
    aL: float = P(3.0, symbol=r"aL", desc="chopped-cosine buckling times length (pi for a full cosine)", units="",
                  dims="1", lo=0.1, hi=3.14, step=0.01, group="Fuel Cell")
    k: float = P(30.0, symbol=r"k", desc="fuel thermal conductivity", units="W·m⁻¹·K⁻¹", dims="M L T⁻³ Θ⁻¹", lo=1.0,
                 hi=150.0, step=0.5, group="Fuel Cell")
    P_ave: float = P(1342.0, symbol=r"P_{ave}", desc="average fuel power density", units="W·cm⁻³", dims="M L⁻¹ T⁻³",
                     lo=100.0, hi=5000.0, step=1.0, group="Power")
    mdot: float = P(1.0, symbol=r"\dot m", desc="propellant flow per channel", units="g·s⁻¹", dims="M T⁻¹", lo=0.1,
                    hi=5.0, step=0.01, group="Flow")
    T_in: float = P(300.0, symbol=r"T_{in}", desc="channel inlet temperature", units="K", dims="Θ", lo=20.0,
                    hi=1000.0, step=5.0, group="Flow")
    p_in: float = P(7.0, symbol=r"P_{in}", desc="channel inlet pressure", units="MPa", dims="M L⁻¹ T⁻²", lo=0.5,
                    hi=20.0, step=0.1, group="Flow")
    cp: float = P(15.2, symbol=r"c_p", desc="mean propellant specific heat", units="J·g⁻¹·K⁻¹", dims="L² T⁻² Θ⁻¹",
                  lo=5.0, hi=20.0, step=0.1, group="Flow")
    hc_mode: str = C("fixed", choices=["fixed", "db"],
                     labels=["Fixed heat transfer coefficient", "Dittus–Boelter (CoolProp hydrogen)"],
                     symbol=r"h_c\ \mathrm{model}", desc="how the film coefficient is found", group="Heat Transfer")
    hc: float = P(10.0, symbol=r"h_c", desc="film heat transfer coefficient (fixed model)", units="W·cm⁻²·K⁻¹",
                  dims="M T⁻³ Θ⁻¹", lo=0.05, hi=30.0, step=0.05, group="Heat Transfer")
    T_lim: float = P(3100.0, symbol=r"T_{lim}", desc="fuel temperature limit", units="K", dims="Θ", lo=1500.0,
                     hi=3500.0, step=10.0, group="Heat Transfer")


ELEMENT_FIELDS = [("Tmax", r"$T_{max}$ (K)"), ("Tp", r"$T_p$ (K)")]


class FuelElementModel(ModelController):
    key = "nuc.element"
    title = "Fuel element thermal hydraulics"
    module = "Nuclear"
    mode = "live"
    Params = ElementParams
    fields = [f[1] for f in ELEMENT_FIELDS]
    legend_labels = {"element": "Fuel element (colour field)", "channels": "Coolant channels"}
    AXIAL_DRAW = 0.1                       # the element is drawn at a tenth of its length

    def __init__(self) -> None:
        self._m: dict = {}
        self._built = None
        _warm_coolprop()

    def _si(self, p):
        return dict(ri=p.ri * 1e-3, ro=max(p.ro, p.ri + 0.05) * 1e-3, L=p.L * 1e-2, a=p.aL / (p.L * 1e-2),
                    mdot=p.mdot * 1e-3, cp=p.cp * 1e3, k=p.k, T_in=p.T_in)

    def evaluate(self, p: ElementParams) -> dict:
        s = self._si(p)
        q0 = th.q0_from_average(p.P_ave * 1e6, s["a"], s["L"])
        hc = None if p.hc_mode == "db" else p.hc * 1e4
        o = th.axial_cell(s["ri"], s["ro"], s["L"], s["a"], q0, s["mdot"], s["cp"], s["k"], s["T_in"], hc=hc,
                          p_in=p.p_in * 1e6, n=121)
        hc_fix = p.hc * 1e4 if hc is not None else float(np.mean(o["hc"]))
        q_lim = th.q0_for_peak(p.T_lim, s["T_in"], s["ri"], s["ro"], s["L"], s["a"], s["mdot"], s["cp"], s["k"], hc_fix)
        md_g = np.linspace(0.2, 5.0, 40) * 1e-3
        Tpk_g, Tout_g = [], []
        for md in md_g:
            c1 = q0 * np.pi * (s["ro"] ** 2 - s["ri"] ** 2) / (s["a"] * md * s["cp"])
            c2 = q0 * float(th.cell_resistance(s["ri"], s["ro"], s["k"], hc_fix))
            Tpk_g.append(s["T_in"] + c1 * np.sin(0.5 * s["a"] * s["L"]) + np.hypot(c1, c2))
            Tout_g.append(s["T_in"] + 2 * c1 * np.sin(0.5 * s["a"] * s["L"]))
        Pave_lim = q_lim * 2 * np.sin(0.5 * s["a"] * s["L"]) / (s["a"] * s["L"])
        return dict(**o, q0=q0, Pave_lim=Pave_lim, md_g=md_g, Tpk_g=np.array(Tpk_g), Tout_g=np.array(Tout_g),
                    over=o["Tpk"] > p.T_lim)

    def tabs(self, p, r, compare=None) -> list[TabSpec]:
        z = r["z"] * 100
        xl = r"$z$ (cm)"
        temp = PlotSpec("Axial temperatures", xl, r"$T$ (K)",
                        [Curve(z, r["Tmax"], "hot", "peak fuel"), Curve(z, r["Tw"], "oxidizer", "channel wall"),
                         Curve(z, r["Tp"], "thrust", "propellant"),
                         Curve(z, np.full_like(z, p.T_lim), "text", "fuel limit", "dash")],
                        markers=[(r["zpk"] * 100, r["Tpk"], "hot")])
        q = PlotSpec("Power density", xl, r"$q'''$ (W·cm⁻³)", [Curve(z, r["q"] / 1e6, "hot")])
        pr = PlotSpec("Channel pressure", xl, r"$P$ (MPa)", [Curve(z, r["p"] / 1e6, "pressure")])
        hc = PlotSpec("Film coefficient and Reynolds number", xl, r"$h_c$ (W·cm⁻²·K⁻¹)",
                      [Curve(z, r["hc"] / 1e4, "fuel", r"$h_c$"), Curve(z, r["Re"] / 1e4, "dim", r"$Re$ / 10⁴", "dash")])
        flow = PlotSpec("Peak fuel and outlet temperature vs flow", r"$\dot m$ (g·s⁻¹)", r"$T$ (K)",
                        [Curve(r["md_g"] * 1e3, r["Tpk_g"], "hot", "peak fuel"),
                         Curve(r["md_g"] * 1e3, r["Tout_g"], "thrust", "propellant outlet"),
                         Curve(r["md_g"] * 1e3, np.full_like(r["md_g"], p.T_lim), "text", "fuel limit", "dash")],
                        markers=[(p.mdot, r["Tpk"], "hot"), (p.mdot, r["Tout"], "thrust")])
        return [TabSpec("Temperatures", [temp, q]), TabSpec("Flow & pressure", [pr, hc, flow])]

    def readout(self, p, r, frame) -> list[str]:
        return [rf"$T_{{max,pk}}$ = {r['Tpk']:.0f} (K)", rf"$z_{{pk}}$ = {r['zpk'] * 100:.1f} (cm)",
                rf"$T_{{out}}$ = {r['Tout']:.0f} (K)", rf"$\Delta P$ = {r['dp'] / 1e3:.0f} (kPa)"]

    def caption(self, p, r, frame) -> str:
        warn = r" · OVER the fuel limit: lower the power or raise the flow" if r["over"] else ""
        model = "Dittus–Boelter" if p.hc_mode == "db" else "fixed $h_c$"
        return (rf"$q_0$ = {r['q0'] / 1e6:.0f} (W·cm⁻³) · limit $P_{{ave}}$ = {r['Pave_lim'] / 1e6:.0f} (W·cm⁻³) · "
                rf"$Ma_{{out}}$ = {r['Ma'][-1]:.2f} · {model} · element drawn at a tenth of its length{warn}")

    def export_data(self, p, r):
        return {"z_m": r["z"], "q_Wm3": r["q"], "Tp_K": r["Tp"], "Tw_K": r["Tw"], "Tmax_K": r["Tmax"], "p_Pa": r["p"],
                "hc_Wm2K": r["hc"], "Re": r["Re"]}

    def _layout(self, p):
        """19 channel centres on a hexagonal lattice with the equivalent-cell pitch, and the hexagon circumradius."""
        A_cell = np.pi * (p.ro * 1e-3) ** 2
        pitch = np.sqrt(A_cell / (np.sqrt(3) / 2))
        pts = []
        for q in range(-2, 3):
            for rr in range(-2, 3):
                s = -q - rr
                if max(abs(q), abs(rr), abs(s)) <= 2:
                    pts.append((pitch * (q + 0.5 * rr), pitch * np.sqrt(3) / 2 * rr))
        flat = np.sqrt(19 * A_cell / (np.sqrt(3) / 2))
        return np.array(pts), flat / np.sqrt(3)              # circumradius of the hexagon

    def build_scene(self, vp, p, r=None) -> None:
        r = r or self.evaluate(p)
        pl = vp.plotter
        pl.clear()
        key, title = ELEMENT_FIELDS[vp.field_index]
        v = r[key]
        clim = (float(min(r["Tp"].min(), r["Tmax"].min())), float(max(r["Tmax"].max(), r["Tp"].max())) + 1e-6)
        x = r["z"] * self.AXIAL_DRAW
        centres, Rh = self._layout(p)
        cut = vp.cutaway
        th_hex = np.radians(np.arange(0.0, 361.0, 60.0) + 30.0)
        if cut > 0:
            th_hex = th_hex[th_hex <= np.radians(30.0 + 360.0 - cut) + 1e-9]
        hexg = rev_grid(x, np.full_like(x, Rh), th_hex)
        hexg["field"] = broadcast_scalar(v, len(th_hex))
        pl.add_mesh(hexg, scalars="field", cmap=CMAPS["temperature"], clim=clim, show_scalar_bar=False, name="element",
                    opacity=0.45, smooth_shading=False)
        th_c = np.radians(np.linspace(0, 360, 16))
        blocks = []
        for cy_, cz in centres:
            g = rev_grid(x, np.full_like(x, p.ri * 1e-3), th_c)
            g.points[:, 1] += cy_
            g.points[:, 2] += cz
            g["field"] = broadcast_scalar(r["Tp"], len(th_c))
            blocks.append(g)
        ch = blocks[0].merge(blocks[1:])
        pl.add_mesh(ch, scalars="field", cmap=CMAPS["temperature"], clim=clim, show_scalar_bar=False, name="channels")
        self._built = (cut, vp.field_index, p.ri, p.ro, p.L)
        self._m = {"built": True}
        vp.reset_view(zoom=1.2)

    def update_scene(self, vp, p, r, frame) -> None:
        self.build_scene(vp, p, r)
        key, title = ELEMENT_FIELDS[vp.field_index]
        lo = float(min(r["Tp"].min(), r["Tmax"].min()))
        hi = float(max(r["Tmax"].max(), r["Tp"].max()))
        vp.set_colorbar(CMAPS["temperature"], lo, hi, title)
        vp.render()


# ====================================================================================== engine cycles (Live)
class CycleParams(ParamModel):
    kind: str = C("hot", choices=cy.CYCLES, labels=cy.CYCLE_LABELS, symbol=r"\mathrm{cycle}",
                  desc="where the turbine gets its power (Emrich Ch. 3)", group="Cycle")
    pc: float = P(7.0, symbol=r"P_c", desc="chamber pressure", units="MPa", dims="M L⁻¹ T⁻²", lo=1.0, hi=15.0,
                  step=0.1, group="Chamber")
    T0: float = P(2750.0, symbol=r"T_0", desc="chamber temperature", units="K", dims="Θ", lo=1500.0, hi=3200.0,
                  step=10.0, group="Chamber")
    eps: float = P(200.0, symbol=r"\varepsilon", desc="main nozzle expansion ratio", units="", dims="1", lo=10.0,
                   hi=500.0, step=1.0, group="Chamber")
    T1: float = P(20.0, symbol=r"T_1", desc="tank temperature (saturated liquid)", units="K", dims="Θ", lo=14.0,
                  hi=30.0, step=0.1, group="Pump")
    p_tank: float = P(0.2, symbol=r"P_1", desc="tank pressure", units="MPa", dims="M L⁻¹ T⁻²", lo=0.05, hi=1.0,
                      step=0.01, group="Pump")
    eta_p: float = P(0.70, symbol=r"\eta_p", desc="pump isentropic efficiency", units="", dims="1", lo=0.3, hi=1.0,
                     step=0.01, group="Pump")
    T3: float = P(200.0, symbol=r"T_3", desc="temperature after regenerative heating (nozzle, reflector, structure)",
                  units="K", dims="Θ", lo=60.0, hi=600.0, step=5.0, group="Regenerative Heating")
    dp_regen: float = P(10.0, symbol=r"\frac{\Delta P_{regen}}{P_c}", desc="regenerative circuit pressure loss",
                        units="%", dims="1", lo=0.0, hi=40.0, step=0.5, group="Regenerative Heating")
    dp_core: float = P(10.0, symbol=r"\frac{\Delta P_{core}}{P_c}", desc="core pressure loss", units="%", dims="1",
                       lo=0.0, hi=40.0, step=0.5, group="Regenerative Heating")
    eta_t: float = P(0.70, symbol=r"\eta_t", desc="turbine isentropic efficiency", units="", dims="1", lo=0.3,
                     hi=1.0, step=0.01, group="Turbine")
    PR_t: float = P(20.0, symbol=r"\frac{P_{in}}{P_{out}}", desc="bleed turbine pressure ratio", units="", dims="1",
                    lo=2.0, hi=60.0, step=0.5, group="Turbine")
    T_turb: float = P(800.0, symbol=r"T_{5}", desc="hot bleed turbine inlet temperature limit", units="K", dims="Θ",
                      lo=300.0, hi=1400.0, step=10.0, group="Turbine")
    eps_aux: float = P(50.0, symbol=r"\varepsilon_{aux}", desc="auxiliary (bleed) nozzle expansion ratio", units="",
                       dims="1", lo=2.0, hi=200.0, step=1.0, group="Turbine")


class CyclesModel(ModelController):
    key = "nuc.cycles"
    title = "NTR engine cycles"
    module = "Nuclear"
    mode = "live"
    Params = CycleParams
    fields = [r"$T$ (K)"]
    legend_labels = {"core": "Reactor core", "tank": "Liquid hydrogen tank", "pump": "Pump", "turbine": "Turbine",
                     "nozzle": "Main nozzle", "pipe": "Pipes (temperature)"}

    def __init__(self) -> None:
        self._m: dict = {}
        self._sweep_key = None
        self._sweep: dict = {}
        self.evaluate(self.Params())               # loads CoolProp hydrogen and caches the default pressure sweep

    @staticmethod
    def _sweep_table(pc_g, T0, a) -> dict:
        """Bleed fraction and delivered Isp of every cycle against chamber pressure."""
        sweep = {}
        for k in cy.CYCLES:
            fb, isp = [], []
            for pcx in pc_g:
                try:
                    c = cy.cycle(k, pcx * 1e6, T0, **a)
                    fb.append(c["fb"] * 100)
                    isp.append(c["isp"])
                except ValueError:
                    fb.append(np.nan)
                    isp.append(np.nan)
            sweep[k] = (np.array(fb), np.array(isp))
        return sweep

    def _args(self, p):
        return dict(T1=p.T1, p_tank=p.p_tank * 1e6, T3=p.T3, dp_core=p.dp_core / 100, dp_regen=p.dp_regen / 100,
                    eta_p=p.eta_p, eta_t=p.eta_t, PR_t=p.PR_t, T_turb=p.T_turb, eps=p.eps, eps_aux=p.eps_aux)

    def evaluate(self, p: CycleParams) -> dict:
        a = self._args(p)
        try:
            o = cy.cycle(p.kind, p.pc * 1e6, p.T0, **a)
            ok = True
        except ValueError:
            o = cy.cycle("cold", p.pc * 1e6, p.T0, **a)
            ok = False
        pc_g = np.linspace(2.0, 15.0, 7)
        key = (p.T0, tuple(sorted(a.items())))
        if key != self._sweep_key:                 # the sweep ignores the chosen cycle and pressure: cache it
            self._sweep_key, self._sweep = key, self._sweep_table(pc_g, p.T0, a)
        sweep = self._sweep
        paths = [cy.path_curve(o["states"], names) for names in o["paths"]]
        return dict(**o, ok=ok, pc_g=pc_g, sweep=sweep, dome=cy.saturation_dome(), path_ts=paths)

    def tabs(self, p, r, compare=None) -> list[TabSpec]:
        Td, sl, sv = r["dome"]
        st = r["states"]
        kJ = 1e-3
        names = ["main flow", "turbine flow"]

        def ts_plot(title, T_cap):
            cut = lambda T: np.where(T <= T_cap, T, np.nan)
            curves = [Curve(sl * kJ, Td, "dim", "saturated liquid"), Curve(sv * kJ, Td, "dim", "saturated vapour", "dash")]
            for k, (ss, tt) in enumerate(r["path_ts"]):
                curves.append(Curve(ss * kJ, cut(tt), "thrust" if k == 0 else "hot", names[min(k, 1)]))
            mk = [(x["s"] * kJ, x["T"], "text") for x in st if x["T"] <= T_cap]
            return PlotSpec(title, r"$s$ (kJ·kg⁻¹·K⁻¹)", r"$T$ (K)", curves, markers=mk)

        ts = ts_plot("Temperature–entropy diagram", 1e9)
        cold = ts_plot("Cold end detail", max(1.3 * p.T3, 1.3 * float(st[2]["T"]), 120.0))
        cols = {"hot": "hot", "cold": "thrust", "expander": "fuel"}
        lab = dict(zip(cy.CYCLES, cy.CYCLE_LABELS))
        fb = PlotSpec("Bleed fraction vs chamber pressure", r"$P_c$ (MPa)", r"$\frac{\dot m_{bleed}}{\dot m}$ (%)",
                      [Curve(r["pc_g"], r["sweep"][k][0], cols[k], lab[k].lower()) for k in cy.CYCLES],
                      markers=[(p.pc, r["fb"] * 100, "text")])
        isp = PlotSpec("Delivered specific impulse vs chamber pressure", r"$P_c$ (MPa)", r"$I_{sp}$ (s)",
                       [Curve(r["pc_g"], r["sweep"][k][1], cols[k], lab[k].lower()) for k in cy.CYCLES],
                       markers=[(p.pc, r["isp"], "text")])
        return [TabSpec("Cycle", [ts, cold]), TabSpec("Chamber pressure trade", [fb, isp])]

    def readout(self, p, r, frame) -> list[str]:
        return [rf"$\frac{{\dot m_{{bleed}}}}{{\dot m}}$ = {r['fb'] * 100:.2f} (%)", rf"$I_{{sp}}$ = {r['isp']:.0f} (s)",
                rf"$P_2$ = {r['p2'] / 1e6:.2f} (MPa)", rf"$w_p$ = {r['w_pump'] / 1e3:.0f} (kJ·kg⁻¹)"]

    def caption(self, p, r, frame) -> str:
        if not r["ok"]:
            return r"expander cycle cannot close at this $T_3$: showing the cold bleed cycle; raise $T_3$ or $\eta_t$"
        extra = rf" · hot gas share of the turbine flow {r['x_hot'] * 100:.1f} (%)" if p.kind == "hot" else ""
        return (rf"main nozzle $I_{{sp}}$ = {r['isp_main']:.0f} (s) · turbine inlet {r['T_turb_in']:.0f} (K) · "
                rf"core heat {r['q_core'] / 1e6:.1f} (MJ·kg⁻¹){extra}")

    def export_data(self, p, r):
        out = {"state_T_K": np.array([s["T"] for s in r["states"]]), "state_p_Pa": np.array([s["p"] for s in r["states"]]),
               "state_h_Jkg": np.array([s["h"] for s in r["states"]]), "state_s_JkgK": np.array([s["s"] for s in r["states"]]),
               "pc_MPa": r["pc_g"]}
        for k in cy.CYCLES:
            out[f"bleed_pct_{k}"], out[f"Isp_s_{k}"] = r["sweep"][k]
        return out

    # 3D: tank -> pump -> regen jacket -> (turbine) -> core -> nozzle, pipes coloured by temperature
    def build_scene(self, vp, p, r=None) -> None:
        r = r or self.evaluate(p)
        pl = vp.plotter
        pl.clear()
        T = {s["name"]: s["T"] for s in r["states"]}
        lo, hi = 15.0, p.T0
        cmap = CMAPS["temperature"]
        clim = (np.log10(lo), np.log10(hi))
        pl.add_mesh(pv.Sphere(radius=0.8, center=(-3.2, 0, 0)), color="#9fc8ff", opacity=0.85, name="tank",
                    smooth_shading=True)
        pump = pv.Cylinder(center=(-1.8, 0, -1.2), direction=(0, 0, 1), radius=0.25, height=0.4)
        pl.add_mesh(pump, color="#8a94a6", name="pump")
        turb = pv.Cylinder(center=(-1.8, 0, -0.6), direction=(0, 0, 1), radius=0.28, height=0.35)
        pl.add_mesh(turb, color="#c0c7d2", name="turbine")
        core_x0, H, R = 0.0, 1.4, 0.55
        x = np.linspace(core_x0, core_x0 + H, 12)
        g = rev_grid(x, np.full_like(x, R), thetas(vp.cutaway))
        g["field"] = broadcast_scalar(np.log10(np.linspace(max(T.get("3", 100), 20), p.T0, 12)), len(thetas(vp.cutaway)))
        pl.add_mesh(g, scalars="field", cmap=cmap, clim=clim, show_scalar_bar=False, name="core", smooth_shading=True)
        xn, rn = contour(0.12, min(p.eps, 60.0), Rc=0.3, f=0.8, g=1.4)
        xn = xn - xn[0] + core_x0 + H + 0.15
        noz = rev_grid(xn, rn, thetas(vp.cutaway))
        Tn = np.interp(xn, [xn[0], xn[-1]], [p.T0, max(0.15 * p.T0, 100.0)])
        noz["field"] = broadcast_scalar(np.log10(Tn), len(thetas(vp.cutaway)))
        pl.add_mesh(noz, scalars="field", cmap=cmap, clim=clim, show_scalar_bar=False, name="nozzle", smooth_shading=True)

        def pipe(pts, Tval, name):
            line = pv.lines_from_points(np.asarray(pts, float))
            tube = line.tube(radius=0.06)
            tube["field"] = np.full(tube.n_points, np.log10(max(Tval, lo)))
            pl.add_mesh(tube, scalars="field", cmap=cmap, clim=clim, show_scalar_bar=False, name=name)

        pipe([(-2.5, 0, -0.5), (-2.5, 0, -1.2), (-2.05, 0, -1.2)], T["1"], "pipe_tank")
        pipe([(-1.55, 0, -1.2), (2.4, 0, -1.2), (2.4, 0, -0.75)], T["2"], "pipe_regen")
        pipe([(2.6, 0, 0.9), (0.7, 0, 0.9), (-0.3, 0, 0.9)], T["3"], "pipe_warm")
        if p.kind == "expander":
            pipe([(-0.3, 0, 0.9), (-1.8, 0, 0.9), (-1.8, 0, -0.42)], T["3"], "pipe_turbine_in")
            pipe([(-1.55, 0, -0.6), (-0.6, 0, -0.6), (-0.6, 0, 0.0), (0.0, 0, 0.0)], T["4"], "pipe_core_in")
        else:
            pipe([(-0.3, 0, 0.9), (-0.3, 0, 0.0), (0.0, 0, 0.0)], T["3"], "pipe_core_in")
            Tt = T.get("5", T["3"]) if p.kind == "hot" else T["3"]
            pipe([(-0.3, 0, 0.9), (-1.8, 0, 0.9), (-1.8, 0, -0.42)], Tt, "pipe_turbine_in")
            Tex = T.get("6", T.get("4", 300.0))
            pipe([(-2.08, 0, -0.6), (-2.6, 0, -0.6), (-2.6, 0, 0.6)], Tex, "pipe_bleed_dump")
            if p.kind == "hot":
                pipe([(core_x0 + H + 0.05, 0, 0.5), (core_x0 + H + 0.05, 0, 1.3), (-1.2, 0, 1.3), (-1.2, 0, 0.9)],
                     p.T0, "pipe_hot_bleed")
        self._m = {"built": (vp.cutaway, p.kind, p.eps)}
        vp.reset_view(zoom=1.0)

    def update_scene(self, vp, p, r, frame) -> None:
        self.build_scene(vp, p, r)
        vp.set_colorbar(CMAPS["temperature"], np.log10(15.0), np.log10(p.T0), r"$\log_{10}T$ (K)")
        vp.render()


# ====================================================================================== poisons (Live)
class PoisonParams(ParamModel):
    phi: float = P(14.0, symbol=r"\log_{10}\phi", desc="thermal flux at full power (log10 of n·cm⁻²·s⁻¹)", units="",
                   dims="1", lo=11.0, hi=16.0, step=0.05, group="Core")
    nu: float = P(2.42, symbol=r"\nu", desc="neutrons per thermal fission of uranium-235", units="", dims="1", lo=2.0,
                  hi=3.0, step=0.01, group="Core")
    t1: float = P(60.0, symbol=r"t_1", desc="first burn duration", units="min", dims="T", lo=1.0, hi=600.0, step=1.0,
                  group="History")
    t_off: float = P(10.0, symbol=r"t_{off}", desc="coast between burns", units="h", dims="T", lo=0.1, hi=200.0,
                     step=0.1, group="History")
    t2: float = P(30.0, symbol=r"t_2", desc="restart burn duration", units="min", dims="T", lo=0.0, hi=600.0, step=1.0,
                  group="History")
    t_after: float = P(48.0, symbol=r"t_{after}", desc="time shown after the restart burn", units="h", dims="T",
                       lo=1.0, hi=200.0, step=1.0, group="History")
    margin: float = P(3.0, symbol=r"\rho_{excess}", desc="excess reactivity held by the control drums", units="$",
                      dims="1", lo=0.0, hi=30.0, step=0.1, group="Control")


class PoisonsModel(ModelController):
    key = "nuc.poisons"
    title = "Xenon & samarium"
    module = "Nuclear"
    mode = "live"
    Params = PoisonParams
    fields = [r"$\frac{\Sigma_a^{Xe}}{\Sigma_f}$ (–)"]
    legend_labels = {**REACTOR_LEGEND, "core": "Core (xenon at the restart)"}

    def __init__(self) -> None:
        self._m: dict = {}

    def evaluate(self, p: PoisonParams) -> dict:
        phi = 10.0**p.phi
        ph = [(p.t1 * 60.0, phi), (p.t_off * po.HOUR, 0.0)]
        if p.t2 > 0:
            ph.append((p.t2 * 60.0, phi))
        ph.append((p.t_after * po.HOUR, 0.0))
        h = po.history(ph, n=500)
        worth = -(h["xe"] + h["sm"]) / (p.nu * kn.BETA)          # dollars
        t_restart = (p.t1 * 60.0 + p.t_off * po.HOUR) / po.HOUR
        i_r = int(np.searchsorted(h["t"] / po.HOUR, t_restart))
        phi_g = np.logspace(11, 16, 80)
        eq = np.array([po.xenon_equilibrium(f) for f in phi_g])
        peak = [po.post_shutdown_peak(f, 72.0, 600) for f in (1e13, 1e14, 1e15)]
        return dict(**h, worth=worth, t_h=h["t"] / po.HOUR, i_r=min(i_r, len(h["t"]) - 1), t_restart=t_restart,
                    phi_g=phi_g, eq=eq, peak=peak, xe_eq=po.xenon_equilibrium(phi),
                    dead=bool(np.any(worth < -p.margin)))

    def tabs(self, p, r, compare=None) -> list[TabSpec]:
        t = r["t_h"]
        xl = r"$t$ (h)"
        poi = PlotSpec("Poison absorption over the history", xl, r"$\frac{\Sigma_a}{\Sigma_f}$ (–)",
                       [Curve(t, r["xe"], "hot", "xenon-135"), Curve(t, r["sm"], "fuel", "samarium-149"),
                        Curve(t, np.full_like(t, r["xe_eq"]), "dim", "xenon equilibrium", "dash")],
                       markers=[(r["t_restart"], float(r["xe"][r["i_r"]]), "text")])
        w = PlotSpec("Poison reactivity", xl, r"$\Delta\rho$ ($)",
                     [Curve(t, r["worth"], "pressure", "xenon + samarium"),
                      Curve(t, np.full_like(t, -p.margin), "text", "drum excess reactivity", "dash")])
        flux = PlotSpec("Flux history", xl, r"$\phi$ (n·cm⁻²·s⁻¹)", [Curve(t, r["phi"], "thrust")])
        eq = PlotSpec("Equilibrium xenon vs flux", r"$\log_{10}\phi$ (–)", r"$\frac{\Sigma_a^{Xe}}{\Sigma_f}$ (–)",
                      [Curve(np.log10(r["phi_g"]), r["eq"], "hot")], markers=[(p.phi, r["xe_eq"], "hot")])
        return [TabSpec("History", [poi, w, flux]), TabSpec("Equilibrium", [eq])]

    def readout(self, p, r, frame) -> list[str]:
        k = int(np.argmax(r["xe"]))
        return [rf"$\frac{{\Sigma_a^{{Xe}}}}{{\Sigma_f}}$ peak = {r['xe'][k]:.4f}", rf"$t_{{peak}}$ = {r['t_h'][k]:.1f} (h)",
                rf"$\Delta\rho_{{min}}$ = {r['worth'].min():.2f} ($)",
                rf"$\Delta\rho$ at restart = {r['worth'][r['i_r']]:.2f} ($)"]

    def caption(self, p, r, frame) -> str:
        warn = (r" · xenon worth exceeds the drum excess reactivity: the restart is blocked until xenon decays"
                if r["dead"] else "")
        pk = r["peak"][1]
        return (rf"after shutdown from equilibrium at 10¹⁴ (n·cm⁻²·s⁻¹) xenon peaks at {pk[0]:.1f} (h), "
                rf"{pk[1]:.1f} times equilibrium{warn}")

    def export_data(self, p, r):
        return {"t_s": r["t"], "phi": r["phi"], "xe_over_Sf": r["xe"], "sm_over_Sf": r["sm"], "worth_dollars": r["worth"]}

    def build_scene(self, vp, p, r=None) -> None:
        r = r or self.evaluate(p)
        vp.plotter.clear()
        xe = float(r["xe"][r["i_r"]])
        reactor_meshes(vp, 0.4, 0.8, 0.12, vp.cutaway, x0=-0.4, core_scalars=lambda x: np.full_like(x, xe),
                       cmap="viridis", clim=(0.0, max(float(r["xe"].max()), 1e-6)))
        self._m = {"built": True}
        vp.reset_view(zoom=1.2)

    def update_scene(self, vp, p, r, frame) -> None:
        self.build_scene(vp, p, r)
        vp.set_colorbar("viridis", 0.0, max(float(r["xe"].max()), 1e-6), r"$\frac{\Sigma_a^{Xe}}{\Sigma_f}$ (–)")
        vp.render()


# ====================================================================================== reactor kinetics (Run)
class KineticsModel(ModelController):
    key = "nuc.kinetics"
    title = "Reactor kinetics & startup"
    module = "Nuclear"
    mode = "run"
    Params = KineticsParams
    fields = [r"$T_f$ (K)"]
    legend_labels = {**REACTOR_LEGEND, "core": "Core (fuel temperature)"}

    def __init__(self) -> None:
        self.sim = ReactorKinetics()
        self._m: dict = {}

    def build_scene(self, vp, p, r=None) -> None:
        self._m = {"built": True}
        self._draw(vp, p, r, 0 if r is not None else None)
        vp.reset_view(zoom=1.2)

    def _draw(self, vp, p, r, frame):
        pl = vp.plotter
        pl.clear()
        if r is None or frame is None:
            Tf, turn, n = p.T_in, 0.0, 0.0
            lo, hi = p.T_in, 3000.0
        else:
            c = r.channels
            Tf = float(c["Tf"][frame])
            # drum angle: absorber faces the core at the most negative control reactivity of the run, turns 180 deg
            re = c["rho_ext"]
            span = max(float(re.max() - re.min()), 1e-9)
            turn = np.pi * float((re[frame] - re.min()) / span)
            n = float(c["P_tot"][frame])
            lo, hi = float(min(c["Tf"].min(), c["Tp"].min())), float(c["Tf"].max()) + 1e-6
        H = 1.3
        shape = lambda x: Tf * (0.55 + 0.45 * np.cos(np.pi * (x - 0.5 * H) / (1.2 * H)))
        reactor_meshes(vp, 0.45, H, 0.15, vp.cutaway, x0=0.0, drum_turn=turn, core_scalars=shape,
                       cmap=CMAPS["temperature"], clim=(lo, max(hi, lo + 1.0)))
        vp.set_colorbar(CMAPS["temperature"], lo, max(hi, lo + 1.0), r"$T_f$ (K)")
        self._m["n"] = n

    def update_scene(self, vp, p, r, frame) -> None:
        if not self._m:
            self.build_scene(vp, p, r)
        self._draw(vp, p, r, frame)
        vp.render()

    def tabs(self, p, r, compare=None) -> list[TabSpec]:
        xl = r"$t$ (s)"
        if r is None:
            e = lambda title, yl: PlotSpec(title, xl, yl, [], cursor=True)
            return [TabSpec("Power & reactivity", [e("Relative power", r"$\frac{P}{P_0}$ (–)"),
                                                   e("Reactivity", r"$\rho$ ($)")]),
                    TabSpec("Temperatures", [e("Core temperatures", r"$T$ (K)")]), TabSpec("Stability", [])]
        c, t = r.channels, r.t
        cur = lambda y, col, label="", style="solid": Curve(t, y, col, label, style)
        pw = PlotSpec("Relative power", xl, r"$\frac{P}{P_0}$ (–)",
                      [cur(c["P_tot"], "hot", "total"), cur(c["n"], "thrust", "fission", "dash"),
                       cur(c["decay"], "fuel", "decay heat", "dash")], cursor=True, cursor_channel="P_tot")
        rho = PlotSpec("Reactivity", xl, r"$\rho$ ($)",
                       [cur(c["rho"], "text", "net"), cur(c["rho_ext"], "pressure", "control drums", "dash"),
                        cur(c["rho_f"], "hot", "fuel"), cur(c["rho_m"], "oxidizer", "moderator"),
                        cur(c["rho_pr"], "thrust", "propellant density")], cursor=True, cursor_channel="rho")
        temps = PlotSpec("Core temperatures", xl, r"$T$ (K)",
                         [cur(c["Tf"], "hot", "fuel"), cur(c["Tm"], "oxidizer", "moderator"),
                          cur(c["Tout"], "thrust", "propellant outlet"),
                          cur(c["T_set"], "dim", "outlet set point", "dash")], cursor=True, cursor_channel="Tf")
        flow = PlotSpec("Flow and chamber pressure", xl, r"$\frac{\dot m}{\dot m_0}$ (–)",
                        [cur(c["mflow"], "fuel", "flow"), cur(c["pc"] / (p.P * 1e6), "pressure", "chamber pressure", "dash")],
                        cursor=True, cursor_channel="mflow")
        w = r.arrays["bode_w"]
        bode = [PlotSpec("Closed-loop gain", r"$\log_{10}\omega$ (–)", r"$|K_{RT}G_{RT}|$ (dB)",
                         [Curve(np.log10(w), r.arrays["bode_gain"], "thrust")]),
                PlotSpec("Closed-loop phase", r"$\log_{10}\omega$ (–)", r"$\angle K_{RT}G_{RT}$ (deg)",
                         [Curve(np.log10(w), r.arrays["bode_phase"], "pressure")])]
        return [TabSpec("Power & reactivity", [pw, rho]), TabSpec("Temperatures", [temps, flow]),
                TabSpec("Stability", bode)]

    def readout(self, p, r, frame) -> list[str]:
        if r is None or frame is None:
            return [r"$\frac{P}{P_0}$ = –", r"$T_f$ = – (K)", r"$T_{out}$ = – (K)", r"$\rho$ = – ($)"]
        c = r.channels
        return [rf"$\frac{{P}}{{P_0}}$ = {c['P_tot'][frame]:.4g}", rf"$T_f$ = {c['Tf'][frame]:.0f} (K)",
                rf"$T_{{out}}$ = {c['Tout'][frame]:.0f} (K)", rf"$\rho$ = {c['rho'][frame]:.3f} ($)"]

    def caption(self, p, r, frame) -> str:
        if r is None or frame is None:
            return r"press Run: six-group point kinetics with fuel, moderator and hydrogen-density feedback"
        c = r.channels
        return (rf"$t$ = {r.t[frame]:.2f} (s) · drums {c['rho_ext'][frame]:.2f} ($) · "
                rf"flow {c['mflow'][frame] * 100:.0f} (%) · chamber pressure {c['pc'][frame] / 1e6:.2f} (MPa)")

    def summary_line(self, p, r) -> str:
        m = r.meta
        stab = "stable" if m["net"] > 0 else "UNSTABLE"
        return (rf"peak fuel {m['Tf_max']:.0f} (K) · peak power {m['P_max']:.3g} $P_0$ · "
                rf"low-frequency feedback {stab} (critical $\beta_p$ = {m['beta_p_crit']:.0f} (cm³·g⁻¹)) · "
                rf"cooldown propellant {m['f_sd'] * 100:.1f} (%)")

    def export_data(self, p, r):
        return {"t_s": r.t, **r.channels}
