"""Run drivers for solid motors: internal ballistics (0-D / spatial, erosive, ignition, tail-off) and 3-D level-set burnback."""
from __future__ import annotations

import json
from dataclasses import replace
from functools import lru_cache

import numpy as np

from .. import data as lib
from ..core.ballistics import solid as sol
from ..core.grain import builder as gb
from ..core.grain import levelset as ls
from ..core.grain import section2d as s2
from ..core.grain import stack as gstack
from ..core.result import Result
from ..core.schema import C, P, ParamModel, T
from .base import Simulation

# burn-rate laws: a and n from the sliders, your own pressure bands, or a library pressure-band law
LAWS = ["single", "bands", "KNSB", "KNDX"]
LAW_LABELS = ["a, n (single law)", "Pressure bands (your table)", "KNSB, Nakka 1999 bands", "KNDX, Nakka 1999 bands"]


def parse_bands(text: str) -> tuple:
    """User pressure bands: JSON ``[[p_lo, p_hi, a', n], ...]`` with p in MPa and r (mm/s) = a' p^n."""
    if not text or not text.strip():
        raise ValueError("the pressure-band table is empty: add bands or pick another burn-rate law")
    try:
        rows = json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"invalid JSON: {e}") from e
    if not isinstance(rows, list) or not rows or not all(isinstance(r, list) and len(r) == 4 for r in rows):
        raise ValueError("pressure bands are a list of [p_lo, p_hi, a, n] rows (MPa, mm/s)")
    return sol.bands_from_mm_MPa(rows)


def law_bands(law: str, bands_json: str = "") -> tuple:
    """The ``SolidProp.bands`` tuple for a burn-rate law choice (empty for the single a, n law)."""
    if law == "single":
        return ()
    if law == "bands":
        return parse_bands(bands_json)
    if law in lib.solid_burn_laws():
        return lib.solid_burn_bands(law)
    raise ValueError(f"unknown burn-rate law {law!r}; use one of {LAWS}")

KIND_LABELS = ["Circular bore (BATES)", "Star", "Wagon wheel (centre + slots)", "Finocyl (centre + fins)",
               "Centre + 4 bores", "Centre + 6 bores", "Moon burner (offset bore)", "C-slot", "D-grain",
               "X-core (arms, no bore)", "End burner (no port)", "Custom (shape list)"]
KINDS = [*s2.KINDS, *s2.PORTLESS, "custom"]
ENDS = ["none", "aft", "head", "both"]
ENDS_LABELS = ["Both ends inhibited", "Aft end burns", "Head end burns", "Both ends burn"]


@lru_cache(maxsize=32)
def _cached_section(kind, R_out, R_p, n, w, r_tip, r_valley, depth, offset=10.0) -> s2.Burnback:
    """Section burnback tables are reused across the many ``to_si`` calls of a session (about 20-50 ms each)."""
    if kind in s2.PORTLESS:
        X, Y, h = s2.grid(R_out)
        inside = np.hypot(X, Y) <= R_out
        A_case = float(np.pi * R_out**2)
        d = np.array([0.0, 2.0 * R_out])
        return s2.Burnback(d, np.array([0.0, A_case]), np.zeros(2), np.ones(2), 2.0 * R_out, A_case,
                           np.full_like(X, 2.0 * R_out), inside, h)
    return s2.section(kind, R_out, R_p=R_p, n=n, w=w, r_tip=r_tip, r_valley=r_valley, depth=depth, offset=offset)


class SolidGrainFields(ParamModel):
    """2-D grain section, extruded along the motor axis."""

    kind: str = C("wagon", choices=KINDS, labels=KIND_LABELS, symbol=r"\mathrm{section}",
                  desc="grain cross-section", group="Grain")
    R_out: float = P(60.0, symbol=r"R_{out}", desc="grain outer radius", units="mm", dims="L", lo=30.0, hi=150.0,
                     step=0.5, group="Grain")
    R_p: float = P(14.0, symbol=r"R_p", desc="centre-bore radius (also the satellite-bore radius)", units="mm",
                   dims="L", lo=6.0, hi=40.0, step=0.5, group="Grain")
    n_el: float = P(6.0, symbol=r"N", desc="number of star points / slots / fins", units="", dims="1", lo=3.0,
                    hi=12.0, step=1.0, group="Grain")
    w_el: float = P(8.0, symbol=r"w_s", desc="slot / fin width", units="mm", dims="L", lo=3.0, hi=24.0, step=0.5,
                    group="Grain")
    depth: float = P(0.55, symbol=r"\frac{r_s}{R_{out}}", desc="slot / fin depth as a fraction of the outer radius",
                     units="", dims="1", lo=0.3, hi=0.9, step=0.01, group="Grain")
    r_tip: float = P(0.7, symbol=r"\frac{r_{tip}}{R_{out}}", desc="star tip radius fraction", units="", dims="1",
                     lo=0.4, hi=0.95, step=0.01, group="Grain")
    r_valley: float = P(0.3, symbol=r"\frac{r_{val}}{R_{out}}", desc="star valley radius fraction", units="", dims="1",
                        lo=0.1, hi=0.6, step=0.01, group="Grain")
    offset: float = P(10.0, symbol=r"e", desc="moon burner: bore offset from the axis; D-grain: distance from the "
                      "axis to the flat (negative cuts past the centre)", units="mm", dims="L", lo=-100.0, hi=100.0,
                      step=0.5, group="Grain")
    L_g: float = P(0.6, symbol=r"L_g", desc="grain (segment) length", units="m", dims="L", lo=0.02, hi=2.0,
                   step=0.005, group="Grain")
    n_seg: float = P(1.0, symbol=r"N_{seg}", desc="number of identical segments, each of length L_g (BATES stack)",
                     units="", dims="1", lo=1.0, hi=12.0, step=1.0, group="Grain")
    ends: str = C("none", choices=ENDS, labels=ENDS_LABELS, symbol=r"\mathrm{ends}",
                  desc="which end faces of each segment burn (an end burner always burns its aft face)",
                  group="Grain")

    section_json: str = T("", label="custom section", desc="custom section drawn as a list of shapes, with optional "
                          "inhibited surfaces (used when the section is custom)", editor="section", group="Grain")

    def custom_spec(self) -> gb.SectionSpec:
        """The custom section; an empty text means the editor's starting shape on this outer radius."""
        spec = gb.spec_from_json(self.section_json) if self.section_json.strip() else gb.default_spec(self.R_out)
        return replace(spec, R_out=self.R_out)               # the form's outer radius always wins

    def sdf_kw(self) -> dict:
        return dict(R_p=self.R_p, n=int(round(self.n_el)), w=self.w_el, r_tip=self.r_tip, r_valley=self.r_valley,
                    depth=self.depth, offset=self.offset)

    def section(self) -> s2.Burnback:
        if self.kind == "custom":
            return gb.cached_section(self.custom_spec().to_json())
        return _cached_section(self.kind, self.R_out, self.R_p, int(round(self.n_el)), self.w_el, self.r_tip,
                               self.r_valley, self.depth, self.offset)

    @property
    def portless(self) -> bool:
        return self.kind in s2.PORTLESS

    def end_flags(self) -> tuple[bool, bool]:
        """``(burn_head, burn_aft)`` of each segment; an end burner always burns its aft face."""
        h, a = self.ends in ("head", "both"), self.ends in ("aft", "both")
        return (h, True) if self.portless and not h else (h, a or self.portless)

    @property
    def faces(self) -> int:
        return sum(self.end_flags())

    def section_table(self, b: s2.Burnback | None = None) -> sol.SectionTable:
        if b is None and self.portless:
            return sol.portless_table(self.R_out * 1e-3, self.L_g)
        b = b or self.section()
        return sol.SectionTable(b.d * 1e-3, b.area * 1e-6, b.perim * 1e-3, b.A_case * 1e-6)


class SolidPropFields(ParamModel):
    rho_p: float = P(1750.0, symbol=r"\rho_p", desc="propellant density", units="kg·m⁻³", dims="M L⁻³", lo=1200.0,
                     hi=2000.0, step=5.0, group="Propellant")
    a_mm: float = P(5.0, symbol=r"a'", desc="burn rate at 1 MPa (r in mm/s with p in MPa)", units="mm·s⁻¹",
                    dims="L T⁻¹", lo=1.0, hi=30.0, step=0.1, group="Propellant")
    n: float = P(0.35, symbol=r"n", desc="pressure exponent", units="", dims="1", lo=0.1, hi=0.95, step=0.01,
                 group="Propellant")
    sigma_p: float = P(2.0, symbol=r"\sigma_p", desc="temperature sensitivity of the burn rate (x 1e-3)",
                       units="10⁻³ K⁻¹", dims="Θ⁻¹", lo=0.0, hi=6.0, step=0.05, group="Propellant")
    T_i: float = P(293.15, symbol=r"T_i", desc="initial grain temperature", units="K", dims="Θ", lo=233.0, hi=343.0,
                   step=0.5, group="Propellant")
    cstar: float = P(1500.0, symbol=r"c^*", desc="characteristic velocity", units="m·s⁻¹", dims="L T⁻¹", lo=1000.0,
                     hi=1800.0, step=5.0, group="Propellant")
    Tc: float = P(3000.0, symbol=r"T_c", desc="flame temperature", units="K", dims="Θ", lo=2000.0, hi=3800.0,
                  step=10.0, group="Propellant")
    gamma: float = P(1.2, symbol=r"\gamma", desc="ratio of specific heats", units="", dims="1", lo=1.1, hi=1.4,
                     step=0.01, group="Propellant")
    gamma_noz: float = P(0.0, symbol=r"\gamma_{2\phi}", desc="ratio of specific heats for the nozzle expansion "
                         "(two-phase value for metallized or KN propellants; 0 = same as gamma)", units="", dims="1",
                         lo=0.0, hi=1.4, step=0.001, group="Propellant")
    law: str = C("single", choices=LAWS, labels=LAW_LABELS, symbol=r"r(p)",
                 desc="burn-rate law: a and n above, your own pressure bands, or a library pressure-band law "
                 "(the bands replace a and n)", group="Propellant")
    bands_json: str = T("", label="pressure bands", desc="burn-rate law by pressure range: p_lo, p_hi (MPa), a' "
                        "(mm/s) and n per band (used when the law is 'Pressure bands')", editor="bands",
                        group="Propellant")

    def prop(self, alpha: float = 6.0, beta: float = 53.0) -> sol.SolidProp:
        return sol.SolidProp(rho_p=self.rho_p, a=sol.a_from_mm_MPa(self.a_mm, self.n), n=self.n,
                             sigma_p=self.sigma_p * 1e-3, Ti=self.T_i, cstar=self.cstar, Tc=self.Tc, gamma=self.gamma,
                             alpha=alpha * 1e-6, beta=beta, bands=law_bands(self.law, self.bands_json),
                             gamma_noz=self.gamma_noz)


class SolidBallisticsParams(SolidGrainFields, SolidPropFields):
    alpha: float = P(6.0, symbol=r"\alpha", desc="Lenoir–Robillard erosive coefficient (x 1e-6, 0 = off)",
                     units="10⁻⁶ SI", dims="–", lo=0.0, hi=30.0, step=0.1, group="Erosive burning")
    beta: float = P(53.0, symbol=r"\beta", desc="Lenoir–Robillard exponent constant", units="", dims="1", lo=20.0,
                    hi=100.0, step=1.0, group="Erosive burning")
    stack_json: str = T("", label="grain stack", desc="optional stack of grains, each with its own length, section, "
                        "propellant and end-face inhibition (empty = one grain from the parameters above)",
                        editor="stack", group="Grain")
    D_t: float = P(36.0, symbol=r"D_t", desc="throat diameter", units="mm", dims="L", lo=10.0, hi=120.0, step=0.5,
                   group="Nozzle")
    eps: float = P(8.0, symbol=r"\varepsilon=\frac{A_e}{A_t}", desc="nozzle expansion ratio", units="", dims="1",
                   lo=2.0, hi=40.0, step=0.1, group="Nozzle")
    eta_c: float = P(1.0, symbol=r"\eta_{c^*}", desc="c* efficiency (delivered / ideal)", units="", dims="1", lo=0.7,
                     hi=1.0, step=0.005, group="Nozzle")
    alpha_div: float = P(0.0, symbol=r"\alpha_{div}", desc="divergence half-angle of a conical nozzle (0 = no "
                         "divergence loss)", units="deg", dims="1", lo=0.0, hi=30.0, step=0.5, group="Nozzle")
    eta_F: float = P(1.0, symbol=r"\eta_{C_F}", desc="other thrust-coefficient losses (friction, two-phase lag)",
                     units="", dims="1", lo=0.8, hi=1.0, step=0.005, group="Nozzle")
    p_amb: float = P(101.325, symbol=r"p_a", desc="ambient pressure (0 = vacuum)", units="kPa", dims="M L⁻¹ T⁻²",
                     lo=0.0, hi=101.325, step=0.5, group="Nozzle")
    edot_t: float = P(0.0, symbol=r"\dot D_t", desc="throat-diameter growth rate at the reference pressure (0 = no "
                      "erosion)", units="mm·s⁻¹", dims="L T⁻¹", lo=0.0, hi=2.0, step=0.005, group="Throat erosion")
    p_ref_t: float = P(5.0, symbol=r"p_{ref}", desc="reference pressure of the erosion law", units="MPa",
                       dims="M L⁻¹ T⁻²", lo=0.5, hi=20.0, step=0.1, group="Throat erosion")
    k_t: float = P(0.8, symbol=r"k", desc="pressure exponent of the erosion law", units="", dims="1", lo=0.0, hi=2.0,
                   step=0.05, group="Throat erosion")
    m_ig: float = P(50.0, symbol=r"m_{ig}", desc="igniter gas mass", units="g", dims="M", lo=5.0, hi=300.0, step=1.0,
                    group="Ignition")
    t_ig: float = P(0.1, symbol=r"t_{ig}", desc="igniter burn time", units="s", dims="T", lo=0.02, hi=0.5, step=0.01,
                    group="Ignition")
    v_fs: float = P(150.0, symbol=r"v_{fs}", desc="flame-spreading speed along the port", units="m·s⁻¹",
                    dims="L T⁻¹", lo=5.0, hi=1000.0, step=5.0, group="Ignition")
    N: float = P(12.0, symbol=r"N_x", desc="axial cells (1 = 0-D ballistics)", units="", dims="1", lo=1.0, hi=40.0,
                 step=1.0, group="Simulation")
    t_end: float = P(7.0, symbol=r"t_{end}", desc="simulated time (includes tail-off)", units="s", dims="T", lo=1.0,
                     hi=40.0, step=0.5, group="Simulation")

    def _grain_section(self, sec: dict | None) -> sol.SectionTable:
        if sec is None:
            return self.section_table()
        if "shapes" in sec:
            return self.section_table(gb.cached_section(replace(gb.spec_from_dict(sec), R_out=self.R_out).to_json()))
        kw = {"R_p": self.R_p * 1.0, "n": int(round(self.n_el)), "w": self.w_el, "r_tip": self.r_tip,
              "r_valley": self.r_valley, "depth": self.depth}
        kw.update({k: v for k, v in sec.items() if k in kw})
        kind = sec.get("kind", self.kind)
        if kind not in s2.KINDS:
            raise ValueError(f"grain section kind {kind!r} is not one of {s2.KINDS}")
        return self.section_table(_cached_section(kind, self.R_out, kw["R_p"], kw["n"], kw["w"], kw["r_tip"],
                                                  kw["r_valley"], kw["depth"]))

    def _grain_prop(self, over: dict) -> sol.SolidProp:
        f = {k: getattr(self, k) for k in gstack.PROP_KEYS if hasattr(self, k)}
        f["alpha"], f["beta"] = self.alpha, self.beta
        f.update(over)
        return sol.SolidProp(rho_p=f["rho_p"], a=sol.a_from_mm_MPa(f["a_mm"], f["n"]), n=f["n"],
                             sigma_p=f["sigma_p"] * 1e-3, Ti=f["T_i"], cstar=f["cstar"], Tc=f["Tc"],
                             gamma=f["gamma"], alpha=f["alpha"] * 1e-6, beta=f["beta"],
                             bands=law_bands(f["law"], self.bands_json), gamma_noz=f["gamma_noz"])

    def to_si(self) -> sol.SolidMotor:
        kw = dict(At=np.pi * (self.D_t * 1e-3) ** 2 / 4.0, eps=self.eps, N=int(round(self.N)),
                  m_ig=self.m_ig * 1e-3, t_ig=self.t_ig, v_fs=self.v_fs, R_case=self.R_out * 1e-3,
                  eta_c=self.eta_c, alpha_div=self.alpha_div, eta_F=self.eta_F, p_amb=self.p_amb * 1e3,
                  edot_t=self.edot_t * 1e-3, p_ref_t=self.p_ref_t * 1e6, k_t=self.k_t)
        defs = gstack.parse_stack(self.stack_json)
        if not defs:
            nseg = int(round(self.n_seg))
            if nseg == 1 and self.faces == 0 and not self.portless:
                return sol.SolidMotor(prop=self.prop(self.alpha, self.beta), table=self.section_table(), L=self.L_g,
                                      **kw)
            h, a = self.end_flags()
            g = sol.Grain(self.prop(self.alpha, self.beta), self.section_table(), self.L_g, h, a)
            return sol.SolidMotor.from_grains([g] * nseg, **kw)
        grains = [sol.Grain(self._grain_prop(dict(d.prop)), self._grain_section(d.section), d.L, d.burn_head,
                            d.burn_aft) for d in defs]
        return sol.SolidMotor.from_grains(grains, **kw)


class SolidBallistics(Simulation):
    N_FRAMES = 601

    def run(self, params: SolidBallisticsParams, progress, cancel) -> Result:
        m = params.to_si()
        ch = sol.integrate(m, params.t_end, self.N_FRAMES, progress, cancel)
        t = np.linspace(0.0, params.t_end, self.N_FRAMES)
        meta = sol.summary(m, t, ch)
        meta["x_cells"] = m.cell_x().tolist()
        meta["L_total"] = m.L
        meta["gamma_mean"] = m.prop.gamma
        law = (f"a' = {params.a_mm:.2f} (mm/s), n = {m.prop.n:.2f}" if not m.prop.bands else
               f"{len(m.prop.bands)} pressure bands ({LAW_LABELS[LAWS.index(params.law)]})")
        setup = (f"setup solid ballistics: {params.kind} section, {len(m.stack)} grain(s), web "
                 f"{meta['web_mm']:.1f} (mm), {m.nc} axial cell(s), "
                 f"{'erosive burning on' if m.prop.alpha > 0 else 'erosive burning off'}")
        eq = (f"K_n {meta['Kn_0']:.0f} at ignition, {meta['Kn_max']:.0f} max  ->  equilibrium P_c "
              f"{meta['Pc_eq_MPa']:.2f} (MPa)")
        throat = f", throat {m.Dt * 1e3:.2f} -> {meta['Dt_end_mm']:.2f} (mm)" if m.edot_t > 0 else ""
        nozzle = (f"nozzle: c* efficiency {m.eta_c:.3f}, divergence {m.alpha_div:.1f} deg, C_F efficiency "
                  f"{m.eta_F:.3f}, ambient {m.p_amb / 1e3:.1f} (kPa){throat}")
        port = (f"port-to-throat area {meta['port_throat_0']:.2f} at the aft end" if meta["port_throat_0"] > 0
                else "no port (end burner)")
        checks = (f"design checks: {port}, peak core mass flux {meta['G_max']:.0f} (kg/m²/s), volumetric loading "
                  f"{100 * meta['vol_loading']:.0f} %, L/D {meta['L_over_D']:.1f}")
        perf = (f"{meta['impulse_class']} motor: I_t {meta['I_t_Ns']:.0f} (N·s), Isp {meta['Isp_avg']:.1f} (s), "
                f"propellant {meta['m_prop'] - m.m_ig:.3f} of {meta['m_loaded']:.3f} (kg) burned")
        log = [setup, f"propellant: {m.prop.rho_p:.0f} (kg/m³), {law} — {meta['note']}", eq, nozzle, checks, perf]
        if meta["port_choked"]:
            log.append("WARNING aft port area below the throat area: the port chokes and this 1-D model is not valid")
        if meta["Pc_max_MPa"] > 1.6 * meta["Pc_eq_MPa"]:
            log.append("WARNING pressure peak above 1.6 x equilibrium: check ignition transient / K_n progression")
        if meta["erosive_peak"] > 1.05:
            log.append(f"erosive burning raises the aft burn rate up to {100 * (meta['erosive_peak'] - 1):.0f} %")
        log.append(f"t = {params.t_end:.1f} (s), {self.N_FRAMES} frames; burn time {meta['t_burn']:.2f} (s)")
        return Result(t, ch, meta, log)


class Solid3DParams(SolidGrainFields):
    cells: float = P(64.0, symbol=r"n_c", desc="cells across the case diameter", units="", dims="1", lo=24.0, hi=128.0,
                     step=4.0, group="Level set")
    fin_frac: float = P(1.0, symbol=r"f_{fin}", desc="finocyl only: fraction of the length (from the head end) that "
                        "carries the fins (< 1 makes a genuinely 3-D grain)", units="", dims="1", lo=0.2, hi=1.0,
                        step=0.05, group="Level set")
    r_nom: float = P(10.0, symbol=r"r_{nom}", desc="nominal burn rate used to convert burn distance to animation "
                     "time", units="mm·s⁻¹", dims="L T⁻¹", lo=1.0, hi=50.0, step=0.5, group="Level set")


class SolidBurnback3D(Simulation):
    """Voxel level-set burnback to burn-through; tables vs burn distance and 3-D snapshots for playback."""

    N_FRAMES = 31

    def run(self, params: Solid3DParams, progress, cancel) -> Result:
        Ro, L = params.R_out, params.L_g * 1e3                     # mm
        h = 2.0 * Ro / params.cells
        kw = params.sdf_kw()
        partial = params.kind == "finocyl" and params.fin_frac < 0.999
        if params.kind == "custom":
            phi, act, _ = gb.phi_3d(params.custom_spec(), L, h)
        elif partial:
            phi, act, _ = ls.phi_partial_fins(Ro, L, h, params.R_p, kw["n"], params.w_el, params.depth,
                                              params.fin_frac)
        elif params.portless:
            X, _, _, act = ls.box_grid(Ro, L, h)
            phi = np.full(X.shape, 1e3, np.float64)
        else:
            phi, act, _ = ls.phi_extruded(lambda Y, Z: s2.sdf_field(params.kind, Y, Z, Ro, **kw), Ro, L, h)
        burn_h, burn_a = params.end_flags()
        if burn_h or burn_a:                                      # burning end faces: the gas lies beyond the face
            X = ls.box_grid(Ro, L, h)[0]
            if burn_h:
                phi = np.minimum(phi, X)
            if burn_a:
                phi = np.minimum(phi, L - X)
        b = params.section()
        tab_si = params.section_table()
        web3 = sol.segment_web(tab_si, params.L_g, params.faces) * 1e3
        lv = ls.LevelSet(phi, act, h)
        stride = max(1, int(round(max(phi.shape) / 64)))
        snaps: list[np.ndarray] = []
        tab = lv.run(web3 * 1.02, self.N_FRAMES, progress=progress, cancel=cancel,
                     on_sample=lambda s: snaps.append(s.phi[::stride, ::stride, ::stride].astype(np.float16)))
        w = tab["w"]
        n = len(w)
        t = w / params.r_nom
        A_b, V_g = tab["A_b"] * 1e-2, tab["V_gas"] * 1e-3                      # cm², cm³
        V_case = b.A_case * L * 1e-3
        mid = tab["A_port"].shape[1] // 2
        ch = {"w": w, "A_b": A_b, "V_gas": V_g, "A_port_mid": tab["A_port"][:, mid] * 1e-2,
              "A_b_section": sol.segment_burn_area(tab_si, params.L_g, w * 1e-3, 1, params.faces) * 1e4,
              "V_gas_section": (tab_si.A_case * 1e6 * L - (tab_si.A_case * 1e6 - tab_si.area(w * 1e-3) * 1e6)
                                * np.maximum(L - params.faces * w, 0.0)) * 1e-3,
              "propellant_left": np.clip(1.0 - (V_g - V_g[0]) / max(V_case - V_g[0], 1e-9), 0.0, 1.0)}
        A_max = float(A_b.max())
        done = np.where(A_b < 0.05 * A_max)[0]
        i_end = int(done[done > int(np.argmax(A_b))][0]) if (done > int(np.argmax(A_b))).any() else n - 1
        sliver = float(ch["propellant_left"][i_end])
        meta = {"web_mm": web3, "A_b0_cm2": float(A_b[0]), "A_b_max_cm2": A_max, "sliver_frac": sliver,
                "progressivity": float(A_max / A_b[0]) if A_b[0] > 0 else float("nan"), "cells": list(phi.shape),
                "h_mm": h, "snap_stride": stride, "partial_fins": bool(partial), "r_nom": params.r_nom,
                "R_out": Ro, "L": L}
        log = [f"setup 3-D level set: grid {phi.shape[0]}x{phi.shape[1]}x{phi.shape[2]} (h = {h:.2f} mm), "
               f"{'Numba' if ls.HAVE_NUMBA else 'NumPy'} kernels",
               f"Godunov upwind, CFL 0.5, unit-speed burnback to w = {b.web * 1.02:.1f} (mm)",
               f"A_b0 = {A_b[0]:.0f} (cm²), peak {A_max:.0f} (cm²), sliver {100 * sliver:.1f} % of the propellant"]
        if not partial and not params.portless:
            ok = (w <= 0.8 * b.web) & (ch["A_b_section"] > 0)
            err = float(np.mean(np.abs(A_b[ok] / ch["A_b_section"][ok] - 1)))
            meta["section_dev_mean"] = err
            log.append(f"cross-check vs L x perimeter of the 2-D section over the first 80 % of the web: mean "
                       f"deviation {100 * err:.1f} % (grid-limited; larger where the front meets the case wall)")
        log.append(f"{n} frames; animation time = w / {params.r_nom:.1f} (mm/s)")
        arrays = {"phi_snaps": np.stack(snaps), "snap_h": np.array([h * stride])}
        return Result(t, ch, meta, log, arrays)
