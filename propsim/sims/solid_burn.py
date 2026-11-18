"""Run drivers for solid motors: internal ballistics (0-D / spatial, erosive, ignition, tail-off) and 3-D level-set burnback."""
from __future__ import annotations

from dataclasses import replace
from functools import lru_cache

import numpy as np

from ..core.ballistics import solid as sol
from ..core.grain import levelset as ls
from ..core.grain import builder as gb
from ..core.grain import section2d as s2
from ..core.grain import stack as gstack
from ..core.result import Result
from ..core.schema import C, P, ParamModel, T
from .base import Simulation

KIND_LABELS = ["Circular bore", "Star", "Wagon wheel (centre + slots)", "Finocyl (centre + fins)",
               "Centre + 4 bores", "Centre + 6 bores", "Custom (shape list)"]
KINDS = [*s2.KINDS, "custom"]


@lru_cache(maxsize=32)
def _cached_section(kind, R_out, R_p, n, w, r_tip, r_valley, depth) -> s2.Burnback:
    """Section burnback tables are reused across the many ``to_si`` calls of a session (about 20-50 ms each)."""
    return s2.section(kind, R_out, R_p=R_p, n=n, w=w, r_tip=r_tip, r_valley=r_valley, depth=depth)


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
    L_g: float = P(0.6, symbol=r"L_g", desc="grain length", units="m", dims="L", lo=0.2, hi=2.0, step=0.01,
                   group="Grain")

    section_json: str = T("", label="custom section", desc="custom section drawn as a list of shapes, with optional "
                          "inhibited surfaces (used when the section is custom)", editor="section", group="Grain")

    def custom_spec(self) -> gb.SectionSpec:
        """The custom section; an empty text means the editor's starting shape on this outer radius."""
        spec = gb.spec_from_json(self.section_json) if self.section_json.strip() else gb.default_spec(self.R_out)
        return replace(spec, R_out=self.R_out)               # the form's outer radius always wins

    def sdf_kw(self) -> dict:
        return dict(R_p=self.R_p, n=int(round(self.n_el)), w=self.w_el, r_tip=self.r_tip, r_valley=self.r_valley,
                    depth=self.depth)

    def section(self) -> s2.Burnback:
        if self.kind == "custom":
            return gb.cached_section(self.custom_spec().to_json())
        return _cached_section(self.kind, self.R_out, self.R_p, int(round(self.n_el)), self.w_el, self.r_tip,
                               self.r_valley, self.depth)

    def section_table(self, b: s2.Burnback | None = None) -> sol.SectionTable:
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

    def prop(self, alpha: float = 6.0, beta: float = 53.0) -> sol.SolidProp:
        return sol.SolidProp(rho_p=self.rho_p, a=sol.a_from_mm_MPa(self.a_mm, self.n), n=self.n,
                             sigma_p=self.sigma_p * 1e-3, Ti=self.T_i, cstar=self.cstar, Tc=self.Tc, gamma=self.gamma,
                             alpha=alpha * 1e-6, beta=beta)


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
                             gamma=f["gamma"], alpha=f["alpha"] * 1e-6, beta=f["beta"])

    def to_si(self) -> sol.SolidMotor:
        kw = dict(At=np.pi * (self.D_t * 1e-3) ** 2 / 4.0, eps=self.eps, N=int(round(self.N)),
                  m_ig=self.m_ig * 1e-3, t_ig=self.t_ig, v_fs=self.v_fs, R_case=self.R_out * 1e-3)
        defs = gstack.parse_stack(self.stack_json)
        if not defs:
            return sol.SolidMotor(prop=self.prop(self.alpha, self.beta), table=self.section_table(), L=self.L_g, **kw)
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
        meta["gamma_mean"] = m.prop.gamma
        log = [f"setup solid ballistics: {params.kind} section, web {meta['web_mm']:.1f} (mm), {m.nc} axial cell(s), "
               f"{'erosive burning on' if m.prop.alpha > 0 else 'erosive burning off'}",
               f"propellant: {m.prop.rho_p:.0f} (kg/m³), a' = {params.a_mm:.2f} (mm/s), n = {m.prop.n:.2f} — "
               f"{meta['note']}", f"K_n max {meta['Kn_max']:.0f}  ->  equilibrium P_c {meta['Pc_eq_MPa']:.2f} (MPa)"]
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
        else:
            phi, act, _ = ls.phi_extruded(lambda Y, Z: s2.sdf_field(params.kind, Y, Z, Ro, **kw), Ro, L, h)
        b = params.section()
        lv = ls.LevelSet(phi, act, h)
        stride = max(1, int(round(max(phi.shape) / 64)))
        snaps: list[np.ndarray] = []
        tab = lv.run(b.web * 1.02, self.N_FRAMES, progress=progress, cancel=cancel,
                     on_sample=lambda s: snaps.append(s.phi[::stride, ::stride, ::stride].astype(np.float16)))
        w = tab["w"]
        n = len(w)
        t = w / params.r_nom
        A_b, V_g = tab["A_b"] * 1e-2, tab["V_gas"] * 1e-3                      # cm², cm³
        V_case = b.A_case * L * 1e-3
        mid = tab["A_port"].shape[1] // 2
        ch = {"w": w, "A_b": A_b, "V_gas": V_g, "A_port_mid": tab["A_port"][:, mid] * 1e-2,
              "A_b_section": L * np.interp(w, b.d, b.perim) * 1e-2,              # L x perimeter (valid when extruded)
              "V_gas_section": L * np.interp(w, b.d, b.area) * 1e-3,
              "propellant_left": np.clip(1.0 - (V_g - V_g[0]) / max(V_case - V_g[0], 1e-9), 0.0, 1.0)}
        A_max = float(A_b.max())
        done = np.where(A_b < 0.05 * A_max)[0]
        i_end = int(done[done > int(np.argmax(A_b))][0]) if (done > int(np.argmax(A_b))).any() else n - 1
        sliver = float(ch["propellant_left"][i_end])
        meta = {"web_mm": b.web, "A_b0_cm2": float(A_b[0]), "A_b_max_cm2": A_max, "sliver_frac": sliver,
                "progressivity": float(A_max / A_b[0]) if A_b[0] > 0 else float("nan"), "cells": list(phi.shape),
                "h_mm": h, "snap_stride": stride, "partial_fins": bool(partial), "r_nom": params.r_nom,
                "R_out": Ro, "L": L}
        log = [f"setup 3-D level set: grid {phi.shape[0]}x{phi.shape[1]}x{phi.shape[2]} (h = {h:.2f} mm), "
               f"{'Numba' if ls.HAVE_NUMBA else 'NumPy'} kernels",
               f"Godunov upwind, CFL 0.5, unit-speed burnback to w = {b.web * 1.02:.1f} (mm)",
               f"A_b0 = {A_b[0]:.0f} (cm²), peak {A_max:.0f} (cm²), sliver {100 * sliver:.1f} % of the propellant"]
        if not partial:
            ok = (w <= 0.8 * b.web) & (ch["A_b_section"] > 0)
            err = float(np.mean(np.abs(A_b[ok] / ch["A_b_section"][ok] - 1)))
            meta["section_dev_mean"] = err
            log.append(f"cross-check vs L x perimeter of the 2-D section over the first 80 % of the web: mean "
                       f"deviation {100 * err:.1f} % (grid-limited; larger where the front meets the case wall)")
        log.append(f"{n} frames; animation time = w / {params.r_nom:.1f} (mm/s)")
        arrays = {"phi_snaps": np.stack(snaps), "snap_h": np.array([h * stride])}
        return Result(t, ch, meta, log, arrays)
