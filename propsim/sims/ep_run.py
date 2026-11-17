"""Run drivers for electric propulsion: gridded-ion beamlet optics (particles + CEX), Hall discharge (1-D fluid),
low-thrust Edelbaum spiral with power degradation."""
from __future__ import annotations

import numpy as np

from ..core.ep import AMU, E, PROPELLANTS, R_EARTH, ion_thrust
from ..core.ep import hall as hl
from ..core.ep import mission as mi
from ..core.ep.optics import Optics, OpticsGeom, perveance_max
from ..core.result import Result
from ..core.schema import C, P, ParamModel
from .base import Simulation

PROP_CHOICES = ["Xe", "Kr", "Ar"]
PROP_LABELS = ["Xenon", "Krypton", "Argon"]


# ====================================================================================== ion optics
class IonOpticsParams(ParamModel):
    prop: str = C("Xe", choices=PROP_CHOICES, labels=PROP_LABELS, symbol=r"\mathrm{propellant}",
                  desc="ion species (singly charged)", group="Propellant")
    V_s: float = P(1100.0, symbol=r"V_s", desc="screen-grid (discharge plasma) potential", units="V",
                   dims="M L² T⁻³ I⁻¹", lo=300.0, hi=3000.0, step=10.0, group="Grid potentials")
    V_a: float = P(-180.0, symbol=r"V_a", desc="accelerator-grid potential (negative to stop electron backstreaming)",
                   units="V", dims="M L² T⁻³ I⁻¹", lo=-600.0, hi=-20.0, step=5.0, group="Grid potentials")
    r_s: float = P(0.95, symbol=r"r_s", desc="screen aperture radius", units="mm", dims="L", lo=0.4, hi=1.2,
                   step=0.01, group="Aperture geometry")
    r_a: float = P(0.57, symbol=r"r_a", desc="accelerator aperture radius", units="mm", dims="L", lo=0.25, hi=1.1,
                   step=0.01, group="Aperture geometry")
    gap: float = P(0.66, symbol=r"l_g", desc="screen to accelerator gap", units="mm", dims="L", lo=0.3, hi=1.5,
                   step=0.01, group="Aperture geometry")
    t_s: float = P(0.38, symbol=r"t_s", desc="screen-grid thickness", units="mm", dims="L", lo=0.2, hi=1.0, step=0.01,
                   group="Aperture geometry")
    t_a: float = P(0.50, symbol=r"t_a", desc="accelerator-grid thickness", units="mm", dims="L", lo=0.2, hi=1.5,
                   step=0.01, group="Aperture geometry")
    pitch: float = P(2.5, symbol=r"l_{cc}", desc="hole pitch (centre to centre)", units="mm", dims="L", lo=1.2, hi=4.0,
                     step=0.05, group="Aperture geometry")
    frac: float = P(0.5, symbol=r"\frac{I_b}{I_{max}}", desc="beamlet current as a fraction of the perveance limit",
                    units="", dims="1", lo=0.1, hi=1.2, step=0.01, group="Operating point")
    Te: float = P(5.0, symbol=r"T_e", desc="discharge electron temperature (Bohm speed)", units="eV",
                  dims="M L² T⁻²", lo=1.0, hi=10.0, step=0.1, group="Operating point")
    n0: float = P(1.0, symbol=r"n_0", desc="neutral density downstream of the grids (x 1e18)", units="m⁻³", dims="L⁻³",
                  lo=0.05, hi=10.0, step=0.05, group="Charge exchange")
    sweep: str = C("on", choices=["on", "off"], labels=["Perveance sweep on (focusing trend)", "Operating point only"],
                   symbol=r"\mathrm{sweep}", desc="also solve a coarse perveance sweep for the focusing curve",
                   group="Simulation")

    def geom(self, n: int = 48) -> OpticsGeom:
        mm = 1e-3
        return OpticsGeom(r_s=self.r_s * mm, t_s=self.t_s * mm, gap=self.gap * mm, r_a=min(self.r_a, self.r_s) * mm,
                          t_a=self.t_a * mm, R_cell=max(0.5 * self.pitch, self.r_s + 0.15) * mm, V_s=self.V_s,
                          V_a=self.V_a, Te=self.Te, M_amu=PROPELLANTS[self.prop], n=n)


class IonOptics(Simulation):
    N_FRAMES = 241
    SWEEP = (0.2, 0.45, 0.7, 0.95)

    def run(self, p: IonOpticsParams, progress, cancel) -> Result:
        g = p.geom(48)
        Vt = p.V_s - p.V_a
        Pmax = perveance_max(g)
        I_max = Pmax * Vt**1.5
        I_b = p.frac * I_max
        progress(0.02)
        opt = Optics(g)
        tr = opt.run(I_b, iters=10)
        if cancel.is_set():
            raise InterruptedError("cancelled")
        progress(0.45)
        cx = opt.cex(tr["phi"], I_b, p.n0 * 1e18)
        progress(0.55)
        sweep_f, sweep_th, sweep_imp, sweep_ok = [], [], [], []
        if p.sweep == "on":
            gc_ = p.geom(32)
            o2 = Optics(gc_)
            Imax2 = perveance_max(gc_) * Vt**1.5
            for k, f in enumerate(self.SWEEP):
                if cancel.is_set():
                    raise InterruptedError("cancelled")
                t2 = o2.run(f * Imax2, iters=6)
                sweep_f.append(f)
                sweep_th.append(np.degrees(t2["theta_med"]))
                sweep_imp.append(t2["impinge"])
                sweep_ok.append(t2["converged"])
                progress(0.55 + 0.43 * (k + 1) / len(self.SWEEP))
        # ion flight animation: tracked rays sampled at a uniform time step
        tracks = tr["tracks"]
        n_s = max(len(t) for t in tracks)
        T = np.full((len(tracks), n_s, 2), np.nan)
        for i, t in enumerate(tracks):
            T[i, :len(t)] = t
            T[i, len(t):] = t[-1]                                    # parked where it left / hit
        dt = tr["dt_track"]
        t_full = np.arange(n_s) * dt
        t = np.linspace(0.0, t_full[-1], self.N_FRAMES)
        idx = np.minimum(np.searchsorted(t_full, t), n_s - 1)
        zf = T[:, idx, 0]                                            # (rays, frames), m
        rf = T[:, idx, 1]
        phi_at = np.array([opt._interp(tr["phi"], zf[:, j], rf[:, j]) for j in range(self.N_FRAMES)]).T
        ke = (opt.phi_fix[0, 0] - phi_at)                            # eV gained (singly charged)
        z_exit = g.z_a0 + g.t_a
        ch = {"z_mean": np.nanmean(zf, axis=0) * 1e3, "ke": np.nanmean(ke, axis=0),
              "passed": np.mean(zf >= z_exit, axis=0), "r_max": np.nanmax(rf, axis=0) * 1e3}
        M = g.M_amu * AMU
        Vb = opt.phi_fix[0, 0]                                       # net beam voltage w.r.t. the neutralized beam
        F_b = float(ion_thrust(I_b, Vb, M)) * np.cos(tr["theta_med"])
        meta = dict(I_b=I_b, I_max=I_max, P_max=Pmax, theta=np.degrees(tr["theta_med"]), impinge=tr["impinge"], screen=tr["screen"],
                    converged=tr["converged"], cex_back=cx["frac_back"], I_cex=cx["I_cex"], I_accel=cx["I_accel"],
                    F_b=F_b, R=p.V_s / Vt, z_exit=z_exit, R_cell=g.R_cell, t_unit="ns")
        arrays = dict(z_tr=zf, r_tr=rf, phi=tr["phi"], grid_z=opt.z, grid_r=opt.r, sweep_f=np.array(sweep_f),
                      sweep_th=np.array(sweep_th), sweep_imp=np.array(sweep_imp), sweep_ok=np.array(sweep_ok, float),
                      theta_hist=np.degrees(np.array(tr["theta_hist"])))
        log = [rf"setup beamlet: {PROP_LABELS[PROP_CHOICES.index(p.prop)].lower()}, $V_s$ {p.V_s:.0f} (V), $V_a$ {p.V_a:.0f} (V), "
               rf"net-to-total ratio $R$ = {meta['R']:.2f}",
               rf"perveance limit {Pmax * 1e9:.2f} ($\mathrm{{nA{{\cdot}}V^{{-3/2}}}}$) $\rightarrow$ $I_{{max}}$ "
               rf"{I_max * 1e3:.3f} (mA); running at {100 * p.frac:.0f} (%)",
               rf"beamlet: rms divergence {meta['theta']:.2f} (deg), accelerator-grid impingement "
               rf"{100 * tr['impinge']:.1f} (%), screen interception {100 * tr['screen']:.1f} (%), thrust "
               rf"{F_b * 1e6:.1f} ($\mathrm{{\mu N}}$)"]
        if not tr["converged"]:
            log.append("WARNING the space-charge iteration did not settle (near or above the perveance limit): "
                       "the beamlet is crossing over / impinging and the divergence is not a steady value")
        if p.frac < 0.25:
            log.append("under-perveance: the sheath is too flat and the beamlet over-focuses (cross-over impingement risk)")
        log.append(f"charge exchange: {100 * cx['frac_back']:.0f} (%) of the CEX ions return to the accelerator grid, "
                   f"accel current {cx['I_accel'] * 1e9:.2f} (nA) (erosion driver)")
        log.append("emission plane is fixed (no plasma-meniscus solve): read the focusing curve as a trend")
        return Result(t * 1e9, ch, meta, log, arrays)


# ====================================================================================== Hall discharge
class HallRunParams(ParamModel):
    Vd: float = P(300.0, symbol=r"V_d", desc="discharge voltage", units="V", dims="M L² T⁻³ I⁻¹", lo=150.0, hi=800.0,
                  step=5.0, group="Operating point")
    mdot: float = P(5.0, symbol=r"\dot m_a", desc="anode xenon flow", units="mg·s⁻¹", dims="M T⁻¹", lo=1.0, hi=20.0,
                    step=0.1, group="Operating point")
    L: float = P(25.0, symbol=r"L_{ch}", desc="channel length", units="mm", dims="L", lo=10.0, hi=50.0, step=0.5,
                 group="Channel")
    A: float = P(40.0, symbol=r"A_{ch}", desc="channel annulus area", units="cm²", dims="L²", lo=5.0, hi=200.0, step=1.0,
                 group="Channel")
    B_x: float = P(22.0, symbol=r"x_B", desc="location of the radial-field peak", units="mm", dims="L", lo=8.0, hi=40.0,
                   step=0.5, group="Magnetic field")
    B_w: float = P(6.0, symbol=r"w_B", desc="width of the radial-field peak", units="mm", dims="L", lo=2.0, hi=15.0,
                   step=0.1, group="Magnetic field")
    Te_max: float = P(25.0, symbol=r"T_{e,max}", desc="peak electron temperature", units="eV", dims="M L² T⁻²", lo=8.0,
                      hi=60.0, step=0.5, group="Electrons")
    Te_x: float = P(14.0, symbol=r"x_{iz}", desc="ionization-zone (electron-temperature peak) location", units="mm",
                    dims="L", lo=4.0, hi=40.0, step=0.5, group="Electrons")
    t_end: float = P(1.5, symbol=r"t_{end}", desc="simulated time", units="ms", dims="T", lo=0.3, hi=4.0, step=0.1,
                     group="Simulation")

    def hall(self) -> hl.HallParams:
        mm = 1e-3
        return hl.HallParams(L=self.L * mm, A=self.A * 1e-4, mdot=self.mdot * 1e-6, Vd=self.Vd, Te_max=self.Te_max,
                             B_peak_x=self.B_x * mm, B_width=self.B_w * mm, Te_peak_x=self.Te_x * mm,
                             Te_width=0.24 * self.L * mm)


class HallDischarge(Simulation):
    N_FRAMES = 601

    def run(self, p: HallRunParams, progress, cancel) -> Result:
        hp = p.hall()
        o = hl.simulate(hp, p.t_end * 1e-3, self.N_FRAMES, progress=progress, cancel=cancel)
        f_fife = hl.fife_frequency(hp)
        t = o["t"]
        M = hp.M_amu * AMU
        ib_ss = float(o["Ib"][-1])
        eta_m = ib_ss * M / (E * hp.mdot)
        n_avg_i = np.trapezoid(o["ni"], o["x"], axis=1) / o["x"][-1]
        n_avg_n = np.trapezoid(o["nn"], o["x"], axis=1) / o["x"][-1]
        ch = {"Iiz": o["Id"], "Ib": o["Ib"], "ni_avg": n_avg_i / 1e17, "nn_avg": n_avg_n / 1e17}
        tail = o["Id"][t >= 0.6 * t[-1]]
        ripple = float((tail.max() - tail.min()) / max(tail.mean(), 1e-12))
        mb = (o["mass_in"] - o["mass_out"] - o["mass_stored"]) / max(o["mass_in"], 1e-30)
        F = ib_ss * np.sqrt(2 * M * 0.9 * hp.Vd / E)
        meta = dict(f_fife=f_fife, Iiz_ss=float(o["Id"][-1]), Ib_ss=ib_ss, eta_m=eta_m, ripple=ripple, mass_err=mb,
                    F=F, P_beam=hp.Vd * ib_ss, t_unit="ms")
        arrays = dict(x=o["x"], nn=o["nn"], ni=o["ni"], phi=o["phi"], vi=o["vi"], Te=o["Te"], Ef=o["Ef"], B=o["B"])
        log = [rf"setup 1-D discharge: $V_d$ {hp.Vd:.0f} (V), anode flow {p.mdot:.1f} (mg·s⁻¹), channel {p.L:.0f} (mm), "
               rf"{hp.N} cells",
               rf"settled: ion production current {meta['Iiz_ss']:.2f} (A), beam current {ib_ss:.2f} (A), utilization "
               rf"{100 * min(eta_m, 9.99):.1f} (%), relative mass-balance error {abs(mb):.1e}",
               "no wall recombination or neutral backflow is modelled, so the utilization approaches 100 % (measured "
               "thrusters: 85 to 95 %); the electron current to the anode is not modelled",
               f"late-time current ripple {100 * ripple:.2f} (%) of the mean: the reduced model (fixed field and electron "
               f"temperature profiles) is overdamped",
               f"breathing mode NOT reproduced by this model; predator-prey (Fife) estimate {f_fife / 1e3:.1f} (kHz)"]
        if abs(mb) > 0.01:
            log.append("WARNING mass balance error above 1 %: lengthen the simulated time (transient not settled)")
        return Result(t * 1e3, ch, meta, log, arrays)


# ====================================================================================== low-thrust spiral
TARGETS = {"geo": 42164.137e3 - R_EARTH, "meo": 20200e3, "custom": None}


class SpiralParams(ParamModel):
    h0: float = P(400.0, symbol=r"h_0", desc="initial circular-orbit altitude", units="km", dims="L", lo=200.0,
                  hi=20000.0, step=10.0, group="Transfer")
    target: str = C("geo", choices=list(TARGETS), labels=["Geostationary orbit", "Medium Earth orbit (20200 km)",
                                                          "Custom altitude"], symbol=r"\mathrm{target}",
                    desc="final circular orbit", group="Transfer")
    h1: float = P(35786.0, symbol=r"h_1", desc="final altitude (custom target)", units="km", dims="L", lo=300.0,
                  hi=100000.0, step=10.0, group="Transfer")
    di: float = P(28.5, symbol=r"\Delta i", desc="inclination change", units="deg", dims="1", lo=0.0, hi=90.0,
                  step=0.1, group="Transfer")
    m0: float = P(2000.0, symbol=r"m_0", desc="initial spacecraft mass", units="kg", dims="M", lo=50.0, hi=20000.0,
                  step=10.0, group="Spacecraft")
    P0: float = P(10.0, symbol=r"P_0", desc="beginning-of-life electric power to the thrusters", units="kW",
                  dims="M L² T⁻³", lo=0.5, hi=200.0, step=0.5, group="Spacecraft")
    eta: float = P(0.6, symbol=r"\eta_T", desc="thruster total efficiency", units="", dims="1", lo=0.2, hi=0.8,
                   step=0.01, group="Thruster")
    isp: float = P(1800.0, symbol=r"I_{sp}", desc="specific impulse", units="s", dims="T", lo=300.0, hi=6000.0,
                   step=10.0, group="Thruster")
    degr: float = P(2.0, symbol=r"\delta", desc="solar-array power loss per year (radiation-belt dose)", units="%",
                    dims="1", lo=0.0, hi=30.0, step=0.1, group="Power")
    duty: float = P(95.0, symbol=r"f_{duty}", desc="thrusting duty cycle (eclipses, operations)", units="%", dims="1",
                    lo=30.0, hi=100.0, step=1.0, group="Power")

    def spiral(self) -> mi.Spiral:
        h1 = self.h1 * 1e3 if self.target == "custom" else TARGETS[self.target]
        return mi.Spiral(r0=R_EARTH + self.h0 * 1e3, r1=R_EARTH + h1, di=np.radians(self.di), m0=self.m0,
                         P0=self.P0 * 1e3, eta=self.eta, isp=self.isp, degr=self.degr / 100.0, duty=self.duty / 100.0)


class LowThrustSpiral(Simulation):
    N_FRAMES = 601

    def run(self, p: SpiralParams, progress, cancel) -> Result:
        s = p.spiral()
        o = mi.simulate(s, self.N_FRAMES)
        progress(0.6)
        t = o["t"] / 86400.0
        # revolutions: integrate the mean motion along the transfer
        n_mot = o["v"] / o["r"]
        theta = np.concatenate([[0.0], np.cumsum(0.5 * (n_mot[1:] + n_mot[:-1]) * np.diff(o["t"]))])
        ch = {"dv": o["dv"] / 1e3, "m": o["m"], "h": (o["r"] - R_EARTH) / 1e3, "inc": np.degrees(o["inc"]),
              "P": o["P"] / 1e3, "F": o["F"] * 1e3, "v": o["v"] / 1e3, "rev": theta / (2 * np.pi)}
        m_hi = s.m0 * (1 - np.exp(-o["dv_tot"] / (9.80665 * s.isp)))
        meta = dict(dv_tot=o["dv_tot"] / 1e3, days=o["t_end"] / 86400.0, m_prop=o["m_prop"], m_rocket=m_hi,
                    revs=float(theta[-1] / (2 * np.pi)), reached=o["reached"], F0=o["F"][0] * 1e3,
                    P_end=o["P"][-1] / 1e3, t_unit="days")
        progress(1.0)
        log = [f"setup Edelbaum spiral: {p.h0:.0f} (km) -> {(s.r1 - R_EARTH) / 1e3:.0f} (km), inclination change {p.di:.1f} (deg)",
               f"total Δv {meta['dv_tot']:.3f} (km·s⁻¹) (Edelbaum), initial thrust {meta['F0']:.0f} (mN), "
               f"{meta['revs']:.0f} revolutions",
               f"trip time {meta['days']:.0f} (days), propellant {meta['m_prop']:.0f} (kg) (rocket equation {m_hi:.0f} (kg))",
               f"power at arrival {meta['P_end']:.2f} (kW) after {p.degr:.1f} (%) per year degradation"]
        if not o["reached"]:
            log.append("WARNING the target was not reached within the integration window")
        return Result(t, ch, meta, log)
