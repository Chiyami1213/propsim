"""Run driver for the vehicle module: 2-D point-mass ascent with staging, drag, wind and thrust profiles imported from
other modules' Run results (hybrid, solid, liquid)."""
from __future__ import annotations

import json

import numpy as np

from ..core.constants import G0
from ..core.result import Result
from ..core.schema import C, H, P, ParamModel
from ..core.trajectory import flight as fl
from ..core.trajectory import engines as lib_engines
from ..core.trajectory import profiles
from .base import Simulation

SRC_CHOICES = ["constant", "profile"]
SRC_LABELS = ["Constant thrust", "Imported thrust profile"]
DRAG_LABELS = ["Slender rocket", "Launch vehicle", "Blunt body"]
ENGINE_CHOICES, ENGINE_LABELS = lib_engines.choices()


class AscentParams(ParamModel):
    # ---------------------------------------------------------------- vehicle
    n_stages: str = C("2", choices=["1", "2", "3"], labels=["One stage", "Two stages", "Three stages"],
                      symbol=r"N_{st}", desc="number of stages", group="Vehicle")
    m_payload: float = P(100.0, symbol=r"m_{pl}", desc="payload mass", units="kg", dims="M", lo=0.0, hi=20000.0,
                         step=0.5, group="Vehicle")
    m_fair: float = P(50.0, symbol=r"m_{fair}", desc="payload fairing mass (jettisoned in flight)", units="kg",
                      dims="M", lo=0.0, hi=5000.0, step=0.5, group="Vehicle")
    h_fair: float = P(110.0, symbol=r"h_{fair}", desc="fairing jettison altitude", units="km", dims="L", lo=20.0,
                      hi=300.0, step=1.0, group="Vehicle")
    # ---------------------------------------------------------------- stage 1
    s1_dry: float = P(1100.0, symbol=r"m_{d,1}", desc="stage 1 dry mass", units="kg", dims="M", lo=1.0, hi=20000.0,
                      step=0.5, group="Stage 1")
    s1_prop: float = P(10000.0, symbol=r"m_{p,1}", desc="stage 1 propellant mass", units="kg", dims="M", lo=1.0,
                       hi=60000.0, step=1.0, group="Stage 1")
    s1_engine: str = C("custom", choices=ENGINE_CHOICES, labels=ENGINE_LABELS, symbol=r"\mathrm{engine}",
                        desc="fill this stage's thrust, specific impulse and exit area from a reference engine of the "
                        "component library (you can edit them afterwards)", group="Stage 1")
    s1_F: float = P(260.0, symbol=r"F_{vac,1}", desc="stage 1 vacuum thrust", units="kN", dims="M L T⁻²", lo=0.1,
                    hi=8000.0, step=0.1, group="Stage 1")
    s1_isp: float = P(300.0, symbol=r"I_{sp,1}", desc="stage 1 vacuum specific impulse (sets the mass flow)", units="s",
                      dims="T", lo=150.0, hi=480.0, step=0.1, group="Stage 1")
    s1_Ae: float = P(2000.0, symbol=r"A_{e,1}", desc="stage 1 nozzle exit area (back-pressure thrust loss)",
                     units="cm²", dims="L²", lo=0.0, hi=60000.0, step=10.0, group="Stage 1")
    s1_d: float = P(1.4, symbol=r"d_1", desc="stage 1 diameter (drag reference area)", units="m", dims="L", lo=0.05,
                    hi=6.0, step=0.01, group="Stage 1")
    s1_src: str = C("constant", choices=SRC_CHOICES, labels=SRC_LABELS, symbol=r"\mathrm{thrust\ source}",
                    desc="constant thrust or a profile sent from a hybrid, solid or liquid Run result (File menu)",
                    group="Stage 1")
    s1_coast: float = P(5.0, symbol=r"t_{c,1}", desc="coast from stage 1 burnout to stage 2 ignition", units="s",
                        dims="T", lo=0.0, hi=120.0, step=0.5, group="Stage 1")
    # ---------------------------------------------------------------- stage 2
    s2_dry: float = P(400.0, symbol=r"m_{d,2}", desc="stage 2 dry mass", units="kg", dims="M", lo=1.0, hi=20000.0,
                      step=0.5, group="Stage 2")
    s2_prop: float = P(3200.0, symbol=r"m_{p,2}", desc="stage 2 propellant mass", units="kg", dims="M", lo=1.0,
                       hi=60000.0, step=1.0, group="Stage 2")
    s2_engine: str = C("custom", choices=ENGINE_CHOICES, labels=ENGINE_LABELS, symbol=r"\mathrm{engine}",
                        desc="fill this stage's thrust, specific impulse and exit area from a reference engine of the "
                        "component library (you can edit them afterwards)", group="Stage 2")
    s2_F: float = P(45.0, symbol=r"F_{vac,2}", desc="stage 2 vacuum thrust", units="kN", dims="M L T⁻²", lo=0.1,
                    hi=8000.0, step=0.1, group="Stage 2")
    s2_isp: float = P(345.0, symbol=r"I_{sp,2}", desc="stage 2 vacuum specific impulse", units="s", dims="T", lo=150.0,
                      hi=480.0, step=0.1, group="Stage 2")
    s2_Ae: float = P(500.0, symbol=r"A_{e,2}", desc="stage 2 nozzle exit area", units="cm²", dims="L²", lo=0.0,
                     hi=60000.0, step=10.0, group="Stage 2")
    s2_d: float = P(1.0, symbol=r"d_2", desc="stage 2 diameter", units="m", dims="L", lo=0.05, hi=6.0, step=0.01,
                    group="Stage 2")
    s2_src: str = C("constant", choices=SRC_CHOICES, labels=SRC_LABELS, symbol=r"\mathrm{thrust\ source}",
                    desc="constant thrust or an imported profile", group="Stage 2")
    s2_coast: float = P(5.0, symbol=r"t_{c,2}", desc="coast from stage 2 burnout to stage 3 ignition", units="s",
                        dims="T", lo=0.0, hi=120.0, step=0.5, group="Stage 2")
    # ---------------------------------------------------------------- stage 3
    s3_dry: float = P(60.0, symbol=r"m_{d,3}", desc="stage 3 dry mass", units="kg", dims="M", lo=1.0, hi=20000.0,
                      step=0.5, group="Stage 3")
    s3_prop: float = P(400.0, symbol=r"m_{p,3}", desc="stage 3 propellant mass", units="kg", dims="M", lo=1.0,
                       hi=60000.0, step=1.0, group="Stage 3")
    s3_engine: str = C("custom", choices=ENGINE_CHOICES, labels=ENGINE_LABELS, symbol=r"\mathrm{engine}",
                        desc="fill this stage's thrust, specific impulse and exit area from a reference engine of the "
                        "component library (you can edit them afterwards)", group="Stage 3")
    s3_F: float = P(6.0, symbol=r"F_{vac,3}", desc="stage 3 vacuum thrust", units="kN", dims="M L T⁻²", lo=0.1,
                    hi=8000.0, step=0.1, group="Stage 3")
    s3_isp: float = P(330.0, symbol=r"I_{sp,3}", desc="stage 3 vacuum specific impulse", units="s", dims="T", lo=150.0,
                      hi=480.0, step=0.1, group="Stage 3")
    s3_Ae: float = P(100.0, symbol=r"A_{e,3}", desc="stage 3 nozzle exit area", units="cm²", dims="L²", lo=0.0,
                     hi=60000.0, step=10.0, group="Stage 3")
    s3_d: float = P(0.8, symbol=r"d_3", desc="stage 3 diameter", units="m", dims="L", lo=0.05, hi=6.0, step=0.01,
                    group="Stage 3")
    s3_src: str = C("constant", choices=SRC_CHOICES, labels=SRC_LABELS, symbol=r"\mathrm{thrust\ source}",
                    desc="constant thrust or an imported profile", group="Stage 3")
    # ---------------------------------------------------------------- launch and guidance
    elevation: float = P(90.0, symbol=r"\gamma_0", desc="launch rail elevation (90 = vertical)", units="deg", dims="1",
                         lo=30.0, hi=90.0, step=0.5, group="Launch")
    rail: float = P(20.0, symbol=r"l_r", desc="launch rail length (flight-path angle held fixed on the rail)", units="m",
                    dims="L", lo=0.0, hi=100.0, step=0.5, group="Launch")
    h_kick: float = P(250.0, symbol=r"h_{kick}", desc="pitch-kick altitude (0 = no kick; vertical launches only)",
                      units="m", dims="L", lo=0.0, hi=5000.0, step=10.0, group="Launch")
    kick: float = P(2.5, symbol=r"\delta_{kick}", desc="pitch-kick angle: instantaneous flight-path change", units="deg",
                    dims="1", lo=0.0, hi=15.0, step=0.1, group="Launch")
    guid: str = C("level", choices=["gt", "level"], labels=["Gravity turn (thrust along velocity)",
                                                              "Gravity turn, then level off"],
                  symbol=r"\mathrm{guidance}", desc="thrust steering after the pitch kick", group="Guidance")
    h_level: float = P(70.0, symbol=r"h_{lvl}", desc="altitude above which the thrust steers the flight path to the "
                       "horizontal (level-off guidance)", units="km", dims="L", lo=20.0, hi=250.0, step=1.0,
                       group="Guidance")
    alpha_max: float = P(25.0, symbol=r"\alpha_{max}", desc="largest thrust angle above the velocity vector", units="deg",
                         dims="1", lo=0.0, hi=60.0, step=1.0, group="Guidance")
    perigee_cut: float = P(200.0, symbol=r"h_{p,cut}", desc="last-stage cutoff when the orbit perigee altitude reaches "
                           "this value (0 = burn to depletion)", units="km", dims="L", lo=0.0, hi=1000.0, step=5.0,
                           group="Guidance")
    # ---------------------------------------------------------------- aerodynamics and wind
    drag: str = C("slender", choices=list(fl.DRAG_TABLES) + ["table"], labels=DRAG_LABELS + ["Table (profiles/drag.csv)"],
                  symbol=r"C_D(M)", desc="drag curve versus Mach number: generic shapes (typical values) or your own "
                  "table, a mach,cd file named drag.csv in the profiles folder of the data folder", group="Aerodynamics")
    cd_scale: float = P(1.0, symbol=r"k_D", desc="drag multiplier", units="", dims="1", lo=0.2, hi=3.0, step=0.01,
                        group="Aerodynamics")
    w_surf: float = P(0.0, symbol=r"w_0", desc="surface wind along the flight direction (tailwind positive)",
                      units="m·s⁻¹", dims="L T⁻¹", lo=-30.0, hi=30.0, step=0.5, group="Wind")
    w_jet: float = P(0.0, symbol=r"w_{jet}", desc="jet-stream wind speed added at its peak altitude (tailwind positive)",
                     units="m·s⁻¹", dims="L T⁻¹", lo=-100.0, hi=100.0, step=1.0, group="Wind")
    h_jet: float = P(11.0, symbol=r"h_{jet}", desc="jet-stream altitude", units="km", dims="L", lo=2.0, hi=30.0,
                     step=0.5, group="Wind")
    w_width: float = P(5.0, symbol=r"\Delta h_{jet}", desc="jet-stream half-width", units="km", dims="L", lo=1.0,
                       hi=20.0, step=0.5, group="Wind")
    t_max: float = P(1500.0, symbol=r"t_{end}", desc="simulated time (the flight ends earlier at ground impact)",
                     units="s", dims="T", lo=60.0, hi=20000.0, step=10.0, group="Simulation")
    # imported thrust profiles (JSON; filled by the app from File > Send Thrust to Vehicle, never shown as sliders)
    prof1: str = H("")
    prof2: str = H("")
    prof3: str = H("")
    drag_table: str = H("")

    # ------------------------------------------------------------------------------------------- build
    def n(self) -> int:
        return int(self.n_stages)

    def _profile(self, k: int):
        raw = getattr(self, f"prof{k}")
        if not raw:
            raise ValueError(f"stage {k} uses an imported thrust profile but none has been sent: run a hybrid, solid or "
                             f"liquid model, then File > Send Thrust to Vehicle > Stage {k}")
        return profiles.arrays(json.loads(raw))

    def vehicle(self, imported: bool = True) -> fl.Vehicle:
        """Vehicle for the flight; ``imported=False`` ignores thrust-profile sources (the Live sketch)."""
        stages = []
        for k in range(1, self.n() + 1):
            g = lambda name, k=k: getattr(self, f"s{k}_{name}")
            prof = self._profile(k) if (imported and g("src") == "profile") else None
            stages.append(fl.StageSpec(g("dry"), g("prop"), g("F") * 1e3, g("isp"), g("Ae") * 1e-4, g("d"),
                                       g("coast") if k < 3 else 0.0, prof))
        return fl.Vehicle(tuple(stages), self.m_payload, self.m_fair, self.h_fair * 1e3)

    def guidance(self) -> fl.Guidance:
        return fl.Guidance(elevation=np.radians(self.elevation), rail=self.rail, h_kick=self.h_kick,
                           kick=np.radians(self.kick), mode=self.guid, h_level=self.h_level * 1e3,
                           alpha_max=np.radians(self.alpha_max), perigee_cut=self.perigee_cut * 1e3)

    def aero(self) -> fl.Aero:
        if self.drag == "table":
            if not self.drag_table:
                raise ValueError("the drag source is a table but none was loaded: put drag.csv (columns mach,cd) in the "
                                 "profiles folder of the data folder (File > Open Data Folder)")
            d = json.loads(self.drag_table)
            return fl.Aero(tuple(d["mach"]), tuple(d["cd"]), self.cd_scale)
        m, c = fl.DRAG_TABLES[self.drag]
        return fl.Aero(tuple(m), tuple(c), self.cd_scale)

    def wind(self) -> fl.Wind:
        return fl.Wind(self.w_surf, self.w_jet, self.h_jet * 1e3, self.w_width * 1e3)


class PointMassAscent(Simulation):
    N_FRAMES = 601

    def run(self, p: AscentParams, progress, cancel) -> Result:
        veh = p.vehicle()
        f = fl.simulate(veh, p.guidance(), p.aero(), p.wind(), t_max=p.t_max, n_out=self.N_FRAMES, progress=progress,
                        cancel=cancel)
        m0 = fl.stack_mass(veh, 0)
        s0 = veh.stages[0]
        F0 = float(np.max(s0.profile[1])) if s0.profile is not None else fl.thrust(s0, 0.0, 101325.0)[0]
        dvs = fl.ideal_dv(veh)
        log = [rf"setup ascent: {p.n()} stage(s), lift-off mass {m0:.0f} (kg), peak thrust-to-weight {F0 / (m0 * G0):.2f}, "
               rf"ideal $\Delta v$ {sum(dvs) / 1e3:.2f} (km·s⁻¹) ({', '.join(f'{d / 1e3:.2f}' for d in dvs)})"]
        for k, s in enumerate(veh.stages, 1):
            if s.profile is not None:
                I, mp = profiles.impulse(dict(t=s.profile[0].tolist(), F=s.profile[1].tolist(),
                                              mdot=None if s.profile[2] is None else s.profile[2].tolist()))
                log.append(rf"stage {k} flies an imported thrust profile: burn time {s.profile[0][-1]:.2f} (s), peak "
                           rf"{s.profile[1].max() / 1e3:.2f} (kN), total impulse {I / 1e3:.1f} (kN·s)")
                if np.isfinite(mp) and mp > 1.02 * s.m_prop:
                    log.append(rf"WARNING the profile consumes {mp:.1f} (kg) but stage {k} carries {s.m_prop:.1f} (kg): "
                               rf"the burn ends when the propellant is gone")
        t, ch = f.t, f.ch
        for name, te in f.events:
            i = int(min(np.searchsorted(t, te), len(t) - 1))
            log.append(rf"{name}: $t$ = {te:.1f} (s), altitude {ch['h'][i]:.1f} (km), speed {ch['V'][i]:.0f} (m·s⁻¹), "
                       rf"Mach {ch['M'][i]:.2f}")
        m = f.meta
        if m["impact"]:
            log.append(rf"suborbital: apogee {m['apogee_km']:.1f} (km), range {m['range_km']:.1f} (km), flight time "
                       rf"{m['t_end']:.0f} (s)")
        elif m["orbit"]:
            log.append(rf"orbit at burnout: perigee {m['perigee_km']:.0f} (km) by apogee {m['apogee_orbit_km']:.0f} (km)")
        else:
            log.append(rf"end of the simulated time: altitude {ch['h'][-1]:.1f} (km), perigee {m['perigee_km']:.0f} (km)")
        log.append("point mass, no lift, non-rotating spherical Earth: the 2-D plane is the launch azimuth")
        meta = dict(m, events=[[n, float(te)] for n, te in f.events], ideal_dv=[float(d) for d in dvs],
                    m0=float(m0), twr0=float(F0 / (m0 * G0)))
        return Result(t, ch, meta, log)
