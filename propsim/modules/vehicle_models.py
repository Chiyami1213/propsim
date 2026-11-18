"""Vehicle module: staging and atmosphere budget (Live) and the 2-D point-mass ascent (Run)."""
from __future__ import annotations

import json

import numpy as np
import pyvista as pv

from ..core.constants import G0
from ..core.result import Curve, PlotSpec, TabSpec
from ..core.trajectory import flight as fl
from ..core.trajectory import profiles
from ..core.trajectory.atmosphere import atmosphere
from ..sims.vehicle_run import AscentParams, PointMassAscent
from .base import RUN_COLORS, ModelController
from .ep_models import polylines
from .geometry import broadcast_scalar, rev_grid, thetas

TT = r"$t$ (s)"
BULK_RHO = 1000.0                              # kg·m⁻³, only to size the stack sketch
EVENT_COLORS = {"Stage": "#ff9d3c", "Apogee": "#b07cff", "Max-Q": "#ff5a3c", "Mach": "#5cc97a", "Fairing": "#e6c34a",
                "Orbit": "#4ea1ff", "Impact": "#ff5a3c", "Pitch": "#8b94a3"}
EVENT_PLOT = {"Stage": "oxidizer", "Apogee": "pressure", "Max-Q": "hot", "Mach": "fuel", "Fairing": "text",
              "Orbit": "thrust", "Impact": "hot", "Pitch": "dim"}


def stage_preset_updates(p, prev: dict) -> dict:
    """A stage's engine preset was just chosen: fill its thrust, Isp and exit area from the component library."""
    from ..core.trajectory import engines as lib
    out, notes = {}, []
    for k in (1, 2, 3):
        name = f"s{k}_engine"
        now = getattr(p, name)
        if now != prev.get(name) and now != lib.CUSTOM:
            pr = lib.preset(now)
            out.update({f"s{k}_F": round(pr["F_vac"] / 1e3, 1), f"s{k}_isp": round(pr["isp"], 1),
                        f"s{k}_Ae": round(pr["A_e"] * 1e4, 0)})
            notes.append(f"stage {k} set from {pr['label']} (component library): thrust {pr['F_vac'] / 1e3:.1f} (kN), "
                         f"Isp {pr['isp']:.1f} (s), exit area {pr['A_e'] * 1e4:.0f} (cm²); {pr['note']}")
    if notes:
        out["_note"] = "\n".join(notes)
    return out


def _ev_key(name: str) -> str | None:
    return next((k for k in EVENT_PLOT if name.startswith(k)), None)


# ============================================================================================== staging (Live)
class VehicleLiveModel(ModelController):
    key = "vehicle.staging"
    title = "Vehicle & staging"
    module = "Vehicle"
    mode = "live"
    Params = AscentParams
    fields = [r"stage (–)"]

    def __init__(self) -> None:
        self._m: dict = {}

    preset_updates = staticmethod(lambda p, prev: stage_preset_updates(p, prev))

    def _build(self, p: AscentParams) -> fl.Vehicle:
        return p.vehicle(imported=False)

    def evaluate(self, p: AscentParams) -> dict:
        veh = self._build(p)
        st = veh.stages
        t, m, v, twr, stage = [0.0], [fl.stack_mass(veh, 0)], [0.0], [], [1]
        burn, mdots = [], []
        t_cur, v_cur = 0.0, 0.0
        for k, s in enumerate(st):
            m0 = fl.stack_mass(veh, k, fairing=(k == 0))
            mdot = s.F_vac / (G0 * s.isp)
            tb = s.m_prop / mdot
            burn.append(tb), mdots.append(mdot)
            if k > 0:                                              # the coast of the previous stage, then ignition
                t.append(t_cur), m.append(m0), v.append(v_cur), stage.append(k + 1)
            tt = np.linspace(0.0, tb, 40)
            mm = m0 - mdot * tt
            vv = v_cur + s.isp * G0 * np.log(m0 / mm)
            t.extend(t_cur + tt), m.extend(mm), v.extend(vv), stage.extend([k + 1] * len(tt))
            v_cur, t_cur = float(vv[-1]), t_cur + tb
            if k < len(st) - 1:
                t_cur += s.coast
                t.append(t_cur), m.append(fl.stack_mass(veh, k + 1, fairing=False)), v.append(v_cur)
                stage.append(k + 1)
        t, m, v = np.array(t), np.array(m), np.array(v)
        # thrust-to-weight on the pad / in vacuum
        F_of = np.array([st[s - 1].F_vac for s in stage])
        twr = F_of / (m * G0)
        F_sl = fl.thrust(st[0], 0.0, 101325.0)[0]
        h = np.linspace(0.0, 100e3, 201)
        a = atmosphere(h)
        mach = np.linspace(0.0, 8.0, 161)
        aero = p.aero()
        hw = np.linspace(0.0, 40e3, 161)
        wnd = p.wind()
        return dict(veh=veh, t=t, m=m, v=v, twr=twr, burn=burn, mdot=mdots, dv=fl.ideal_dv(veh), h=h / 1e3, atm=a,
                    mach=mach, cd=np.array([aero.cd_of(x) for x in mach]), hw=hw / 1e3,
                    wind=np.array([wnd.at(x) for x in hw]), m0=m[0], twr0=F_sl / (m[0] * G0), stage=np.array(stage))

    def tabs(self, p, r, compare=None) -> list[TabSpec]:
        t = r["t"]
        mk_pad = [(0.0, float(r["twr"][0]), "thrust")]
        stg = [PlotSpec("Ideal Velocity", TT, r"$v$ (km·s⁻¹)", [Curve(t, r["v"] / 1e3, "thrust")],
                        markers=[(float(t[-1]), float(r["v"][-1] / 1e3), "hot")]),
               PlotSpec("Vehicle Mass", TT, r"$m$ (kg)", [Curve(t, r["m"], "fuel")]),
               PlotSpec("Thrust-to-Weight", TT, r"$\frac{F_{vac}}{m g_0}$ (–)",
                        [Curve(t, r["twr"], "oxidizer"), Curve(t, np.ones_like(t), "dim", "weight", "dash")],
                        markers=mk_pad)]
        a, h = r["atm"], r["h"]
        atm_ = [PlotSpec("Temperature", r"$T$ (K)", r"$h$ (km)", [Curve(a["T"], h, "hot")]),
                PlotSpec("Density", r"$\rho$ (kg·m⁻³)", r"$h$ (km)", [Curve(a["rho"], h, "pressure")]),
                PlotSpec("Speed of Sound", r"$a$ (m·s⁻¹)", r"$h$ (km)", [Curve(a["a"], h, "thrust")])]
        aero = [PlotSpec("Drag Coefficient", r"$M$ (–)", r"$C_D$ (–)", [Curve(r["mach"], r["cd"], "hot")]),
                PlotSpec("Pressure", r"$p$ (kPa)", r"$h$ (km)", [Curve(a["p"] / 1e3, h, "oxidizer")]),
                PlotSpec("Wind Profile", r"$w$ (m·s⁻¹)", r"$h$ (km)", [Curve(r["wind"], r["hw"], "fuel")])]
        return [TabSpec("Staging", stg), TabSpec("Atmosphere", atm_), TabSpec("Aerodynamics & Wind", aero)]

    def readout(self, p, r, frame) -> list[str]:
        return [rf"$m_0$ = {r['m0'] / 1e3:.2f} (t)", rf"$\frac{{F}}{{m_0g_0}}$ = {r['twr0']:.2f}",
                rf"$\Delta v$ = {sum(r['dv']) / 1e3:.2f} (km·s⁻¹)", rf"$t_b$ = {sum(r['burn']):.0f} (s)"]

    def caption(self, p, r, frame) -> str:
        warn = " · thrust-to-weight below 1: the vehicle cannot lift off" if r["twr0"] <= 1.0 else ""
        parts = ", ".join(f"{d / 1e3:.2f}" for d in r["dv"])
        return (rf"stage $\Delta v$ {parts} (km·s⁻¹) · burn times " + ", ".join(f"{b:.0f}" for b in r["burn"])
                + rf" (s) · sketch assumes {BULK_RHO:.0f} (kg·m⁻³) bulk propellant density{warn}")

    def export_data(self, p, r):
        return {"t_s": r["t"], "mass_kg": r["m"], "v_ideal_ms": r["v"], "TWR": r["twr"], "h_km": r["h"],
                "T_K": r["atm"]["T"], "p_Pa": r["atm"]["p"], "rho": r["atm"]["rho"], "a_ms": r["atm"]["a"]}

    # ---------------------------------------------------------------- 3D stack sketch
    def build_scene(self, vp, p, r=None) -> None:
        vp.plotter.clear()
        self._m = {"built": True}
        self.update_scene(vp, p, r or self.evaluate(p), None)
        vp.reset_view(position="iso", zoom=1.0)

    def update_scene(self, vp, p, r, frame) -> None:
        if not self._m:
            return self.build_scene(vp, p, r)
        pl = vp.plotter
        veh = r["veh"]
        n = len(veh.stages)
        th = thetas(vp.cutaway)
        y = 0.0
        d_top = veh.stages[-1].d
        parts = []                                                  # (name, x, radius, stage index)
        for k in range(n - 1, -1, -1):                              # top stage first, stacked downward
            s = veh.stages[k]
            R = 0.5 * s.d
            L = max(1.15 * s.m_prop / (BULK_RHO * np.pi * R * R) + 0.6 * s.d, 1.0 * s.d)
            nozzle = 0.5 * s.d
            parts.append((k, y - L, y, R, nozzle))
            y -= L + 0.0
        top = parts[0][2]
        nose = 2.0 * d_top
        shift = -parts[-1][1]
        for k, y0, y1, R, nozzle in parts:
            y0, y1 = y0 + shift, y1 + shift
            ax = np.array([y0 - nozzle, y0, y0, y1])
            rad = np.array([0.55 * R, 0.30 * R, R, R])
            g = rev_grid(ax, rad, th)
            g.rotate_z(90, inplace=True)                           # axis vertical
            g["s"] = broadcast_scalar(np.full(len(ax), k + 1.0), len(th))
            pl.add_mesh(g, scalars="s", cmap="viridis", clim=(1, n + 1), show_scalar_bar=False, name=f"stage{k}",
                        smooth_shading=True)
        yt = parts[0][2] + shift
        s_ = np.linspace(0, 1, 14)
        ax = yt + nose * s_
        rad = 0.5 * d_top * np.sqrt(np.maximum(1 - s_**2, 0)) ** 1.0
        g = rev_grid(ax, np.maximum(rad, 1e-4), th)
        g.rotate_z(90, inplace=True)
        g["s"] = broadcast_scalar(np.full(len(ax), n + 1.0), len(th))
        pl.add_mesh(g, scalars="s", cmap="viridis", clim=(1, n + 1), show_scalar_bar=False, name="payload",
                    smooth_shading=True)
        vp.set_colorbar("viridis", 1.0, float(n + 1), r"stage (–)")
        vp.render()

    def reset_scene(self) -> None:
        self._m = {}


# ============================================================================================== ascent (Run)
FIELDS_ASCENT = [("M", "viridis", r"$M$ (–)"), ("q", "plasma", r"$q$ (kPa)"), ("V", "inferno", r"$V$ (m·s⁻¹)"),
                 ("h", "cividis", r"$h$ (km)")]


class AscentModel(ModelController):
    key = "vehicle.ascent"
    title = "Point-mass ascent"
    module = "Vehicle"
    mode = "run"
    Params = AscentParams
    fields = [f[2] for f in FIELDS_ASCENT]

    def __init__(self) -> None:
        self.sim = PointMassAscent()
        self._m: dict = {}

    def reset_scene(self) -> None:
        self._m = {}

    preset_updates = staticmethod(lambda p, prev: stage_preset_updates(p, prev))

    def prepare_params(self, p, home) -> None:
        """Load the thrust profiles sent from other modules (``<data folder>/profiles/stage<k>.json|csv``)."""
        for k in (1, 2, 3):
            if k <= p.n() and getattr(p, f"s{k}_src") == "profile":
                path = profiles.find(home / "profiles", k)
                if path is None:
                    raise ValueError(f"stage {k} is set to an imported thrust profile but none was sent: run a hybrid, "
                                     f"solid or liquid model, then File > Send Thrust to Vehicle > Stage {k}")
                setattr(p, f"prof{k}", json.dumps(profiles.load(path)))
        if p.drag == "table":
            f = home / "profiles" / "drag.csv"
            if not f.is_file():
                raise ValueError("the drag source is a table but profiles/drag.csv was not found: create it with the "
                                 "columns mach,cd in the data folder (File > Open Data Folder)")
            p.drag_table = json.dumps(profiles.load_drag(f))

    # ---------------------------------------------------------------- 3D (local tangent frame, km)
    @staticmethod
    def _path(r) -> np.ndarray:
        c, R = r.channels, fl.R_E / 1e3
        th, rr = c["x"] / R, R + c["h"]
        return np.c_[rr * np.sin(th), rr * np.cos(th) - R, np.zeros(len(th))]

    def _field(self, vp, r):
        key, cmap, title = FIELDS_ASCENT[vp.field_index]
        return r.channels[key], cmap, title

    def _ground(self, pl, ex: float, th_lo: float, th_hi: float, orbital: bool) -> None:
        R = fl.R_E / 1e3
        if orbital:
            pl.add_mesh(pv.Sphere(radius=R, center=(0.0, -R, 0.0), theta_resolution=120, phi_resolution=60),
                        color="#2f6db5", opacity=0.9, smooth_shading=True, name="ground")
            return
        pad = 0.15 * ex / R
        phi = np.linspace(th_lo - pad, th_hi + pad, 240)
        depth = 0.07 * ex
        A = np.c_[R * np.sin(phi), R * np.cos(phi) - R]
        B = A - depth * np.c_[np.sin(phi), np.cos(phi)]
        g = pv.StructuredGrid()
        g.points = np.r_[np.c_[A, np.zeros(len(phi))], np.c_[B, np.zeros(len(phi))]]
        g.dimensions = (len(phi), 2, 1)
        pl.add_mesh(g, color="#4a5d3a", name="ground")
        pl.add_mesh(pv.lines_from_points(np.c_[A, np.zeros(len(phi))]), color="#8fb08a", line_width=2, name="horizon")

    def build_scene(self, vp, p, r=None) -> None:
        pl = vp.plotter
        pl.clear()
        R = fl.R_E / 1e3
        if r is None:
            self._ground(pl, 2.0, -0.0005, 0.0005, False)
            pl.add_mesh(pv.Arrow(start=(0, 0, 0), direction=(0, 1, 0), scale=0.4), color="#ffd23f", name="craft")
            self._m = dict(built=False, fi=vp.field_index)
            vp.set_colorbar("viridis", 0.0, 1.0, FIELDS_ASCENT[vp.field_index][2])
            vp.reset_view(position="xy", zoom=1.0)
            return
        P = self._path(r)
        ex = max(float(np.ptp(P[:, 0])), float(np.ptp(P[:, 1])), 1.0)
        orbital = ex > 0.12 * R
        th = r.channels["x"] / R
        self._ground(pl, ex, float(th.min()), float(th.max()), orbital)
        f, cmap, title = self._field(vp, r)
        lo, hi = float(np.min(f)), float(np.max(f)) + 1e-9
        poly = pv.lines_from_points(P)
        poly["f"] = f
        pl.add_mesh(poly, scalars="f", cmap=cmap, clim=(lo, hi), line_width=3, opacity=0.35, show_scalar_bar=False,
                    name="path")
        ev = [(n, te) for n, te in r.meta.get("events", []) if _ev_key(n) and n not in ("Liftoff", "Rail exit")]
        if ev:
            idx = [int(min(np.searchsorted(r.t, te), len(r.t) - 1)) for _, te in ev]
            rgb = np.array([[int(EVENT_COLORS[_ev_key(n)][i:i + 2], 16) for i in (1, 3, 5)] for n, _ in ev], np.uint8)
            pl.add_points(P[idx], scalars=rgb, rgb=True, point_size=9, render_points_as_spheres=True, name="events")
        self._m = dict(built=True, P=P, ex=ex, f=f, lo=lo, hi=hi, fi=vp.field_index, cmap=cmap, title=title)
        vp.reset_view(position="xy", zoom=1.1)

    def update_scene(self, vp, p, r, frame) -> None:
        if not self._m or self._m.get("built") != (r is not None) or self._m.get("fi") != vp.field_index:
            self.build_scene(vp, p, r)
        if r is None or frame is None:
            vp.render()
            return
        m = self._m
        P, i = m["P"], int(min(max(frame, 0), len(m["P"]) - 1))
        pl = vp.plotter
        if i >= 1:
            fl_ = pv.lines_from_points(P[: i + 1])
            fl_["f"] = m["f"][: i + 1]
            pl.add_mesh(fl_, scalars="f", cmap=m["cmap"], clim=(m["lo"], m["hi"]), line_width=4, show_scalar_bar=False,
                        name="flown")
        d = P[min(i + 1, len(P) - 1)] - P[max(i - 1, 0)]
        nd = float(np.linalg.norm(d))
        d = d / nd if nd > 0 else np.array([0.0, 1.0, 0.0])
        pl.add_mesh(pv.Arrow(start=P[i], direction=d, scale=0.05 * m["ex"], tip_length=0.45, tip_radius=0.16,
                             shaft_radius=0.05), color="#ffd23f", name="craft")
        vp.set_colorbar(m["cmap"], m["lo"], m["hi"], m["title"])
        vp.render()

    # ---------------------------------------------------------------- plots
    def _markers(self, r, xch, ych, which):
        out = []
        for n, te in r.meta.get("events", []):
            k = _ev_key(n)
            if k in which:
                i = int(min(np.searchsorted(r.t, te), r.n - 1))
                out.append((float(r.channels[xch][i]) if xch else float(te), float(r.channels[ych][i]), EVENT_PLOT[k]))
        return out

    def _overlay(self, compare, r, ch, title, yl):
        runs = compare or [("current", r)]
        cs = [Curve(rr.t, rr.channels[ch], RUN_COLORS[i % len(RUN_COLORS)], name) for i, (name, rr) in enumerate(runs)]
        return PlotSpec(title, TT, yl, cs, cursor=False)

    def tabs(self, p, r, compare=None) -> list[TabSpec]:
        if r is None:
            e = lambda title, xl, yl: PlotSpec(title, xl, yl, [], cursor=(xl == TT))
            return [TabSpec("Trajectory", [e("Altitude vs Downrange", r"$x$ (km)", r"$h$ (km)"), e("Altitude", TT, r"$h$ (km)"),
                                           e("Speed", TT, r"$V$ (m·s⁻¹)")]),
                    TabSpec("Loads", []), TabSpec("Propulsion", []), TabSpec("Orbit", []), TabSpec("Compare", [])]
        c, t = r.channels, r.t
        cur = lambda y, col, label="", style="solid": Curve(t, y, col, label, style)
        traj = [PlotSpec("Altitude vs Downrange", r"$x$ (km)", r"$h$ (km)", [Curve(c["x"], c["h"], "thrust")],
                         markers=self._markers(r, "x", "h", ("Stage", "Apogee", "Max-Q", "Impact", "Orbit"))),
                PlotSpec("Altitude", TT, r"$h$ (km)", [cur(c["h"], "thrust")], cursor=True,
                         markers=self._markers(r, None, "h", ("Stage", "Apogee", "Orbit"))),
                PlotSpec("Speed", TT, r"$V$ (m·s⁻¹)", [cur(c["V"], "fuel")], cursor=True)]
        loads = [PlotSpec("Dynamic Pressure", TT, r"$q$ (kPa)", [cur(c["q"], "hot")], cursor=True,
                          markers=self._markers(r, None, "q", ("Max-Q",))),
                 PlotSpec("Mach Number", TT, r"$M$ (–)", [cur(c["M"], "pressure")], cursor=True,
                          markers=self._markers(r, None, "M", ("Mach",))),
                 PlotSpec("Acceleration", TT, r"$\frac{a}{g_0}$ (–)", [cur(c["g_load"], "oxidizer")], cursor=True)]
        prop = [PlotSpec("Thrust and Drag", TT, r"$F$ (kN)", [cur(c["F"], "thrust", "thrust"), cur(c["D"], "hot", "drag", "dash")],
                         cursor=True),
                PlotSpec("Vehicle Mass", TT, r"$m$ (kg)", [cur(c["m"], "fuel")], cursor=True,
                         markers=self._markers(r, None, "m", ("Stage", "Fairing"))),
                PlotSpec("Flight-Path and Thrust Angles", TT, r"$\gamma$ (deg)",
                         [cur(c["gamma"], "pressure", r"$\gamma$"), cur(c["alpha"], "oxidizer", r"$\alpha$", "dash")], cursor=True)]
        orb = [PlotSpec("Orbit Altitudes", TT, r"$h$ (km)",
                        [cur(np.clip(c["perigee"], -200.0, 3000.0), "fuel", "perigee"),
                         cur(np.clip(c["apogee_alt"], -200.0, 3000.0), "pressure", "apogee"),
                         cur(np.zeros_like(t), "dim", "surface", "dash")], cursor=True),
               PlotSpec("Altitude vs Speed", r"$V$ (m·s⁻¹)", r"$h$ (km)", [Curve(c["V"], c["h"], "thrust")],
                        markers=self._markers(r, "V", "h", ("Max-Q", "Stage")))]
        cmp_ = [self._overlay(compare, r, "h", "Altitude", r"$h$ (km)"), self._overlay(compare, r, "V", "Speed", r"$V$ (m·s⁻¹)"),
                self._overlay(compare, r, "q", "Dynamic Pressure", r"$q$ (kPa)")]
        return [TabSpec("Trajectory", traj), TabSpec("Loads", loads), TabSpec("Propulsion", prop), TabSpec("Orbit", orb),
                TabSpec("Compare", cmp_)]

    def readout(self, p, r, frame) -> list[str]:
        if r is None or frame is None:
            return [r"$h$ = – (km)", r"$V$ = – (m·s⁻¹)", r"$M$ = –", r"$q$ = – (kPa)"]
        c = r.channels
        return [rf"$h$ = {c['h'][frame]:.1f} (km)", rf"$V$ = {c['V'][frame]:.0f} (m·s⁻¹)", rf"$M$ = {c['M'][frame]:.2f}",
                rf"$q$ = {c['q'][frame]:.1f} (kPa)"]

    def caption(self, p, r, frame) -> str:
        if r is None or frame is None:
            return r"press Run: ascent with staging, drag, wind and imported thrust profiles"
        c = r.channels
        last = [n for n, te in r.meta.get("events", []) if te <= r.t[frame]]
        what = last[-1] if last else "launch"
        return (rf"$t$ = {r.t[frame]:.1f} (s) · stage {int(c['stage'][frame])} · "                rf"$m$ = {c['m'][frame]:.0f} (kg) · {what}")

    def summary_line(self, p, r) -> str:
        m = r.meta
        if m["impact"]:
            return (rf"apogee {m['apogee_km']:.1f} (km) · range {m['range_km']:.1f} (km) · max-$q$ {m['maxq_kpa']:.1f} (kPa) · "
                    rf"flight time {m['t_end']:.0f} (s)")
        if m["orbit"]:
            return (rf"orbit {m['perigee_km']:.0f} × {m['apogee_orbit_km']:.0f} (km) · burnout {m['v_burnout']:.0f} (m·s⁻¹) · "
                    rf"max-$q$ {m['maxq_kpa']:.1f} (kPa)")
        return rf"apogee {m['apogee_km']:.1f} (km) · burnout {m['v_burnout']:.0f} (m·s⁻¹) · max-$q$ {m['maxq_kpa']:.1f} (kPa)"

    def export_data(self, p, r):
        return {"t_s": r.t, **r.channels}
