r"""Start-up, throttling and shut-down of an engine graph: a lumped transient on top of the steady network.

The hydraulics are quasi-steady: at every time step the network is solved for its pressures and flows with
the valve positions, tank pressures and shaft speeds of that instant held fixed.  What carries the history is

* **shaft speed** of every shaft group, from the power left over on the shaft,

  .. math:: I\,\omega\frac{d\omega}{dt}=\sum P_{turbine}-\sum P_{pump}-P_{aux}+P_{starter}(t),

* **propellant mass** of every tank, :math:`dm/dt=-\dot m_{out}`; a regulated tank holds its set pressure, a
  blowdown tank's ullage expands polytropically, :math:`p=p_0\,(V_{u0}/V_u)^n`; the run stops when a tank runs dry,
* **valve positions** from the start/shut-down sequence (open time, ramp, close time, ramp) and the throttle
  command, or, for a valve the design targets size (an engine controller valve), from an integral controller that
  holds the target (chamber pressure scaled by the throttle, mixture ratio) once mainstage is reached.

Chamber filling, line inertia, ignition delay and two-phase priming are not modelled: their time constants
(milliseconds) are short against the turbomachinery spin-up (tenths of a second to seconds), which this model is for.
A shaft without a library inertia gets one from its wheels, each a steel disk of thickness 0.1 D
(``PROPSIM-EST``: :math:`I=\pi\rho t D^4/32`, :math:`\rho` = 8000 kg/m³).
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field

import numpy as np

from .edit import param_value
from .network import GraphError, Network, Solution
from .props import props

import os
_DEBUG = bool(os.environ.get('PROPSIM_GRAPH_DEBUG'))
POS_MIN = 0.02            # a "closed" valve still passes a trickle: the network has no zero-flow solution
STEEL_RHO = 8000.0


@dataclass
class ValveStep:
    t_open: float = 0.0
    ramp_open: float = 0.5
    t_close: float | None = None
    ramp_close: float = 0.3
    position: float = 1.0           # fully open position of this valve in mainstage
    throttle: bool = False          # its open position scales with the throttle command

    def at(self, t: float) -> float:
        if t < self.t_open:
            p = 0.0
        else:
            p = self.position * min(1.0, (t - self.t_open) / max(self.ramp_open, 1e-6))
        if self.t_close is not None and t > self.t_close:
            p *= max(0.0, 1.0 - (t - self.t_close) / max(self.ramp_close, 1e-6))
        return p


@dataclass
class Sequence:
    valves: dict[str, ValveStep]
    t_end: float = 3.0
    throttle: list[tuple[float, float]] = field(default_factory=lambda: [(0.0, 1.0)])   # (t, fraction) points
    starter: dict[int, tuple[float, float]] = field(default_factory=dict)  # shaft group -> (power W, duration s)
    control_from: float | None = None   # controller valves hold the targets from this time (None: after the ramps)
    loads: dict[str, float] = field(default_factory=dict)   # tank id -> loaded propellant mass (kg)
    dt_max: float = 0.02

    def throttle_at(self, t: float) -> float:
        ts, fs = zip(*sorted(self.throttle))
        return float(np.interp(t, ts, fs))


def wheel_inertia(D: float) -> float:
    t = 0.1 * D
    return math.pi * STEEL_RHO * t * D**4 / 32.0


def default_sequence(doc: dict, t_end: float = 3.0) -> Sequence:
    """A start-up a user can edit: gas generator / preburner valves open first, then the main valves; the controller
    valves (sized by the design targets) take over once the main valves are open; nothing closes before ``t_end``.
    For a pressure-fed engine the main valves carry the throttle; otherwise the gas-generator valves do."""
    g = doc["graph"]
    nodes = {n["id"]: n for n in g["nodes"]}
    free = {(f["node"], f["param"]) for f in doc.get("free", [])}
    feeds_gg = _feeds(g, ("gas_generator",))
    has_turbine = any(n["type"] == "turbine_impulse" for n in g["nodes"])
    valves = {}
    for nid, n in nodes.items():
        if not n["type"].startswith("valve"):
            continue
        spec = {k: param_value(n, k)[0] for k in ("t_open", "ramp_open", "t_close", "ramp_close")}
        gg = nid in feeds_gg
        step = ValveStep(t_open=spec["t_open"] if spec["t_open"] is not None else (0.05 if gg else 0.25),
                         ramp_open=spec["ramp_open"] or (0.4 if gg else 0.5),
                         t_close=spec["t_close"], ramp_close=spec["ramp_close"] or 0.3,
                         position=float(n.get("options", {}).get("position", 1.0)),
                         throttle=(gg and has_turbine and (nid, "position") not in free) or not has_turbine)
        valves[nid] = step
    seq = Sequence(valves, t_end=t_end)
    return seq


def surge_notes(g: dict, seq: Sequence, sol) -> list[str]:
    """Water hammer at shut-down: a valve that closes faster than the pressure wave's round trip along the line
    feeding it sees the full Joukowsky surge rho a V (Rennels & Hudson ch. 21; rigid-wall wave speed unless the
    line gives its wall); slower closures cut it in the ratio 2L/(a t_close)."""
    out = []
    for e in g["edges"]:
        v, ln = e["to"]["node"], e["from"]["node"]
        st = seq.valves.get(v)
        c = sol.components.get(ln, {})
        if st is None or st.t_close is None or c.get("type") != "line" or "surge_dp" not in c:
            continue
        t_rt = c["round_trip_time"]
        dp = c["surge_dp"] * min(1.0, t_rt / max(st.ramp_close, 1e-9))
        msg = (f"{v} closes in {st.ramp_close:g} s; the wave round trip in {ln} is {1e3 * t_rt:.1f} ms, so the surge "
               f"there is up to {dp / 1e6:.2f} MPa above the line pressure (not in the quasi-steady run)")
        out.append(msg)
    return out


def _feeds(g: dict, types: tuple[str, ...]) -> set[str]:
    """Components upstream of any node of ``types`` without passing a chamber or pump (their feed valves)."""
    nodes = {n["id"]: n for n in g["nodes"]}
    up: dict[str, list[str]] = {}
    for e in g["edges"]:
        if e.get("kind") == "fluid":
            up.setdefault(e["to"]["node"], []).append(e["from"]["node"])
    out, stack = set(), [n for n in nodes if nodes[n]["type"] in types]
    while stack:
        b = stack.pop()
        for a in up.get(b, []):
            if a in out or nodes[a]["type"] in ("chamber", "pump_centrifugal", "tank_regulated", "tank_blowdown",
                                                 "gas_generator"):
                continue
            out.add(a)
            stack.append(a)
    return out


@dataclass
class Transient:
    t: list[float]
    solutions: list[Solution]
    speeds: list[list[float]]             # rpm per shaft group
    tank_mass: dict[str, list[float]]
    positions: dict[str, list[float]]
    throttle: list[float]
    message: str
    notes: list[str]

    def series(self, node: str, output: str) -> np.ndarray:
        return np.array([s.components.get(node, {}).get(output, np.nan) for s in self.solutions], float)

    def station_series(self, index: int, key: str) -> np.ndarray:
        return np.array([s.stations[index].get(key, np.nan) if index < len(s.stations) else np.nan
                         for s in self.solutions], float)


def run(doc: dict, seq: Sequence, progress=None, cancelled=None) -> Transient:
    """Integrate the start-up (and any throttling or shut-down) of sandbox document ``doc`` over ``seq``."""
    g = copy.deepcopy(doc["graph"])
    notes: list[str] = []
    # design point: the shaft speeds the start-up heads for, the controller set points and gains
    design = Network(copy.deepcopy(g), doc.get("targets") or None, doc.get("free") or None)
    sol_d = design.solve()
    if not sol_d.converged:
        raise GraphError(f"the design point does not solve ({sol_d.message}); fix it before running a start-up")
    N_des = [s["speed"] for s in sol_d.shafts]
    for f, v in zip(doc.get("free", []), [sol_d.free[f"{f['node']}.{f['param']}"] for f in doc.get("free", [])]):
        if f["param"] == "position" and f["node"] in seq.valves:
            seq.valves[f["node"]].position = float(v)
        elif f["param"] != "position":
            next(n for n in g["nodes"] if n["id"] == f["node"])["params"][f["param"]] = {"value": float(v)}
    net = Network(g)                                       # the hardware, no targets: the run is open-loop
    net.m_ref, net.P_ref = design.m_ref, design.P_ref
    nfree = len(doc.get("free") or [])
    x_design = design.x[:len(design.x) - nfree].copy() if design.x is not None else None
    _calibrate_turbines(net, seq, notes, x_design)
    groups = net.shaft_groups
    inertia = []
    for grp in groups:
        I = 0.0
        given = False
        for nid in grp:
            n = net.nodes[nid]
            if n["type"] == "shaft":
                v = net.param(nid, "inertia", 0.0)
                if v:
                    I += float(v)
                    given = True
        if not given:
            for nid in grp:
                D = net.param(nid, "D_tip", 0.0) if net.nodes[nid]["type"] == "pump_centrifugal" else \
                    net.param(nid, "D_mean", 0.0) if net.nodes[nid]["type"] == "turbine_impulse" else 0.0
                I += wheel_inertia(float(D or 0.0))
            notes.append(f"shaft {'/'.join(grp)}: inertia {I:.3g} kg m² estimated from its wheels (steel disks)")
        inertia.append(max(I, 1e-4))
    # controllers: each free valve holds its target from mainstage on
    ctrl = []
    for f, t in zip(doc.get("free", []), doc.get("targets", [])):
        if f["param"] != "position" or f["node"] not in seq.valves:
            continue
        ctrl.append({"valve": f["node"], "node": t["node"], "output": t["output"], "value": float(t["value"]),
                     "throttled": t["output"] in ("p_c", "thrust")})
    if ctrl:
        _gains(doc, design, sol_d, ctrl)
    t_ctrl = seq.control_from
    if t_ctrl is None:
        t_ctrl = max([v.t_open + v.ramp_open for v in seq.valves.values()] or [0.0]) + 0.1
    # tanks
    tanks = {}
    for nid, m in net.models.items():
        if m.type not in ("tank_regulated", "tank_blowdown"):
            continue
        fid = m.fluid()
        V, ff = float(net.param(nid, "V_tank")), float(net.param(nid, "fill_fraction", 0.9))
        p0 = float(m.p_tank())
        T = net.param(nid, "T_prop", None) if m.type == "tank_regulated" else None
        T = float(T) if T else float(m.opt.get("T_prop", 293.15))
        pr = props(fid)
        rho = pr.at(p0, pr.h_pT(p0, T), T)["rho"]
        m0 = float(seq.loads.get(nid, ff * V * rho))
        if m0 > V * rho:
            notes.append(f"{nid}: loaded {m0:.4g} kg exceeds the tank ({V * rho:.4g} kg); limited to full")
        m0 = min(m0, V * rho * 0.999)
        tanks[nid] = dict(m=m0, rho=rho, V=V, Vu0=V - m0 / rho, p0=p0, kind=m.type,
                          n=float(net.param(nid, "n_poly", 1.2) or 1.2) if m.type == "tank_blowdown" else 0.0)
    notes.extend(surge_notes(g, seq, sol_d))
    # start state
    N = [0.03 * max(abs(n), 100.0) for n in N_des]
    pos = {v: POS_MIN for v in seq.valves}
    x_prev = _wind_down(net, seq, N_des, N, pos, notes)
    out = Transient([], [], [], {k: [] for k in tanks}, {k: [] for k in seq.valves}, [], "", notes)
    t, dt = 0.0, 0.005
    fails = 0
    prev_dN: list[float] = []
    while t <= seq.t_end + 1e-9:
        if cancelled and cancelled():
            out.message = f"stopped at {t:.3f} s"
            break
        thr = seq.throttle_at(t)
        for v, step in seq.valves.items():
            if t >= t_ctrl and any(c["valve"] == v for c in ctrl):
                continue                                     # held by its controller
            p = step.at(t) * (thr if step.throttle else 1.0)
            pos[v] = max(p, POS_MIN)
        for v, p in pos.items():
            net.models[v].opt["position"] = p
        for nid, tk in tanks.items():
            if tk["kind"] == "tank_blowdown":
                Vu = tk["V"] - tk["m"] / tk["rho"]
                net.overrides[(nid, "p0")] = tk["p0"] * (tk["Vu0"] / max(Vu, 1e-9)) ** tk["n"]
        net.speed_set = list(N)
        sol = _solve(net, x_prev)
        if not sol.converged:
            fails += 1
            if fails > 6 or dt < 2e-4:
                out.message = f"the network stops solving at {t:.3f} s: {sol.message}"
                break
            dt *= 0.5
            t = out.t[-1] + dt if out.t else t
            continue
        fails = 0
        x_prev = net.x.copy()
        # shaft speeds
        dN = []
        for k, s in enumerate(sol.shafts):
            # accessory load (gearbox, lube, hydraulics) falls with speed like a pump: cubic (PROPSIM-EST)
            P = s["turbine_power"] - s["pump_power"] - s["aux_power"] * (N[k] / max(N_des[k], 1.0)) ** 3
            Pst, tst = seq.starter.get(k, (0.0, 0.0))
            if t < tst:
                P += Pst
            dN.append(P / (inertia[k] * max(N[k], 0.02 * max(N_des[k], 100.0))))
        if _DEBUG:
            print(f"t={t:.4f} dt={dt:.4f} N={[round(n) for n in N]} dN={[round(d, 1) for d in dN]} "
                  f"Pt={[round(sh['turbine_power']) for sh in sol.shafts]} Pp={[round(sh['pump_power']) for sh in sol.shafts]}")
        # record
        out.t.append(t)
        out.solutions.append(sol)
        out.speeds.append([n * 30 / math.pi for n in N])
        for nid in tanks:
            out.tank_mass[nid].append(tanks[nid]["m"])
        for v in seq.valves:
            out.positions[v].append(pos[v])
        out.throttle.append(thr)
        if progress:
            progress(t, seq.t_end)
        # advance (explicit, step limited to 8 % speed change)
        h = dt
        for k in range(len(N)):
            lim = 0.08 * max(N[k], 0.05 * max(N_des[k], 100.0))
            if abs(dN[k]) * h > lim:
                h = lim / abs(dN[k])
        h = max(h, 1e-4)
        for k in range(len(N)):
            N[k] = max(N[k] + dN[k] * h, 0.0)
        empty = None
        for nid, tk in tanks.items():
            tk["m"] -= max(sol.components[nid].get("mdot", 0.0), 0.0) * h
            if tk["m"] <= 0:
                empty = nid
        if t >= t_ctrl:
            for c in ctrl:
                y = sol.components.get(c["node"], {}).get(c["output"])
                if y is None:
                    continue
                cmd = c["value"] * (thr if c["throttled"] else 1.0)
                e = (cmd - y) / abs(c["value"])
                pos[c["valve"]] = float(np.clip(pos[c["valve"]] + c["ki"] * e * h, POS_MIN, 1.0))
        if empty:
            out.message = f"{empty} ran dry at {t + h:.2f} s"
            break
        t += h
        # explicit Euler rings when the step is long against a shaft's time constant: a reversed acceleration
        # that matters is the sign of an overshoot, so shorten the step
        ring = any(prev_dN[k] * dN[k] < 0 and abs(dN[k]) * h > 2e-3 * max(N[k], 100.0) for k in range(len(N))) \
            if prev_dN else False
        prev_dN = dN
        dt = max(dt * 0.5, 4e-3) if ring else min(dt * 1.25, seq.dt_max)
    else:
        out.message = f"ran to {seq.t_end:g} s"
    if not out.message:
        out.message = f"ran to {out.t[-1] if out.t else 0:.2f} s"
    if out.speeds and out.t and out.t[-1] >= 0.9 * seq.t_end and N_des:
        low = [k for k in range(len(N_des)) if N[k] < 0.5 * N_des[k]]
        if low and not any(x[0] for x in seq.starter.values()):
            out.notes.append("a shaft is still below half its design speed at the end of the run: this engine may "
                             "need a starter (Starter tab) or a longer run to bootstrap")
    return out


def _calibrate_turbines(net: Network, seq: Sequence, notes: list[str], x0=None) -> None:
    """A fixed-pressure-ratio turbine cannot start (its inlet pressure is pinned at a multiple of its exhaust pressure
    whatever the flow).  Each turbine becomes a flow element calibrated at the open-loop design point: a choked
    nozzle for a gas turbine (A_nozzle), a loss coefficient for a liquid one (K_turb), and an efficiency that follows
    the blade-speed ratio about the design value."""
    from ..nozzle import Gamma as _G
    for v in seq.valves:
        net.models[v].opt["position"] = max(seq.valves[v].position, POS_MIN)
    net.speed_set = None
    sol = net.solve(x0=x0)
    if not sol.converged:
        raise GraphError(f"the engine at full open does not solve ({sol.message})")
    ev = net.ev
    for nid, m in net.models.items():
        if m.type != "turbine_impulse":
            continue
        if m.opt.get("flow_model", "auto") not in ("auto", "pressure_ratio"):
            continue
        s_in = ev.inlet(nid, m.pin)
        p1, p2 = ev.p(nid, m.pin), ev.p(nid, m.pout)
        md = float(sol.components[nid]["mdot"])
        if s_in.kind == "gas" or p1 / max(p2, 1.0) > 2.0:
            if s_in.kind == "gas":
                R, T, g = s_in.R, s_in.T, s_in.gamma
            else:
                a = props(s_in.fluid).at(p1, s_in.h, s_in.T)
                g, T = 1.4, a["T"]
                R = p1 / (a["rho"] * T)
            net.overrides[(nid, "A_nozzle")] = md * math.sqrt(R * T) / (p1 * _G(g))
            m.opt["flow_model"] = "choked"
        else:
            rho = ev.inlet(nid, m.pin)
            from .props import density
            net.overrides[(nid, "K_turb")] = (p1 - p2) * density(rho, p1) / md**2
            m.opt["flow_model"] = "orifice"
        nu = sol.components[nid].get("u_over_c0")
        if nu:
            net.overrides[(nid, "nu_design")] = float(nu)
    notes.append("turbines run as calibrated flow elements (choked nozzle or loss coefficient) with efficiency "
                 "following the blade-speed ratio about the design value")


def _wind_down(net: Network, seq: Sequence, N_des, N0, pos0, notes) -> np.ndarray:
    """The pre-start state (valves at their trickle, shafts barely turning) reached from the open-loop design point
    in small steps, each solve starting from the last: a cold solve so far off design does not converge."""
    open_pos = {v: max(st.position, POS_MIN) for v, st in seq.valves.items()}
    for v, p in open_pos.items():
        net.models[v].opt["position"] = p
    net.speed_set = None
    s = net.solve(x0=net.x.copy() if getattr(net, 'x', None) is not None else None)
    if not s.converged:
        raise GraphError(f"the engine at full open does not solve ({s.message})")
    x = net.x.copy()
    N_open = [sh["speed"] for sh in s.shafts]
    lam, dl = 0.0, 0.08
    while lam < 1.0 - 1e-12:
        trial = min(lam + dl, 1.0)
        # geometric in position and speed: the engine's behaviour scales with their ratio to the open values
        for v in open_pos:
            net.models[v].opt["position"] = open_pos[v] * (pos0[v] / open_pos[v]) ** trial
        net.speed_set = [N_open[k] * (N0[k] / N_open[k]) ** trial for k in range(len(N0))]
        sol = _solve(net, x, cheap=True)
        if sol.converged:
            lam, x = trial, net.x.copy()
            dl = min(dl * 1.5, 0.2)
        else:
            dl *= 0.5
            if dl < 2e-3:
                raise GraphError(f"cannot reach the pre-start state from the design point ({sol.message}); "
                                 f"{100 * lam:.0f} % of the way")
    return x


def _solve(net: Network, x_prev, cheap: bool = False) -> Solution:
    """One quasi-steady network solve, warm from the last step; a failed solve comes back unconverged.  ``cheap``
    skips the slow least-squares rescue (the caller shortens its step instead)."""
    attempts = [(x_prev, 1, False)] if cheap else \
        [(x_prev, 1, False), (x_prev, 3, True)] if x_prev is not None else [(None, 6, True)]
    sol = None
    for x0, outer, rescue in attempts:
        try:
            sol = net._solve_from(x0, outer=outer, rescue=rescue)
        except (ValueError, ArithmeticError, GraphError) as e:
            sol = Solution(False, str(e), [], {}, [], [], {}, float("inf"), 0)
        if not sol.converged and sol.stations and sol.residual < 1e-2:
            sol.converged = True               # relative residuals: 1e-2 is inside the model's own accuracy
        if sol.converged:
            break
    return sol


def _gains(doc: dict, design: Network, sol_d: Solution, ctrl: list[dict], tau: float = 0.25) -> None:
    """Integral gain of each controller from the design-point sensitivity of its output to its valve, so the loop
    closes with time constant ``tau`` (diagonal: each valve answers for its own target)."""
    for c in ctrl:
        g = copy.deepcopy(doc["graph"])
        pos0 = sol_d.free.get(f"{c['valve']}.position", 1.0)
        for f in doc.get("free", []):
            node = next(n for n in g["nodes"] if n["id"] == f["node"])
            v = sol_d.free[f"{f['node']}.{f['param']}"]
            if f["param"] == "position":
                node.setdefault("options", {})["position"] = v
            else:
                node["params"][f["param"]] = {"value": v}
        node = next(n for n in g["nodes"] if n["id"] == c["valve"])
        dp = 0.03 if pos0 < 0.97 else -0.03
        node["options"]["position"] = pos0 + dp
        net = Network(g)
        net.m_ref, net.P_ref = design.m_ref, design.P_ref
        s = net.solve()
        y0 = sol_d.components[c["node"]][c["output"]]
        y1 = s.components.get(c["node"], {}).get(c["output"], y0) if s.converged else y0
        sens = (y1 - y0) / abs(c["value"]) / dp
        if abs(sens) < 1e-4:
            sens = 1.0
        c["ki"] = 1.0 / (tau * sens)
