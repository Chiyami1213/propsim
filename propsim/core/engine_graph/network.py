r"""Steady operating point of any engine graph (``schemas/graph.schema.json``): the liquid-engine cycle sandbox solver.

The graph is reduced to **stations** (fluid ports joined by edges share one pressure; a fluid out-port feeding several
edges is a tee, several out-ports into one in-port a junction) and **shaft groups** (machines joined by shaft edges
turn at one speed).  Unknowns: station pressures, component flows (``models.py``) and shaft speeds.  Equations:

* mass balance at every station (or the fixed pressure of a ``fixed_pressure`` boundary, where flow leaves the graph),
* one hydraulic law per component (loss, pump curve, turbine flow, chamber throat),
* power balance on every shaft group: :math:`\sum P_{turbine}=\sum P_{pump}+P_{aux}`.

Fluid states (enthalpy, temperature, oxidizer fraction for the mixture ratio) are swept downstream through the graph on
every evaluation; heat loads of cooling jackets and drive-gas curves are updated between outer iterations.  The
nonlinear system is solved by Powell's hybrid method with a trust-region least-squares fallback.

**Design mode.**  ``targets`` fix outputs (``{"node": "chamber", "output": "p_c", "value": 7e6}``) and ``free``
releases the same number of parameters (``{"node": "chamber", "param": "A_t"}``); the extra unknowns and equations
are appended, so the solver sizes hardware to the targets (for example throat area for chamber pressure, valve
positions for preburner mixture ratios, pump diameter for discharge pressure).
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import least_squares, root

from .models import MODELS, Model
from .props import State, density, mix, props


_DEBUG = bool(__import__("os").environ.get("PROPSIM_GRAPH_DEBUG"))


class GraphError(ValueError):
    """The graph cannot be solved as built (missing connection, unknown type, unbalanced design problem)."""


@dataclass
class Solution:
    converged: bool
    message: str
    stations: list[dict]
    components: dict[str, dict]
    shafts: list[dict]
    warnings: list[str]
    free: dict[str, float] = field(default_factory=dict)
    residual: float = 0.0
    iterations: int = 0
    profiles: dict[str, dict] = field(default_factory=dict)
    edges: list[dict] = field(default_factory=list)

    def value(self, node: str, output: str) -> float:
        return float(self.components[node][output])


class _UF:
    def __init__(self):
        self.p: dict = {}

    def find(self, a):
        self.p.setdefault(a, a)
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a, b):
        self.p[self.find(a)] = self.find(b)


class Network:
    def __init__(self, graph: dict, targets: list[dict] | None = None, free: list[dict] | None = None) -> None:
        from ...data import components
        self.graph = copy.deepcopy(graph)
        self.lib = components()
        self.nodes = {n["id"]: n for n in self.graph["nodes"]}
        self.ox = self.graph["propellants"]["oxidizer"]
        self.fuel = self.graph["propellants"]["fuel"]
        for nid, n in self.nodes.items():
            if n["type"] not in self.lib:
                raise GraphError(f"{nid}: unknown component type {n['type']!r}")
            if n["type"] not in MODELS:
                raise GraphError(f"{nid}: component type {n['type']!r} has no steady model yet")
        self.targets = list(targets or [])
        self.free = list(free or [])
        if len(self.targets) != len(self.free):
            raise GraphError(f"design mode needs as many free parameters as targets ({len(self.free)} free, "
                             f"{len(self.targets)} targets)")
        self._free_val: dict[tuple[str, str], float] = {}
        self.overrides: dict[tuple[str, str], float] = {}   # time-dependent values set by the transient (tank p, ...)
        self.speed_set: list[float] | None = None          # transient: shaft speeds held, not balanced
        self._build_topology()
        self.models: dict[str, Model] = {nid: MODELS[n["type"]](self, nid) for nid, n in self.nodes.items()}
        self._layout()

    # ------------------------------------------------------------------------------------------------ parameters
    def param(self, nid: str, name: str, default=None):
        key = (nid, name)
        if key in self.overrides:
            return self.overrides[key]
        if key in self._free_val:
            return self._free_val[key]
        n = self.nodes[nid]
        if name in n.get("params", {}):
            v = n["params"][name]["value"]
            if v is not None:
                return v
        for prm in self.lib[n["type"]]["parameters"]:
            if prm["name"] == name:
                v = prm["default"]["value"]
                if v is not None:
                    return v
                if default is None and prm.get("user_input"):
                    raise GraphError(f"{nid}: parameter {name!r} must be given")
                return default
        return default

    # ------------------------------------------------------------------------------------------------ topology
    def ports(self, nid: str) -> dict[str, dict]:
        return {p["name"]: p for p in self.lib[self.nodes[nid]["type"]]["ports"]}

    def _build_topology(self) -> None:
        uf, sh = _UF(), _UF()
        self.fluid_ports: list[tuple[str, str]] = []
        for nid in self.nodes:
            for pn, p in self.ports(nid).items():
                if p["kind"] == "fluid" and pn != "pressurant_in":
                    self.fluid_ports.append((nid, pn))
                    uf.find((nid, pn))
        self.thermal: dict[str, str] = {}                     # jacket -> chamber
        self.edge_list = []
        connected = set()
        for e in self.graph["edges"]:
            a = (e["from"]["node"], e["from"]["port"])
            b = (e["to"]["node"], e["to"]["port"])
            for nid, pn in (a, b):
                if nid not in self.nodes or pn not in self.ports(nid):
                    raise GraphError(f"edge {e['id']}: {nid}.{pn} does not exist")
            kind = e["kind"]
            if kind == "fluid":
                uf.union(a, b)
                connected |= {a, b}
                self.edge_list.append(e)
            elif kind == "shaft":
                sh.union(a, b)
            elif kind == "thermal":
                ta, tb = self.nodes[a[0]]["type"], self.nodes[b[0]]["type"]
                if ta == "regen_jacket":
                    self.thermal[a[0]] = b[0]
                elif tb == "regen_jacket":
                    self.thermal[b[0]] = a[0]
        bcs = {(b["node"], b["port"]): b for b in self.graph.get("boundary_conditions", [])}
        self.fixed_p: dict[int, float] = {}
        self.ambient_p: dict[str, float] = {}
        for (nid, pn), b in bcs.items():
            if b["kind"] == "ambient_pressure":
                self.ambient_p[nid] = b["value"]["value"]
        missing = [f"{n}.{p}" for n, p in self.fluid_ports if (n, p) not in connected and (n, p) not in bcs]
        if missing:
            raise GraphError("unconnected fluid ports (connect them or give a boundary condition): "
                             + ", ".join(missing))
        roots = sorted({uf.find(fp) for fp in self.fluid_ports}, key=str)
        self.station_of = {fp: roots.index(uf.find(fp)) for fp in self.fluid_ports}
        self.n_st = len(roots)
        for (nid, pn), b in bcs.items():
            if b["kind"] == "fixed_pressure" and (nid, pn) in self.station_of:
                self.fixed_p[self.station_of[(nid, pn)]] = b["value"]["value"]
        self.st_ports: list[list[tuple[str, str]]] = [[] for _ in range(self.n_st)]
        for fp, s in self.station_of.items():
            self.st_ports[s].append(fp)
        # shaft groups: every shaft-capable port, shaft components join their drive and load ports
        groups: dict = {}
        shaft_ports = {nid: [pn for pn, p in self.ports(nid).items() if p["kind"] == "shaft"] for nid in self.nodes}
        for nid, sp in shaft_ports.items():
            if self.nodes[nid]["type"] == "shaft" and len(sp) == 2:
                sh.union((nid, sp[0]), (nid, sp[1]))
        for nid, sp in shaft_ports.items():
            for pn in sp:
                groups.setdefault(sh.find((nid, pn)), set()).add(nid)
        self.shaft_groups = [sorted(v) for v in groups.values()
                             if any(self.nodes[i]["type"] in ("pump_centrifugal", "turbine_impulse") for i in v)]
        self.shaft_of = {nid: k for k, g in enumerate(self.shaft_groups) for nid in g}
        for nid, n in self.nodes.items():
            if n["type"] in ("pump_centrifugal", "turbine_impulse") and nid not in self.shaft_of:
                raise GraphError(f"{nid}: connect its shaft port to a turbine, pump or shaft")
        for k, g in enumerate(self.shaft_groups):
            types = {self.nodes[i]["type"] for i in g}
            if "pump_centrifugal" not in types:
                raise GraphError(f"shaft {'/'.join(g)} drives no pump")
            if "turbine_impulse" not in types:
                raise GraphError(f"shaft {'/'.join(g)} has no turbine to drive it")

    def port_fluid(self, nid: str, port: str) -> str:
        """Fluid of a tank: its ``fluid`` option, else the medium of its outlet edge, else the propellant role."""
        opt = self.nodes[nid].get("options", {})
        if "fluid" in opt:
            return opt["fluid"]
        for e in self.edge_list:
            if e["from"]["node"] == nid and e["from"]["port"] == port and e.get("medium") not in (None, "combustion_gas"):
                return e["medium"]
        return self.ox if "ox" in nid else self.fuel

    def ox_fraction(self, fid: str) -> float:
        return 1.0 if fid == self.ox else 0.0

    def wall_source(self, jacket: str) -> str | None:
        return self.thermal.get(jacket)

    def upstream_chamber(self, nozzle: str) -> str:
        s = self.station_of[(nozzle, "inlet")]
        for nid, pn in self.st_ports[s]:
            if self.nodes[nid]["type"] == "chamber":
                return nid
        raise GraphError(f"{nozzle}: its inlet must be joined to a chamber throat")

    def downstream_nozzle(self, chamber: str) -> str | None:
        s = self.station_of.get((chamber, "throat"))
        if s is None:
            return None
        for nid, pn in self.st_ports[s]:
            if self.nodes[nid]["type"] == "nozzle_bell":
                return nid
        return None

    def ambient(self, nozzle: str) -> float:
        if nozzle in self.ambient_p:
            return self.ambient_p[nozzle]
        return float(self.graph.get("ambient_pressure", 0.0))

    # ------------------------------------------------------------------------------------------------ layout
    def _layout(self) -> None:
        self.slices: dict[str, slice] = {}
        i = self.n_st
        for nid, m in self.models.items():
            self.slices[nid] = slice(i, i + len(m.unknowns))
            i += len(m.unknowns)
        self.i_shaft = i
        i += len(self.shaft_groups)
        self.i_free = i
        i += len(self.free)
        self.n_x = i
        n_eq = self.n_st + len(self.shaft_groups) + len(self.targets)
        n_eq += sum(len(m.unknowns) for m in self.models.values())
        if n_eq != self.n_x:
            raise GraphError(f"the graph gives {n_eq} equations for {self.n_x} unknowns")
        for m in self.models.values():
            if m.type == "chamber" and self.downstream_nozzle(m.id) is None:
                raise GraphError(f"{m.id}: join a nozzle to the chamber throat")
        # downstream order for the state sweep (Kahn on component -> component through stations)
        succ = {nid: set() for nid in self.nodes}
        for s, plist in enumerate(self.st_ports):
            outs = [n for n, pn in plist if self.ports(n)[pn]["direction"] == "out"]
            ins = [n for n, pn in plist if self.ports(n)[pn]["direction"] == "in"]
            for a in outs:
                succ[a] |= set(ins)
        indeg = {n: 0 for n in self.nodes}
        for a, bs in succ.items():
            for b in bs:
                indeg[b] += 1
        order, ready = [], sorted([n for n, d in indeg.items() if d == 0])
        seen = set()
        while len(order) < len(self.nodes):
            if not ready:                                        # a loop: break it at the least-blocked node
                rest = [n for n in self.nodes if n not in seen]
                ready = [min(rest, key=lambda n: (indeg[n], n))]
            n = ready.pop(0)
            if n in seen:
                continue
            seen.add(n)
            order.append(n)
            for b in sorted(succ[n]):
                indeg[b] -= 1
                if indeg[b] <= 0 and b not in seen:
                    ready.append(b)
        self.order = order
        self.p_ref = 1e6
        self.m_ref = 1.0
        self.P_ref = 1e6

    # ------------------------------------------------------------------------------------------------ solve
    def solve(self, x0: np.ndarray | None = None, outer: int = 12, tol: float = 1e-7, steps: int = 4) -> Solution:
        """Operating point; in design mode the targets are reached by continuation from the operating point of the
        hardware as given (each step moves every target a fraction of the way and restarts Newton from the last
        solution)."""
        if self.targets and x0 is None:
            base = Network(self.graph)
            sol0 = base.solve(outer=outer)
            if not sol0.converged:
                return sol0
            x0 = np.concatenate([base.x, [self._free_start(f) for f in self.free]])
            self.m_ref, self.P_ref = base.m_ref, base.P_ref
            goal = [t["value"] for t in self.targets]
            start = []
            for t in self.targets:
                try:
                    start.append(float(sol0.components[t["node"]][t["output"]]))
                except KeyError:
                    raise GraphError(f"target {t['node']}.{t['output']}: no such output") from None
            sol, lam, dl = sol0, 0.0, 1.0 / steps
            while lam < 1.0 - 1e-12:
                trial = min(lam + dl, 1.0)
                for t, a, b in zip(self.targets, start, goal):
                    t["value"] = a + trial * (b - a)
                sol = self._solve_from(x0, outer, rescue=dl < 0.05)
                if sol.converged:
                    lam, x0 = trial, self.x
                    dl = min(2.0 * dl, 0.5)
                elif dl > 0.02:
                    dl /= 2.0                                   # shorter step from the last converged point
                else:
                    sol.message = f"design targets not reached ({100 * lam:.0f} % of the way): " + sol.message
                    break
            for t, b in zip(self.targets, goal):
                t["value"] = b
            return sol
        if x0 is None and any(m.type == "pump_centrifugal" for m in self.models.values()):
            # a cold start far from design sends the off-design pump efficiency to its floor; start from the
            # constant-efficiency answer and switch the efficiency curves on from there
            self.const_eta = True
            try:
                sol0 = self._solve_from(None, outer)
            finally:
                self.const_eta = False
            if sol0.converged:
                sol = self._solve_from(self.x, outer)
                return sol
        return self._solve_from(x0, outer)

    def _free_start(self, f: dict) -> float:
        if f["param"] == "position":
            return float(f.get("guess", self.nodes[f["node"]].get("options", {}).get("position", 1.0)))
        v = self.param(f["node"], f["param"], f.get("guess"))
        if v is None:
            raise GraphError(f"free parameter {f['node']}.{f['param']} needs a starting value")
        return float(v)

    def _solve_from(self, x0: np.ndarray | None, outer: int, rescue: bool = True) -> Solution:
        g = _Guess(self)
        if x0 is None:
            x = g.vector()
            self.m_ref = max(g.m_scale, 1e-3)
            self.P_ref = max(g.P_scale, 1e3)
        else:
            x = np.array(x0, float)
        ev = _Eval(self)
        warnings: list[str] = []
        msg, ok, it = "", False, 0
        xs = self._scales(x)
        for it in range(outer):
            ev.set(x)
            ev.sweep(passes=3)
            for m in self.models.values():
                try:
                    m.update_slow(ev.local(m.id), ev)
                except (ValueError, ZeroDivisionError) as exc:
                    warnings.append(f"{m.id}: {exc}")
            ev.frozen_states = dict(ev.port_state)
            z0 = x / xs
            f = lambda z: ev.residuals(z * xs)
            opts = {"xtol": 1e-9, "maxfev": 40 * (self.n_x + 1)}
            sol = root(f, z0, method="hybr", options=opts)
            r = float(np.max(np.abs(f(sol.x)))) if np.all(np.isfinite(sol.x)) else np.inf
            if r > 1e-6 and it == 0 and x0 is None:
                z_rel = self._relax(f, z0)                      # pseudo-transient start-up toward the steady state
                sol2 = root(f, z_rel, method="hybr", options=opts)
                r2 = float(np.max(np.abs(f(sol2.x)))) if np.all(np.isfinite(sol2.x)) else np.inf
                if r2 < r:
                    sol, r = sol2, r2
            sol_x = sol.x
            if r > 1e-6 and not rescue:
                break
            if r > 1e-6:
                lsq = least_squares(f, z0 if not np.isfinite(r) or r > 1e3 else sol.x, method="trf", xtol=1e-12,
                                    ftol=1e-12, gtol=1e-12, max_nfev=25)
                r2 = float(np.max(np.abs(lsq.fun)))
                if r2 < r:
                    sol_x, r = lsq.x, r2
            if _DEBUG:
                print(f"  outer {it}: residual {r:.2e}, evaluations {ev.count}")
            x_new = sol_x * xs
            change = float(np.max(np.abs(x_new - x) / np.maximum(np.abs(xs), 1e-12)))
            x = x_new
            xs = self._scales(x)
            if r < 1e-6 and change < 1e-3 and it > 0:
                ok = True
                break
        ev.set(x)
        ev.sweep(passes=3)
        res = float(np.max(np.abs(ev.residuals(x))))
        ok = ok or res < 1e-5
        msg = "converged" if ok else f"not converged (largest residual {res:.2e})"
        self.x = x
        self.ev = ev
        return self._solution(ev, ok, msg, res, it + 1, warnings)

    def signs(self) -> np.ndarray:
        """Relaxation direction of each unknown: d(unknown)/d(tau) = sign x its residual row."""
        sg = np.ones(self.n_x)
        for s in self.fixed_p:
            sg[s] = -1.0
        for nid, m in self.models.items():
            sg[self.slices[nid]] = m.signs
        sg[self.i_free:] = 0.0
        return sg

    def _relax(self, f, z0: np.ndarray, tau: float = 200.0, budget: int = 1500) -> np.ndarray:
        """Pseudo-transient continuation: integrate dz/dtau = S f(z) (each flow speeds up while pressure is left
        over, each station fills while more enters than leaves, each shaft spins up while the turbine has power to
        spare) with a stiff BDF method, stopping once the residual is small."""
        from scipy.integrate import solve_ivp
        sg = self.signs()

        count = [0]

        class _Budget(Exception):
            pass

        best = [z0, float(np.max(np.abs(f(z0))))]

        def rhs(t, z):
            count[0] += 1
            if count[0] > budget:
                raise _Budget
            r = f(z)
            rm = float(np.max(np.abs(r)))
            if rm < best[1]:
                best[0], best[1] = z.copy(), rm
            return sg * r

        def done(t, z):
            return float(np.max(np.abs(f(z)))) - 1e-3
        done.terminal = True
        try:
            solve_ivp(rhs, (0.0, tau), z0, method="BDF", events=done, rtol=1e-4, atol=1e-6)
        except (_Budget, ValueError, RuntimeError, FloatingPointError):
            pass
        return best[0]

    def _scales(self, x: np.ndarray) -> np.ndarray:
        xs = np.ones(self.n_x)
        xs[: self.n_st] = self.p_ref
        for nid, sl in self.slices.items():
            xs[sl] = self.m_ref
        for k, grp in enumerate(self.shaft_groups):
            xs[self.i_shaft + k] = max(abs(x[self.i_shaft + k]), 100.0)
        for k, fr in enumerate(self.free):
            xs[self.i_free + k] = max(abs(x[self.i_free + k]), 1e-9)
        return xs

    # ------------------------------------------------------------------------------------------------ results
    def _solution(self, ev: "_Eval", ok: bool, msg: str, res: float, it: int, warnings: list[str]) -> Solution:
        comps: dict[str, dict] = {}
        profiles = {}
        for nid, m in self.models.items():
            try:
                comps[nid] = {k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                              for k, v in m.outputs(ev.local(nid), ev).items()}
            except (ValueError, ZeroDivisionError) as exc:
                comps[nid] = {}
                warnings.append(f"{nid}: {exc}")
            comps[nid]["type"] = m.type
            if "profile" in m.slow:
                profiles[nid] = m.slow["profile"]
        stations = []
        for s, plist in enumerate(self.st_ports):
            p = ev.P[s]
            st = ev.station_state(s)
            flow = sum(max(ev.port_flow(n, pn), 0.0) for n, pn in plist)
            d = dict(index=s, p=float(p), mdot=float(flow), ports=[f"{n}.{pn}" for n, pn in plist])
            if st is not None:
                d.update(T=float(st.T), fluid=st.fluid or "combustion gas", ox_frac=float(st.ox_frac))
                if st.kind == "fluid":
                    a = props(st.fluid).at(p, st.h, st.T)
                    d.update(rho=float(a["rho"]), phase=a["phase"], h=float(st.h))
                else:
                    d.update(rho=float(density(st, p)), phase="gas", cp=st.cp, gamma=st.gamma)
            stations.append(d)
        shafts = []
        for k, grp in enumerate(self.shaft_groups):
            N = ev.N[k]
            P_t = sum(self.models[i].shaft_power(ev.local(i), ev) for i in grp if self.models[i].shaft_role == "turbine")
            P_p = -sum(self.models[i].shaft_power(ev.local(i), ev) for i in grp if self.models[i].shaft_role == "pump")
            aux = -sum(self.models[i].shaft_power(ev.local(i), ev) for i in grp if self.models[i].type == "shaft")
            shafts.append(dict(members=grp, speed=float(N), rpm=float(N * 30 / np.pi), turbine_power=float(P_t),
                               pump_power=float(P_p), aux_power=float(aux)))
            for i in grp:
                if self.models[i].type == "shaft":
                    comps[i].update(speed=float(N), power=float(P_t))
        warnings += self._checks(comps, stations)
        free = {f"{f['node']}.{f['param']}": float(ev.free[k]) for k, f in enumerate(self.free)}
        edges = []
        for e in self.edge_list:
            s = self.station_of[(e["from"]["node"], e["from"]["port"])]
            edges.append(dict(id=e["id"], station=s, mdot=float(abs(ev.port_flow(e["from"]["node"], e["from"]["port"])))))
        return Solution(ok, msg, stations, comps, shafts, warnings, free, res, it, profiles, edges)

    def _checks(self, comps: dict, stations: list[dict]) -> list[str]:
        out = []
        for nid, c in comps.items():
            t = c.get("type")
            if t == "pump_centrifugal" and c.get("npsh_available", 1.0) < 0:
                out.append(f"{nid}: inlet below the vapour pressure (cavitation)")
            known = t == "inducer" or (t == "pump_centrifugal" and nid in self.models
                                       and self.models[nid].omega_ss_known())
            if known and c.get("npsh_available", 1.0) >= 0 and c.get("npsh_margin", 1.0) < 0:
                out.append(f"{nid}: NPSH available {c['npsh_available']:.1f} m below the {c['npsh_required']:.1f} m "
                           "required at its suction specific speed (cavitation)")
            if t == "pump_centrifugal" and not 0.5 < c.get("flow_ratio", 1.0) < 1.5:
                out.append(f"{nid}: running at {c['flow_ratio']:.2f} of design flow (curve extrapolated)")
            if t == "injector" and c.get("stiffness", 1.0) < 0.15 and c.get("dp", 0) > 0:
                out.append(f"{nid}: injector drop {100 * c['stiffness']:.0f} % of downstream pressure (< 15 %, "
                           "chugging risk)")
            if t == "gas_generator":
                T_max = self.param(nid, "T_max")
                if T_max and c.get("T_out", 0) > T_max:
                    out.append(f"{nid}: drive gas {c['T_out']:.0f} K above the {T_max:.0f} K limit")
            if t == "turbine_impulse" and c.get("pressure_ratio", 2) < 1.0:
                out.append(f"{nid}: pressure rises across the turbine (it absorbs power)")
            if c.get("mdot", 0.0) < -1e-6:
                out.append(f"{nid}: reverse flow")
        for s in stations:
            if s["p"] <= 0:
                out.append(f"station {s['index']} ({', '.join(s['ports'])}): non-positive pressure")
        return out


class _Eval:
    """One evaluation of the system at ``x``: station pressures, flows, speeds and swept port states."""

    def __init__(self, net: Network) -> None:
        self.net = net
        self.p_ref = net.p_ref
        self.port_state: dict[tuple[str, str], State] = {}
        self.frozen_states: dict[tuple[str, str], State] = {}
        self._in_cache: dict = {}
        self.count = 0

    @property
    def m_ref(self) -> float:
        return self.net.m_ref

    def set(self, x: np.ndarray) -> None:
        net = self.net
        self.x = x
        self.P = x[: net.n_st]
        self.N = x[net.i_shaft: net.i_shaft + len(net.shaft_groups)]
        self.free = x[net.i_free: net.i_free + len(net.free)]
        net._free_val = {(f["node"], f["param"]): float(v) for f, v in zip(net.free, self.free)}
        self._flows = {nid: m.flows(x[net.slices[nid]]) for nid, m in net.models.items()}
        self._in_cache = {}

    def local(self, nid: str) -> np.ndarray:
        return self.x[self.net.slices[nid]]

    def model(self, nid: str) -> Model:
        return self.net.models[nid]

    def p(self, nid: str, port: str) -> float:
        return float(self.P[self.net.station_of[(nid, port)]])

    def speed(self, nid: str) -> float:
        return float(self.N[self.net.shaft_of[nid]])

    def port_flow(self, nid: str, port: str) -> float:
        return self._flows[nid].get(port, 0.0)

    def outlet_state(self, nid: str, port: str) -> State | None:
        return self.port_state.get((nid, port)) or self.frozen_states.get((nid, port))

    def station_state(self, s: int) -> State | None:
        net = self.net
        parts = []
        for n, pn in net.st_ports[s]:
            if net.ports(n)[pn]["direction"] == "out":
                st = self.outlet_state(n, pn)
                if st is not None:
                    parts.append((max(self.port_flow(n, pn), 0.0), st))
        return mix(parts, float(self.P[s])) if parts else None

    def inlet(self, nid: str, port: str) -> State:
        s = self.net.station_of[(nid, port)]
        if s not in self._in_cache:
            st = self.station_state(s)
            if st is None:
                st = self._default_state(nid, port)
            self._in_cache[s] = st
        return self._in_cache[s]

    def _default_state(self, nid: str, port: str) -> State:
        """Before the first sweep reaches a port: the propellant that feeds it (ox side if the port says so)."""
        from .props import liquid_state
        fid = self.net.fuel if ("fuel" in port or "coolant" in port or "fuel" in nid) else self.net.ox
        T = 90.0 if fid == "LOX" else 21.0 if fid == "LH2" else 112.0 if fid == "LCH4" else 293.15
        return liquid_state(fid, max(self.p(nid, port), 2e5), T, self.net.ox_fraction(fid))

    def sweep(self, passes: int = 2) -> None:
        net = self.net
        for _ in range(passes):
            for nid in net.order:
                m = net.models[nid]
                try:
                    outs = m.outlets(self.local(nid), self)
                except (ValueError, ZeroDivisionError):
                    continue
                for pn, st in outs.items():
                    self.port_state[(nid, pn)] = st
                for pn in outs:
                    self._in_cache.pop(net.station_of[(nid, pn)], None)

    def residuals(self, x: np.ndarray) -> np.ndarray:
        net = self.net
        self.count += 1
        self.set(x)
        self.port_state = dict(self.frozen_states)
        self.sweep(passes=1)
        r = np.zeros(net.n_x)
        k = 0
        for s in range(net.n_st):
            if s in net.fixed_p:
                r[k] = (self.P[s] - net.fixed_p[s]) / net.p_ref
            else:
                r[k] = sum(self.port_flow(n, pn) for n, pn in net.st_ports[s]) / net.m_ref
            k += 1
        for nid in net.models:
            m = net.models[nid]
            try:
                rr = m.residuals(self.local(nid), self)
            except (ValueError, ZeroDivisionError, FloatingPointError):
                rr = [1e3] * len(m.unknowns)
            for v in rr:
                if k < len(r):
                    r[k] = v
                k += 1
        for g_i, grp in enumerate(net.shaft_groups):
            P = 0.0
            for i in grp:
                try:
                    P += net.models[i].shaft_power(self.local(i), self)
                except (ValueError, ZeroDivisionError):
                    P += 1e9
            if k < len(r):
                if net.speed_set is not None:          # transient: the speed is a state, the power is an output
                    N_set = net.speed_set[g_i]
                    r[k] = (self.N[g_i] - N_set) / max(abs(N_set), 100.0)
                else:
                    r[k] = P / net.P_ref
            k += 1
        for t in net.targets:
            m = net.models[t["node"]]
            try:
                val = m.outputs(self.local(t["node"]), self)[t["output"]]
            except (KeyError, ValueError, ZeroDivisionError):
                val = 0.0
            scale = abs(t["value"]) if t["value"] else 1.0
            if k < len(r):
                r[k] = (val - t["value"]) / scale
            k += 1
        return np.where(np.isfinite(r), r, 1e6)


class _Guess:
    """Starting point for the Newton solve.

    1. Flows: the chamber flow at a guessed pressure is split by a mid-range mixture ratio and pushed upstream as
       *demands*: a through component passes its outlet demand to its inlet, a gas generator splits its demand by its
       own mixture ratio, a junction shares a demand among the out-ports that feed it, a tee adds the demands of its
       branches.  Open-cycle drive gas leaving through a fixed-pressure boundary is guessed at 3 % of the chamber
       flow.
    2. Pressures: walked upstream from the chamber injector end through every loss, turbine pressure ratio and
       gas-generator injector, and downstream from the tanks to the pump inlets.
    3. Shaft speeds: the speed at which each pump gives the rise the walk asks of it.
    """

    GG_OPEN = 0.03

    def __init__(self, net: Network) -> None:
        self.net = net
        self._chamber_flows()

    # ------------------------------------------------------------------------------------------------ helpers
    def _out_ports(self, s: int):
        net = self.net
        return [(n, pn) for n, pn in net.st_ports[s] if net.ports(n)[pn]["direction"] == "out"]

    def _in_ports(self, s: int):
        net = self.net
        return [(n, pn) for n, pn in net.st_ports[s] if net.ports(n)[pn]["direction"] == "in"]

    def _chamber_flows(self) -> None:
        net = self.net
        chambers = [m for m in net.models.values() if m.type == "chamber"]
        pumps = [m for m in net.models.values() if m.type == "pump_centrifugal"]
        tanks = [m for m in net.models.values() if m.type in ("tank_regulated", "tank_blowdown")]
        p_tank = max([t.p_tank() for t in tanks] or [3e6])
        self.mr = 2.5
        if pumps:
            rise = 0.0
            for pmp in pumps:
                fid = net.ox if "ox" in pmp.id else net.fuel
                rho = {"LOX": 1141.0, "LH2": 70.0, "LCH4": 420.0}.get(fid, 800.0)
                rise = max(rise, pmp.design(rho)[0])
            self.pc = 0.55 * (p_tank + rise)
        else:
            self.pc = 0.6 * p_tank
        self.m_o, self.m_f = 2.5, 1.0
        self.cs = 1700.0
        if chambers:
            ch = chambers[0]
            try:
                from ..thermo.cea import PAIRS
                from .models import PAIR_KEYS
                pr = PAIRS[PAIR_KEYS[(net.ox, net.fuel)]]
                self.mr = 0.5 * (pr.mr_lo + pr.mr_hi)
                self.cs = ch.lut()(self.mr, self.pc)[0] * ch.par("eta_cstar")
            except (GraphError, ValueError, KeyError):
                pass
            m = self.pc * ch.par("A_t") / self.cs
            self.m_o, self.m_f = m * self.mr / (1 + self.mr), m / (1 + self.mr)
        self.m_scale = max(self.m_o + self.m_f, 1e-3)
        self.P_scale = 1e3
        self._demands()
        Pp = 0.0
        for pmp in pumps:
            Pp += self.d_through.get(pmp.id, self.m_f) * self.pc / 500.0
        self.P_scale = max(Pp / max(len(net.shaft_groups), 1), 1e3)

    def _demands(self) -> None:
        """Flow demand of every component inlet, pushed upstream from the chambers and exhaust boundaries."""
        net = self.net
        closed = [m for m in net.models.values() if m.type == "gas_generator" and self._feeds_chamber(m.id)]
        mr_gg = {m.id: (net.param(m.id, "mixture_ratio") or 0.5) for m in net.models.values()
                 if m.type == "gas_generator"}
        pb_ox = sum(mr_gg[m.id] * self.m_f / max(len(closed), 1) for m in closed) if closed else 0.0
        port_d: dict[tuple[str, str], float] = {}            # demand on each in-port
        for m in net.models.values():
            if m.type == "chamber":
                if closed:
                    port_d[(m.id, "ox_in")] = max(self.m_o - pb_ox, 0.05 * self.m_o)
                    port_d[(m.id, "fuel_in")] = self.m_f + pb_ox
                else:
                    port_d[(m.id, "ox_in")], port_d[(m.id, "fuel_in")] = self.m_o, self.m_f
        self.d_through: dict[str, float] = {}
        self.d_port = port_d
        for _ in range(4 * len(net.nodes)):
            changed = False
            for s in range(net.n_st):
                outs, ins = self._out_ports(s), self._in_ports(s)
                if not outs:
                    continue
                if s in net.fixed_p and not ins:
                    total = self.GG_OPEN * (self.m_o + self.m_f)
                else:
                    total = sum(port_d.get(ip, 0.0) for ip in ins)
                # share among feeders: recirculating feeders (fed from this station's downstream) get a fifth
                share = {op: 1.0 for op in outs}
                if len(outs) > 1:
                    for op in outs:
                        if net.nodes[op[0]]["type"] == "turbine_impulse" and self._recirculates(op[0], s):
                            share[op] = 0.25
                tot_w = sum(share.values())
                for (n, pn) in outs:
                    d = total * share[(n, pn)] / tot_w
                    m = net.models[n]
                    if m.type in ("chamber",):
                        continue
                    if m.type == "gas_generator":
                        r = mr_gg[n]
                        new = {(n, "ox_in"): d * r / (1 + r), (n, "fuel_in"): d / (1 + r)}
                    elif hasattr(m, "pin"):
                        new = {(n, m.pin): d}
                        self.d_through[n] = d
                    else:
                        new = {}
                        self.d_through[n] = d
                    for k, v in new.items():
                        if abs(port_d.get(k, -1.0) - v) > 1e-9 * max(v, 1.0):
                            port_d[k] = v
                            changed = True
            if not changed:
                break

    def _feeds_chamber(self, nid: str) -> bool:
        net = self.net
        seen, stack = set(), [nid]
        while stack:
            a = stack.pop()
            if a in seen:
                continue
            seen.add(a)
            if net.nodes[a]["type"] == "chamber" and a != nid:
                return True
            for pn, p in net.ports(a).items():
                if p["kind"] == "fluid" and p["direction"] == "out" and (a, pn) in net.station_of:
                    for b, bp in self._in_ports(net.station_of[(a, pn)]):
                        stack.append(b)
        return False

    def _recirculates(self, nid: str, station: int) -> bool:
        """True when ``nid`` is fed (through its inlet) from somewhere downstream of ``station``."""
        net = self.net
        seen, stack = set(), [b for b, _ in self._in_ports(station)]
        while stack:
            a = stack.pop()
            if a in seen:
                continue
            seen.add(a)
            if a == nid:
                return True
            if net.nodes[a]["type"] in ("chamber",):
                continue
            for pn, p in net.ports(a).items():
                if p["kind"] == "fluid" and p["direction"] == "out" and (a, pn) in net.station_of:
                    for b, bp in self._in_ports(net.station_of[(a, pn)]):
                        stack.append(b)
        return False

    def flow(self, nid: str, which: str | None = None) -> float:
        if which == "ox":
            return self.d_port.get((nid, "ox_in"), self.m_o)
        if which == "fuel":
            return self.d_port.get((nid, "fuel_in"), self.m_f)
        if which == "total":
            return self.m_o + self.m_f
        if nid in self.d_through:
            return self.d_through[nid]
        m = self.net.models[nid]
        if m.type.startswith("tank"):
            return self.m_o if self.net.port_fluid(nid, "outlet") == self.net.ox else self.m_f
        return 0.5 * (self.m_o + self.m_f)

    # ------------------------------------------------------------------------------------------------ vector
    def vector(self) -> np.ndarray:
        net = self.net
        x = np.zeros(net.n_x)
        for nid, m in net.models.items():
            x[net.slices[nid]] = m.guess(self)
        for nid, m in net.models.items():
            if m.type.startswith("tank"):                            # tank flow = what leaves its station
                s = net.station_of[(nid, "outlet")]
                x[net.slices[nid]] = sum(self.d_port.get(ip, 0.0) for ip in self._in_ports(s)) or x[net.slices[nid]]
        for k, grp in enumerate(net.shaft_groups):
            Ns = [net.param(i, "N_design") for i in grp if net.nodes[i]["type"] == "pump_centrifugal"]
            x[net.i_shaft + k] = float(np.mean(Ns)) if Ns else 1000.0
        for k, f in enumerate(net.free):
            if f["param"] == "position":
                v = f.get("guess", float(net.nodes[f["node"]].get("options", {}).get("position", 1.0)))
            else:
                v = net.param(f["node"], f["param"], f.get("guess"))
            if v is None:
                raise GraphError(f"free parameter {f['node']}.{f['param']} needs a starting value")
            x[net.i_free + k] = float(v)
        self._pressures(x)
        return x

    def _pressures(self, x: np.ndarray) -> None:
        net = self.net
        known: dict[int, float] = dict(net.fixed_p)
        for nid, m in net.models.items():
            if m.type.startswith("tank"):
                known[net.station_of[(nid, "outlet")]] = m.p_tank()
        ev = _Eval(net)
        x[: net.n_st] = self.pc
        for s, v in known.items():
            x[s] = v
        for nid, m in net.models.items():                           # chamber stations
            if m.type == "chamber":
                x[net.station_of[(nid, "throat")]] = self.pc
                for pn in ("ox_in", "fuel_in"):
                    x[net.station_of[(nid, pn)]] = self.pc * 1.03
                    known[net.station_of[(nid, pn)]] = self.pc * 1.03
                known[net.station_of[(nid, "throat")]] = self.pc
        # forward from tanks through passive parts to the pump inlets (pressure-fed lines end at the chamber)
        for _ in range(len(net.nodes)):
            ev.set(x)
            ev.sweep(1)
            changed = False
            for nid in net.order:
                m = net.models[nid]
                if m.type == "pump_centrifugal":                     # booster pump feeding another pump
                    si, so = net.station_of[(nid, m.pin)], net.station_of[(nid, m.pout)]
                    if si in known and so not in known and self._upstream_of_pump(so):
                        rho = m.rho_in(ev)
                        x[so] = x[si] + m.design(rho)[0]
                        known[so] = x[so]
                        changed = True
                    continue
                if not hasattr(m, "pin") or m.type == "turbine_impulse":
                    continue
                si, so = net.station_of[(nid, m.pin)], net.station_of[(nid, m.pout)]
                if si in known and so not in known and self._upstream_of_pump(so):
                    try:
                        x[so] = x[si] - max(m.dp(ev.local(nid)[0], ev), 0.0)
                    except (ValueError, ZeroDivisionError):
                        x[so] = x[si]
                    known[so] = x[so]
                    changed = True
            if not changed:
                break
        # backward from the chambers: each inlet station needs its outlet pressure plus the component's drop
        for _ in range(len(net.nodes)):
            ev.set(x)
            ev.sweep(1)
            changed = False
            for nid in reversed(net.order):
                m = net.models[nid]
                ports = []
                if m.type == "gas_generator":
                    so = net.station_of[(nid, "outlet")]
                    if so in known:
                        for pn in ("ox_in", "fuel_in"):
                            ports.append((net.station_of[(nid, pn)], x[so] * (1.0 + m.inj())))
                elif m.type == "turbine_impulse":
                    si, so = net.station_of[(nid, "inlet")], net.station_of[(nid, "outlet")]
                    if so in known and m.model() == "pressure_ratio":
                        ports.append((si, x[so] * m.par("pressure_ratio")))
                    elif so in known:
                        ports.append((si, x[so] * 2.0))
                elif hasattr(m, "pin") and m.type != "pump_centrifugal":
                    si, so = net.station_of[(nid, m.pin)], net.station_of[(nid, m.pout)]
                    if so in known:
                        try:
                            ports.append((si, x[so] + max(m.dp(ev.local(nid)[0], ev), 0.0)))
                        except (ValueError, ZeroDivisionError):
                            ports.append((si, x[so]))
                for si, v in ports:
                    if si in net.fixed_p or (si in known and known[si] >= v - 1.0):
                        continue
                    if si in known and any(net.nodes[n]["type"].startswith("tank") for n, _ in net.st_ports[si]):
                        continue
                    x[si] = v
                    known[si] = v
                    changed = True
            if not changed:
                break
        # shaft speeds for the asked pump rises
        ev.set(x)
        ev.sweep(2)
        for k, grp in enumerate(net.shaft_groups):
            Ns = []
            for i in grp:
                m = net.models[i]
                if m.type != "pump_centrifugal":
                    continue
                si, so = net.station_of[(i, "inlet")], net.station_of[(i, "outlet")]
                need = x[so] - x[si]
                rho = m.rho_in(ev)
                dp_d, Q_d = m.design(rho)
                sr = float(m.opt.get("shutoff_ratio", 1.25))
                q = ev.local(i)[0] / rho / Q_d
                val = (need / dp_d + (sr - 1.0) * q * q) / sr
                Ns.append(m.par("N_design") * np.sqrt(max(val, 0.05)))
            if Ns:
                x[net.i_shaft + k] = max(Ns)

    def _upstream_of_pump(self, s: int) -> bool:
        net = self.net
        seen, stack = set(), [s]
        while stack:
            a = stack.pop()
            if a in seen:
                continue
            seen.add(a)
            for n, pn in self._in_ports(a):
                t = net.nodes[n]["type"]
                if t == "pump_centrifugal":
                    return True
                m = net.models[n]
                if hasattr(m, "pout") and t not in ("turbine_impulse",):
                    stack.append(net.station_of[(n, m.pout)])
        return False
