"""Cycle sandbox *3D Model* panel: the selected component drawn to its own size, with the solved state on it."""
from __future__ import annotations

import numpy as np
import pyvista as pv
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QFileDialog, QHBoxLayout, QPushButton, QVBoxLayout, QWidget

from ..modules.component_shapes import FIELD_TITLES, build, merged
from .viewport3d import Viewport3D

ROTOR_PARTS = {"hub", "blade", "vane", "disk", "shaft"}          # the parts that spin in pumps, inducers and turbines

# solved outputs shown in the readout card, per component type: (key, label, scale, unit, digits)
READOUT = {
    "pump_centrifugal": [("speed", "Speed", 60 / (2 * np.pi), "rpm", 0), ("dp", "Pressure rise", 1e-6, "MPa", 2),
                         ("power", "Power", 1e-6, "MW", 2), ("npsh_available", "NPSH available", 1, "m", 0)],
    "inducer": [("dp", "Pressure rise", 1e-6, "MPa", 3), ("mdot", "Flow", 1, "kg/s", 1)],
    "turbine_impulse": [("speed", "Speed", 60 / (2 * np.pi), "rpm", 0), ("power", "Power", 1e-6, "MW", 2),
                        ("pressure_ratio", "Pressure ratio", 1, "", 2), ("T_inlet", "Inlet T", 1, "K", 0)],
    "chamber": [("p_c", "Chamber pressure", 1e-6, "MPa", 2), ("mixture_ratio", "O/F", 1, "", 2),
                ("T_c", "Gas T", 1, "K", 0), ("mdot", "Flow", 1, "kg/s", 1)],
    "nozzle_bell": [("thrust", "Thrust", 1e-3, "kN", 0), ("isp", "Isp", 1, "s", 1), ("CF", "CF", 1, "", 3),
                    ("p_exit", "Exit pressure", 1e-3, "kPa", 1)],
    "regen_jacket": [("Q", "Heat load", 1e-6, "MW", 1), ("T_wg_max", "Max gas-side wall T", 1, "K", 0),
                     ("q_max", "Peak heat flux", 1e-6, "MW/m$^2$", 1), ("T_out", "Coolant out", 1, "K", 0)],
    "gas_generator": [("p_gg", "Pressure", 1e-6, "MPa", 2), ("mixture_ratio", "O/F", 1, "", 3),
                      ("T_out", "Gas T", 1, "K", 0), ("mdot", "Flow", 1, "kg/s", 2)],
    "injector": [("dp", "Pressure drop", 1e-6, "MPa", 2), ("stiffness", "Stiffness Δp/pc", 100, "%", 0),
                 ("velocity", "Injection velocity", 1, "m/s", 1), ("mdot", "Flow", 1, "kg/s", 2)],
    "shaft": [("speed", "Speed", 60 / (2 * np.pi), "rpm", 0), ("power", "Power", 1e-6, "MW", 2)],
    "tank_regulated": [("p", "Pressure", 1e-6, "MPa", 3), ("mdot", "Outflow", 1, "kg/s", 1)],
    "tank_blowdown": [("p", "Pressure", 1e-6, "MPa", 3), ("mdot", "Outflow", 1, "kg/s", 1)],
    "line": [("dp", "Pressure drop", 1e-3, "kPa", 1), ("velocity", "Velocity", 1, "m/s", 2),
             ("mdot", "Flow", 1, "kg/s", 2)],
}
DEFAULT_READOUT = [("position", "Position", 100, "% open", 0), ("dp", "Pressure drop", 1e-3, "kPa", 1),
                   ("mdot", "Flow", 1, "kg/s", 2)]


class ComponentView(QWidget):
    """3-D viewport for one component of the sandbox graph."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.vp = Viewport3D(self)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.vp, 1)
        bar = QHBoxLayout()
        bar.setContentsMargins(6, 0, 6, 4)
        self.flow_btn = QPushButton("Play Flow")
        self.flow_btn.setCheckable(True)
        self.flow_btn.setToolTip("animate the flow through the component")
        self.flow_btn.toggled.connect(self.set_flow)
        bar.addWidget(self.flow_btn)
        bar.addStretch(1)
        lay.addLayout(bar)
        self.flowing = False
        self.phase = 0.0
        self._timer = QTimer(self)
        self._timer.setInterval(40)
        self._timer.timeout.connect(self._tick)
        self._path: np.ndarray | None = None
        self._rotors: list = []
        self._merged: list = []
        self._flow_color = "#ffffff"
        self.graph: dict | None = None
        self.nid: str | None = None
        self.sol = None
        self.model = None
        self._field_keys: list[str] = []
        self._stale = False
        self.vp.resetRequested.connect(lambda: self.vp.reset_view(zoom=1.2))
        self.vp.cutawayChanged.connect(lambda *_: self.refresh(keep_camera=True))
        self.vp.fieldChanged.connect(lambda *_: self._paint(keep_camera=True))
        self.vp.b_snap.clicked.connect(self._snapshot)
        self.vp.set_fields([])
        self.vp.set_colorbar_visible(False)
        self.vp._next_cut()                                      # open at a 90° cutaway so the inside shows

    def show_component(self, graph: dict | None, nid: str | None, sol=None) -> None:
        """Draw ``nid`` (or clear the view); deferred while the panel is hidden."""
        same = nid is not None and nid == self.nid
        self.graph, self.nid, self.sol = graph, nid, sol
        if not self.isVisible():
            self._stale = True
            return
        self.refresh(keep_camera=same)

    def showEvent(self, ev) -> None:
        super().showEvent(ev)
        if self._stale:
            self.refresh()

    def refresh(self, keep_camera: bool = False) -> None:
        self._stale = False
        pl = self.vp.plotter
        if self.graph is None or self.nid is None or not any(n["id"] == self.nid for n in self.graph["nodes"]):
            pl.clear()
            self.model = None
            self.vp.set_readout(["Select a component on the diagram", "to see its 3D model."])
            self.vp.set_caption("")
            self.vp.set_fields([])
            self.vp.set_colorbar_visible(False)
            self.vp.render()
            return
        try:
            self.model = build(self.graph, self.nid, self.sol, self.vp.cutaway)
        except Exception as e:                                   # noqa: BLE001 - a half-edited part must not break the UI
            pl.clear()
            self.model = None
            self.vp.set_readout([f"No model: {e}"])
            self.vp.render()
            return
        self._path = None
        self._state_fields()
        keys = list(self.model.fields)
        if keys != self._field_keys:
            self._field_keys = keys
            self.vp.set_fields([FIELD_TITLES[k][0] for k in keys])
        self._paint(keep_camera)

    def _paint(self, keep_camera: bool = False) -> None:
        m = self.model
        if m is None:
            return
        pl = self.vp.plotter
        cam = pl.camera_position if keep_camera else None
        pl.clear()
        field = self._field_keys[self.vp.field_index] if self._field_keys else None
        if field == "model":                                     # plain appearance, no field painted on it
            field = None
        lo = hi = None
        if field:
            vals = np.concatenate([np.asarray(p.mesh.point_data[field]) for p in m.parts if field in p.mesh.point_data])
            vals = vals[np.isfinite(vals)]
            if vals.size:
                lo, hi = float(vals.min()), float(vals.max())
                if hi <= lo:
                    hi = lo + 1.0
        self._merged = merged(m)
        self._rotors = []
        spin = self._node_type() in ("pump_centrifugal", "inducer", "turbine_impulse", "shaft")
        for p in self._merged:
            if spin and p.name in ROTOR_PARTS:
                self._rotors.append((p.mesh, np.array(p.mesh.points)))
            if p.scalars and p.scalars != field and p.name == "coolant":
                continue                                         # the coolant layer shows only with its own field
            if field and lo is not None and field in p.mesh.point_data:
                pl.add_mesh(p.mesh, scalars=field, cmap=FIELD_TITLES[field][1], clim=(lo, hi), nan_color="#7d848c",
                            show_scalar_bar=False, smooth_shading=True, opacity=p.opacity, name=p.name + "_field")
            else:
                pl.add_mesh(p.mesh, color=p.color, opacity=p.opacity, smooth_shading=True, name=p.name)
        self.vp.legend_labels = dict(m.labels)
        if field and lo is not None:
            self.vp.set_colorbar(FIELD_TITLES[field][1], lo, hi, FIELD_TITLES[field][0])
            self.vp.set_colorbar_visible(True)
        else:
            self.vp.set_colorbar_visible(False)
        self.vp.set_readout([m.title] + self._readout())
        self.vp.set_caption(m.size + ("  ·  " + "; ".join(m.notes) if m.notes else ""))
        if cam is not None:
            pl.camera_position = cam
            self.vp.render()
        else:
            self.vp.reset_view(zoom=1.2)

    def _port_states(self) -> tuple[dict | None, dict | None]:
        """(inlet, outlet) solved stations of the selected component: mean pressure and temperature on its in and out
        ports; ``None`` where the component has no such port or the solution lacks it."""
        if self.sol is None or self.graph is None:
            return None, None
        from ..core.engine_graph import edit as E
        node = next(n for n in self.graph["nodes"] if n["id"] == self.nid)
        spec = E.library()[node["type"]]["ports"]
        sides: dict[str, list[dict]] = {"in": [], "out": []}
        for st in self.sol.stations:
            for port in st["ports"]:
                nid, _, pname = port.partition(".")
                if nid != self.nid:
                    continue
                ps = next((q for q in spec if q["name"] == pname and q["kind"] == "fluid"), None)
                if ps is not None:
                    sides["in" if ps["direction"] == "in" else "out"].append(st)
        mean = lambda sts: None if not sts else {"p": float(np.mean([s["p"] for s in sts])),
                                                 "T": float(np.mean([s["T"] for s in sts]))}
        return mean(sides["in"]), mean(sides["out"])

    def _state_fields(self) -> None:
        """Pressure and fluid-temperature fields for every component: the solved inlet and outlet values blended along
        the flow axis (the longest extent of the model), so any part can be colored by pressure or temperature."""
        m = self.model
        if m is None:
            return
        m.fields = {"model": FIELD_TITLES["model"], **{k: v for k, v in m.fields.items() if k != "model"}}
        if self.sol is None:
            return
        a, b = self._port_states()
        a = a or b
        b = b or a
        if a is None:
            return
        pts = np.vstack([np.asarray(p.mesh.points) for p in m.parts if p.mesh.n_points])
        lo, hi = pts.min(axis=0), pts.max(axis=0)
        ax = int(np.argmax(hi - lo))
        span = max(float(hi[ax] - lo[ax]), 1e-12)
        for part in m.parts:
            t = (np.asarray(part.mesh.points)[:, ax] - lo[ax]) / span
            part.mesh.point_data["p"] = (a["p"] + (b["p"] - a["p"]) * t) * 1e-6
            part.mesh.point_data["T_f"] = a["T"] + (b["T"] - a["T"]) * t
        m.fields = {"model": FIELD_TITLES["model"], "p": FIELD_TITLES["p"], "T_f": FIELD_TITLES["T_f"],
                    **{k: v for k, v in m.fields.items() if k not in ("model", "p", "T_f")}}

    # ---- 3D flow animation -------------------------------------------------------------------------------
    def set_flow(self, on: bool) -> None:
        """Play or stop the flow: arrow heads travel along the component's flow axis, in the color of its fluid."""
        self.flowing = on
        self.flow_btn.setText("Pause Flow" if on else "Play Flow")
        if on:
            self._timer.start()
        else:
            self._timer.stop()
            self.vp.plotter.remove_actor("flow", render=False) if "flow" in self.vp.plotter.actors else None
            self.vp.render()

    def _flow_setup(self) -> None:
        """Flow axis from the model: the longest extent of its parts, through the middle; color from the fluid."""
        self._path = None
        m = self.model
        if m is None or self.graph is None:
            return
        pts = np.vstack([np.asarray(p.mesh.points) for p in m.parts if p.mesh.n_points])
        lo, hi = pts.min(axis=0), pts.max(axis=0)
        ax = int(np.argmax(hi - lo))
        mid = (lo + hi) / 2.0
        a, b = mid.copy(), mid.copy()
        a[ax], b[ax] = lo[ax], hi[ax]
        self._path = np.vstack([a, b])
        node = next(n for n in self.graph["nodes"] if n["id"] == self.nid)
        from .cycle_sandbox import MEDIUM_COLORS
        med = next((e.get("medium") for e in self.graph["edges"]
                    if e["kind"] == "fluid" and self.nid in (e["from"]["node"], e["to"]["node"])), None)
        if node["type"] in ("chamber", "nozzle_bell", "gas_generator"):
            self._flow_color = MEDIUM_COLORS["gas"]
        elif med == self.graph["propellants"]["oxidizer"]:
            self._flow_color = MEDIUM_COLORS["ox"]
        elif med is not None:
            self._flow_color = MEDIUM_COLORS["fuel"]
        self._size = float(np.linalg.norm(hi - lo))

    def _node_type(self) -> str:
        if self.graph is None or self.nid is None:
            return ""
        return next((n["type"] for n in self.graph["nodes"] if n["id"] == self.nid), "")

    def _particles(self) -> None:
        """Reduced-order flow visualization (not CFD): particles follow an analytic streamline field along the flow axis,
        with a swirl in rotating machinery, colored by the solved pressure (or the fluid color without a solution)."""
        if self._path is None:
            return
        rng = np.random.default_rng(3)
        n = 420
        self._u = rng.random(n)                                  # axial position (0..1) at phase 0
        self._rf = np.sqrt(rng.random(n)) * 0.85                 # radial fraction of the bore
        self._th = rng.random(n) * 2 * np.pi
        self._vf = 0.7 + 0.6 * rng.random(n)                     # each particle's own speed
        pts = np.vstack([np.asarray(p.mesh.points) for p in self.model.parts if p.mesh.n_points])
        lo, hi = pts.min(axis=0), pts.max(axis=0)
        ext = hi - lo
        ax = int(np.argmax(ext))
        self._ax = ax
        self._lo, self._hi = lo, hi
        others = [i for i in range(3) if i != ax]
        self._R = 0.45 * float(min(ext[others])) if ext[others].min() > 0 else 0.1 * float(ext[ax])
        self._swirl = 1.8 if self._node_type() in ("pump_centrifugal", "inducer", "turbine_impulse") else 0.25

    def _tick(self) -> None:
        if not self.isVisible() or self.model is None:
            return
        if self._path is None:
            self._flow_setup()
            if self._path is None:
                return
            self._particles()
        self.phase = (self.phase + 0.012) % 1.0
        for mesh, orig in getattr(self, "_rotors", []):          # turbomachinery spins about its axis
            a = self.phase * 2 * np.pi * 8.0
            c, s_ = np.cos(a), np.sin(a)
            q = orig.copy()
            ax = self._ax
            i, j = [k for k in range(3) if k != ax]
            q[:, i], q[:, j] = c * orig[:, i] - s_ * orig[:, j], s_ * orig[:, i] + c * orig[:, j]
            mesh.points = q
        u = (self._u + self.phase * self._vf * 3.0) % 1.0
        ax = self._ax
        i, j = [k for k in range(3) if k != ax]
        mid = (self._lo + self._hi) / 2.0
        th = self._th + self._swirl * 2 * np.pi * u + self.phase * 2 * np.pi * 4.0 * (self._swirl > 1.0)
        pos = np.zeros((len(u), 3))
        pos[:, ax] = self._lo[ax] + (self._hi[ax] - self._lo[ax]) * u
        pos[:, i] = mid[i] + self._R * self._rf * np.cos(th)
        pos[:, j] = mid[j] + self._R * self._rf * np.sin(th)
        cloud = pv.PolyData(pos)
        a_, b_ = self._port_states()
        a_ = a_ or b_
        b_ = b_ or a_
        if a_ is not None:
            cloud["p"] = (a_["p"] + (b_["p"] - a_["p"]) * u) * 1e-6
            self.vp.plotter.add_mesh(cloud, scalars="p", cmap="turbo", point_size=5, render_points_as_spheres=True,
                                     show_scalar_bar=False, name="flow")
        else:
            self.vp.plotter.add_mesh(cloud, color=self._flow_color, point_size=5, render_points_as_spheres=True,
                                     name="flow")
        self.vp.render()

    def _readout(self) -> list[str]:
        if self.sol is None:
            return ["Solve to see its operating state."]
        node = next(n for n in self.graph["nodes"] if n["id"] == self.nid)
        c = self.sol.components.get(self.nid, {})
        lines = []
        for key, label, scale, unit, dg in READOUT.get(node["type"], DEFAULT_READOUT):
            v = c.get(key)
            if isinstance(v, (int, float)) and np.isfinite(v):
                lines.append(f"{label}: {v * scale:,.{dg}f} {unit}".rstrip())
        return lines[:3]

    def _snapshot(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Save 3D snapshot", f"{self.nid or 'component'}.png", "PNG (*.png)")
        if path:
            self.vp.snapshot(path)
