"""Cycle sandbox *3D Model* panel: the selected component drawn to its own size, with the solved state on it."""
from __future__ import annotations

import numpy as np
from PySide6.QtWidgets import QFileDialog, QVBoxLayout, QWidget

from ..modules.component_shapes import FIELD_TITLES, build, merged
from .viewport3d import Viewport3D

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
        lay.addWidget(self.vp)
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
        lo = hi = None
        if field:
            vals = np.concatenate([np.asarray(p.mesh.point_data[field]) for p in m.parts if field in p.mesh.point_data])
            vals = vals[np.isfinite(vals)]
            if vals.size:
                lo, hi = float(vals.min()), float(vals.max())
                if hi <= lo:
                    hi = lo + 1.0
        for p in merged(m):
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
