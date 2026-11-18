"""Liquid-engine cycle sandbox: build an engine from library components, solve its steady operating point and watch
every component's state on the flow diagram.

Left: the component palette (the component library, grouped by category); double-click or drag a part onto the
canvas.  Centre: the flow diagram; drag from a port to another port to connect (fluid outlet to inlet, shaft to shaft,
chamber wall to a jacket), Delete removes the selection, the wheel zooms, the right or middle button pans.  Right: the
inspector of the selected component (parameters with their library provenance, categorical options, and after a
solve its state: inlet and outlet station, outputs).  Bottom: design targets, stations, components, shafts and
messages.  The solver is :mod:`propsim.core.engine_graph`.
"""
from __future__ import annotations

import copy
import math

import numpy as np
from PySide6.QtCore import QObject, QPointF, QRectF, Qt, QThread, Signal
from PySide6.QtGui import QAction, QBrush, QColor, QFont, QKeySequence, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import (QAbstractItemView, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout,
                               QGraphicsItem, QGraphicsPathItem, QGraphicsScene, QGraphicsView, QHBoxLayout,
                               QHeaderView, QLabel, QLineEdit, QMainWindow, QMenu, QMessageBox, QPlainTextEdit,
                               QPushButton, QSplitter, QTableWidget, QTableWidgetItem, QTabWidget, QToolBar,
                               QDockWidget, QStyle, QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget)

from ..core.engine_graph import edit as E
from .pid import INSTRUMENTS, draw_bubble, draw_symbol, mechanical_pen, tag_for
from .panels import PAD, PanelTitle, draw_guide
from .text import apa
from .theme import theme

NODE_W = 168
PORT_R = 5.5
CAT_COLORS = {"tank": "#5b8def", "line": "#7f8a99", "valve": "#c9a227", "volume": "#7f8a99",
              "turbomachinery": "#b07cff", "combustion": "#ff5a3c", "nozzle": "#ff9d3c", "heat_transfer": "#36b3a8"}
MEDIUM_COLORS = {"ox": "#ff9d3c", "fuel": "#5cc97a", "gas": "#ff5a3c", "shaft": "#a0a8b8", "heat": "#e05a8a"}
COLOR_BY = {"Propellant": None, "Pressure (MPa)": "p", "Temperature (K)": "T", "Mass flow (kg/s)": "mdot"}
# display units for parameters (SI value * factor = shown value)
DISPLAY = {"Pa": ("MPa", 1e-6), "rad/s": ("rpm", 30.0 / math.pi), "W": ("kW", 1e-3)}
SUMMARY = {  # one-line state on the node after a solve: (output, label, factor, unit, digits)
    "tank_regulated": [("mdot", "ṁ", 1, "kg/s", 1)], "tank_blowdown": [("mdot", "ṁ", 1, "kg/s", 1)],
    "pump_centrifugal": [("dp", "Δp", 1e-6, "MPa", 2), ("power", "P", 1e-6, "MW", 2), ("speed", "N", 30 / math.pi,
                                                                                         "rpm", 0)],
    "turbine_impulse": [("power", "P", 1e-6, "MW", 2), ("pressure_ratio", "PR", 1, "", 2), ("T_inlet", "T", 1, "K", 0)],
    "gas_generator": [("p_gg", "p", 1e-6, "MPa", 2), ("mixture_ratio", "O/F", 1, "", 3), ("T_out", "T", 1, "K", 0)],
    "chamber": [("p_c", "pc", 1e-6, "MPa", 2), ("mixture_ratio", "O/F", 1, "", 2), ("T_c", "T", 1, "K", 0)],
    "nozzle_bell": [("thrust", "F", 1e-3, "kN", 1), ("isp", "Isp", 1, "s", 1)],
    "regen_jacket": [("Q", "Q", 1e-6, "MW", 1), ("dT", "ΔT", 1, "K", 0), ("T_wg_max", "Twg", 1, "K", 0)],
    "injector": [("dp", "Δp", 1e-6, "MPa", 2), ("stiffness", "Δp/p", 100, "%", 0)],
    "shaft": [("speed", "N", 30 / math.pi, "rpm", 0), ("power", "P", 1e-6, "MW", 2)],
}


SI_UNITS = {"p": "Pa", "T": "K", "thrust": "N", "isp": "s", "mdot": "kg/s", "mixture_ratio": "–", "power": "W", "Q": "W",
            "speed": "rad/s", "dp": "Pa", "dT": "K", "pressure_ratio": "–", "stiffness": "–", "cstar": "m/s"}


def output_unit(output: str) -> str:
    """SI unit of a design-target output (``p_c`` and ``T_wg_max`` follow their first letter)."""
    if output in SI_UNITS:
        return SI_UNITS[output]
    return SI_UNITS.get(output.split("_")[0], "")


def _fmt(v, digits=3) -> str:
    if isinstance(v, str):
        return v
    if v is None or not np.isfinite(v):
        return "–"
    a = abs(v)
    if a != 0 and (a >= 1e6 or a < 1e-3):
        return f"{v:.{digits}e}"
    return f"{v:.{digits}f}" if a < 100 else f"{v:,.{max(digits - 2, 0)}f}"


def _ramp(t: float) -> QColor:
    """Perceptual blue -> cyan -> yellow -> red ramp for 0 <= t <= 1."""
    stops = [(0.0, (49, 54, 149)), (0.25, (69, 117, 180)), (0.5, (116, 196, 118)), (0.75, (253, 174, 97)),
             (1.0, (215, 48, 39))]
    t = float(np.clip(t, 0.0, 1.0))
    for (t0, c0), (t1, c1) in zip(stops, stops[1:]):
        if t <= t1:
            w = (t - t0) / (t1 - t0)
            return QColor(*[int(a + w * (b - a)) for a, b in zip(c0, c1)])
    return QColor(*stops[-1][1])


# ================================================================================================= canvas items
class PortItem(QGraphicsItem):
    def __init__(self, node: "NodeItem", spec: dict, pos: QPointF) -> None:
        super().__init__(node)
        self.node, self.spec, self.name = node, spec, spec["name"]
        self.setPos(pos)
        self.setAcceptHoverEvents(True)
        self.setToolTip(f"{spec['name']} ({spec['kind']}, {spec.get('direction', '')})")
        self.hot = False

    def boundingRect(self) -> QRectF:
        return QRectF(-PORT_R - 2, -PORT_R - 2, 2 * PORT_R + 4, 2 * PORT_R + 4)

    def paint(self, p: QPainter, *_):
        k = self.spec["kind"]
        col = QColor({"fluid": "#8fb8ff", "shaft": MEDIUM_COLORS["shaft"], "thermal": MEDIUM_COLORS["heat"],
                      "signal": "#cccccc"}.get(k, "#cccccc"))
        if self.hot:
            col = QColor(theme.color("accent"))
        p.setPen(QPen(col.darker(160), 1.2))
        p.setBrush(QBrush(col if self.spec.get("direction") != "in" else QColor(theme.color("panel"))))
        if k == "shaft":
            p.drawRect(QRectF(-PORT_R, -PORT_R, 2 * PORT_R, 2 * PORT_R))
        else:
            p.drawEllipse(QPointF(0, 0), PORT_R, PORT_R)

    def hoverEnterEvent(self, e):
        self.hot = True
        self.update()

    def hoverLeaveEvent(self, e):
        self.hot = False
        self.update()

    def scene_center(self) -> QPointF:
        return self.mapToScene(QPointF(0, 0))


class NodeItem(QGraphicsItem):
    def __init__(self, view: "Canvas", node: dict) -> None:
        super().__init__()
        self.view, self.node = view, node
        self.spec = E.library()[node["type"]]
        self.setFlags(QGraphicsItem.ItemIsMovable | QGraphicsItem.ItemIsSelectable |
                      QGraphicsItem.ItemSendsGeometryChanges)
        self.state: list[str] = []
        self.alert = False
        ports = self.spec["ports"]
        ins = [p for p in ports if p["kind"] == "fluid" and p["direction"] == "in"]
        outs = [p for p in ports if p["kind"] == "fluid" and p["direction"] == "out"]
        shafts = [p for p in ports if p["kind"] == "shaft"]
        therm = [p for p in ports if p["kind"] == "thermal"]
        self.h = E.node_height(node["type"])
        self.ports: dict[str, PortItem] = {}
        for lst, x in ((ins, 0.0), (outs, NODE_W)):
            for i, sp in enumerate(lst):
                self.ports[sp["name"]] = PortItem(self, sp, QPointF(x, 48.0 + 18.0 * i))
        for lst, y in ((shafts, self.h), (therm, 0.0)):
            for i, sp in enumerate(lst):
                self.ports[sp["name"]] = PortItem(self, sp, QPointF(NODE_W * (i + 1) / (len(lst) + 1), y))
        ui = node.setdefault("ui", {"x": 0.0, "y": 0.0})
        self.setPos(float(ui.get("x", 0.0)), float(ui.get("y", 0.0)))
        self.setToolTip(self.spec["description"])

    def boundingRect(self) -> QRectF:
        return QRectF(-PORT_R - 3, -PORT_R - 3, NODE_W + 2 * PORT_R + 6, self.h + 2 * PORT_R + 6)

    def paint(self, p: QPainter, *_):
        """ANSI/ISA-5.1 P&ID presentation: the component's process symbol, its tag, instrument bubbles, short stubs to
        the ports and a faint outline only while selected."""
        t = theme.t
        r = QRectF(0, 0, NODE_W, self.h)
        ink, bg = QColor(t["text"]), QColor(t["viewport_bg"])
        cat = QColor(CAT_COLORS.get(self.spec["category"], "#888888"))
        p.setRenderHint(QPainter.Antialiasing)
        if self.isSelected() or self.alert:
            p.setPen(QPen(QColor("#ff5a3c" if self.alert else t["accent"]), 1.2, Qt.DotLine))
            p.setBrush(Qt.NoBrush)
            p.drawRoundedRect(r, 6, 6)
        sym = QRectF(NODE_W / 2 - 24, 32, 48, 44)
        for name, it in self.ports.items():                       # process connections: solid line to the symbol
            if it.spec["kind"] == "fluid":
                left = it.spec["direction"] == "in"
                y = it.pos().y()
                p.setPen(QPen(ink, 1.4))
                p.drawLine(QPointF(0 if left else NODE_W, y), QPointF(sym.left() if left else sym.right(), y))
        draw_symbol(p, self.node["type"], sym, ink, bg, phase=getattr(self, "phase", 0.0),
                    command=any(q["kind"] == "signal" for q in self.spec["ports"]))
        idx = self.view.graph["nodes"].index(self.node) if self.node in self.view.graph["nodes"] else 0
        f = QFont(p.font())
        f.setPointSizeF(8.2)
        f.setBold(True)
        p.setFont(f)
        p.setPen(ink)
        p.drawText(QRectF(6, 2, NODE_W - 70, 18), Qt.AlignVCenter | Qt.AlignLeft,
                   self.node.get("label") or self.node["id"])
        f.setBold(False)
        f.setPointSizeF(7.4)
        p.setFont(f)
        p.setPen(QColor(t["dim"]))
        p.drawText(QRectF(sym.left() - 20, sym.bottom() + 3, sym.width() + 40, 12), Qt.AlignCenter,
                   tag_for(self.node["type"], idx))
        for k, letters in enumerate(INSTRUMENTS.get(self.node["type"], [])[:2]):     # ISA-5.1 bubbles, top right
            c = QPointF(NODE_W - 13 - 26 * k, 14)
            p.setPen(QPen(ink, 0.9))
            p.drawLine(QPointF(c.x(), c.y() + 12), QPointF(sym.center().x() + 8 * (1 - 2 * k), sym.top() + 4))
            draw_bubble(p, c, letters, str(100 + idx), ink, bg)
        p.setPen(QColor(t["dim"]))
        f.setPointSizeF(6.8)
        p.setFont(f)
        for name, it in self.ports.items():
            if it.spec["kind"] == "fluid" and len(self.spec["ports"]) > 3:
                left = it.spec["direction"] == "in"
                p.drawText(QRectF(4 if left else NODE_W / 2 + 26, it.pos().y() + 3, NODE_W / 2 - 30, 11),
                           (Qt.AlignLeft if left else Qt.AlignRight) | Qt.AlignVCenter, name.replace("_", " "))
        if self.state:
            p.setPen(QColor(t["text"]))
            f.setPointSizeF(7.8)
            p.setFont(f)
            y = self.h - 12.0 * len(self.state) - 4
            for line in self.state:
                p.drawText(QRectF(8, y, NODE_W - 16, 12), Qt.AlignLeft | Qt.AlignVCenter, line)
                y += 12.0

    def itemChange(self, change, value):
        if change == QGraphicsItem.ItemPositionHasChanged:
            self.node["ui"] = {"x": round(self.pos().x(), 1), "y": round(self.pos().y(), 1)}
            self.view.refresh_edges(self.node["id"])
        return super().itemChange(change, value)


class EdgeItem(QGraphicsPathItem):
    def __init__(self, view: "Canvas", edge: dict, a: PortItem, b: PortItem) -> None:
        super().__init__()
        self.view, self.edge, self.a, self.b = view, edge, a, b
        self.setZValue(-1)
        self.setFlags(QGraphicsItem.ItemIsSelectable)
        self.color: QColor | None = None
        self.width = 2.4
        self.update_path()

    def paint(self, p, option, widget=None) -> None:
        option.state &= ~QStyle.State_Selected          # a selected connection is drawn thicker, never with a box
        super().paint(p, option, widget)
        if self.edge["kind"] == "fluid":                # P&ID flow arrow at mid-line
            path = self.path()
            pt, ang = path.pointAtPercent(0.5), path.angleAtPercent(0.5)
            p.save()
            p.translate(pt)
            p.rotate(-ang)
            p.setPen(Qt.NoPen)
            p.setBrush(self.pen().color())
            p.drawPolygon([QPointF(6, 0), QPointF(-4, -4), QPointF(-4, 4)])
            p.restore()

    def update_path(self) -> None:
        p1, p2 = self.a.scene_center(), self.b.scene_center()
        path = QPainterPath(p1)
        k = self.edge["kind"]
        if k == "fluid":
            dx = max(abs(p2.x() - p1.x()) * 0.5, 40.0)
            path.cubicTo(p1 + QPointF(dx, 0), p2 - QPointF(dx, 0), p2)
        else:
            dy = max(abs(p2.y() - p1.y()) * 0.5, 30.0)
            s1 = 1 if self.a.spec["kind"] == "shaft" else -1
            s2 = 1 if self.b.spec["kind"] == "shaft" else -1
            path.cubicTo(p1 + QPointF(0, s1 * dy), p2 + QPointF(0, s2 * dy), p2)
        self.setPath(path)
        self._restyle()

    def base_color(self) -> QColor:
        m, k = self.edge.get("medium"), self.edge["kind"]
        if k == "shaft":
            return QColor(MEDIUM_COLORS["shaft"])
        if k == "thermal":
            return QColor(MEDIUM_COLORS["heat"])
        if m == "combustion_gas":
            return QColor(MEDIUM_COLORS["gas"])
        g = self.view.graph
        return QColor(MEDIUM_COLORS["ox"] if m == g["propellants"]["oxidizer"] else MEDIUM_COLORS["fuel"])

    def _restyle(self) -> None:
        col = self.color or self.base_color()
        pen = QPen(col, self.width + (1.5 if self.isSelected() else 0.0))
        pen.setCapStyle(Qt.RoundCap)
        if self.edge["kind"] == "shaft":
            pen.setStyle(Qt.DashDotLine)                    # ISA mechanical link
        elif self.edge["kind"] == "thermal":
            pen.setStyle(Qt.DotLine)
        self.setPen(pen)

    def itemChange(self, change, value):
        if change == QGraphicsItem.ItemSelectedHasChanged:
            self._restyle()
        return super().itemChange(change, value)


class Canvas(QGraphicsView):
    """The flow diagram: nodes, port-to-port edges, drag-to-connect, coloring by the solved stations."""

    selected = Signal(object)            # node id or None
    changed = Signal()
    message = Signal(str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setScene(QGraphicsScene(self))
        self.scene().setSceneRect(-4000, -4000, 8000, 8000)
        self.setRenderHints(QPainter.Antialiasing | QPainter.TextAntialiasing)
        self.setDragMode(QGraphicsView.RubberBandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setAcceptDrops(True)
        self.graph: dict = E.blank_graph()
        self.nodes: dict[str, NodeItem] = {}
        self.edges: list[EdgeItem] = []
        self._drag_port: PortItem | None = None
        self._rubber: QGraphicsPathItem | None = None
        self._pan: QPointF | None = None
        self.legend: tuple[str, float, float] | None = None
        self.scene().selectionChanged.connect(self._sel)
        theme.changed.connect(lambda *_: self._restyle_all())
        self._restyle_all()

    def _restyle_all(self) -> None:
        self.setBackgroundBrush(QBrush(QColor(theme.color("viewport_bg"))))
        self.scene().update()

    # ------------------------------------------------------------------------------------------ build
    def set_graph(self, graph: dict) -> None:
        self.scene().clear()
        self.graph = graph
        self.nodes, self.edges = {}, []
        for n in graph["nodes"]:
            self._add_item(n)
        for e in graph["edges"]:
            self._add_edge_item(e)
        self.fit()

    def fit(self, min_scale: float = 0.62) -> None:
        """Whole diagram in view, but never so small that the labels cannot be read (then start at the tanks)."""
        if self.nodes:
            r = self.scene().itemsBoundingRect().adjusted(-60, -60, 60, 60)
            self.fitInView(r, Qt.KeepAspectRatio)
            k = self.transform().m11()
            if k > 1.2 or k < min_scale:
                self.resetTransform()
                k = min(max(k, min_scale), 1.2)
                self.scale(k, k)
                vw = self.viewport().width() / k
                self.centerOn(QPointF(r.left() + vw / 2, r.center().y()))

    def _add_item(self, n: dict) -> NodeItem:
        it = NodeItem(self, n)
        self.scene().addItem(it)
        self.nodes[n["id"]] = it
        return it

    def _add_edge_item(self, e: dict) -> None:
        a, b = self.nodes.get(e["from"]["node"]), self.nodes.get(e["to"]["node"])
        if a is None or b is None or e["from"]["port"] not in a.ports or e["to"]["port"] not in b.ports:
            return
        it = EdgeItem(self, e, a.ports[e["from"]["port"]], b.ports[e["to"]["port"]])
        self.scene().addItem(it)
        self.edges.append(it)

    def add_component(self, ntype: str, pos: QPointF | None = None) -> str:
        if pos is None:
            pos = self.mapToScene(self.viewport().rect().center()) - QPointF(NODE_W / 2, 40)
        n = E.add_node(self.graph, ntype, round(pos.x(), 1), round(pos.y(), 1))
        it = self._add_item(n)
        self.scene().clearSelection()
        it.setSelected(True)
        self.changed.emit()
        return n["id"]

    def refresh_edges(self, nid: str | None = None) -> None:
        for e in self.edges:
            if nid is None or nid in (e.edge["from"]["node"], e.edge["to"]["node"]):
                e.update_path()

    def rebuild_edges(self) -> None:
        for e in self.edges:
            self.scene().removeItem(e)
        self.edges = []
        for e in self.graph["edges"]:
            self._add_edge_item(e)

    def delete_selected(self) -> None:
        sel = self.scene().selectedItems()
        if not sel:
            return
        for it in sel:
            if isinstance(it, EdgeItem):
                E.disconnect(self.graph, it.edge["id"])
        for it in sel:
            if isinstance(it, NodeItem):
                E.remove_node(self.graph, it.node["id"])
                self.scene().removeItem(it)
                self.nodes.pop(it.node["id"], None)
        self.rebuild_edges()
        self.changed.emit()

    def _sel(self) -> None:
        try:
            items = [i for i in self.scene().selectedItems() if isinstance(i, NodeItem)]
        except RuntimeError:
            return
        self.selected.emit(items[0].node["id"] if len(items) == 1 else None)

    # ------------------------------------------------------------------------------------------ results
    def show_results(self, sol, color_by: str | None) -> None:
        for it in self.nodes.values():
            c = sol.components.get(it.node["id"], {}) if sol else {}
            lines = []
            for key, lab, fac, unit, dig in SUMMARY.get(it.node["type"], [("mdot", "ṁ", 1, "kg/s", 1)]):
                v = c.get(key)
                if isinstance(v, (int, float)):
                    lines.append(f"{lab} {_fmt(v * fac, dig)} {unit}".rstrip())
            it.state = [" · ".join(lines[:2])] + ([lines[2]] if len(lines) > 2 else []) if lines else []
            it.alert = bool(sol) and any(w.startswith(it.node["id"] + ":") for w in sol.warnings)
            it.update()
        self.legend = None
        vals = {}
        if sol and color_by:
            st = {s["index"]: s for s in sol.stations}
            for e in sol.edges:
                v = st[e["station"]].get(color_by) if color_by != "mdot" else e["mdot"]
                if v is not None:
                    vals[e["id"]] = v
        if vals:
            lo, hi = min(vals.values()), max(vals.values())
            fac = {"p": 1e-6}.get(color_by, 1.0)
            self.legend = (next(k for k, v in COLOR_BY.items() if v == color_by), lo * fac, hi * fac)
        for e in self.edges:
            v = vals.get(e.edge["id"])
            if v is None:
                e.color, e.width = None, 2.4
            else:
                t = (v - lo) / (hi - lo) if hi > lo else 0.5
                e.color = _ramp(t)
                e.width = 2.4 if color_by != "mdot" else 1.5 + 6.0 * t
            e._restyle()
        self.viewport().update()

    LEGEND = [("Oxidizer line", "ox", Qt.SolidLine), ("Fuel line", "fuel", Qt.SolidLine),
              ("Combustion gas line", "gas", Qt.SolidLine), ("Mechanical link (shaft)", "shaft", Qt.DashDotLine),
              ("Heat transfer (wall)", "heat", Qt.DotLine)]
    BUBBLES = "PT pressure · TT temperature · ST speed · FT flow · PDT differential pressure (ISA-5.1)"

    def paintEvent(self, e) -> None:
        super().paintEvent(e)
        p = QPainter(self.viewport())
        try:
            p.setRenderHint(QPainter.Antialiasing)
            t = theme.t
            f = QFont(p.font())
            f.setPointSizeF(8.0)
            p.setFont(f)
            rows = len(self.LEGEND) + 1
            w, h = 250, 10 + 18 * rows + 8
            box = QRectF(self.viewport().width() - w - 12, self.viewport().height() - h - 12, w, h)
            c = QColor(t["card"])
            c.setAlpha(225)
            p.setBrush(c)
            p.setPen(QColor(t["edge"]))
            p.drawRoundedRect(box, 6, 6)
            for i, (name, key, style) in enumerate(self.LEGEND):
                y = box.top() + 14 + 18 * i
                pen = QPen(QColor(MEDIUM_COLORS[key]), 2.4, style)
                p.setPen(pen)
                p.drawLine(QPointF(box.left() + 10, y), QPointF(box.left() + 52, y))
                if style == Qt.SolidLine:                               # flow direction arrow
                    p.setPen(Qt.NoPen)
                    p.setBrush(QColor(MEDIUM_COLORS[key]))
                    p.drawPolygon([QPointF(box.left() + 36, y), QPointF(box.left() + 29, y - 3.5),
                                   QPointF(box.left() + 29, y + 3.5)])
                p.setPen(QColor(t["text"]))
                p.drawText(QRectF(box.left() + 60, y - 8, w - 64, 16), Qt.AlignVCenter | Qt.AlignLeft, name)
            y = box.top() + 14 + 18 * len(self.LEGEND)
            p.setPen(QPen(QColor(t["dim"]), 1.0))
            p.setBrush(Qt.NoBrush)
            p.drawEllipse(QPointF(box.left() + 31, y), 7, 7)
            p.setPen(QColor(t["text"]))
            p.drawText(QRectF(box.left() + 60, y - 8, w - 64, 16), Qt.AlignVCenter | Qt.AlignLeft, "Instrument bubble")
        finally:
            p.end()

    def drawForeground(self, p: QPainter, rect: QRectF) -> None:
        if not self.legend:
            return
        name, lo, hi = self.legend
        p.save()
        p.resetTransform()
        w, h = 180, 12
        x, y = self.viewport().width() - w - 16, 14
        for i in range(w):
            p.setPen(_ramp(i / (w - 1)))
            p.drawLine(x + i, y + 16, x + i, y + 16 + h)
        p.setPen(QColor(theme.color("text")))
        f = QFont(p.font())
        f.setPointSizeF(8.0)
        p.setFont(f)
        p.drawText(QRectF(x, y, w, 14), Qt.AlignCenter, name)
        p.drawText(QRectF(x - 40, y + 30, 80, 14), Qt.AlignCenter, _fmt(lo, 3))
        p.drawText(QRectF(x + w - 40, y + 30, 80, 14), Qt.AlignCenter, _fmt(hi, 3))
        p.restore()

    # ------------------------------------------------------------------------------------------ mouse
    def _port_at(self, pos) -> PortItem | None:
        for it in self.items(pos):
            if isinstance(it, PortItem):
                return it
        return None

    def mousePressEvent(self, e):
        if e.button() in (Qt.MiddleButton, Qt.RightButton) and not self._port_at(e.position().toPoint()):
            if e.button() == Qt.RightButton and self._context(e):
                return
            self._pan = e.position()
            self.setCursor(Qt.ClosedHandCursor)
            return
        port = self._port_at(e.position().toPoint()) if e.button() == Qt.LeftButton else None
        if port is not None:
            self._drag_port = port
            self._rubber = QGraphicsPathItem()
            self._rubber.setPen(QPen(QColor(theme.color("accent")), 2, Qt.DashLine))
            self.scene().addItem(self._rubber)
            return
        super().mousePressEvent(e)

    def _context(self, e) -> bool:
        items = [i for i in self.items(e.position().toPoint()) if isinstance(i, (NodeItem, EdgeItem))]
        if not items:
            return False
        it = items[0]
        m = QMenu(self)
        if isinstance(it, NodeItem):
            nid = it.node["id"]
            m.addAction("Delete component", lambda: (it.setSelected(True), self.delete_selected()))
            openp = [f"{nid}.{p}" for p in (pn for pn in it.ports) if f"{nid}.{p}" in E.unconnected(self.graph)
                     and it.ports[p].spec["direction"] == "out"]
            for op in openp:
                m.addAction(f"Exhaust {op.split('.')[1]} overboard (fixed pressure)",
                            lambda op=op: self._exhaust(op))
        else:
            m.addAction("Delete connection", lambda: (E.disconnect(self.graph, it.edge["id"]), self.rebuild_edges(),
                                                       self.changed.emit()))
        m.exec(e.globalPosition().toPoint())
        return True

    def _exhaust(self, op: str) -> None:
        nid, pn = op.split(".")
        E.set_exhaust(self.graph, nid, pn, 101325.0)
        self.message.emit(f"{op} exhausts at 101.3 kPa; change it in the inspector")
        self.changed.emit()

    def mouseMoveEvent(self, e):
        if self._pan is not None:
            d = e.position() - self._pan
            self._pan = e.position()
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - int(d.x()))
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - int(d.y()))
            return
        if self._drag_port is not None and self._rubber is not None:
            p1 = self._drag_port.scene_center()
            p2 = self.mapToScene(e.position().toPoint())
            path = QPainterPath(p1)
            path.lineTo(p2)
            self._rubber.setPath(path)
            return
        super().mouseMoveEvent(e)

    def mouseReleaseEvent(self, e):
        if self._pan is not None:
            self._pan = None
            self.setCursor(Qt.ArrowCursor)
            return
        if self._drag_port is not None:
            target = self._port_at(e.position().toPoint())
            self.scene().removeItem(self._rubber)
            self._rubber = None
            src, self._drag_port = self._drag_port, None
            if target is not None and target is not src:
                a = (src.node.node["id"], src.name)
                b = (target.node.node["id"], target.name)
                ok, why, *_ = E.can_connect(self.graph, a, b)
                if ok:
                    E.connect(self.graph, a, b)
                    self.rebuild_edges()
                    self.changed.emit()
                else:
                    self.message.emit(f"cannot connect {a[0]}.{a[1]} to {b[0]}.{b[1]}: {why}")
            return
        super().mouseReleaseEvent(e)

    def wheelEvent(self, e):
        f = 1.0015 ** e.angleDelta().y()
        self.scale(f, f)

    def keyPressEvent(self, e):
        if e.key() in (Qt.Key_Delete, Qt.Key_Backspace):
            self.delete_selected()
            return
        super().keyPressEvent(e)

    def dragEnterEvent(self, e):
        if e.mimeData().hasText():
            e.acceptProposedAction()

    def dragMoveEvent(self, e):
        if e.mimeData().hasText():
            e.acceptProposedAction()

    def dropEvent(self, e):
        t = e.mimeData().text()
        if t in E.library():
            self.add_component(t, self.mapToScene(e.position().toPoint()) - QPointF(NODE_W / 2, 20))
            e.acceptProposedAction()


# ================================================================================================= solver thread
class _Solver(QObject):
    done = Signal(object, str)

    def __init__(self, doc: dict) -> None:
        super().__init__()
        self.doc = doc

    def run(self) -> None:
        from ..core.engine_graph import GraphError, solve
        try:
            g = self.doc["graph"]
            sol = solve(g, self.doc.get("targets") or None, self.doc.get("free") or None)
            self.done.emit(sol, "")
        except (GraphError, ValueError, KeyError, ZeroDivisionError, RuntimeError) as exc:
            self.done.emit(None, str(exc))


class Palette(QTreeWidget):
    add = Signal(str)

    def drawRow(self, painter, option, index) -> None:
        super().drawRow(painter, option, index)
        draw_guide(self, painter, option, index)         # the grey guide line down each expanded group, as in the module tree

    def __init__(self) -> None:
        super().__init__()
        self.setHeaderHidden(True)
        self.setIndentation(18)
        self.setDragEnabled(True)
        self.setDragDropMode(QAbstractItemView.DragOnly)
        lib = E.library()
        for head, types in E.palette().items():
            top = QTreeWidgetItem([head])
            top.setFlags(Qt.ItemIsEnabled)
            for t in types:
                it = QTreeWidgetItem([E.TYPE_LABELS.get(t, t)])
                it.setData(0, Qt.UserRole, t)
                it.setToolTip(0, lib[t]["description"])
                top.addChild(it)
            self.addTopLevelItem(top)
        self.expandAll()
        self.itemDoubleClicked.connect(lambda it, _c: it.data(0, Qt.UserRole) and self.add.emit(it.data(0, Qt.UserRole)))

    def mimeData(self, items):
        md = super().mimeData(items)
        t = items[0].data(0, Qt.UserRole) if items else None
        if t:
            md.setText(t)
        return md


# ================================================================================================= window
class CycleSandbox(QMainWindow):
    """The sandbox window (one per main window)."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Engine Cycle Sandbox")
        self.resize(1500, 920)
        self.doc = E.new_document()
        self.sol = None
        self.path: str | None = None
        self._thread: QThread | None = None
        self._job: _Solver | None = None
        self._loading = False
        self.canvas = Canvas()
        self.palette_tree = Palette()
        self.palette_tree.add.connect(self.canvas.add_component)
        self.inspector = QWidget()
        self.insp_layout = QVBoxLayout(self.inspector)
        self.insp_layout.setContentsMargins(6, 6, 6, 6)
        self.tabs = QTabWidget()
        self.targets_tab = self._targets_widget()
        self.st_table = self._table(["Station", "Ports", "Fluid", "Phase", "p (MPa)", "T (K)", "ṁ (kg/s)",
                                     "ρ (kg/m³)"])
        self.comp_table = self._table(["Component", "Type", "State"])
        self.shaft_table = self._table(["Shaft", "Speed (rpm)", "Turbine (MW)", "Pumps (MW)", "Auxiliary (kW)"])
        self.msgs = QPlainTextEdit()
        self.msgs.setReadOnly(True)
        for w, name in ((self.targets_tab, "Design Targets"), (self.st_table, "Stations"),
                        (self.comp_table, "Components"), (self.shaft_table, "Shafts"), (self.msgs, "Messages")):
            self.tabs.addTab(w, name)
        left = QWidget()
        ll = QVBoxLayout(left)
        ll.setContentsMargins(PAD, 0, PAD, PAD)
        ll.addWidget(self.palette_tree)
        hint = QLabel("Double-click or drag a part onto the diagram. Drag from port to port to connect.")
        hint.setWordWrap(True)
        hint.setObjectName("hint")
        ll.addWidget(hint)
        self.setCentralWidget(self.canvas)
        self.dock_parts = self._dock("Parts", "Components", left)
        self.dock_insp = self._dock("Inspector", "Inspector", self.inspector)
        self.dock_res = self._dock("Results", "Results", self.tabs)
        from .cycle_animation import CycleAnimation
        self.anim = CycleAnimation(self.canvas, lambda: self.doc.get("title") or self.doc["graph"].get("title") or self.doc["graph"].get("name") or self.windowTitle())
        self.dock_anim = self._dock("Animation", "Cycle Animation", self.anim)
        self.docks = [self.dock_parts, self.dock_insp, self.dock_res, self.dock_anim]
        self.default_layout()
        self._toolbar()
        self.canvas.selected.connect(self.inspect)
        self.canvas.changed.connect(self._changed)
        self.canvas.message.connect(self.log)
        self.inspect(None)
        self._style()
        theme.changed.connect(lambda *_: self._style())
        self.load_template("gg_kerolox")

    def _dock(self, name: str, title: str, widget: QWidget) -> QDockWidget:
        d = QDockWidget(apa(title), self)
        d.setObjectName("sb" + name)
        d.setFeatures(QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable | QDockWidget.DockWidgetClosable)
        d.setTitleBarWidget(PanelTitle(d, title))               # drag grip top left, close button top right
        d.setWidget(widget)
        return d

    def default_layout(self) -> None:
        """Components | diagram | Inspector, with the results under the diagram; every panel can be dragged."""
        for d in self.docks:
            d.setFloating(False)
            self.removeDockWidget(d)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.dock_parts)
        self.addDockWidget(Qt.RightDockWidgetArea, self.dock_insp)
        self.addDockWidget(Qt.BottomDockWidgetArea, self.dock_res)
        self.addDockWidget(Qt.RightDockWidgetArea, self.dock_anim)
        self.tabifyDockWidget(self.dock_insp, self.dock_anim)
        self.dock_insp.raise_()
        for d in self.docks:
            d.show()
        self.resizeDocks([self.dock_parts, self.dock_insp], [230, 370], Qt.Horizontal)
        self.resizeDocks([self.dock_res], [280], Qt.Vertical)

    def _style(self) -> None:
        t = theme.t
        self.setStyleSheet(
            f"QTableWidget {{ background: {t['panel']}; color: {t['text']}; gridline-color: {t['edge']}; "
            f"alternate-background-color: {t['card']}; border: 1px solid {t['edge']}; }}"
            f"QTableWidget::item:selected {{ background: {t['select_bg']}; color: {t['accent']}; }}"
            f"QHeaderView::section {{ background: {t['card']}; color: {t['dim']}; border: none; "
            f"border-bottom: 1px solid {t['edge']}; padding: 3px 6px; }}"
            f"QPlainTextEdit {{ background: {t['panel']}; color: {t['text']}; }}"
            f"QLineEdit, QDoubleSpinBox {{ background: {t['panel']}; color: {t['text']}; "
            f"border: 1px solid {t['edge']}; border-radius: 3px; padding: 2px 4px; }}"
            f"QLabel#hint {{ color: {t['dim']}; }}")

    # ------------------------------------------------------------------------------------------ chrome
    def _toolbar(self) -> None:
        tb = QToolBar("Sandbox")
        tb.setMovable(False)
        self.addToolBar(tb)
        tb.addAction(self._action("New", self.new_doc, QKeySequence.New))
        tb.addAction(self._action("Open", self.open_doc, QKeySequence.Open))
        tb.addAction(self._action("Save", self.save_doc, QKeySequence.Save))
        tmpl = QPushButton("Templates")
        menu = QMenu(tmpl)
        for gid, name in E.templates().items():
            menu.addAction(name, lambda gid=gid: self.load_template(gid))
        tmpl.setMenu(menu)
        tb.addWidget(tmpl)
        tb.addSeparator()
        tb.addWidget(QLabel(" Oxidizer "))
        from ..data import fluids
        fl = fluids()
        self.ox_box, self.fu_box = QComboBox(), QComboBox()
        for fid, r in fl.items():
            if "oxidizer" in r["role"]:
                self.ox_box.addItem(fid)
            if "fuel" in r["role"]:
                self.fu_box.addItem(fid)
        tb.addWidget(self.ox_box)
        tb.addWidget(QLabel(" Fuel "))
        tb.addWidget(self.fu_box)
        self.ox_box.currentTextChanged.connect(lambda v: self._propellant("oxidizer", v))
        self.fu_box.currentTextChanged.connect(lambda v: self._propellant("fuel", v))
        tb.addWidget(QLabel(" Ambient (kPa) "))
        self.pa_box = QDoubleSpinBox()
        self.pa_box.setRange(0.0, 200.0)
        self.pa_box.setDecimals(2)
        self.pa_box.valueChanged.connect(self._ambient)
        tb.addWidget(self.pa_box)
        tb.addSeparator()
        self.solve_btn = QPushButton("Solve")
        self.solve_btn.setObjectName("run")
        self.solve_btn.setShortcut("Ctrl+R")
        self.solve_btn.clicked.connect(self.solve)
        tb.addWidget(self.solve_btn)
        panels = QPushButton("Panels")                                 # the way back after a panel was closed
        pm = QMenu(panels)
        for d in self.docks:
            pm.addAction(d.toggleViewAction())
        pm.addSeparator()
        pm.addAction("Restore Default Layout", self.default_layout)
        panels.setMenu(pm)
        tb.addWidget(panels)
        tb.addWidget(QLabel(" Color by "))
        self.color_box = QComboBox()
        self.color_box.addItems(list(COLOR_BY))
        self.color_box.setCurrentText("Pressure (MPa)")
        self.color_box.currentTextChanged.connect(lambda _: self.canvas.show_results(self.sol, self._color_key()))
        tb.addWidget(self.color_box)
        tb.addAction(self._action("Fit", self.canvas.fit, "F"))
        self.status = QLabel("")
        self.statusBar().addWidget(self.status, 1)

    def _action(self, text, slot, key=None) -> QAction:
        a = QAction(text, self)
        a.triggered.connect(lambda _=False: slot())
        if key:
            a.setShortcut(key)
        return a

    def _table(self, heads: list[str]) -> QTableWidget:
        t = QTableWidget(0, len(heads))
        t.setHorizontalHeaderLabels(heads)
        t.setAlternatingRowColors(True)
        t.verticalHeader().setVisible(False)
        t.setEditTriggers(QAbstractItemView.NoEditTriggers)
        t.setSelectionBehavior(QAbstractItemView.SelectRows)
        t.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        t.horizontalHeader().setStretchLastSection(True)
        t.itemSelectionChanged.connect(lambda t=t: self._table_pick(t))
        return t

    def _color_key(self):
        return COLOR_BY[self.color_box.currentText()]

    def log(self, text: str) -> None:
        self.msgs.appendPlainText(text)
        self.status.setText(text)

    # ------------------------------------------------------------------------------------------ document
    def set_doc(self, doc: dict) -> None:
        self._loading = True
        try:
            self.doc = doc
            g = doc["graph"]
            E.infer_media(g)
            self.ox_box.setCurrentText(g["propellants"]["oxidizer"])
            self.fu_box.setCurrentText(g["propellants"]["fuel"])
            pa = next((b["value"]["value"] for b in g.get("boundary_conditions", [])
                       if b["kind"] == "ambient_pressure"), 0.0)
            self.pa_box.setValue(pa / 1e3)
            self.canvas.set_graph(g)
            self.sol = None
            self.canvas.show_results(None, None)
            self._fill_targets()
            self._fill_results()
            self.inspect(None)
        finally:
            self._loading = False

    def new_doc(self) -> None:
        self.path = None
        self.set_doc(E.new_document(self.ox_box.currentText() or "LOX", self.fu_box.currentText() or "RP1"))
        self.log("new engine: add a tank, feed, injector, chamber and nozzle")

    def load_template(self, gid: str) -> None:
        self.path = None
        self.set_doc(E.template(gid))
        self.log(f"template: {self.doc['graph']['name']}")

    def open_doc(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Open engine", "", "PropSim engine (*.json)")
        if path:
            try:
                self.set_doc(E.load(path))
                self.path = path
            except (OSError, ValueError, KeyError) as exc:
                QMessageBox.warning(self, "Open engine", str(exc))

    def save_doc(self) -> None:
        path = self.path
        if not path:
            path, _ = QFileDialog.getSaveFileName(self, "Save engine", "engine.json", "PropSim engine (*.json)")
        if path:
            E.save(self.doc, path)
            self.path = path
            self.log(f"saved {path}")

    def _propellant(self, role: str, fid: str) -> None:
        if self._loading or not fid:
            return
        g = self.doc["graph"]
        old = g["propellants"][role]
        g["propellants"][role] = fid
        for n in g["nodes"]:
            if n["type"].startswith("tank") and n.get("options", {}).get("fluid") == old:
                n["options"]["fluid"] = fid
        E.infer_media(g)
        self.canvas.rebuild_edges()
        self._changed()

    def _ambient(self, kpa: float) -> None:
        if not self._loading:
            E.set_ambient(self.doc["graph"], kpa * 1e3)
            self._changed()

    def _changed(self) -> None:
        if self._loading:
            return
        if self.sol is not None:
            self.sol = None
            self.canvas.show_results(None, None)
            self.status.setText("changed: press Solve")
        sel = [i for i in self.canvas.scene().selectedItems() if isinstance(i, NodeItem)]
        self.inspect(sel[0].node["id"] if len(sel) == 1 else None)

    # ------------------------------------------------------------------------------------------ inspector
    def _clear_inspector(self) -> None:
        while self.insp_layout.count():
            it = self.insp_layout.takeAt(0)
            w = it.widget()
            if w is not None:
                w.hide()
                w.setParent(None)
                w.deleteLater()

    def inspect(self, nid: str | None) -> None:
        self._clear_inspector()
        head = QLabel("Inspector")
        head.setObjectName("heading")
        self.insp_layout.addWidget(head)
        g = self.doc["graph"]
        node = next((n for n in g["nodes"] if n["id"] == nid), None) if nid else None
        if node is None:
            info = QLabel(f"{len(g['nodes'])} components, {len(g['edges'])} connections.\n"
                          "Select a component to edit it.")
            info.setWordWrap(True)
            self.insp_layout.addWidget(info)
            open_ports = E.unconnected(g)
            if open_ports:
                w = QLabel("Open fluid ports: " + ", ".join(open_ports[:12]) + (" …" if len(open_ports) > 12 else ""))
                w.setWordWrap(True)
                self.insp_layout.addWidget(w)
            self.insp_layout.addStretch(1)
            return
        spec = E.library()[node["type"]]
        form = QFormLayout()
        idbox = QLineEdit(node["id"])
        idbox.editingFinished.connect(lambda: self._rename(node, idbox.text()))
        form.addRow("Id", idbox)
        lab = QLineEdit(node.get("label", ""))
        lab.editingFinished.connect(lambda: self._relabel(node, lab.text()))
        form.addRow("Label", lab)
        form.addRow("Type", QLabel(E.TYPE_LABELS.get(node["type"], node["type"])))
        for name, choices in E.OPTIONS.get(node["type"], {}).items():
            cur = node.get("options", {}).get(name, choices[0] if choices else "")
            if name == "fluid":
                from ..data import fluids
                choices = list(fluids())
            if len(choices) > 1 or name == "fluid":
                box = QComboBox()
                box.addItems(choices)
                box.setCurrentText(cur)
                box.currentTextChanged.connect(lambda v, n=name: self._set_option(node, n, v))
            else:
                box = QLineEdit(cur)
                box.editingFinished.connect(lambda n=name, b=box: self._set_option(node, n, b.text()))
            box.setToolTip(E.OPTION_HELP.get(name, ""))
            form.addRow(name.replace("_", " "), box)
        for b in g.get("boundary_conditions", []):
            if b["node"] == node["id"] and b["kind"] == "fixed_pressure":
                sp = QDoubleSpinBox()
                sp.setRange(0.0, 50000.0)
                sp.setDecimals(1)
                sp.setValue(b["value"]["value"] / 1e3)
                sp.valueChanged.connect(lambda v, pn=b["port"]: (E.set_exhaust(g, node["id"], pn, v * 1e3),
                                                                 self._changed()))
                form.addRow(f"{b['port']} exhaust (kPa)", sp)
        fw = QWidget()
        fw.setLayout(form)
        self.insp_layout.addWidget(fw)
        t = QTableWidget(len(spec["parameters"]), 4)
        t.setHorizontalHeaderLabels(["Parameter", "Value", "Unit", "Source"])
        t.verticalHeader().setVisible(False)
        t.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        t.horizontalHeader().setStretchLastSection(True)
        for i, prm in enumerate(spec["parameters"]):
            val, prov, default = E.param_value(node, prm["name"])
            unit, fac = DISPLAY.get(prm["unit"], (prm["unit"], 1.0))
            it0 = QTableWidgetItem(prm["name"])
            it0.setFlags(Qt.ItemIsEnabled)
            it0.setToolTip(prm.get("meaning", ""))
            itv = QTableWidgetItem("" if not isinstance(val, (int, float)) else _fmt(val * fac, 5))
            if default:
                itv.setForeground(QBrush(QColor(theme.color("dim"))))
                itv.setToolTip("library default: type a value to override, clear it to go back")
            itu = QTableWidgetItem(unit)
            itu.setFlags(Qt.ItemIsEnabled)
            its = QTableWidgetItem(prov)
            its.setFlags(Qt.ItemIsEnabled)
            its.setToolTip(prov)
            for j, it in enumerate((it0, itv, itu, its)):
                t.setItem(i, j, it)
        t.itemChanged.connect(lambda it, node=node, spec=spec, t=t: self._edit_param(node, spec, t, it))
        self.insp_layout.addWidget(t, 2)
        self.insp_layout.addWidget(self._state_box(node), 3)

    def _state_box(self, node: dict) -> QWidget:
        t = QTableWidget(0, 2)
        t.setHorizontalHeaderLabels(["State", "Value"])
        t.verticalHeader().setVisible(False)
        t.horizontalHeader().setStretchLastSection(True)
        t.setEditTriggers(QAbstractItemView.NoEditTriggers)
        if self.sol is None:
            t.setRowCount(1)
            t.setItem(0, 0, QTableWidgetItem("press Solve to see this component's state"))
            return t
        rows = []
        for k, v in self.sol.components.get(node["id"], {}).items():
            if k == "type":
                continue
            rows.append((k, _fmt(v, 4) if isinstance(v, (int, float)) else str(v)))
        for s in self.sol.stations:
            for pt in s["ports"]:
                if pt.startswith(node["id"] + "."):
                    pn = pt.split(".", 1)[1]
                    rows.append((f"{pn}: p (MPa)", _fmt(s["p"] * 1e-6, 4)))
                    if "T" in s:
                        rows.append((f"{pn}: T (K)", _fmt(s["T"], 1)))
                        rows.append((f"{pn}: fluid, phase", f"{s.get('fluid', '')}, {s.get('phase', '')}"))
        t.setRowCount(len(rows))
        for i, (a, b) in enumerate(rows):
            t.setItem(i, 0, QTableWidgetItem(a))
            t.setItem(i, 1, QTableWidgetItem(b))
        t.resizeColumnToContents(0)
        return t

    def _edit_param(self, node, spec, t, it) -> None:
        if it.column() != 1:
            return
        prm = spec["parameters"][it.row()]
        unit, fac = DISPLAY.get(prm["unit"], (prm["unit"], 1.0))
        txt = it.text().strip().replace(",", "")
        try:
            val = None if txt == "" else float(txt) / fac
        except ValueError:
            self.log(f"{node['id']}.{prm['name']}: '{txt}' is not a number")
            return
        E.set_param(node, prm["name"], val, prm["unit"])
        self.log(f"{node['id']}.{prm['name']} = {txt} {unit}" if val is not None else
                 f"{node['id']}.{prm['name']}: back to the library default")
        self._invalidate()

    def _invalidate(self) -> None:
        if self.sol is not None:
            self.sol = None
            self.canvas.show_results(None, None)
            self.status.setText("changed: press Solve")

    def _set_option(self, node, name, value) -> None:
        node.setdefault("options", {})[name] = value
        if name == "fluid":
            E.infer_media(self.doc["graph"])
            self.canvas.rebuild_edges()
        self._invalidate()

    def _rename(self, node, new) -> None:
        old = node["id"]
        if new and new != old:
            nid = E.rename_node(self.doc["graph"], old, new)
            for t in self.doc.get("targets", []) + self.doc.get("free", []):
                if t["node"] == old:
                    t["node"] = nid
            self.canvas.nodes[nid] = self.canvas.nodes.pop(old)
            self.canvas.nodes[nid].update()
            self.canvas.rebuild_edges()
            self._fill_targets()
            self._invalidate()

    def _relabel(self, node, text) -> None:
        node["label"] = text
        if node["id"] in self.canvas.nodes:
            self.canvas.nodes[node["id"]].update()

    # ------------------------------------------------------------------------------------------ targets
    def _targets_widget(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        info = QLabel("Design mode: each target fixes an output and releases one parameter for the solver to size "
                      "(for example chamber p_c by a valve position, mixture ratio by another valve, thrust by "
                      "the throat area A_t). Leave empty to solve the hardware as built.")
        info.setWordWrap(True)
        lay.addWidget(info)
        self.tg_table = QTableWidget(0, 5)
        self.tg_table.setHorizontalHeaderLabels(["Target output (node.output)", "Value (SI)", "Unit",
                                                 "Free parameter (node.param)", "Sized value"])
        self.tg_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.tg_table.itemChanged.connect(self._target_edit)
        lay.addWidget(self.tg_table)
        row = QHBoxLayout()
        add, rem = QPushButton("Add Target"), QPushButton("Remove Target")
        add.clicked.connect(self._add_target)
        rem.clicked.connect(self._remove_target)
        row.addWidget(add)
        row.addWidget(rem)
        row.addStretch(1)
        lay.addLayout(row)
        return w

    def _fill_targets(self) -> None:
        self._loading_t = True
        tg, fr = self.doc.get("targets", []), self.doc.get("free", [])
        self.tg_table.setRowCount(len(tg))
        for i, (t, f) in enumerate(zip(tg, fr)):
            vals = [f"{t['node']}.{t['output']}", repr(t["value"]), output_unit(t["output"]),
                    f"{f['node']}.{f['param']}",
                    _fmt(self.sol.free.get(f"{f['node']}.{f['param']}"), 5) if self.sol else ""]
            for j, v in enumerate(vals):
                it = QTableWidgetItem(v)
                if j in (2, 4):
                    it.setFlags(Qt.ItemIsEnabled)
                self.tg_table.setItem(i, j, it)
        self._loading_t = False

    def _add_target(self) -> None:
        ch = next((n["id"] for n in self.doc["graph"]["nodes"] if n["type"] == "chamber"), "chamber")
        self.doc.setdefault("targets", []).append({"node": ch, "output": "p_c", "value": 7e6})
        self.doc.setdefault("free", []).append({"node": ch, "param": "A_t"})
        self._fill_targets()
        self._invalidate()

    def _remove_target(self) -> None:
        r = self.tg_table.currentRow()
        if 0 <= r < len(self.doc.get("targets", [])):
            del self.doc["targets"][r]
            del self.doc["free"][r]
            self._fill_targets()
            self._invalidate()

    def _target_edit(self, it) -> None:
        if getattr(self, "_loading_t", False):
            return
        r, c = it.row(), it.column()
        t, f = self.doc["targets"][r], self.doc["free"][r]
        txt = it.text().strip()
        try:
            if c == 0:
                t["node"], t["output"] = txt.split(".", 1)
            elif c == 1:
                t["value"] = float(txt)
            elif c == 3:
                f["node"], f["param"] = txt.split(".", 1)
        except ValueError:
            self.log(f"design target row {r + 1}: write node.output, a number, and node.param")
        self._invalidate()

    # ------------------------------------------------------------------------------------------ solve
    def solve(self) -> None:
        if self._thread is not None:
            return
        self.solve_btn.setEnabled(False)
        self.status.setText("solving …")
        self.msgs.appendPlainText("solving …")
        self._thread = QThread(self)
        self._job = _Solver(copy.deepcopy(self.doc))
        self._job.moveToThread(self._thread)
        self._thread.started.connect(self._job.run)
        self._job.done.connect(self._solved)
        self._thread.start()

    def wait(self, ms: int = 600000) -> None:
        """Block until a running solve finishes (tests)."""
        from PySide6.QtWidgets import QApplication
        import time
        t0 = time.time()
        while self._thread is not None and (time.time() - t0) * 1e3 < ms:
            QApplication.processEvents()
            time.sleep(0.01)

    def _solved(self, sol, err: str) -> None:
        self._thread.quit()
        self._thread.wait()
        self._thread, self._job = None, None
        self.solve_btn.setEnabled(True)
        if sol is None:
            self.log(f"cannot solve: {err}")
            QMessageBox.warning(self, "Engine Cycle Sandbox", err)
            return
        self.sol = sol
        lines = [f"{sol.message} after {sol.iterations} outer iterations (largest residual {sol.residual:.1e})"]
        noz = [c for c in sol.components.values() if c.get("type") == "nozzle_bell"]
        ch = [c for c in sol.components.values() if c.get("type") == "chamber"]
        if noz and ch:
            lines.append(f"thrust {noz[0]['thrust'] / 1e3:.1f} kN, Isp {noz[0]['isp']:.1f} s, "
                         f"pc {ch[0]['p_c'] / 1e6:.3f} MPa, O/F {ch[0]['mixture_ratio']:.3f}")
        for k, v in sol.free.items():
            lines.append(f"sized {k} = {v:.6g}")
        lines += [f"warning: {w}" for w in sol.warnings]
        for line in lines:
            self.msgs.appendPlainText(line)
        self.status.setText(lines[0] + (" · " + lines[1] if len(lines) > 1 else ""))
        self.canvas.show_results(sol, self._color_key())
        self._fill_results()
        self._fill_targets()
        sel = [i for i in self.canvas.scene().selectedItems() if isinstance(i, NodeItem)]
        self.inspect(sel[0].node["id"] if len(sel) == 1 else None)

    def _fill_results(self) -> None:
        sol = self.sol
        for t in (self.st_table, self.comp_table, self.shaft_table):
            t.setRowCount(0)
        if sol is None:
            return
        t = self.st_table
        t.setRowCount(len(sol.stations))
        for i, s in enumerate(sol.stations):
            vals = [str(s["index"]), ", ".join(s["ports"]), s.get("fluid", ""), s.get("phase", ""),
                    _fmt(s["p"] * 1e-6, 4), _fmt(s.get("T"), 1), _fmt(s["mdot"], 3), _fmt(s.get("rho"), 3)]
            for j, v in enumerate(vals):
                t.setItem(i, j, QTableWidgetItem(v))
        t = self.comp_table
        t.setRowCount(len(sol.components))
        for i, (nid, c) in enumerate(sol.components.items()):
            state = ", ".join(f"{k} {_fmt(v, 4) if isinstance(v, (int, float)) else v}" for k, v in c.items()
                              if k != "type")
            for j, v in enumerate((nid, E.TYPE_LABELS.get(c.get("type"), c.get("type")), state)):
                t.setItem(i, j, QTableWidgetItem(v))
        t = self.shaft_table
        t.setRowCount(len(sol.shafts))
        for i, s in enumerate(sol.shafts):
            vals = [" + ".join(s["members"]), _fmt(s["rpm"], 0), _fmt(s["turbine_power"] * 1e-6, 3),
                    _fmt(s["pump_power"] * 1e-6, 3), _fmt(s["aux_power"] * 1e-3, 1)]
            for j, v in enumerate(vals):
                t.setItem(i, j, QTableWidgetItem(v))

    def _table_pick(self, t: QTableWidget) -> None:
        """Selecting a component row selects it on the diagram."""
        if t is not self.comp_table or not t.selectedItems():
            return
        nid = t.item(t.currentRow(), 0).text() if t.item(t.currentRow(), 0) else None
        if nid in self.canvas.nodes:
            self.canvas.scene().clearSelection()
            self.canvas.nodes[nid].setSelected(True)
            self.canvas.centerOn(self.canvas.nodes[nid])


def open_sandbox(parent=None) -> CycleSandbox:
    w = CycleSandbox(parent)
    w.show()
    return w
