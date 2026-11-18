"""Realistic animated schematic of a liquid-engine cycle (the cycle panel of the sandbox).

The cycle in the sandbox is identified from its graph (the ``cycle`` field, else the components present):

* ``pressure_fed``      tanks, valves, injector, chamber, nozzle
* ``gas_generator``     pumps on one shaft with a turbine driven by a gas generator, exhaust overboard
* ``expander``          the regenerative jacket heats the fuel that then drives the turbine and feeds the chamber
* ``staged``            a preburner drives the turbine and its exhaust goes on into the chamber
* anything else is a ``custom`` cycle: the diagram is then drawn from the graph itself (P&ID view)

The drawing follows the textbook cycle diagrams: pumps, turbine and shaft along the top, control valves, gas generator or
preburner, combustion chamber with the nozzle and the regenerative cooling tubes hugging the nozzle wall.  Flow arrows move
along every line; the shaft turns; the exhaust moves down the nozzle.  Coordinates are in a 100 x 135 design box.
"""
from __future__ import annotations

import math

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import (QBrush, QColor, QFont, QLinearGradient, QPainter, QPainterPath, QPen, QPolygonF,
                           QRadialGradient)

from .theme import theme

FUEL, OX, GAS, RICH, OXRICH, ELEC = "#ff2a2a", "#3d6bff", "#ff9a1f", "#ff8fa3", "#a58bff", "#ffd23f"
KIND_NAMES = {"pressure_fed": "Pressure-fed cycle", "gas_generator": "Gas-generator cycle (open)",
              "tap_off": "Tap-off cycle", "expander_bleed": "Expander bleed cycle (open)",
              "expander": "Expander cycle (closed)", "staged_fuel": "Fuel-rich staged-combustion cycle",
              "staged_ox": "Oxidizer-rich staged-combustion cycle", "full_flow": "Full-flow staged-combustion cycle",
              "electric_pump": "Electric pump-fed cycle", "custom": "Custom cycle"}


def identify_cycle(graph: dict) -> str:
    """One of :data:`KIND_NAMES`: from the graph's ``cycle`` field, else from the components present."""
    types = [n["type"] for n in graph.get("nodes", [])]
    has = lambda t: t in types
    c = (graph.get("cycle") or "").lower()
    if "full_flow" in c or "full-flow" in c:
        return "full_flow"
    if "staged" in c:
        return "staged_ox" if ("ox_rich" in c or "oxidizer_rich" in c) else "staged_fuel"
    if "tap" in c:
        return "tap_off"
    if "bleed" in c:
        return "expander_bleed"
    if "expander" in c:
        return "expander"
    if "electric" in c:
        return "electric_pump"
    if "pressure" in c:
        return "pressure_fed"
    if "gas_generator" in c:
        return "gas_generator"
    if not has("pump_centrifugal") and has("chamber"):
        return "pressure_fed"
    if has("turbine_impulse") and sum(t == "gas_generator" for t in types) >= 2:
        return "full_flow"
    if has("turbine_impulse") and has("gas_generator"):
        return "gas_generator"
    if has("turbine_impulse") and has("regen_jacket"):
        return "expander"
    return "custom"


def _bezier(p0, p1, p2, n=8):
    return [(((1 - t) ** 2) * p0[0] + 2 * (1 - t) * t * p1[0] + t * t * p2[0],
             ((1 - t) ** 2) * p0[1] + 2 * (1 - t) * t * p1[1] + t * t * p2[1]) for t in [i / n for i in range(n + 1)]]


# chamber / nozzle outline (left half, top to bottom); the right half is the mirror image about x = 50
LEFT = [(36, 78), (36, 90), (46.5, 98.5), (48, 101)] + _bezier((48, 101), (45, 109), (31, 125))[1:]
SH_Y = 21.0                                                  # shaft height


def _flows(kind: str, regen: bool, g: dict) -> tuple[list, list, list]:
    """(flows, valves, labels): flows are ``(points, color)``; valves are centres; labels ``(text, x, y, align)``."""
    off = [(x - 2.9, y) for x, y in LEFT]
    up = list(reversed(off))                                  # from the nozzle exit up to the chamber
    ox_name, fu_name = g["propellants"]["oxidizer"], g["propellants"]["fuel"]
    flows, valves, labels = [], [], []
    to_cc_fuel = ([(20, 125)] + up + [(33.1, 76), (33.1, 70), (44, 70), (44, 77.5)]) if regen else \
                 [(20, 74), (44, 74), (44, 77.5)]
    top_labels = [("Fuel pump", 18, 4, "c"), ("Turbine", 50, 4, "c"), ("Oxidizer pump", 82, 4, "c"),
                  (fu_name, 1, 30, "l"), (ox_name, 99, 30, "r")]
    if kind == "pressure_fed":
        flows += [([(23, 36), (23, 74), (44, 74), (44, 77.5)], FUEL), ([(77, 36), (77, 74), (56, 74), (56, 77.5)], OX)]
        if regen:
            flows[0] = ([(23, 36), (23, 125)] + up + [(33.1, 76), (33.1, 70), (44, 70), (44, 77.5)], FUEL)
        valves += [(23, 50), (77, 50)]
        labels += [("Fuel tank", 23, 4, "c"), ("Oxidizer tank", 77, 4, "c"), ("Control" + chr(10) + "valves", 83, 58, "l")]
        return flows, valves, labels
    labels += top_labels
    flows += [([(5, 8), (5, SH_Y), (13, SH_Y)], FUEL), ([(95, 8), (95, SH_Y), (87, SH_Y)], OX)]
    if kind == "gas_generator":
        flows += [([(20, 34), (20, 53), (39, 53)], FUEL), ([(80, 34), (80, 53), (61, 53)], OX),
                  ([(20, 53)] + to_cc_fuel, FUEL), ([(80, 53), (80, 68), (56, 68), (56, 77.5)], OX),
                  ([(50, 44), (50, 34)], GAS), ([(44, 27), (30, 27), (30, 38)], GAS)]
        valves += [(29, 53), (71, 53)]
        labels += [("Exhaust", 30, 42, "c"), ("GG", 50, 53.5, "c")]
    elif kind in ("staged_fuel", "staged_ox"):
        gas = RICH if kind == "staged_fuel" else OXRICH
        flows += [([(20, 34), (20, 53), (39, 53)], FUEL), ([(80, 34), (80, 53), (61, 53)], OX),
                  ([(20, 53)] + to_cc_fuel, FUEL), ([(80, 53), (80, 68), (56, 68), (56, 77.5)], OX),
                  ([(50, 44), (50, 34)], gas), ([(56, 27), (66, 27), (66, 72), (50, 72), (50, 77.5)], gas)]
        valves += [(29, 53), (71, 53)]
        labels += [("Pre-burner", 50, 53.5, "c"),
                   ("Fuel-rich gas" if kind == "staged_fuel" else "Oxidizer-rich gas", 68, 44, "l")]
    elif kind == "tap_off":                                    # hot gas bled from the chamber wall drives the turbine
        flows += [([(20, 34), (20, 53)] + to_cc_fuel, FUEL), ([(80, 34), (80, 68), (56, 68), (56, 77.5)], OX),
                  ([(36, 84), (29, 84), (29, 27), (44, 27)], GAS), ([(50, 34), (50, 52)], GAS)]
        valves += [(20, 50), (80, 48)]
        labels += [("Exhaust", 53, 46, "l"), ("Tapped gas", 31, 66, "l")]
    elif kind == "expander_bleed":                             # part of the heated fuel drives the turbine, vented
        flows += [([(20, 34), (20, 125)] + up + [(33.1, 76), (33.1, 70), (44, 70), (44, 77.5)], FUEL),
                  ([(33.1, 70), (33.1, 27), (44, 27)], FUEL), ([(80, 34), (80, 68), (56, 68), (56, 77.5)], OX),
                  ([(50, 34), (50, 52)], GAS)]
        valves += [(20, 48), (80, 48)]
        labels += [("Vented", 53, 46, "l"), ("Heated fuel", 31, 50, "r")]
    elif kind == "full_flow":                                  # two preburners, two turbines, gas-gas injection
        flows = [([(5, 8), (5, SH_Y), (11, SH_Y)], FUEL), ([(95, 8), (95, SH_Y), (89, SH_Y)], OX),
                 ([(18, 34), (18, 53), (25, 53)], FUEL), ([(82, 34), (82, 53), (75, 53)], OX),
                 ([(36, 44), (36, 34)], RICH), ([(64, 44), (64, 34)], OXRICH),
                 ([(42, 27), (46, 27), (46, 72), (46, 77.5)], RICH), ([(58, 27), (54, 27), (54, 72), (54, 77.5)], OXRICH)]
        valves += [(18, 46), (82, 46)]
        labels = [("Fuel pump", 18, 4, "c"), ("Fuel" + chr(10) + "turbine", 36, 6, "c"), ("Ox" + chr(10) + "turbine", 64, 6, "c"),
                  ("Oxidizer pump", 82, 4, "c"), (fu_name, 1, 30, "l"), (ox_name, 99, 30, "r"),
                  ("Fuel-rich" + chr(10) + "preburner", 36, 53.5, "c"), ("Ox-rich" + chr(10) + "preburner", 64, 53.5, "c"),
                  ("Control" + chr(10) + "valves", 83, 42, "l")]
        return flows, valves, labels
    elif kind == "electric_pump":                              # batteries and motors replace the turbine
        flows = [([(5, 8), (5, SH_Y), (13, SH_Y)], FUEL), ([(95, 8), (95, SH_Y), (87, SH_Y)], OX),
                 ([(20, 34), (20, 74)] + ([(20, 125)] + up + [(33.1, 76), (33.1, 70), (44, 70), (44, 77.5)] if regen else
                                            [(44, 74), (44, 77.5)]), FUEL),
                 ([(80, 34), (80, 68), (56, 68), (56, 77.5)], OX),
                 ([(44, 20), (32, 20)], ELEC), ([(56, 20), (68, 20)], ELEC)]
        if regen:
            flows[2] = ([(20, 34), (20, 125)] + up + [(33.1, 76), (33.1, 70), (44, 70), (44, 77.5)], FUEL)
        valves += [(20, 48), (80, 48)]
        labels = [("Fuel pump", 18, 4, "c"), ("Battery", 50, 4, "c"), ("Oxidizer pump", 82, 4, "c"),
                  (fu_name, 1, 30, "l"), (ox_name, 99, 30, "r"), ("Motor", 33, 33, "c"), ("Motor", 67, 33, "c"),
                  ("Control" + chr(10) + "valves", 83, 48, "l")]
        return flows, valves, labels
    else:                                                      # expander
        flows += [([(20, 34), (20, 125)] + up + [(33.1, 76), (33.1, 27), (44, 27)], FUEL),
                  ([(80, 34), (80, 68), (56, 68), (56, 77.5)], OX), ([(50, 34), (50, 62), (50, 77.5)], GAS)]
        valves += [(20, 48), (80, 48)]
        labels += [("Heated fuel", 53, 56, "l")]
    labels += [("Control" + chr(10) + "valves", 86 if kind in ("expander", "expander_bleed", "tap_off") else 83, 48 if kind in ("expander", "expander_bleed", "tap_off") else 60, "l")]
    return flows, valves, labels


def _pump(path_x: float, y: float, flip: bool) -> QPainterPath:
    """The flared pump / turbine casing of the textbook diagrams, 12 wide and 26 tall."""
    path = QPainterPath(QPointF(4, 0))
    path.lineTo(12, 0)
    path.lineTo(12, 26)
    path.lineTo(4, 26)
    path.quadTo(4, 17, 0, 17)
    path.lineTo(0, 9)
    path.quadTo(4, 9, 4, 0)
    path.closeSubpath()
    from PySide6.QtGui import QTransform
    tr = QTransform()
    tr.translate(path_x, y)
    if flip:
        tr.translate(12, 0)
        tr.scale(-1, 1)
    return tr.map(path)


class Schematic:
    """Draws one cycle at one animation phase; ``graph`` is the sandbox graph."""

    def __init__(self, graph: dict) -> None:
        self.graph = graph
        self.kind = identify_cycle(graph)
        self.regen = any(n["type"] == "regen_jacket" for n in graph["nodes"])
        if self.kind in ("expander", "expander_bleed") and not self.regen:
            self.kind = "custom"

    # ------------------------------------------------------------------------------------------ drawing
    def draw(self, p: QPainter, size: QRectF, phase: float) -> None:
        t = theme.t
        ink, bg = QColor(t["text"]), QColor(t["viewport_bg"])
        p.setRenderHint(QPainter.Antialiasing)
        p.fillRect(size, bg)
        top = 26.0
        k = min(size.width() / 112.0, (size.height() - top) / 135.0)
        p.save()
        p.translate(size.left() + (size.width() - 112 * k) / 2 + 4 * k, size.top() + top)
        p.scale(k, k)
        flows, valves, labels = _flows(self.kind, self.regen, self.graph)
        self._machinery(p, ink, bg, phase)
        for pts, col in flows:                                  # lines under the equipment
            self._line(p, pts, QColor(col), 1.5)
        self._chamber(p, ink, phase)
        for pts, col in flows:
            self._arrows(p, pts, QColor(col), phase)
        for cx, cy in valves:
            self._valve(p, cx, cy, ink, bg)
        self._labels(p, ink, labels)
        p.restore()
        f = QFont(p.font())
        f.setPointSizeF(11.0)
        f.setBold(True)
        p.setFont(f)
        p.setPen(ink)
        p.drawText(QRectF(size.left() + 10, size.top() + 4, size.width() - 20, 22), Qt.AlignVCenter | Qt.AlignLeft,
                   f"{self.graph.get('name', '')}  -  {KIND_NAMES[self.kind]}")

    def _line(self, p, pts, color, w) -> None:
        path = QPainterPath(QPointF(*pts[0]))
        for q in pts[1:]:
            path.lineTo(QPointF(*q))
        pen = QPen(color, w)
        pen.setJoinStyle(Qt.MiterJoin)
        pen.setCapStyle(Qt.FlatCap)
        p.setPen(pen)
        p.setBrush(Qt.NoBrush)
        p.drawPath(path)

    def _arrows(self, p, pts, color, phase) -> None:
        segs = [(pts[i], pts[i + 1], math.dist(pts[i], pts[i + 1])) for i in range(len(pts) - 1)]
        total = sum(s[2] for s in segs)
        n = max(1, round(total / 15.0))
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(theme.t["text"]))                      # plain line color, arrow heads in the text color
        for i in range(n):
            d = ((i + phase) / n) * total
            for a, b, ln in segs:
                if d <= ln:
                    ang = math.degrees(math.atan2(b[1] - a[1], b[0] - a[0]))
                    q = (a[0] + (b[0] - a[0]) * d / ln, a[1] + (b[1] - a[1]) * d / ln)
                    p.save()
                    p.translate(*q)
                    p.rotate(ang)
                    p.drawPolygon(QPolygonF([QPointF(2.2, 0), QPointF(-1.4, -1.7), QPointF(-1.4, 1.7)]))
                    p.restore()
                    break
                d -= ln

    def _valve(self, p, cx, cy, ink, bg) -> None:
        p.setPen(QPen(ink, 0.7))
        p.setBrush(bg)
        p.drawEllipse(QPointF(cx, cy), 2.7, 2.7)
        p.drawLine(QPointF(cx - 1.9, cy - 1.9), QPointF(cx + 1.9, cy + 1.9))
        p.drawLine(QPointF(cx - 1.9, cy + 1.9), QPointF(cx + 1.9, cy - 1.9))

    def _machinery(self, p, ink, bg, phase) -> None:
        if self.kind == "pressure_fed":
            for x in (14, 68):                                  # propellant tanks
                p.setPen(QPen(ink, 0.8))
                p.setBrush(Qt.NoBrush)
                p.drawRoundedRect(QRectF(x + 0.0, 8, 18, 28), 7, 7)
            return
        k = self.kind
        bars = {"full_flow": [(24, 30), (70, 76)], "electric_pump": [(26, 32), (68, 74)]}.get(k, [(26, 74)])
        pumps = {"full_flow": [(12, False), (30, False), (58, True), (76, True)],
                 "electric_pump": [(14, False), (74, True)]}.get(k, [(14, False), (44, False), (74, True)])
        # shafts: plain bars with ticks that run along them
        for x0, x1 in bars:
            p.setPen(QPen(ink, 0.8))
            p.setBrush(Qt.NoBrush)
            p.drawRect(QRectF(x0, SH_Y - 2.0, x1 - x0, 4.0))
            p.setPen(QPen(ink, 0.5))
            n = max(2, int((x1 - x0) / 4))
            for i in range(n):
                x = x0 + ((i + phase) % n) * (x1 - x0) / n
                p.drawLine(QPointF(x, SH_Y - 2.0), QPointF(x, SH_Y + 2.0))
        for x, flip in pumps:
            p.setPen(QPen(ink, 0.9))
            p.setBrush(bg)
            p.drawPath(_pump(x, 8, flip))
        p.setPen(QPen(ink, 0.9))
        p.setBrush(Qt.NoBrush)
        if k in ("gas_generator", "staged_fuel", "staged_ox"):
            p.drawRoundedRect(QRectF(39, 44, 22, 18), 5, 5)
        elif k == "full_flow":
            p.drawRoundedRect(QRectF(25, 44, 22, 18), 5, 5)
            p.drawRoundedRect(QRectF(53, 44, 22, 18), 5, 5)
        elif k == "electric_pump":
            p.drawRect(QRectF(44, 10, 12, 20))                   # battery
            p.drawLine(QPointF(48, 10), QPointF(48, 7))
            p.drawLine(QPointF(52, 10), QPointF(52, 7))
            for x in (28, 66):                                    # motors
                p.drawRect(QRectF(x, 13, 8, 14))

    @staticmethod
    def _wall_x(y: float) -> float:
        ys = [q[1] for q in LEFT]
        return float(__import__("numpy").interp(y, ys, [q[0] for q in LEFT]))

    def _chamber(self, p, ink, phase) -> None:
        left = LEFT
        right = [(100 - x, y) for x, y in reversed(left)]
        body = QPainterPath(QPointF(*left[0]))
        for q in left[1:]:
            body.lineTo(QPointF(*q))
        for q in right:
            body.lineTo(QPointF(*q))
        body.closeSubpath()
        p.setBrush(Qt.NoBrush)
        p.setPen(QPen(ink, 0.7))
        p.drawPath(body)
        if self.regen:                                          # regenerative cooling tubes hugging the nozzle wall
            self._line(p, [(x - 1.4, y) for x, y in left], QColor(FUEL), 1.4)
            self._line(p, [(100 - x + 1.4, y) for x, y in left], QColor(FUEL), 1.4)
        # exhaust: fixed plain lines across the nozzle, an arrow on each that moves to the right
        for i in range(6):
            y = 103.0 + i * 4.2
            half = 50.0 - self._wall_x(y + 1.2) - 1.5          # half width of the bell at that height
            p.setPen(QPen(QColor("#ff6a2a"), 0.7))
            p.drawLine(QPointF(50 - half, y), QPointF(50 + half, y))
            x = 50 - half + ((phase + i / 6.0) % 1.0) * 2 * half
            p.setPen(Qt.NoPen)
            p.setBrush(QColor("#ff6a2a"))
            p.drawPolygon(QPolygonF([QPointF(x + 1.6, y), QPointF(x - 1.2, y - 1.6), QPointF(x - 1.2, y + 1.6)]))
            p.setBrush(Qt.NoBrush)

    def _labels(self, p, ink, labels) -> None:
        f = QFont(p.font())
        f.setPixelSize(4)
        f.setBold(True)
        p.setFont(f)
        p.setPen(ink)
        for text, x, y, al in labels:
            p.setPen(ink)
            w = 40.0
            h = 5.0 * (text.count(chr(10)) + 1)                    # a two-line label needs a taller box
            rect = QRectF(x - (w / 2 if al == "c" else (0 if al == "l" else w)), y - h / 2, w, h)
            p.drawText(rect, {"c": Qt.AlignCenter, "l": Qt.AlignLeft, "r": Qt.AlignRight}[al] | Qt.AlignVCenter, text)
        f.setPixelSize(5)
        p.setFont(f)
        p.drawText(QRectF(36, 79.5, 28, 7), Qt.AlignCenter, "CC")
        f.setPixelSize(4)
        p.setFont(f)
        p.drawText(QRectF(30, 128, 40, 6), Qt.AlignCenter, "Nozzle")
        if self.regen:
            p.drawText(QRectF(70, 102, 40, 12), Qt.AlignLeft | Qt.AlignVCenter, "Regenerative\ncooling")
