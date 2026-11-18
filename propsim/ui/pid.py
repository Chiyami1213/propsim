r"""P&ID symbols after ANSI/ISA-5.1 (instrumentation symbols and identification) and the usual ISO 10628 process
equipment shapes, drawn as line art with a QPainter.

* Process equipment: vessels, valves (gate/globe/ball/butterfly/check, with a diaphragm actuator), pumps, turbines,
  orifices, venturis, combustors, nozzles and heat exchangers.
* Instruments: a *bubble* (circle) with the function letters on top and the loop number below, field mounted (no line
  across the bubble).  First letter = measured variable (P pressure, T temperature, F flow, S speed), succeeding
  letters = function (T transmitter, I indicator, C controller, V valve, E primary element).
* Lines: process line solid, instrument signal dashed (electric signal), mechanical link dash-dot.

``draw_symbol`` draws one component centred in a rectangle; ``phase`` (0..1) turns the rotating parts so the same
symbols serve the animation panel.
"""
from __future__ import annotations

import math

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPainterPath, QPen

# ISA-5.1 tag letters of the instruments shown on each component: (function letters, measured connection)
INSTRUMENTS = {
    "tank_blowdown": ["PT"], "tank_regulated": ["PT"], "pump_centrifugal": ["PT", "ST"], "inducer": ["PT"],
    "turbine_impulse": ["ST"], "chamber": ["PT", "TT"], "gas_generator": ["PT", "TT"], "injector": ["PDT"],
    "cavitating_venturi": ["FT"], "orifice": ["PDT"], "regen_jacket": ["TT"], "nozzle_bell": [],
}
# equipment tag prefix (ISA-5.1 has none for equipment; the common P&ID practice is used)
EQUIPMENT = {"tank_blowdown": "T", "tank_regulated": "T", "pump_centrifugal": "P", "inducer": "P",
             "turbine_impulse": "TB", "chamber": "TC", "gas_generator": "GG", "injector": "INJ", "manifold": "M",
             "regen_jacket": "E", "nozzle_bell": "N", "shaft": "SH", "line": "L", "orifice": "FO",
             "cavitating_venturi": "FE", "check_valve": "CKV", "valve_ball": "XV", "valve_butterfly": "XV",
             "valve_poppet": "XV"}


def tag_for(ntype: str, index: int) -> str:
    """ISA-style tag: prefix and a loop number from 101 up, e.g. ``XV-103``."""
    return f"{EQUIPMENT.get(ntype, 'X')}-{100 + index}"


def _pen(color: QColor, w: float = 1.4) -> QPen:
    pen = QPen(color, w)
    pen.setJoinStyle(Qt.MiterJoin)
    pen.setCapStyle(Qt.FlatCap)
    return pen


def _bowtie(p: QPainter, r: QRectF) -> None:
    c = r.center()
    p.drawPolygon([r.topLeft(), r.bottomLeft(), c])
    p.drawPolygon([r.topRight(), r.bottomRight(), c])


def _actuator(p: QPainter, r: QRectF) -> None:
    """Diaphragm actuator on a stem (ISA-5.1 control valve body symbol)."""
    c = r.center()
    p.drawLine(QPointF(c.x(), c.y()), QPointF(c.x(), r.top() - 8))
    d = QRectF(c.x() - 7, r.top() - 15, 14, 14)
    p.drawChord(d, 0, 180 * 16)


def draw_symbol(p: QPainter, ntype: str, rect: QRectF, color: QColor, fill: QColor, phase: float = 0.0,
                command: bool = False) -> None:
    """One component as an ISA/ISO P&ID symbol inside ``rect`` (line art, ``fill`` behind it)."""
    p.save()
    p.setRenderHint(QPainter.Antialiasing)
    p.setPen(_pen(color))
    p.setBrush(fill)
    r, c = rect, rect.center()
    if ntype.startswith("tank"):                                           # vessel with domed ends
        p.drawRoundedRect(QRectF(r.left() + 6, r.top(), r.width() - 12, r.height()), 14, 14)
        if ntype == "tank_regulated":                                      # pressure regulator on the pressurant line
            p.drawLine(QPointF(r.left() + 6, c.y() + 10), QPointF(r.left() - 8, c.y() + 10))
    elif ntype in ("valve_ball", "valve_butterfly", "valve_poppet", "check_valve"):
        v = QRectF(c.x() - 15, c.y() - 11, 30, 22)
        if ntype == "check_valve":                                         # triangle against a stop bar
            p.drawPolygon([QPointF(v.left(), v.top()), QPointF(v.left(), v.bottom()), QPointF(v.right(), c.y())])
            p.drawLine(QPointF(v.right(), v.top()), QPointF(v.right(), v.bottom()))
        else:
            _bowtie(p, v)
            if ntype == "valve_ball":                                      # ball: open circle on the bow-tie
                p.setBrush(Qt.NoBrush)
                p.drawEllipse(c, 6, 6)
            elif ntype == "valve_poppet":                                  # globe/poppet: solid disc
                p.setBrush(color)
                p.drawEllipse(c, 4.5, 4.5)
            else:                                                          # butterfly: disc edge-on as a bar
                p.drawLine(QPointF(c.x(), v.top() - 1), QPointF(c.x(), v.bottom() + 1))
            if command:
                p.setBrush(fill)
                _actuator(p, v)
    elif ntype in ("pump_centrifugal", "inducer"):                         # circle, tangential discharge, vanes
        rad = min(r.width(), r.height()) / 2 - 2
        p.drawEllipse(c, rad, rad)
        p.drawLine(QPointF(c.x() + rad * 0.2, c.y() - rad), QPointF(c.x() + rad * 1.1, c.y() - rad))
        p.setPen(_pen(color, 1.1))
        for k in range(3):
            a = 2 * math.pi * (k / 3 + phase)
            p.drawLine(c, QPointF(c.x() + rad * 0.8 * math.cos(a), c.y() + rad * 0.8 * math.sin(a)))
        p.setBrush(color)
        p.drawEllipse(c, 2, 2)
    elif ntype == "turbine_impulse":                                       # trapezoid opening toward the exhaust
        poly = [QPointF(r.left() + 6, c.y() - 8), QPointF(r.right() - 6, r.top() + 1),
                QPointF(r.right() - 6, r.bottom() - 1), QPointF(r.left() + 6, c.y() + 8)]
        p.drawPolygon(poly)
        p.setPen(_pen(color, 1.0))
        for k in range(4):
            x = r.left() + 14 + k * (r.width() - 28) / 3
            h = 8 + 3 * k
            dy = h * math.sin(2 * math.pi * (phase + k / 8))
            p.drawLine(QPointF(x, c.y() - h + dy * 0.0), QPointF(x + 4, c.y() + h * 0.0 + dy * 0.0 + h))
    elif ntype in ("orifice",):                                            # two plates in the line
        p.drawLine(QPointF(c.x() - 4, r.top() + 4), QPointF(c.x() - 4, r.bottom() - 4))
        p.drawLine(QPointF(c.x() + 4, r.top() + 4), QPointF(c.x() + 4, r.bottom() - 4))
        p.drawLine(QPointF(r.left(), c.y()), QPointF(c.x() - 4, c.y()))
        p.drawLine(QPointF(c.x() + 4, c.y()), QPointF(r.right(), c.y()))
    elif ntype == "cavitating_venturi":                                    # converging then diverging section
        path = QPainterPath(QPointF(r.left(), c.y() - 12))
        path.lineTo(c.x() - 4, c.y() - 4)
        path.lineTo(r.right(), c.y() - 12)
        path.lineTo(r.right(), c.y() + 12)
        path.lineTo(c.x() - 4, c.y() + 4)
        path.lineTo(r.left(), c.y() + 12)
        path.closeSubpath()
        p.drawPath(path)
    elif ntype == "line":
        p.drawLine(QPointF(r.left(), c.y()), QPointF(r.right(), c.y()))
        p.setBrush(color)
        p.drawPolygon([QPointF(c.x() - 5, c.y() - 4), QPointF(c.x() - 5, c.y() + 4), QPointF(c.x() + 5, c.y())])
    elif ntype == "manifold":                                              # header
        p.drawRoundedRect(QRectF(r.left(), c.y() - 7, r.width(), 14), 3, 3)
    elif ntype == "injector":                                              # shower-head plate
        p.drawRect(QRectF(c.x() - 5, r.top() + 2, 10, r.height() - 4))
        for k in range(4):
            y = r.top() + 8 + k * (r.height() - 16) / 3
            p.drawLine(QPointF(c.x() - 5, y), QPointF(c.x() + 5, y))
    elif ntype in ("chamber", "gas_generator"):                            # combustor (and, for a chamber, the bell)
        body = QRectF(r.left() + 4, r.top() + 2, r.width() * (0.55 if ntype == "chamber" else 0.75), r.height() - 4)
        p.drawRoundedRect(body, 8, 8)
        if ntype == "chamber":
            x0 = body.right()
            path = QPainterPath(QPointF(x0, c.y() - 7))
            path.lineTo(x0 + 6, c.y() - 4)
            path.quadTo(x0 + 12, c.y() - 6, r.right(), r.top() + 2)
            path.lineTo(r.right(), r.bottom() - 2)
            path.quadTo(x0 + 12, c.y() + 6, x0 + 6, c.y() + 4)
            path.lineTo(x0, c.y() + 7)
            p.drawPath(path)
    elif ntype == "nozzle_bell":
        path = QPainterPath(QPointF(r.left() + 6, c.y() - 5))
        path.quadTo(c.x(), c.y() - 7, r.right() - 6, r.top() + 2)
        path.lineTo(r.right() - 6, r.bottom() - 2)
        path.quadTo(c.x(), c.y() + 7, r.left() + 6, c.y() + 5)
        path.closeSubpath()
        p.drawPath(path)
    elif ntype == "regen_jacket":                                          # heat exchanger: circle with zig-zag
        rad = min(r.width(), r.height()) / 2 - 3
        p.drawEllipse(c, rad, rad)
        zig = QPainterPath(QPointF(c.x() - rad, c.y()))
        for k, x in enumerate((-0.6, -0.3, 0.0, 0.3, 0.6, 1.0)):
            zig.lineTo(QPointF(c.x() + x * rad, c.y() + (-6 if k % 2 == 0 else 6)))
        p.setBrush(Qt.NoBrush)
        p.drawPath(zig)
    elif ntype == "shaft":                                                 # mechanical coupling
        p.drawRect(QRectF(r.left(), c.y() - 3, r.width(), 6))
        for k in range(1, 5):
            x = r.left() + k * r.width() / 5
            p.drawLine(QPointF(x - 3, c.y() + 3), QPointF(x + 3, c.y() - 3))
    else:
        p.drawRect(QRectF(c.x() - 16, c.y() - 12, 32, 24))
    p.restore()


def draw_bubble(p: QPainter, center: QPointF, letters: str, number: str, color: QColor, fill: QColor,
                radius: float = 12.0) -> None:
    """ANSI/ISA-5.1 field-mounted discrete instrument: circle, function letters above and loop number below."""
    p.save()
    p.setRenderHint(QPainter.Antialiasing)
    p.setPen(_pen(color, 1.1))
    p.setBrush(fill)
    p.drawEllipse(center, radius, radius)
    f = QFont(p.font())
    f.setPointSizeF(6.6)
    f.setBold(False)
    p.setFont(f)
    p.drawText(QRectF(center.x() - radius, center.y() - radius + 1, 2 * radius, radius), Qt.AlignCenter, letters)
    p.drawText(QRectF(center.x() - radius, center.y(), 2 * radius, radius - 1), Qt.AlignCenter, number)
    p.restore()


def signal_pen(color: QColor) -> QPen:
    """Instrument signal line (ISA-5.1: electric signal, dashed)."""
    pen = QPen(color, 1.0, Qt.DashLine)
    return pen


def mechanical_pen(color: QColor, w: float = 2.0) -> QPen:
    pen = QPen(color, w, Qt.DashDotLine)
    return pen
