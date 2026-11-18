"""Guided build of a liquid engine in the cycle sandbox: pick the cycle type, add the required and optional components
(with a live checklist), connect them (with the order to follow and the ports still open), then solve."""
from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QButtonGroup, QFrame, QHBoxLayout, QLabel, QPushButton, QRadioButton, QScrollArea,
                               QStackedWidget, QVBoxLayout, QWidget)

from ..core.engine_graph import edit as E

# cycle -> description, required components (type, count, role), optional components (type, role), wiring order
CYCLES = {
    "pressure_fed": dict(
        name="Pressure-fed",
        about="Tank pressure alone pushes the propellants into the chamber. Simplest and most reliable, limited to low "
              "chamber pressure because the tanks must be heavy.",
        required=[("tank_regulated", 2, "propellant tanks, one for the oxidizer and one for the fuel"),
                  ("valve_ball", 2, "a main valve in each propellant line"),
                  ("injector", 2, "an injector for each propellant"),
                  ("chamber", 1, "the combustion chamber"), ("nozzle_bell", 1, "the nozzle")],
        optional=[("line", "feed lines (friction and inertia)"), ("orifice", "calibration orifices"),
                  ("manifold", "injector manifolds"), ("check_valve", "check valves"),
                  ("regen_jacket", "regenerative cooling jacket (fuel cools the chamber before injection)")],
        wiring=["Tank outlet to the main valve inlet, for each propellant.",
                "Valve outlet to the injector inlet (through a line, orifice or manifold if you added them).",
                "Oxidizer injector outlet to the chamber ox_in, fuel injector outlet to fuel_in.",
                "Chamber throat to the nozzle inlet.",
                "Optional jacket: tank > valve > jacket coolant_in, jacket coolant_out > injector, chamber wall to "
                "the jacket wall."]),
    "gas_generator": dict(
        name="Gas generator (open)",
        about="Turbopumps raise the pressure. A small gas generator burns a little propellant fuel-rich to drive the "
              "turbine, and the turbine exhaust is dumped overboard (the Saturn V F-1 and the Merlin use it).",
        required=[("tank_regulated", 2, "propellant tanks"), ("pump_centrifugal", 2, "one pump per propellant"),
                  ("turbine_impulse", 1, "the turbine that drives the pumps"),
                  ("shaft", 1, "the shaft joining the turbine and the pumps"),
                  ("gas_generator", 1, "the gas generator (drives the turbine)"),
                  ("valve_poppet", 2, "main valves, plus valves on the gas generator feed"),
                  ("injector", 2, "main injectors"), ("chamber", 1, "the combustion chamber"),
                  ("nozzle_bell", 1, "the nozzle")],
        optional=[("inducer", "inducers ahead of the pumps"), ("orifice", "orifices that set the gas generator flow"),
                  ("line", "feed lines"), ("manifold", "manifolds"),
                  ("regen_jacket", "regenerative jacket on the chamber or nozzle")],
        wiring=["Tank outlet to the pump inlet, for each propellant.",
                "Pump outlet splits: one branch to the main valve and injector, one branch through a valve and "
                "orifice to the gas generator (ox_in and fuel_in).",
                "Gas generator outlet to the turbine inlet. Leave the turbine outlet open (exhaust overboard).",
                "Join the turbine and both pumps with the shaft (shaft ports).",
                "Main injectors to the chamber, chamber throat to the nozzle.",
                "Optional jacket between the fuel pump and the fuel injector, wall port to the chamber wall."]),
    "expander": dict(
        name="Expander (closed)",
        about="The fuel is heated in the cooling jacket, expands through the turbine, then burns in the chamber. No "
              "gas generator, so there is no wasted flow, but the chamber heat limits the power (RL10, LE-5).",
        required=[("tank_regulated", 2, "propellant tanks"), ("pump_centrifugal", 2, "one pump per propellant"),
                  ("turbine_impulse", 1, "the turbine"), ("shaft", 1, "the shaft"),
                  ("regen_jacket", 1, "the cooling jacket that heats the fuel (required: it is the heat source)"),
                  ("valve_butterfly", 2, "control valves"), ("injector", 2, "main injectors"),
                  ("chamber", 1, "the combustion chamber"), ("nozzle_bell", 1, "the nozzle")],
        optional=[("inducer", "inducers"), ("orifice", "orifices"), ("line", "feed lines"),
                  ("manifold", "manifolds")],
        wiring=["Fuel: tank, pump, valve, jacket coolant_in; jacket coolant_out to the turbine inlet; turbine outlet "
                "to the fuel injector.",
                "Oxidizer: tank, pump, valve, injector.",
                "Join the turbine and both pumps with the shaft.",
                "Jacket wall port to the chamber wall port.",
                "Injectors to the chamber, chamber throat to the nozzle."]),
    "staged": dict(
        name="Staged combustion (closed)",
        about="A preburner burns part of one propellant (fuel-rich here) to drive the turbine, and ALL of that gas then "
              "goes on into the main chamber. Highest performance, highest pressures (SSME, RD-180).",
        required=[("tank_regulated", 2, "propellant tanks"), ("pump_centrifugal", 2, "one pump per propellant"),
                  ("turbine_impulse", 1, "the turbine"), ("shaft", 1, "the shaft"),
                  ("gas_generator", 1, "the preburner (same component as a gas generator)"),
                  ("valve_ball", 2, "preburner and main valves"), ("injector", 2, "main injectors"),
                  ("chamber", 1, "the main combustion chamber"), ("nozzle_bell", 1, "the nozzle")],
        optional=[("inducer", "inducers"), ("pump_centrifugal", "boost pumps"), ("regen_jacket", "cooling jacket"),
                  ("orifice", "orifices"), ("manifold", "manifolds")],
        wiring=["Tank to pump, for each propellant.",
                "Fuel pump outlet: most to the preburner fuel_in, the rest through the jacket to the main injector.",
                "Oxidizer pump outlet: a small branch to the preburner ox_in, the rest to the main injector.",
                "Preburner outlet to the turbine inlet; turbine outlet to the main chamber (not overboard).",
                "Join the turbine and pumps with the shaft; injectors to the chamber; throat to the nozzle."]),
}
STEPS = ["Cycle", "Components", "Connect", "Solve"]


def counts(graph: dict) -> dict[str, int]:
    out: dict[str, int] = {}
    for n in graph["nodes"]:
        out[n["type"]] = out.get(n["type"], 0) + 1
    return out


def checklist(graph: dict, cycle: str) -> tuple[list[tuple[str, int, int, str]], list[tuple[str, bool, str]]]:
    """(required: type, have, need, role) and (optional: type, present, role) for ``cycle`` against ``graph``."""
    have = counts(graph)
    spec = CYCLES[cycle]
    req = [(t, have.get(t, 0), need, role) for t, need, role in spec["required"]]
    opt = [(t, have.get(t, 0) > 0, role) for t, role in spec["optional"]]
    return req, opt


class CycleGuide(QWidget):
    """Wizard panel; ``sandbox`` is the CycleSandbox it drives."""

    finished = Signal()

    def __init__(self, sandbox, parent=None) -> None:
        super().__init__(parent)
        self.sb = sandbox
        self.cycle = "gas_generator"
        self.step = 0
        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 4, 8, 8)
        self.crumbs = QLabel("")
        self.crumbs.setObjectName("caption")
        lay.addWidget(self.crumbs)
        self.stack = QStackedWidget()
        lay.addWidget(self.stack, 1)
        self.stack.addWidget(self._page_cycle())
        self.page_comp = QWidget()
        self.comp_lay = QVBoxLayout(self.page_comp)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.page_comp)
        self.stack.addWidget(scroll)
        self.page_conn = QLabel()
        self.page_conn.setWordWrap(True)
        self.page_conn.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.page_conn.setTextFormat(Qt.RichText)
        cs = QScrollArea()
        cs.setWidgetResizable(True)
        cs.setWidget(self.page_conn)
        self.stack.addWidget(cs)
        self.page_solve = QLabel()
        self.page_solve.setWordWrap(True)
        self.page_solve.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.page_solve.setTextFormat(Qt.RichText)
        self.stack.addWidget(self.page_solve)
        nav = QHBoxLayout()
        self.back = QPushButton("Back")
        self.next = QPushButton("Next")
        self.back.clicked.connect(lambda: self.go(self.step - 1))
        self.next.clicked.connect(self._advance)
        nav.addWidget(self.back)
        nav.addStretch(1)
        nav.addWidget(self.next)
        lay.addLayout(nav)
        sandbox.canvas.changed.connect(self.refresh)
        self.go(0)

    # ------------------------------------------------------------------ pages
    def _page_cycle(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        v.addWidget(QLabel("<b>Choose the engine cycle</b>"))
        self.group = QButtonGroup(self)
        self.about = QLabel()
        self.about.setWordWrap(True)
        for key, spec in CYCLES.items():
            rb = QRadioButton(spec["name"])
            rb.setChecked(key == self.cycle)
            rb.toggled.connect(lambda on, k=key: on and self._pick(k))
            self.group.addButton(rb)
            v.addWidget(rb)
        v.addWidget(self.about)
        hint = QLabel("Pick the propellants in the toolbar (Oxidizer, Fuel). <b>Start Blank Diagram</b> clears the "
                      "canvas for a new engine; the next step lists what to add.")
        hint.setWordWrap(True)
        v.addWidget(hint)
        blank = QPushButton("Start Blank Diagram")
        blank.clicked.connect(self.sb.new_doc)
        v.addWidget(blank)
        v.addStretch(1)
        self._pick(self.cycle)
        return w

    def _pick(self, key: str) -> None:
        self.cycle = key
        self.about.setText(CYCLES[key]["about"])
        if hasattr(self, "page_comp"):
            self.refresh()

    def refresh(self, *_) -> None:
        g = self.sb.doc["graph"]
        req, opt = checklist(g, self.cycle)
        while self.comp_lay.count():
            it = self.comp_lay.takeAt(0)
            if it.widget():
                it.widget().deleteLater()
        self.comp_lay.addWidget(QLabel("<b>Required components</b>"))
        for t, have, need, role in req:
            self._row(t, have, need, role, True)
        self.comp_lay.addWidget(QLabel("<b>Optional components</b>"))
        for t, present, role in opt:
            self._row(t, 1 if present else 0, 1, role, False)
        self.comp_lay.addStretch(1)
        missing = [E.TYPE_LABELS.get(t, t) for t, have, need, _ in req if have < need]
        self.next.setEnabled(self.step != 1 or not missing)
        self.missing = missing
        self._connect_page()
        self._solve_page()

    def _row(self, ntype: str, have: int, need: int, role: str, required: bool) -> None:
        row = QFrame()
        h = QHBoxLayout(row)
        h.setContentsMargins(0, 0, 0, 0)
        ok = have >= need
        mark = "✓" if ok else ("✗" if required else "○")
        lab = QLabel(f"{mark}  <b>{E.TYPE_LABELS.get(ntype, ntype)}</b> ({have}/{need}): {role}")
        lab.setWordWrap(True)
        add = QPushButton("Add")
        add.setFixedWidth(60)
        add.clicked.connect(lambda _=False, t=ntype: self.sb.canvas.add_component(t))
        h.addWidget(lab, 1)
        h.addWidget(add)
        self.comp_lay.addWidget(row)

    def _connect_page(self) -> None:
        g = self.sb.doc["graph"]
        steps = "".join(f"<li>{s}</li>" for s in CYCLES[self.cycle]["wiring"])
        open_ports = E.unconnected(g)
        shown = ", ".join(open_ports[:14]) + (" ..." if len(open_ports) > 14 else "")
        self.page_conn.setText(
            f"<b>Connect the components</b><p>Drag from an output port to an input port. Follow this order:</p>"
            f"<ol>{steps}</ol><p><b>Ports still open ({len(open_ports)}):</b> {shown or 'none'}</p>"
            "<p>An open turbine outlet is fine in a gas-generator cycle (exhaust overboard); every other open fluid "
            "port needs a connection or an ambient/exhaust boundary.</p>")

    def _solve_page(self) -> None:
        self.page_solve.setText(
            "<b>Solve</b><ol><li>Select a part and set its parameters in the Inspector (tank pressure, pump speed, "
            "throat area, valve position).</li><li>Optional: in Design Targets fix an output (for example chamber "
            "<i>p_c</i>, thrust, mixture ratio) and free the parameter the solver may size.</li>"
            "<li>Press <b>Solve</b>. Messages lists any problem; the diagram then colors by the solved pressures, "
            "temperatures or flows.</li><li>Press <b>Play Flow</b> or open Cycle Animation to watch the cycle.</li></ol>")

    # ------------------------------------------------------------------ navigation
    def go(self, i: int) -> None:
        self.step = max(0, min(len(STEPS) - 1, i))
        self.stack.setCurrentIndex(self.step)
        self.crumbs.setText("  >  ".join(f"<{s}>" if k == self.step else s for k, s in enumerate(STEPS)))
        self.back.setEnabled(self.step > 0)
        self.next.setText("Done" if self.step == len(STEPS) - 1 else "Next")
        self.refresh()

    def _advance(self) -> None:
        if self.step == len(STEPS) - 1:
            self._done()
        else:
            self.go(self.step + 1)

    def _done(self) -> None:
        self.finished.emit()
        self.go(0)
