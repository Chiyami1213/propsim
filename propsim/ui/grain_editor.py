"""Dialogs for the user-built grain: the section editor (shapes, inhibitors) and the grain-stack editor.

Both store plain JSON text (see :mod:`propsim.core.grain.builder` and :mod:`propsim.core.grain.stack`), so the result is
saved with the case.  :func:`edit` opens the dialog named by a ``T`` field's ``editor`` and returns the new text, or
``None`` when cancelled; :func:`summary` gives the one-line description shown beside the button.
"""
from __future__ import annotations

import json

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox, QHBoxLayout, QLabel,
                               QPushButton, QSpinBox, QTableWidget, QTableWidgetItem, QVBoxLayout)

from ..core.grain import builder as gb
from ..core.grain import stack as gs

HINT = {"circle": "r; cx; cy (mm)", "box": "w; h; cx; cy; angle (mm, deg)", "slot": "r1; w; r0; angle (mm, deg)",
        "polygon": "pts=[[x,y],...] (mm)", "star": "n; r_tip; r_valley (mm)"}
EXAMPLE = {"circle": "r=14", "box": "w=10; h=30", "slot": "r1=40; w=8", "polygon": "pts=[[0,0],[20,0],[10,15]]",
           "star": "n=6; r_tip=40; r_valley=18"}


def _fmt_params(params: dict) -> str:
    return "; ".join(f"{k}={json.dumps(v)}" for k, v in params.items())


def _parse_params(text: str) -> dict:
    out = {}
    for part in filter(None, (t.strip() for t in text.split(";"))):
        if "=" not in part:
            raise ValueError(f"'{part}': write name=value, separated by ;")
        k, v = part.split("=", 1)
        try:
            out[k.strip()] = json.loads(v)
        except json.JSONDecodeError:
            out[k.strip()] = v.strip()                                # bare words: law=marxman, pair=LOX/HTPB
    return out


def summary(editor: str, text: str) -> str:
    if not text.strip():
        return "default"
    try:
        if editor == "section":
            sp = gb.spec_from_json(text)
            n_in = sum(s.inhibited for s in sp.shapes)
            return f"{len(sp.shapes)} shape(s)" + (f", {n_in} inhibited" if n_in else "")
        defs = gs.parse_stack(text, gs.FUEL_KEYS if editor == "hstack" else gs.PROP_KEYS)
        return f"{len(defs)} grain(s), {sum(d.L for d in defs) * 1e3:.0f} mm"
    except ValueError:
        return "invalid"


def _image(spec: gb.SectionSpec, n: int = 200) -> QImage:
    """Section picture: propellant grey, port blue (coated surfaces orange), case outline."""
    R = spec.R_out
    xs = np.linspace(-R * 1.04, R * 1.04, n)
    X, Y = np.meshgrid(xs, xs[::-1])
    F, S = gb.front_field(spec, X, Y)
    burn = np.zeros(X.shape, bool)
    for s in spec.shapes:
        if not s.inhibited:
            burn |= gb.shape_sdf(s, X, Y) < 0
    coat = (S < 0) if S is not None else np.zeros(X.shape, bool)
    r = np.hypot(X, Y)
    img = np.full((n, n, 3), 30, np.uint8)
    img[r <= R] = (150, 150, 150)
    img[burn & (r <= R)] = (78, 161, 255)
    img[coat & (r <= R)] = (255, 157, 60)
    img[np.abs(r - R) < 0.012 * R * 2] = (240, 240, 240)
    return QImage(img.data, n, n, 3 * n, QImage.Format_RGB888).copy()


class SectionEditor(QDialog):
    """Shapes table with a live picture of the section; the port is the union of the shapes."""

    def __init__(self, text: str, ctx: dict, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Custom grain section")
        self.R_out = float(ctx["R_out"] if "R_out" in ctx else ctx.get("R_p0", 25.0) + ctx.get("web", 35.0))
        spec = gb.spec_from_json(text) if text.strip() else gb.default_spec(self.R_out)
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["Shape", "Inhibited", "Repeat", "Parameters"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setColumnWidth(0, 130)
        self.pic = QLabel()
        self.pic.setFixedSize(300, 300)
        self.pic.setAlignment(Qt.AlignCenter)
        self.info = QLabel()
        self.info.setWordWrap(True)
        self.outer = QCheckBox("Outer surface burns too (free-standing grain)")
        self.outer.setChecked(spec.outer_burn)
        add, rm = QPushButton("Add shape"), QPushButton("Remove shape")
        add.clicked.connect(lambda: self._add("circle", False, 1, {"r": 8.0}))
        rm.clicked.connect(self._remove)
        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        left, bar = QVBoxLayout(), QHBoxLayout()
        bar.addWidget(add)
        bar.addWidget(rm)
        bar.addStretch(1)
        left.addWidget(QLabel(f"Port shapes inside the case of radius {self.R_out:g} (mm); the port is their union."))
        left.addWidget(self.table, 1)
        left.addLayout(bar)
        left.addWidget(self.outer)
        left.addWidget(QLabel("Parameters, separated by ;  "
                              + "  ".join(f"{HINT[k]}" for k in ("circle", "slot")) + " ... (hover a shape)"))
        right = QVBoxLayout()
        right.addWidget(self.pic)
        right.addWidget(self.info, 1)
        row = QHBoxLayout()
        row.addLayout(left, 1)
        row.addLayout(right)
        lay = QVBoxLayout(self)
        lay.addLayout(row)
        lay.addWidget(self.buttons)
        self.resize(900, 460)
        for s in spec.shapes:
            self._add(s.kind, s.inhibited, s.repeat, {k: (list(map(list, v)) if k == "pts" else v)
                                                      for k, v in s.params})
        self.table.cellChanged.connect(lambda *_: self.refresh())
        self.outer.toggled.connect(lambda *_: self.refresh())
        self.refresh()

    def _add(self, kind, inhibited, repeat, params) -> None:
        r = self.table.rowCount()
        self.table.blockSignals(True)
        self.table.insertRow(r)
        cb = QComboBox()
        for k in gb.SHAPES:
            cb.addItem(gb.SHAPE_LABELS[k], k)
        cb.setCurrentIndex(list(gb.SHAPES).index(kind))
        cb.setToolTip("; ".join(f"{gb.SHAPE_LABELS[k]}: {HINT[k]}" for k in gb.SHAPES))
        cb.currentIndexChanged.connect(lambda *_: self.refresh())
        self.table.setCellWidget(r, 0, cb)
        chk = QCheckBox()
        chk.setChecked(inhibited)
        chk.setToolTip("Inhibited: the surface is coated and does not regress")
        chk.toggled.connect(lambda *_: self.refresh())
        self.table.setCellWidget(r, 1, chk)
        sp = QSpinBox()
        sp.setRange(1, 48)
        sp.setValue(repeat)
        sp.setToolTip("Copies around the axis")
        sp.valueChanged.connect(lambda *_: self.refresh())
        self.table.setCellWidget(r, 2, sp)
        self.table.setItem(r, 3, QTableWidgetItem(_fmt_params(params)))
        self.table.blockSignals(False)
        self.refresh()

    def _remove(self) -> None:
        r = self.table.currentRow()
        if r >= 0:
            self.table.removeRow(r)
            self.refresh()

    def spec(self) -> gb.SectionSpec:
        shapes = []
        for r in range(self.table.rowCount()):
            kind = self.table.cellWidget(r, 0).currentData()
            shapes.append(gb.make_shape(kind, self.table.cellWidget(r, 1).isChecked(),
                                        self.table.cellWidget(r, 2).value(),
                                        **_parse_params(self.table.item(r, 3).text())))
        if not shapes:
            raise ValueError("add at least one shape")
        return gb.SectionSpec(self.R_out, tuple(shapes), self.outer.isChecked())

    def text(self) -> str:
        return self.spec().to_json()

    def refresh(self) -> None:
        ok = self.buttons.button(QDialogButtonBox.Ok)
        try:
            spec = self.spec()
            b = gb.build_section(spec, 96)
            self.pic.setPixmap(QPixmap.fromImage(_image(spec)).scaled(300, 300, Qt.KeepAspectRatio,
                                                                       Qt.SmoothTransformation))
            self.info.setText(f"web {b.web:.1f} (mm), initial burning perimeter {b.perim[0]:.0f} (mm), port "
                              f"{100 * b.area[0] / b.A_case:.0f} % of the case area.  Blue: burning port; orange: "
                              f"inhibited; grey: propellant.")
            ok.setEnabled(True)
        except (ValueError, TypeError, KeyError, json.JSONDecodeError) as e:
            self.info.setText(f"Cannot build this section: {e}")
            ok.setEnabled(False)


class StackEditor(QDialog):
    """One row per grain, head end first: length, end faces, section, propellant overrides."""

    COLS = ["Length (mm)", "Head end", "Aft end", "Section", "Propellant overrides"]
    hybrid = False
    keys = gs.PROP_KEYS
    HELP = ("Grains from the head end to the aft end.  Overrides change only the named propellant values and are "
            "separated by ;  e.g.  a_mm=9; cstar=1400; gamma=1.25; rho_p=1650.\nFields: rho_p, a_mm, n, sigma_p, T_i, "
            "cstar, Tc, gamma, alpha, beta (same units as the propellant parameters).  Empty = the motor's propellant.")

    def __init__(self, text: str, ctx: dict, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Fuel grain stack" if self.hybrid else "Grain stack")
        self.ctx = ctx
        self.sections: list[dict | None] = []
        self.table = QTableWidget(0, len(self.COLS))
        self.table.setHorizontalHeaderLabels(self.COLS)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.info = QLabel()
        self.info.setWordWrap(True)
        add, dup, rm = QPushButton("Add grain"), QPushButton("Duplicate"), QPushButton("Remove")
        clear = QPushButton("Single grain (clear stack)")
        add.clicked.connect(lambda: self._add(gs.GrainDef(float(ctx.get("L_g", 0.2)))))
        dup.clicked.connect(self._dup)
        rm.clicked.connect(self._remove)
        clear.clicked.connect(self._clear)
        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        bar = QHBoxLayout()
        for b in (add, dup, rm, clear):
            bar.addWidget(b)
        bar.addStretch(1)
        lay = QVBoxLayout(self)
        help_lbl = QLabel(self.HELP)
        help_lbl.setWordWrap(True)
        lay.addWidget(help_lbl)
        lay.addWidget(self.table, 1)
        lay.addLayout(bar)
        lay.addWidget(self.info)
        lay.addWidget(self.buttons)
        self.resize(900, 420)
        if self.hybrid:
            self.table.setHorizontalHeaderLabels(["Length (mm)", "", "", "Port", "Fuel overrides"])
            self.table.setColumnHidden(1, True)
            self.table.setColumnHidden(2, True)
        defs = gs.parse_stack(text, self.keys)
        if not defs:
            defs = [gs.GrainDef(float(ctx.get("L_g", 0.6)))]
        for d in defs:
            self._add(d)
        self.table.cellChanged.connect(lambda *_: self.refresh())

    def _add(self, d: gs.GrainDef) -> None:
        r = self.table.rowCount()
        self.table.blockSignals(True)
        self.table.insertRow(r)
        self.table.setCellWidget(r, 0, self._length(d.L * 1e3))
        for c, burning in ((1, d.burn_head), (2, d.burn_aft)):
            cb = QComboBox()
            cb.addItems(["Inhibited", "Burning"])
            cb.setCurrentIndex(int(burning))
            cb.currentIndexChanged.connect(lambda *_: self.refresh())
            self.table.setCellWidget(r, c, cb)
        self.sections.append(d.section)
        btn = QPushButton()
        btn.clicked.connect(lambda _=False, b=btn: self._edit_section(b))
        self.table.setCellWidget(r, 3, btn)
        self._label_section(btn, d.section)
        self.table.setItem(r, 4, QTableWidgetItem(_fmt_params(dict(d.prop))))
        self.table.blockSignals(False)
        self.refresh()

    def _length(self, v: float) -> QDoubleSpinBox:
        sp = QDoubleSpinBox()
        sp.setRange(5.0, 5000.0)
        sp.setDecimals(1)
        sp.setValue(v)
        sp.valueChanged.connect(lambda *_: self.refresh())
        return sp

    def _label_section(self, btn: QPushButton, sec: dict | None) -> None:
        own = "Motor port" if self.hybrid else "Motor section"
        btn.setText(own if sec is None else "Custom shapes…" if "shapes" in sec else f"{sec.get('kind')}")

    def _row_of(self, btn) -> int:
        return next(r for r in range(self.table.rowCount()) if self.table.cellWidget(r, 3) is btn)

    def _edit_section(self, btn: QPushButton) -> None:
        r = self._row_of(btn)
        cur = self.sections[r]
        dlg = SectionEditor(json.dumps(cur) if cur and "shapes" in cur else "", self.ctx, self)
        dlg.setWindowTitle("Custom grain section (this grain)")
        if dlg.exec() == QDialog.Accepted:
            self.sections[r] = dlg.spec().to_dict()
            self._label_section(btn, self.sections[r])
        self.refresh()

    def _dup(self) -> None:
        r = self.table.currentRow()
        if r >= 0:
            self._add(self.defs()[r])

    def _remove(self) -> None:
        r = self.table.currentRow()
        if r >= 0 and self.table.rowCount() > 1:
            self.table.removeRow(r)
            del self.sections[r]
            self.refresh()

    def _clear(self) -> None:
        self._cleared = True
        self.accept()

    _cleared = False

    def defs(self) -> list[gs.GrainDef]:
        out = []
        for r in range(self.table.rowCount()):
            prop = _parse_params(self.table.item(r, 4).text())
            out.append(gs.GrainDef(self.table.cellWidget(r, 0).value() * 1e-3, self.sections[r],
                                   tuple(sorted((k, gs._freeze(v) if k in gs._TEXT_KEYS else float(v))
                                                for k, v in prop.items())),
                                   self.table.cellWidget(r, 1).currentIndex() == 1,
                                   self.table.cellWidget(r, 2).currentIndex() == 1))
        return out

    def text(self) -> str:
        if self._cleared:
            return ""
        text = gs.dump_stack(self.defs())
        gs.parse_stack(text, self.keys)                               # validates field names and lengths
        return text

    def refresh(self) -> None:
        ok = self.buttons.button(QDialogButtonBox.Ok)
        try:
            d = self.defs()
            gs.parse_stack(gs.dump_stack(d), self.keys)
            n_burn = sum(x.burn_head + x.burn_aft for x in d)
            self.info.setText(f"{len(d)} grain(s), total length {sum(x.L for x in d) * 1e3:.0f} mm"
                              + ("." if self.hybrid else f", {n_burn} burning end face(s)."))
            ok.setEnabled(True)
        except (ValueError, TypeError, json.JSONDecodeError) as e:
            self.info.setText(f"Cannot use this stack: {e}")
            ok.setEnabled(False)


class HybridStackEditor(StackEditor):
    """Fuel grains of a hybrid: length, port and fuel overrides (end faces of hybrid grains are not modelled)."""

    hybrid = True
    keys = gs.FUEL_KEYS
    HELP = ("Fuel grains from the head end to the aft end.  Overrides change only the named fuel values, separated "
            "by ;  e.g.  rho_f=900; a=0.1171; n=0.62  or  law=marxman; B=8.\nFields: rho_f (kg/m³), law (power or "
            "marxman), a (mm/s with G in kg/m²/s), n, B, pair (a CEA pair such as LOX/HTPB), chem (your own "
            "chemistry: chem={\"of\": [1, 2, 3], \"cstar\": [...], \"Tc\": [...], \"gamma\": [...]}, c* in m/s, Tc "
            "in K).  Empty = the motor's fuel.  With several fuels the gas properties are mixed by fuel mass flow.")


EDITORS = {"section": SectionEditor, "stack": StackEditor, "hstack": HybridStackEditor}


def edit(editor: str, text: str, ctx: dict, parent=None) -> str | None:
    dlg = EDITORS[editor](text, ctx, parent)
    return dlg.text() if dlg.exec() == QDialog.Accepted else None
