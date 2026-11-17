"""Schema -> slider form. Labels are LaTeX pixmaps; hover emits a LaTeX description with [dimensions]."""
from __future__ import annotations

import math
import re

from PySide6.QtCore import QEvent, Qt, Signal
from PySide6.QtWidgets import QHBoxLayout, QLabel, QSlider, QVBoxLayout, QWidget

from ..core.schema import ParamModel
from .latex import LatexLabel
from .latex_widgets import LatexComboBox
from .text import apa, hover_message, to_math


def _decimals(step: float) -> int:
    return max(0, int(round(-math.log10(step))))


class _Row(QWidget):
    changed = Signal()

    def __init__(self, meta: dict, value: float, parent=None) -> None:
        super().__init__(parent)
        self.meta = meta
        self.lo, self.hi, self.step = meta["lo"], meta["hi"], meta["step"]
        self.n = int(math.ceil((self.hi - self.lo) / self.step - 1e-9))
        self.dec = _decimals(self.step)
        lab = rf"${meta['symbol']}$ ({meta['units']})" if meta["units"] else rf"${meta['symbol']}$ (–)"
        self.label = LatexLabel(lab, "text", 10.0)
        self.value_lbl = QLabel()
        self.value_lbl.setObjectName("value")
        self.value_lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, self.n)
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 2, 0, 2)
        v.setSpacing(2)
        top = QHBoxLayout()
        top.addWidget(self.label)
        top.addStretch(1)
        top.addWidget(self.value_lbl)
        v.addLayout(top)
        v.addWidget(self.slider)
        self.set_value(value)
        self.slider.valueChanged.connect(self._moved)

    def set_value(self, x: float) -> None:
        self.slider.blockSignals(True)
        self.slider.setValue(int(round((x - self.lo) / self.step)))
        self.slider.blockSignals(False)
        self._show()

    def value(self) -> float:
        return min(round(self.lo + self.slider.value() * self.step, 10), self.hi)   # last step may be partial

    def _show(self) -> None:
        self.value_lbl.setText(f"{self.value():.{self.dec}f}")

    def _moved(self, *_) -> None:
        self._show()
        self.changed.emit()


class _ChoiceRow(QWidget):
    """Combo box for a discrete choice (propellant pair, equilibrium/frozen, ...). Same interface as ``_Row``."""

    changed = Signal()

    def __init__(self, meta: dict, value: str, parent=None) -> None:
        super().__init__(parent)
        self.meta = meta
        sym = meta["symbol"]
        m = re.fullmatch(r"\\mathrm\{(.+)\}", sym)                # a worded label, e.g. \mathrm{nozzle\ flow}
        self.label = LatexLabel(apa(m.group(1).replace(r"\ ", " ")) if m else rf"${sym}$", "text", 10.0)
        self.slider = LatexComboBox(9.5)                 # named ``slider`` so the form treats all rows alike
        self.slider.addItems(meta["labels"])
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 2, 0, 2)
        v.setSpacing(2)
        v.addWidget(self.label)
        v.addWidget(self.slider)
        self.set_value(value)
        self.slider.currentIndexChanged.connect(lambda *_: self.changed.emit())

    def set_value(self, x: str) -> None:
        self.slider.blockSignals(True)
        self.slider.setCurrentIndex(self.meta["choices"].index(x))
        self.slider.blockSignals(False)

    def value(self) -> str:
        return self.meta["choices"][self.slider.currentIndex()]


class ParamForm(QWidget):
    """Auto-generated parameter form for a ``ParamModel`` subclass."""

    changed = Signal()
    hovered = Signal(str)

    def __init__(self, model_cls: type[ParamModel], values: ParamModel | None = None, parent=None) -> None:
        super().__init__(parent)
        self.model_cls = model_cls
        self.rows: dict[str, _Row | _ChoiceRow] = {}
        values = values or model_cls()
        lay = QVBoxLayout(self)
        lay.setContentsMargins(6, 2, 6, 6)
        lay.setSpacing(0)
        group = None
        for meta in model_cls.fields_meta():
            if meta["group"] != group:
                group = meta["group"]
                title = apa(group)                                   # section headers: APA title case
                g = LatexLabel(title, "accent", 10.0) if "$" in to_math(title) else QLabel(title)
                g.setObjectName("groupTitle")
                g.setContentsMargins(0, 8, 0, 2)
                lay.addWidget(g)
            row = (_ChoiceRow if meta["kind"] == "choice" else _Row)(meta, getattr(values, meta["name"]))
            row.changed.connect(self.changed)
            for w in (row, row.label, row.slider):
                w.installEventFilter(self)
            self.rows[meta["name"]] = row
            lay.addWidget(row)
        lay.addStretch(1)

    def eventFilter(self, obj, ev):
        if ev.type() == QEvent.Enter:
            for r in self.rows.values():
                if obj in (r, r.label, r.slider):
                    m = r.meta
                    self.hovered.emit(hover_message(m["symbol"], m["desc"], m["dims"]))
                    break
        return False

    def values(self) -> ParamModel:
        return self.model_cls(**{k: r.value() for k, r in self.rows.items()})

    def set_values(self, obj: ParamModel | dict) -> None:
        d = obj if isinstance(obj, dict) else obj.model_dump()
        for k, v in d.items():
            if k in self.rows:
                self.rows[k].set_value(v)
