"""Cycle sandbox *Trade Study* panel: sweep one input and plot the engine's response; compare whole designs."""
from __future__ import annotations

import copy
import math

import numpy as np
from PySide6.QtCore import QObject, QThread, Signal
from PySide6.QtWidgets import (QComboBox, QFileDialog, QHBoxLayout, QHeaderView, QLabel, QLineEdit, QMenu,
                               QProgressBar, QPushButton, QSpinBox, QSplitter, QTableWidget, QTableWidgetItem,
                               QTabWidget, QVBoxLayout, QWidget)
from PySide6.QtCore import Qt

from ..core.engine_graph import edit as E
from ..core.engine_graph import trade as T
from ..core.result import Curve, PlotSpec
from .plots import GridPlot

UNITS = {"Pa": ("MPa", 1e-6), "rad/s": ("rpm", 30.0 / math.pi), "W": ("kW", 1e-3), "m2": ("cm²", 1e4),
         "m3/s": ("L/s", 1e3)}
OUT_UNITS = {"p": ("MPa", 1e-6), "dp": ("MPa", 1e-6), "p_c": ("MPa", 1e-6), "p_gg": ("MPa", 1e-6),
             "p_injector_end": ("MPa", 1e-6), "p_exit": ("kPa", 1e-3), "power": ("MW", 1e-6), "thrust": ("kN", 1e-3),
             "speed": ("rpm", 30.0 / math.pi), "Q": ("MW", 1e-6), "q_max": ("MW/m²", 1e-6), "torque": ("N m", 1.0)}


def input_unit(kind: str, name: str, unit: str) -> tuple[str, float]:
    if kind == "target":
        return OUT_UNITS.get(name, ("", 1.0))
    if name == "position":
        return ("", 1.0)
    return UNITS.get(unit, (unit if unit not in ("1", None) else "", 1.0))


class _Job(QObject):
    progress = Signal(int, int)
    done = Signal(object, str)

    def __init__(self, fn) -> None:
        super().__init__()
        self.fn = fn
        self.stop = False

    def run(self) -> None:
        try:
            self.done.emit(self.fn(self), "")
        except Exception as e:                                   # noqa: BLE001 - reported in the panel
            self.done.emit(None, f"{type(e).__name__}: {e}")


class TradePanel(QWidget):
    """Sweep and Compare tabs; ``doc_getter`` returns the sandbox document being edited."""

    message = Signal(str)

    def __init__(self, doc_getter, parent=None) -> None:
        super().__init__(parent)
        self.doc_getter = doc_getter
        self.result: dict | None = None
        self.designs: dict[str, dict] = {}
        self.compared: dict[str, dict] = {}
        self._thread: QThread | None = None
        self._job: _Job | None = None
        tabs = QTabWidget()
        tabs.addTab(self._sweep_tab(), "Sweep")
        tabs.addTab(self._compare_tab(), "Compare Designs")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(tabs)
        self.tabs = tabs

    # ------------------------------------------------------------------------------------------------ sweep
    def _sweep_tab(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(6, 6, 6, 6)
        row = QHBoxLayout()
        self.in_box = QComboBox()
        self.in_box.setMinimumWidth(260)
        self.in_box.currentIndexChanged.connect(self._input_changed)
        self.lo, self.hi = QLineEdit(), QLineEdit()
        for e in (self.lo, self.hi):
            e.setFixedWidth(90)
        self.unit_lbl = QLabel("")
        self.n_box = QSpinBox()
        self.n_box.setRange(2, 60)
        self.n_box.setValue(9)
        self.run_btn = QPushButton("Run Sweep")
        self.run_btn.clicked.connect(self.run_sweep)
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._stop)
        for x in (QLabel("Vary"), self.in_box, QLabel("from"), self.lo, QLabel("to"), self.hi, self.unit_lbl,
                  QLabel("points"), self.n_box, self.run_btn, self.stop_btn):
            row.addWidget(x)
        row.addStretch(1)
        v.addLayout(row)
        row2 = QHBoxLayout()
        self.y_box = QComboBox()
        self.y_box.setMinimumWidth(260)
        self.y_box.currentIndexChanged.connect(self._plot)
        self.prog = QProgressBar()
        self.prog.setMaximumWidth(220)
        self.prog.setTextVisible(True)
        self.csv_btn = QPushButton("Export CSV")
        self.csv_btn.setEnabled(False)
        self.csv_btn.clicked.connect(self._export)
        self.status = QLabel("")
        self.status.setObjectName("hint")
        for x in (QLabel("Plot"), self.y_box, self.prog, self.csv_btn, self.status):
            row2.addWidget(x)
        row2.addStretch(1)
        v.addLayout(row2)
        split = QSplitter(Qt.Horizontal)
        self.plot = GridPlot(PlotSpec("Sweep", "", ""))
        self.table = QTableWidget(0, 0)
        self.table.setAlternatingRowColors(True)
        self.table.verticalHeader().setVisible(False)
        split.addWidget(self.plot)
        split.addWidget(self.table)
        split.setSizes([520, 480])
        v.addWidget(split, 1)
        return w

    def refresh_inputs(self) -> None:
        """Re-list what can be swept (after a template is loaded or the diagram is edited)."""
        doc = self.doc_getter()
        cur = self.in_box.currentData()
        self.in_box.blockSignals(True)
        self.in_box.clear()
        items = T.inputs(doc)
        targets = [i for i in items if i[0] == "target"]
        for kind, node, name, unit in targets + [i for i in items if i[0] == "param"]:
            u, _ = input_unit(kind, name, unit)
            label = f"{'Target' if kind == 'target' else node} · {name if kind == 'param' else node + '.' + name}"
            self.in_box.addItem(label + (f" ({u})" if u else ""), (kind, node, name, unit))
        self.in_box.blockSignals(False)
        i = self.in_box.findData(cur) if cur else -1
        self.in_box.setCurrentIndex(max(i, 0))
        self._input_changed()

    def _input_changed(self, *_) -> None:
        d = self.in_box.currentData()
        if not d:
            return
        kind, node, name, unit = d
        u, f = input_unit(kind, name, unit)
        self.unit_lbl.setText(u)
        v = T.current_value(self.doc_getter(), kind, node, name)
        if v is None:
            return
        if name == "position":
            lo, hi = max(0.05, v - 0.4), min(1.0, max(v, 0.6))
        else:
            lo, hi = 0.8 * v, 1.2 * v
        self.lo.setText(f"{lo * f:.6g}")
        self.hi.setText(f"{hi * f:.6g}")

    def run_sweep(self) -> None:
        if self._thread is not None:
            return
        d = self.in_box.currentData()
        if not d:
            self.status.setText("nothing to sweep: build or load an engine first")
            return
        kind, node, name, unit = d
        _, f = input_unit(kind, name, unit)
        try:
            lo, hi = float(self.lo.text()) / f, float(self.hi.text()) / f
        except ValueError:
            self.status.setText("the range must be two numbers")
            return
        values = list(np.linspace(lo, hi, self.n_box.value()))
        doc = copy.deepcopy(self.doc_getter())
        self.status.setText(f"sweeping {node}.{name} …")
        self.prog.setRange(0, len(values))
        self.prog.setValue(0)
        self._start(lambda job: T.sweep(doc, kind, node, name, values,
                                        progress=lambda a, b: job.progress.emit(a, b),
                                        cancelled=lambda: job.stop), self._swept)

    def _start(self, fn, slot) -> None:
        self.run_btn.setEnabled(False)
        self.cmp_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self._thread = QThread(self)
        self._job = _Job(fn)
        self._job.moveToThread(self._thread)
        self._thread.started.connect(self._job.run)
        self._job.progress.connect(lambda a, b: (self.prog.setRange(0, b), self.prog.setValue(a)))
        self._job.done.connect(slot)
        self._thread.start()

    def _finish(self) -> None:
        self._thread.quit()
        self._thread.wait()
        self._thread, self._job = None, None
        self.run_btn.setEnabled(True)
        self.cmp_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

    def _stop(self) -> None:
        if self._job is not None:
            self._job.stop = True
            self.status.setText("stopping after the current point …")

    def wait(self, ms: int = 900000) -> None:
        """Block until a running sweep or comparison finishes (tests)."""
        import time
        from PySide6.QtWidgets import QApplication
        t0 = time.time()
        while self._thread is not None and (time.time() - t0) * 1e3 < ms:
            QApplication.processEvents()
            time.sleep(0.01)

    def _swept(self, res, err: str) -> None:
        self._finish()
        if res is None:
            self.status.setText(f"sweep failed: {err}")
            self.message.emit(f"sweep failed: {err}")
            return
        self.result = res
        n_ok = sum(res["converged"])
        self.status.setText(f"{n_ok} of {len(res['values'])} points converged")
        self.message.emit(f"sweep of {res['node']}.{res['name']}: {n_ok} of {len(res['values'])} points converged")
        cur = self.y_box.currentData()
        self.y_box.blockSignals(True)
        self.y_box.clear()
        for k, label, _, u in T.SUMMARY_KEYS:
            if any(np.isfinite(res["summary"][k])):
                self.y_box.addItem(f"{label}" + (f" ({u})" if u else ""), ("summary", k))
        for k in sorted(res["outputs"]):
            u, _ = OUT_UNITS.get(k.split(".")[-1], ("", 1.0))
            self.y_box.addItem(k + (f" ({u})" if u else ""), ("output", k))
        self.y_box.blockSignals(False)
        i = self.y_box.findData(cur) if cur else -1
        self.y_box.setCurrentIndex(max(i, 0))
        self.csv_btn.setEnabled(True)
        self._fill_table()
        self._plot()

    def _y(self) -> tuple[np.ndarray, str]:
        res = self.result
        src, k = self.y_box.currentData()
        if src == "summary":
            _, label, f, u = next(s for s in T.SUMMARY_KEYS if s[0] == k)
            return np.asarray(res["summary"][k], float) * f, label + (f" ({u})" if u else "")
        u, f = OUT_UNITS.get(k.split(".")[-1], ("", 1.0))
        return np.asarray(res["outputs"][k], float) * f, k + (f" ({u})" if u else "")

    def _plot(self, *_) -> None:
        res = self.result
        if res is None or self.y_box.currentData() is None:
            return
        kind, node, name = res["kind"], res["node"], res["name"]
        unit = next((d[3] for d in (self.in_box.itemData(i) for i in range(self.in_box.count()))
                     if d and d[:3] == (kind, node, name)), "")
        u, f = input_unit(kind, name, unit)
        x = np.asarray(res["values"], float) * f
        y, ylab = self._y()
        ok = np.asarray(res["converged"], bool)
        xlab = f"{node}.{name}" + (f" ({u})" if u else "")
        spec = PlotSpec(ylab, xlab, ylab, [Curve(x, np.where(ok, y, np.nan), "accent")],
                        [(float(a), float(b), "accent") for a, b, o in zip(x, y, ok) if o and np.isfinite(b)])
        self.plot.set_spec(spec)

    def _fill_table(self) -> None:
        res = self.result
        keys = [k for k, *_ in T.SUMMARY_KEYS if any(np.isfinite(res["summary"][k]))]
        unit = next((d[3] for d in (self.in_box.itemData(i) for i in range(self.in_box.count()))
                     if d and d[:3] == (res["kind"], res["node"], res["name"])), "")
        xu, xf = input_unit(res["kind"], res["name"], unit)
        heads = [f"{res['node']}.{res['name']}" + (f" ({xu})" if xu else ""), "Converged"] + [
            next(f"{lab}" + (f" ({u})" if u else "") for kk, lab, _, u in T.SUMMARY_KEYS if kk == k) for k in keys]
        t = self.table
        t.setColumnCount(len(heads))
        t.setHorizontalHeaderLabels(heads)
        t.setRowCount(len(res["values"]))
        for i, v in enumerate(res["values"]):
            cells = [f"{v * xf:.5g}", "yes" if res["converged"][i] else "no"]
            for k in keys:
                f = next(s[2] for s in T.SUMMARY_KEYS if s[0] == k)
                val = res["summary"][k][i]
                cells.append("" if not np.isfinite(val) else f"{val * f:,.4g}")
            for j, c in enumerate(cells):
                t.setItem(i, j, QTableWidgetItem(c))
        t.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)

    def _export(self) -> None:
        if self.result is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export sweep", f"sweep_{self.result['node']}_{self.result['name']}.csv",
                                              "CSV (*.csv)")
        if path:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(T.to_csv(self.result))

    # ------------------------------------------------------------------------------------------------ compare
    def _compare_tab(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(6, 6, 6, 6)
        row = QHBoxLayout()
        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("design name")
        self.name_edit.setFixedWidth(200)
        add = QPushButton("Add Current Design")
        add.clicked.connect(self.add_current)
        tmpl = QPushButton("Add Template")
        tm = QMenu(tmpl)
        for gid, name in E.templates().items():
            tm.addAction(name, lambda g=gid, n=name: self.add_design(n, E.template(g)))
        tmpl.setMenu(tm)
        rem = QPushButton("Remove Selected")
        rem.clicked.connect(self._remove)
        self.cmp_btn = QPushButton("Solve and Compare")
        self.cmp_btn.clicked.connect(self.run_compare)
        for x in (self.name_edit, add, tmpl, rem, self.cmp_btn):
            row.addWidget(x)
        row.addStretch(1)
        v.addLayout(row)
        hint = QLabel("Each design is a snapshot of the diagram, propellants and design targets when it was added. "
                      "Change the cycle or the propellants, add it again under a new name, and compare.")
        hint.setWordWrap(True)
        hint.setObjectName("hint")
        v.addWidget(hint)
        self.cmp_table = QTableWidget(len(T.SUMMARY_KEYS), 0)
        self.cmp_table.setVerticalHeaderLabels([lab + (f" ({u})" if u else "") for _, lab, _, u in T.SUMMARY_KEYS])
        self.cmp_table.setAlternatingRowColors(True)
        v.addWidget(self.cmp_table, 1)
        return w

    def add_current(self) -> None:
        doc = self.doc_getter()
        g = doc["graph"]
        name = self.name_edit.text().strip() or f"{g.get('name', 'design')[:28]} ({g['propellants']['oxidizer']}/" \
                                                 f"{g['propellants']['fuel']})"
        self.add_design(name, copy.deepcopy(doc))
        self.name_edit.clear()

    def add_design(self, name: str, doc: dict) -> None:
        base, k = name, 2
        while name in self.designs:
            name = f"{base} {k}"
            k += 1
        self.designs[name] = doc
        self._fill_compare()

    def _remove(self) -> None:
        cols = sorted({i.column() for i in self.cmp_table.selectedItems()}, reverse=True)
        names = list(self.designs)
        for c in cols:
            if c < len(names):
                self.designs.pop(names[c], None)
                self.compared.pop(names[c], None)
        self._fill_compare()

    def run_compare(self) -> None:
        if self._thread is not None or not self.designs:
            return
        docs = copy.deepcopy(self.designs)
        self.prog.setRange(0, 0)
        self._start(lambda job: T.compare(docs), self._compared)

    def _compared(self, res, err: str) -> None:
        self._finish()
        self.prog.setRange(0, 1)
        self.prog.setValue(1)
        if res is None:
            self.message.emit(f"comparison failed: {err}")
            return
        self.compared.update(res)
        self._fill_compare()

    def _fill_compare(self) -> None:
        t = self.cmp_table
        names = list(self.designs)
        t.setColumnCount(len(names))
        t.setHorizontalHeaderLabels(names)
        for j, n in enumerate(names):
            s = self.compared.get(n)
            for i, (k, _, f, _) in enumerate(T.SUMMARY_KEYS):
                if s is None:
                    txt = "press Solve and Compare" if i == 0 else ""
                elif not s.get("converged", 1.0):
                    txt = "did not converge" if i == 0 else ""
                else:
                    val = s.get(k, math.nan)
                    txt = "" if not np.isfinite(val) else f"{val * f:,.4g}"
                t.setItem(i, j, QTableWidgetItem(txt))
        t.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
