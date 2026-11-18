"""Cycle sandbox *Start-up* panel: edit the start sequence, throttle and propellant load, run the transient, plot every
pressure, temperature, speed and flow against time, and scrub or play the diagram through it."""
from __future__ import annotations

import copy
import math

import numpy as np
from PySide6.QtCore import QObject, Qt, QThread, QTimer, Signal
from PySide6.QtGui import QColor, QIcon, QPixmap
from PySide6.QtWidgets import (QCheckBox, QDoubleSpinBox, QFileDialog, QHBoxLayout, QHeaderView, QLabel, QListWidget,
                               QListWidgetItem, QProgressBar, QPushButton, QSlider, QSplitter, QTableWidget,
                               QTableWidgetItem, QTabWidget, QVBoxLayout, QWidget)

from ..core.engine_graph import edit as E
from ..core.engine_graph import transient as TR
from ..core.result import Curve, PlotSpec
from .plots import GridPlot

PALETTE = ["#4ea1ff", "#ff8a3d", "#6fd08c", "#e5484d", "#c08bff", "#f2c94c", "#2dd4bf", "#f472b6", "#a3a3a3",
           "#60a5fa", "#fb923c", "#34d399"]


class _Job(QObject):
    progress = Signal(float, float)
    done = Signal(object, str)

    def __init__(self, doc, seq) -> None:
        super().__init__()
        self.doc, self.seq, self.stop = doc, seq, False

    def run(self) -> None:
        try:
            r = TR.run(self.doc, self.seq, progress=lambda t, T: self.progress.emit(t, T),
                       cancelled=lambda: self.stop)
            self.done.emit(r, "")
        except Exception as e:                                   # noqa: BLE001 - shown in the panel
            self.done.emit(None, f"{type(e).__name__}: {e}")


def _swatch(color: str) -> QIcon:
    pm = QPixmap(14, 14)
    pm.fill(QColor(color))
    return QIcon(pm)


class StartupPanel(QWidget):
    """``doc_getter`` returns the sandbox document; ``show_point(solution)`` paints one instant on the diagram."""

    message = Signal(str)

    def __init__(self, doc_getter, show_point, parent=None) -> None:
        super().__init__(parent)
        self.doc_getter, self.show_point = doc_getter, show_point
        self.result: TR.Transient | None = None
        self._thread: QThread | None = None
        self._job: _Job | None = None
        self.series: dict[str, dict] = {}                         # tab -> {key: (label, values, unit scale)}
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        split = QSplitter(Qt.Horizontal)
        split.addWidget(self._setup())
        split.addWidget(self._results())
        split.setSizes([520, 700])
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(split)

    # ------------------------------------------------------------------------------------------------ setup
    def _setup(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(6, 6, 6, 6)
        row = QHBoxLayout()
        self.t_end = QDoubleSpinBox()
        self.t_end.setRange(0.2, 60.0)
        self.t_end.setValue(3.0)
        self.t_end.setSuffix(" s")
        self.run_btn = QPushButton("Run Start-up")
        self.run_btn.clicked.connect(self.run)
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._stop)
        for x in (QLabel("Run for"), self.t_end, self.run_btn, self.stop_btn):
            row.addWidget(x)
        row.addStretch(1)
        v.addLayout(row)
        self.status = QLabel("")
        self.status.setObjectName("hint")
        self.status.setWordWrap(True)
        v.addWidget(self.status)
        tabs = QTabWidget()
        self.valve_table = QTableWidget(0, 6)
        self.valve_table.setHorizontalHeaderLabels(["Valve", "Opens at (s)", "Ramp (s)", "Closes at (s)",
                                                    "Ramp (s)", "Throttles"])
        self.throttle_table = QTableWidget(0, 2)
        self.throttle_table.setHorizontalHeaderLabels(["Time (s)", "Throttle (%)"])
        self.load_table = QTableWidget(0, 4)
        self.load_table.setHorizontalHeaderLabels(["Tank", "Propellant", "Loaded (kg)", "Full (kg)"])
        self.start_table = QTableWidget(0, 3)
        self.start_table.setHorizontalHeaderLabels(["Shaft", "Starter power (kW)", "Duration (s)"])
        for name, t in (("Sequence", self.valve_table), ("Throttle", self._with_buttons(self.throttle_table)),
                        ("Load", self.load_table), ("Starter", self.start_table)):
            tabs.addTab(t, name)
        for t in (self.valve_table, self.throttle_table, self.load_table, self.start_table):
            t.setAlternatingRowColors(True)
            t.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
            t.verticalHeader().setVisible(False)
        v.addWidget(tabs, 1)
        hint = QLabel("A valve the design targets size (an engine-controller valve) holds chamber pressure and mixture "
                      "ratio from mainstage; the throttle scales that target. Other valves open on this schedule.")
        hint.setWordWrap(True)
        hint.setObjectName("hint")
        v.addWidget(hint)
        return w

    def _with_buttons(self, table: QTableWidget) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(0, 0, 0, 0)
        v.addWidget(table)
        row = QHBoxLayout()
        add, rem = QPushButton("Add Point"), QPushButton("Remove Point")
        add.clicked.connect(lambda: self._add_throttle(None))
        rem.clicked.connect(lambda: table.removeRow(table.currentRow()) if table.currentRow() >= 0 else None)
        row.addWidget(add)
        row.addWidget(rem)
        row.addStretch(1)
        v.addLayout(row)
        return w

    def _add_throttle(self, pt) -> None:
        t = self.throttle_table
        r = t.rowCount()
        t.insertRow(r)
        a, b = pt if pt else ((float(t.item(r - 1, 0).text()) + 1.0) if r else 0.0, 100.0)
        t.setItem(r, 0, QTableWidgetItem(f"{a:g}"))
        t.setItem(r, 1, QTableWidgetItem(f"{b:g}"))

    def refresh(self) -> None:
        """Fill the tables from the sandbox document (after a template is loaded or the diagram is edited)."""
        doc = self.doc_getter()
        try:
            seq = TR.default_sequence(doc, self.t_end.value())
        except Exception as e:                                   # noqa: BLE001 - a half-built diagram
            self.status.setText(f"no start-up yet: {e}")
            return
        g = doc["graph"]
        t = self.valve_table
        t.setRowCount(len(seq.valves))
        for i, (v, st) in enumerate(seq.valves.items()):
            cells = [v, f"{st.t_open:g}", f"{st.ramp_open:g}", "" if st.t_close is None else f"{st.t_close:g}",
                     f"{st.ramp_close:g}"]
            for j, c in enumerate(cells):
                it = QTableWidgetItem(c)
                if j == 0:
                    it.setFlags(it.flags() & ~Qt.ItemIsEditable)
                t.setItem(i, j, it)
            cb = QCheckBox()
            cb.setChecked(st.throttle)
            t.setCellWidget(i, 5, cb)
        self.throttle_table.setRowCount(0)
        self._add_throttle((0.0, 100.0))
        # tanks
        from ..core.engine_graph.props import props
        tanks = [n for n in g["nodes"] if n["type"].startswith("tank")]
        self.load_table.setRowCount(len(tanks))
        self._tanks = {}
        for i, n in enumerate(tanks):
            fid = n.get("options", {}).get("fluid") or g["propellants"]["oxidizer"]
            V = E.param_value(n, "V_tank")[0] or 1.0
            ff = E.param_value(n, "fill_fraction")[0] or 0.9
            T = (E.param_value(n, "T_prop")[0] or 293.15)
            p = (E.param_value(n, "p_set")[0] or E.param_value(n, "p0")[0] or 3e5)
            try:
                pr = props(fid)
                rho = pr.at(p, pr.h_pT(p, T), T)["rho"]
            except Exception:                                    # noqa: BLE001
                rho = 1000.0
            full = float(V) * rho
            self._tanks[n["id"]] = (float(V), rho)
            for j, c in enumerate([n["id"], fid, f"{ff * full:.4g}", f"{full:.4g}"]):
                it = QTableWidgetItem(c)
                if j != 2:
                    it.setFlags(it.flags() & ~Qt.ItemIsEditable)
                self.load_table.setItem(i, j, it)
        # shafts
        net_groups = self._shaft_groups(g)
        self.start_table.setRowCount(len(net_groups))
        for i, grp in enumerate(net_groups):
            for j, c in enumerate([" + ".join(grp), "0", "0"]):
                it = QTableWidgetItem(c)
                if j == 0:
                    it.setFlags(it.flags() & ~Qt.ItemIsEditable)
                self.start_table.setItem(i, j, it)
        self.status.setText("Edit the sequence, throttle and propellant load, then Run Start-up.")

    @staticmethod
    def _shaft_groups(g: dict) -> list[list[str]]:
        from ..core.engine_graph.network import Network
        try:
            return [list(x) for x in Network(copy.deepcopy(g)).shaft_groups]
        except Exception:                                        # noqa: BLE001
            return []

    def _num(self, table: QTableWidget, r: int, c: int, default=None):
        it = table.item(r, c)
        try:
            return float(it.text()) if it and it.text().strip() else default
        except ValueError:
            return default

    def sequence(self) -> TR.Sequence:
        doc = self.doc_getter()
        seq = TR.default_sequence(doc, self.t_end.value())
        t = self.valve_table
        for r in range(t.rowCount()):
            st = seq.valves[t.item(r, 0).text()]
            st.t_open = self._num(t, r, 1, st.t_open)
            st.ramp_open = max(self._num(t, r, 2, st.ramp_open), 1e-3)
            st.t_close = self._num(t, r, 3, None)
            st.ramp_close = max(self._num(t, r, 4, st.ramp_close), 1e-3)
            st.throttle = bool(t.cellWidget(r, 5).isChecked())
        pts = []
        for r in range(self.throttle_table.rowCount()):
            a, b = self._num(self.throttle_table, r, 0), self._num(self.throttle_table, r, 1)
            if a is not None and b is not None:
                pts.append((a, max(b, 5.0) / 100.0))
        seq.throttle = sorted(pts) or [(0.0, 1.0)]
        seq.loads = {}
        for r in range(self.load_table.rowCount()):
            nid = self.load_table.item(r, 0).text()
            m = self._num(self.load_table, r, 2)
            if m is not None and nid in self._tanks:
                seq.loads[nid] = m
        for r in range(self.start_table.rowCount()):
            P, d = self._num(self.start_table, r, 1, 0.0), self._num(self.start_table, r, 2, 0.0)
            if P and d:
                seq.starter[r] = (P * 1e3, d)
        return seq

    # ------------------------------------------------------------------------------------------------ run
    def run(self) -> None:
        if self._thread is not None:
            return
        self.timer.stop()
        doc = copy.deepcopy(self.doc_getter())
        seq = self.sequence()
        self.prog.setRange(0, 1000)
        self.prog.setValue(0)
        self.status.setText("preparing the pre-start state …")
        self.run_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self._thread = QThread(self)
        self._job = _Job(doc, seq)
        self._job.moveToThread(self._thread)
        self._thread.started.connect(self._job.run)
        self._job.progress.connect(self._on_progress)       # a bound slot: runs in the GUI thread
        self._job.done.connect(self._done)
        self._thread.start()

    def _on_progress(self, t: float, T: float) -> None:
        self.prog.setValue(int(1000 * t / max(T, 1e-9)))
        self.status.setText(f"running: t = {t:.2f} s")

    def _stop(self) -> None:
        if self._job is not None:
            self._job.stop = True

    def wait(self, ms: int = 1200000) -> None:
        """Block until a running start-up finishes (tests)."""
        import time
        from PySide6.QtWidgets import QApplication
        t0 = time.time()
        while self._thread is not None and (time.time() - t0) * 1e3 < ms:
            QApplication.processEvents()
            time.sleep(0.01)

    def _done(self, res, err: str) -> None:
        self._thread.quit()
        self._thread.wait()
        self._thread, self._job = None, None
        self.run_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        if res is None:
            self.status.setText(f"start-up failed: {err}")
            self.message.emit(f"start-up failed: {err}")
            return
        self.result = res
        self.prog.setValue(1000)
        self.status.setText(f"{res.message} ({len(res.t)} time points)")
        self.message.emit(f"start-up: {res.message}")
        self._build_series()
        self.slider.setRange(0, max(len(res.t) - 1, 0))
        self.slider.setValue(len(res.t) - 1)
        self._plot_all()
        self._scrub(len(res.t) - 1)

    # ------------------------------------------------------------------------------------------------ results
    def _results(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(6, 6, 6, 6)
        row = QHBoxLayout()
        self.prog = QProgressBar()
        self.prog.setMaximumWidth(200)
        self.play_btn = QPushButton("Play")
        self.play_btn.setCheckable(True)
        self.play_btn.toggled.connect(self._play)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.valueChanged.connect(self._scrub)
        self.time_lbl = QLabel("t = –")
        self.time_lbl.setMinimumWidth(120)
        self.csv_btn = QPushButton("Export CSV")
        self.csv_btn.clicked.connect(self._export)
        for x in (self.prog, self.play_btn, self.slider, self.time_lbl, self.csv_btn):
            row.addWidget(x, 1 if x is self.slider else 0)
        v.addLayout(row)
        self.tabs = QTabWidget()
        self.plots: dict[str, GridPlot] = {}
        self.lists: dict[str, QListWidget] = {}
        for name in ("Pressure", "Temperature", "Speed", "Flow", "Tanks and Valves"):
            sp = QSplitter(Qt.Horizontal)
            plot = GridPlot(PlotSpec(name, "t (s)", ""))
            lst = QListWidget()
            lst.setMaximumWidth(260)
            lst.itemChanged.connect(lambda *_, n=name: self._plot(n))
            sp.addWidget(plot)
            sp.addWidget(lst)
            sp.setSizes([560, 240])
            self.plots[name], self.lists[name] = plot, lst
            self.tabs.addTab(sp, name)
        v.addWidget(self.tabs, 1)
        return w

    def _build_series(self) -> None:
        r = self.result
        sols = r.solutions
        t = np.array(r.t)
        fl = lambda f: np.array([f(s) for s in sols], float)
        S = {k: [] for k in self.lists}                          # tab -> [(label, values, unit, default on)]
        st0 = sols[0].stations
        for i, s in enumerate(st0):
            lab = f"{s['ports'][0]}" + (f" (+{len(s['ports']) - 1})" if len(s["ports"]) > 1 else "")
            pv = fl(lambda q, i=i: q.stations[i]["p"] * 1e-6)
            Tv = fl(lambda q, i=i: q.stations[i].get("T", np.nan))
            S["Pressure"].append((f"station {i}: {lab}", pv, "MPa", False))
            S["Temperature"].append((f"station {i}: {lab}", Tv, "K", False))
            S["Flow"].append((f"station {i}: {lab}", fl(lambda q, i=i: q.stations[i]["mdot"]), "kg/s", False))
        for nid, c in sols[0].components.items():
            ty = c.get("type")
            if ty == "chamber":
                S["Pressure"].insert(0, (f"{nid}: chamber pressure", r.series(nid, "p_c") * 1e-6, "MPa", True))
                S["Temperature"].insert(0, (f"{nid}: gas temperature", r.series(nid, "T_c"), "K", True))
            if ty == "nozzle_bell":
                S["Flow"].insert(0, (f"{nid}: engine flow", r.series(nid, "mdot"), "kg/s", True))
                S["Speed"].append((f"{nid}: thrust (kN)", r.series(nid, "thrust") * 1e-3, "kN", True))
            if ty == "pump_centrifugal":
                S["Pressure"].insert(1, (f"{nid}: pressure rise", r.series(nid, "dp") * 1e-6, "MPa", False))
            if ty in ("turbine_impulse", "gas_generator"):
                S["Temperature"].append((f"{nid}: inlet / outlet T", r.series(nid, "T_inlet" if ty != "gas_generator"
                                                                              else "T_out"), "K", ty == "gas_generator"))
            if ty == "regen_jacket":
                S["Temperature"].append((f"{nid}: coolant out", r.series(nid, "T_out"), "K", False))
                S["Temperature"].append((f"{nid}: peak gas-side wall", r.series(nid, "T_wg_max"), "K", False))
        for k in range(len(r.speeds[0])):
            S["Speed"].insert(0, (f"shaft {k}: speed", np.array([sp[k] for sp in r.speeds]), "rpm", True))
        for nid, m in r.tank_mass.items():
            S["Tanks and Valves"].append((f"{nid}: propellant left", np.array(m), "kg", True))
        for v_, pos in r.positions.items():
            S["Tanks and Valves"].append((f"{v_}: position", np.array(pos), "", False))
        S["Tanks and Valves"].append(("throttle command", np.array(r.throttle), "", False))
        self.series = {k: [(a, b, c, d) for a, b, c, d in v] for k, v in S.items()}
        self._t = t
        for name, lst in self.lists.items():
            lst.blockSignals(True)
            lst.clear()
            for i, (lab, vals, unit, on) in enumerate(self.series[name]):
                it = QListWidgetItem(_swatch(PALETTE[i % len(PALETTE)]), lab + (f" ({unit})" if unit else ""))
                it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
                it.setCheckState(Qt.Checked if on and np.isfinite(vals).any() else Qt.Unchecked)
                lst.addItem(it)
            lst.blockSignals(False)

    def _plot_all(self) -> None:
        for n in self.plots:
            self._plot(n)

    def _plot(self, name: str) -> None:
        if self.result is None:
            return
        lst = self.lists[name]
        curves = []
        units = set()
        for i, (lab, vals, unit, _) in enumerate(self.series[name]):
            if i < lst.count() and lst.item(i).checkState() == Qt.Checked:
                curves.append(Curve(self._t, vals, PALETTE[i % len(PALETTE)], lab))
                units.add(unit)
        ylab = ", ".join(sorted(u for u in units if u)) or ""
        self.plots[name].set_spec(PlotSpec(name, "t (s)", ylab, curves, [], cursor=True))
        self.plots[name].set_cursor(float(self._t[self.slider.value()]) if len(self._t) else 0.0)

    def _scrub(self, i: int) -> None:
        r = self.result
        if r is None or not r.solutions:
            return
        i = int(np.clip(i, 0, len(r.solutions) - 1))
        self.time_lbl.setText(f"t = {r.t[i]:.3f} s")
        for p in self.plots.values():
            p.set_cursor(float(r.t[i]))
        self.show_point(r.solutions[i])

    def _play(self, on: bool) -> None:
        self.play_btn.setText("Pause" if on else "Play")
        if on and self.result is not None:
            if self.slider.value() >= self.slider.maximum():
                self.slider.setValue(0)
            self.timer.start(40)
        else:
            self.timer.stop()

    def _tick(self) -> None:
        r = self.result
        if r is None:
            self.timer.stop()
            return
        # advance by wall-clock: the run plays at 1/3 real speed so the ramp is easy to follow
        t_now = r.t[self.slider.value()] + 0.04 / 3.0
        j = int(np.searchsorted(r.t, t_now))
        if j >= len(r.t) - 1:
            self.slider.setValue(len(r.t) - 1)
            self.play_btn.setChecked(False)
            return
        self.slider.setValue(max(j, self.slider.value() + 1))

    def _export(self) -> None:
        r = self.result
        if r is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export start-up", "startup.csv", "CSV (*.csv)")
        if not path:
            return
        cols = ["t (s)"]
        data = [np.array(r.t)]
        for tab in self.series.values():
            for lab, vals, unit, _ in tab:
                cols.append(lab + (f" ({unit})" if unit else ""))
                data.append(vals)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(",".join(c.replace(",", ";") for c in cols) + "\n")
            for i in range(len(r.t)):
                fh.write(",".join(repr(float(d[i])) for d in data) + "\n")
