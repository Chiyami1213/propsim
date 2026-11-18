"""UI acceptance for the Liquid 'Cycle sandbox' (offscreen): template solve, editing, save/load."""

import pytest
from PySide6.QtWidgets import QApplication

from propsim.core.engine_graph import edit as E
from propsim.ui.cycle_sandbox import CycleSandbox


@pytest.fixture(scope="module")
def box():
    app = QApplication.instance() or QApplication([])
    w = CycleSandbox()
    w.show()
    app.processEvents()
    yield w
    w.close()


def test_opens_with_gas_generator_template(box):
    assert any(n["type"] == "gas_generator" for n in box.doc["graph"]["nodes"])
    assert set(box.canvas.nodes) == {n["id"] for n in box.doc["graph"]["nodes"]}
    assert E.unconnected(box.doc["graph"]) == []


def test_solve_fills_diagram_and_tables(box):
    box.solve()
    box.wait()
    sol = box.sol
    assert sol is not None and sol.converged
    assert box.st_table.rowCount() == len(sol.stations) > 0
    assert box.comp_table.rowCount() == len(sol.components)
    assert box.shaft_table.rowCount() >= 1
    noz = next(c for c in sol.components.values() if c.get("type") == "nozzle_bell")
    assert noz["thrust"] > 0 and noz["isp"] > 200
    assert "converged" in box.msgs.toPlainText()
    box.inspect("chamber")                      # inspector shows the solved state without raising


def test_add_connect_delete(box):
    g = box.doc["graph"]
    n0 = len(g["nodes"])
    a = box.canvas.add_component("valve_ball")
    b = box.canvas.add_component("line")
    assert len(g["nodes"]) == n0 + 2 and a in box.canvas.nodes
    out = next(p["name"] for p in E.library()["valve_ball"]["ports"] if p.get("direction") == "out")
    inn = next(p["name"] for p in E.library()["line"]["ports"] if p.get("direction") == "in")
    e = E.connect(g, (b, inn), (a, out))     # given backwards, stored in flow order
    assert e["from"]["node"] == a and e["to"]["node"] == b
    with pytest.raises(ValueError):
        E.connect(g, (a, out), (b, inn))
    box.canvas.rebuild_edges()
    box.canvas.scene().clearSelection()
    for nid in (a, b):
        box.canvas.nodes[nid].setSelected(True)
    box.canvas.delete_selected()
    assert len(g["nodes"]) == n0 and not any(a in (x["from"]["node"], x["to"]["node"]) for x in g["edges"])


def test_save_load_round_trip(box, tmp_path):
    box.load_template("ssme_staged_combustion")
    assert [t["output"] for t in box.doc["targets"]] == ["p_c", "mixture_ratio"]
    p = tmp_path / "ssme.sandbox.json"
    E.save(box.doc, p)
    d = E.load(p)
    assert d["graph"]["nodes"] == box.doc["graph"]["nodes"] and d["free"] == box.doc["free"]
    with pytest.raises(ValueError):
        (tmp_path / "bad.json").write_text('{"format": "other"}')
        E.load(tmp_path / "bad.json")


def test_every_component_has_an_isa_symbol(qapp):
    from PySide6.QtCore import QRectF
    from PySide6.QtGui import QColor, QImage, QPainter
    from propsim.core.engine_graph import edit as E
    from propsim.ui.pid import draw_symbol
    img = QImage(80, 60, QImage.Format_RGB32)
    for ntype in E.library():
        img.fill(0)
        p = QPainter(img)
        draw_symbol(p, ntype, QRectF(10, 8, 60, 44), QColor("white"), QColor("black"), phase=0.3, command=True)
        p.end()
        assert any(img.pixelColor(x, y).red() > 100 for x in range(80) for y in range(60)), ntype


def test_panels_are_dockable_and_animation_exports(qapp, tmp_path):
    from PySide6.QtWidgets import QDockWidget
    from propsim.ui.cycle_sandbox import CycleSandbox
    w = CycleSandbox()
    for d in w.docks:
        assert d.features() & QDockWidget.DockWidgetMovable and d.features() & QDockWidget.DockWidgetFloatable
    gif = w.anim.export_gif(str(tmp_path / "c.gif"), frames=4)
    png = w.anim.save_image(str(tmp_path / "c.png"))
    assert gif.stat().st_size > 1000 and png.stat().st_size > 1000


def test_cycles_are_identified_and_flow_plays(qapp):
    from propsim.core.engine_graph import edit as E
    from propsim.ui.cycle_sandbox import CycleSandbox
    from propsim.ui.cycle_schematic import identify_cycle
    w = CycleSandbox()
    want = {"gg_kerolox": "gas_generator", "expander_hydrolox": "expander", "pressure_fed_kerolox": "pressure_fed",
            "ssme_staged_combustion": "staged_fuel"}
    for gid, kind in want.items():
        w.load_template(gid)
        assert identify_cycle(w.doc["graph"]) == kind
    w.play_btn.setChecked(True)
    w.canvas._flow_tick()
    assert w.canvas.playing and w.canvas.flow_phase > 0
    w.play_btn.setChecked(False)
    assert not w.canvas.playing
    g = E.new_document()["graph"]
    assert identify_cycle(g) == "custom"


def test_every_component_has_a_3d_model_sized_from_its_parameters():
    from propsim.modules.component_shapes import build, merged
    for gid in ("gg_kerolox", "ssme_staged_combustion", "expander_hydrolox", "pressure_fed_kerolox"):
        g = E.template(gid)["graph"]
        for n in g["nodes"]:
            for cut in (0, 90):
                m = build(g, n["id"], None, cut)
                assert m.parts and all(p.mesh.n_points > 0 for p in merged(m)), (gid, n["id"])
    g = E.template("gg_kerolox")["graph"]
    line = next(n for n in g["nodes"] if n["type"] == "line")
    b = build(g, line["id"], None, 0).parts[0].mesh.bounds
    d = E.param_value(line, "diameter")[0]
    assert abs((b[3] - b[2]) - d * 1.08) / d < 0.01             # bore plus two walls of 8 % of the radius
    lib = {k for k in E.library()}
    from propsim.modules.component_shapes import BUILDERS
    assert lib <= set(BUILDERS)


def test_3d_panel_follows_selection_and_solution(box):
    box.load_template("gg_kerolox")
    box.solve()
    box.wait()
    v = box.view3d
    box.dock_3d.show()
    box.canvas.scene().clearSelection()
    box.canvas.nodes["regen_jacket"].setSelected(True)
    QApplication.processEvents()
    assert v.nid == "regen_jacket" and v.model is not None
    assert {"T_wg", "q", "T_cool"} <= set(v.model.fields)       # wall and coolant temperatures from the jacket march
    assert any("Heat load" in lbl.latex() for lbl in v.vp.readout_lbls)
    box.canvas.scene().clearSelection()
    box.canvas.nodes["mov"].setSelected(True)
    QApplication.processEvents()
    assert v.model.title == "Butterfly valve" and set(v.model.fields) == {"model", "p", "T_f"}   # no wall heat transfer on a valve


def test_build_guide_lists_required_and_optional_components(qapp):
    from propsim.ui.cycle_guide import CYCLES, checklist
    from propsim.ui.cycle_sandbox import CycleSandbox
    w = CycleSandbox()
    w.new_doc()
    g = w.guide
    g._pick("expander")
    req, opt = checklist(w.doc["graph"], "expander")
    assert any(t == "regen_jacket" and need == 1 for t, _, need, _ in req) and opt
    assert all(have == 0 for _, have, _, _ in req)
    g.go(1)
    assert not g.next.isEnabled()                              # required parts are missing
    for t, _, need, _ in req:
        for _ in range(need):
            w.canvas.add_component(t)
    g.refresh()
    assert g.next.isEnabled() and not g.missing
    g.go(2)
    assert "Ports still open" in g.page_conn.text()
    assert {"pressure_fed", "gas_generator", "expander", "tap_off", "expander_bleed", "staged_fuel", "staged_ox",
            "full_flow", "electric_pump", "cold_gas"} <= set(CYCLES)


def test_trade_study_sweeps_and_compares(box):
    box.load_template("gg_kerolox")
    tp = box.trade
    i = next(i for i in range(tp.in_box.count()) if tp.in_box.itemData(i)[1:3] == ("nozzle", "expansion_ratio"))
    tp.in_box.setCurrentIndex(i)
    tp.n_box.setValue(3)
    tp.run_sweep()
    tp.wait()
    assert tp.result is not None and all(tp.result["converged"]) and tp.table.rowCount() == 3
    assert tp.y_box.count() > 10 and tp.plot.spec.curves
    tp.add_current()
    tp.add_design("Pressure-fed", E.template("pressure_fed_kerolox"))
    tp.run_compare()
    tp.wait()
    assert tp.cmp_table.columnCount() == 2
    assert all(tp.compared[n]["converged"] for n in tp.designs)
    assert tp.cmp_table.item(0, 0).text().replace(",", "").replace(".", "").isdigit()


def test_component_3d_flow_animation(qapp):
    from propsim.ui.cycle_sandbox import CycleSandbox
    w = CycleSandbox()
    w.show()
    nid = next(n["id"] for n in w.doc["graph"]["nodes"] if n["type"] == "valve_poppet")
    w.view3d.show_component(w.doc["graph"], nid, None)
    w.view3d.show()
    w.view3d.refresh()
    w.view3d.flow_btn.setChecked(True)
    w.view3d._tick()
    assert w.view3d.flowing and "flow" in w.view3d.vp.plotter.actors
    w.view3d.flow_btn.setChecked(False)
    assert "flow" not in w.view3d.vp.plotter.actors


def test_component_3d_has_pressure_and_temperature_fields(qapp):
    from propsim.core.engine_graph import solve
    from propsim.ui.cycle_sandbox import CycleSandbox
    w = CycleSandbox()
    w.show()
    sol = solve(w.doc["graph"], None, None)
    for ntype in ("valve_poppet", "pump_centrifugal", "chamber"):
        nid = next(n["id"] for n in w.doc["graph"]["nodes"] if n["type"] == ntype)
        w.view3d.show()
        w.view3d.show_component(w.doc["graph"], nid, sol)
        w.view3d.refresh()
        assert {"p", "T_f"} <= set(w.view3d._field_keys), ntype
    assert {"q"} <= set(w.view3d._field_keys)                 # the chamber also has its heat flux


def test_every_module_has_a_lowercase_hover_description(qapp, tmp_path):
    from PySide6.QtCore import Qt
    from propsim.modules import descriptions, registry
    from propsim.ui.main_window import MainWindow
    keys = {c.key for entries in registry.TREE.values() for _, _, c in entries if c is not None}
    assert keys <= set(descriptions.MODELS)
    assert set(registry.TREE) <= set(descriptions.MODULES)
    assert all(t == t.lower() for t in list(descriptions.MODELS.values()) + list(descriptions.MODULES.values()))
    w = MainWindow(home=tmp_path, use_settings=False)
    tips = [w.tree_model.item(r).child(c).toolTip() for r in range(w.tree_model.rowCount())
            for c in range(w.tree_model.item(r).rowCount())]
    assert tips and all(tips)


def test_undo_and_redo_of_the_diagram(qapp):
    from propsim.ui.cycle_sandbox import CycleSandbox
    w = CycleSandbox()
    w.new_doc()
    n0 = len(w.doc["graph"]["nodes"])
    w.canvas.add_component("tank_regulated")
    w.canvas.add_component("valve_ball")
    assert len(w.doc["graph"]["nodes"]) == n0 + 2
    w.undo()
    assert len(w.doc["graph"]["nodes"]) == n0 + 1
    w.undo()
    assert len(w.doc["graph"]["nodes"]) == n0
    w.redo()
    assert len(w.doc["graph"]["nodes"]) == n0 + 1


def test_turbine_spins_and_particles_flow_in_the_3d_model(qapp):
    import numpy as np
    from propsim.core.engine_graph import solve
    from propsim.ui.cycle_sandbox import CycleSandbox
    w = CycleSandbox()
    w.show()
    sol = solve(w.doc["graph"], None, None)
    nid = next(n["id"] for n in w.doc["graph"]["nodes"] if n["type"] == "turbine_impulse")
    w.view3d.show()
    w.view3d.show_component(w.doc["graph"], nid, sol)
    w.view3d.refresh()
    assert w.view3d._field_keys[0] == "model" and {"p", "T_f"} <= set(w.view3d._field_keys)
    assert w.view3d._rotors
    before = np.array(w.view3d._rotors[0][0].points)
    w.view3d.flow_btn.setChecked(True)
    w.view3d._tick()
    w.view3d._tick()
    after = np.array(w.view3d._rotors[0][0].points)
    assert not np.allclose(before, after)                     # the rotor turned
    assert "flow" in w.view3d.vp.plotter.actors


def test_ambient_presets_and_helper_box(qapp):
    from propsim.ui.cycle_sandbox import CycleSandbox
    w = CycleSandbox()
    w.amb_preset.setCurrentText("Vacuum")
    assert w.pa_box.value() == 0.0
    w.amb_preset.setCurrentText("Sea level")
    assert abs(w.pa_box.value() - 101.325) < 1e-3
    w.pa_box.setValue(50.0)
    w._sync_ambient_preset(50.0)
    assert w.amb_preset.currentText() == "Custom"
    nid = next(n["id"] for n in w.doc["graph"]["nodes"] if n["type"] == "tank_regulated")
    w.inspect(nid)
    assert hasattr(w, "hint_box")
    from PySide6.QtCore import Qt
    assert w.ox_box.itemData(0, Qt.ToolTipRole)


def test_designer_title_busy_bar_and_docks_stay_docked(qapp):
    import time
    from propsim.ui.cycle_sandbox import CycleSandbox
    w = CycleSandbox()
    w.show()
    assert w.windowTitle() == "Liquid Engine Designer"
    w.solve()
    assert not w.busy.isHidden()                               # busy bar while solving
    w.wait()
    assert w.busy.isHidden()
    w._state_t = time.monotonic()                              # just maximized
    d = w.dock_insp
    d.setFloating(True)
    for _ in range(20):
        qapp.processEvents()
    assert not d.isFloating()                                  # put back after the window state change


def test_legend_keeps_propellant_lines_when_colored_by_pressure(qapp):
    from propsim.ui.cycle_sandbox import CycleSandbox
    w = CycleSandbox()
    w.solve()
    w.wait()
    w.show()
    w.color_box.setCurrentText("Pressure (MPa)")
    w.canvas._place_legend()
    names = [r[1] for r in w.canvas.legend_box.rows()]
    assert names[0].startswith("Process line, colored by pressure")
    assert "Oxidizer line" in names and "Fuel line" in names


def test_busy_bar_for_export_and_import(qapp, tmp_path):
    from propsim.ui.cycle_sandbox import CycleSandbox
    from propsim.ui.main_window import MainWindow
    w = MainWindow(home=tmp_path, use_settings=False)
    w.show()
    seen = []
    from propsim.ui import busy as B
    real = B.busy
    from contextlib import contextmanager

    @contextmanager
    def spy(win, text="Working"):
        seen.append(text)
        with real(win, text) as bar:
            yield bar
    B.busy = spy
    import propsim.ui.main_window as MW
    MW.busy = spy
    try:
        w.select_model("hybrid.regression")
        w.export("CSV", str(tmp_path / "x.csv"))               # a given path: the bar starts at once
        w.save_case(str(tmp_path / "c.psim.json"))
        w.open_case(str(tmp_path / "c.psim.json"))
    finally:
        B.busy = real
        MW.busy = real
    assert (tmp_path / "x.csv").exists() and {"Exporting", "Saving case", "Opening case"} <= set(seen)
    assert w.statusBar().isHidden() or not w.statusBar().currentMessage()
    sb = CycleSandbox()
    gif = sb.anim.export_gif(str(tmp_path / "a.gif"), frames=3)
    assert gif.exists()


def test_3d_legend_toggle_comes_back_even_in_a_small_view(qapp):
    from propsim.ui.viewport3d import Viewport3D
    v = Viewport3D()
    v.resize(500, 260)
    v.show()
    v.legend.set_rows([("#ffffff", "Part")])
    v._place()
    v._toggle_legend()
    assert not v.legend.isVisible()
    v._toggle_legend()                                         # off then on again: it must be back
    assert v.legend.isVisible()
