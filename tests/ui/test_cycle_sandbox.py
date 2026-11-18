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
            "ssme_staged_combustion": "staged"}
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
    assert set(v.model.fields) == {"T_wg", "q", "T_cool"}       # wall and coolant temperatures from the jacket march
    assert any("Heat load" in lbl.latex() for lbl in v.vp.readout_lbls)
    box.canvas.scene().clearSelection()
    box.canvas.nodes["mov"].setSelected(True)
    QApplication.processEvents()
    assert v.model.title == "Butterfly valve" and not v.model.fields


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
    assert set(CYCLES) == {"pressure_fed", "gas_generator", "expander", "staged"}


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


def test_startup_panel_runs_and_scrubs(box):
    box.load_template("pressure_fed_kerolox")
    sp = box.startup
    assert sp.valve_table.rowCount() == 2 and sp.load_table.rowCount() == 2     # two valves, two tanks
    sp.t_end.setValue(1.0)
    sp.run()
    sp.wait()
    r = sp.result
    assert r is not None and r.message.startswith("ran to") and len(r.t) > 5
    assert sp.slider.maximum() == len(r.t) - 1 and set(sp.plots) >= {"Pressure", "Speed", "Tanks and Valves"}
    sp._scrub(0)
    p0 = box.sol.components["chamber"]["p_c"]
    sp._scrub(len(r.t) - 1)
    assert box.sol.components["chamber"]["p_c"] > 3 * p0                        # the diagram follows the scrubber
    sp.slider.setValue(2)
    assert "t = " in sp.time_lbl.text()
