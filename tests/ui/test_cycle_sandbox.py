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
