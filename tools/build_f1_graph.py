"""Build the F-1 (Saturn V first stage) example graph from the Huzel & Huang A-1 graph, scaled to the F-1.

The library holds only the top-level F-1 figures (thrust, chamber pressure, mixture ratio, expansion ratio; NASA
SP-2009-4545), not its turbopump, valve or injector data.  This graph is therefore an ESTIMATE: the A-1 gas-generator
kerolox topology (which has the F-1's cycle) with every area scaled by the thrust ratio, diameters by its square root, volumes by
its 3/2 power, volumetric flows by the ratio and pump speeds by its inverse square root, each scaled value labelled ``estimated`` with its A-1 source in ``formula``.  The F-1's own
published figures are the validation targets.  Rerun:  python tools/build_f1_graph.py
"""
from __future__ import annotations

import copy
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LIB = ROOT / "propsim" / "data" / "library"
SRC = LIB / "examples" / "gg_kerolox.graph.json"
OUT = LIB / "examples" / "f1_saturn_v.graph.json"
BY_UNIT = {"m2": 1.0, "m3/s": 1.0, "m3": 1.5, "kg m2": 2.5}               # unit -> power of the thrust ratio
BY_NAME = {"diameter": 0.5, "D_tip": 0.5, "D_mean": 0.5, "N_design": -0.5}  # geometric similarity at equal head


def f1_values() -> dict[str, dict]:
    rows = csv.DictReader((LIB / "engines" / "engine_params.csv").open(encoding="utf-8"))
    return {f"{r['subsystem']}.{r['property']}": r for r in rows if r["engine_id"] == "F1"}


def main() -> None:
    g = json.loads(SRC.read_text(encoding="utf-8"))
    f1 = f1_values()
    thrust_a1 = next(t for t in g["validation_targets"] if t["output"] == "thrust")["value"]["value"]
    k = float(f1["engine.thrust"]["value"]) / thrust_a1
    out = copy.deepcopy(g)
    out["graph_id"] = "f1_saturn_v"
    out["name"] = "F-1 gas-generator kerolox engine (Saturn V first stage, scaled from A-1, estimated)"
    out["reference_engine"] = "F1"
    out["description"] = (
        "LOX/RP-1 gas-generator cycle on one turbopump shaft, fuel-cooled tubular chamber: the F-1 cycle. ESTIMATE: the "
        f"Huzel & Huang A-1 graph with areas scaled by {k:.3f} (F-1 thrust / A-1 thrust), volumes by {k ** 1.5:.3f}; the F-1 "
        "published figures (thrust, chamber pressure, mixture ratio) are only validation targets.")
    for n in out["nodes"]:
        for name, prm in n["params"].items():
            power = BY_NAME.get(name, BY_UNIT.get(prm["unit"]))
            if power is None:
                continue
            f = k ** power
            prm["value"] = prm["value"] * f
            prm["formula"] = f"A-1 value x {f:.4g} (F-1/A-1 thrust ratio {k:.4f}^{power}); was {prm.get('formula', '')}"
            prm["quality"] = "estimated"
            prm["source_id"] = "PROPSIM-EST"
            prm["location"] = "examples/f1_saturn_v.graph.json (scaled from gg_kerolox)"
    src = f1["engine.thrust"]["source_id"], f1["engine.thrust"]["location"]
    mk = lambda key, val, unit, node, output: {"node": node, "output": output, "value": {
        "value": val, "unit": unit, "quality": "published", "source_id": f1[key]["source_id"],
        "location": f1[key]["location"], "formula": f"engines/engine_params.csv F1.{key}"}}
    out["validation_targets"] = [
        mk("thrust_chamber.pc", float(f1["thrust_chamber.pc"]["value"]), "Pa", "chamber", "p_c"),
        mk("engine.thrust", float(f1["engine.thrust"]["value"]), "N", "nozzle", "thrust")]
    out["known_limitations"] = [
        "Estimated, not published: every area and volume is the A-1 value scaled by the thrust ratio. The F-1's "
        "turbopump (5,500 rpm class), injector and valve data are not in the library.",
        f"Published F-1 figures used only as targets: thrust {float(f1['engine.thrust']['value']):.4g} N ({src[0]}), "
        f"chamber pressure {float(f1['thrust_chamber.pc']['value']):.4g} Pa, Isp {f1['engine.Isp']['value']} s, "
        f"mixture ratio {f1['engine.MR']['value']}, expansion ratio {f1['thrust_chamber.expansion_ratio']['value']}."] + \
        g["known_limitations"]
    OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("wrote", OUT, f"(scale {k:.4f})")


if __name__ == "__main__":
    main()
