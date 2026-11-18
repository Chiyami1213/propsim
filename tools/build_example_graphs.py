"""Build the block H files: graph and dashboard schemas, default panels and the example graphs.

Run from the repository root (rewrites the files under propsim/data/library)::

    python tools/build_example_graphs.py propsim/data/library
"""

# ruff: noqa: E501  (long provenance strings)

import json
import math
import sys
from pathlib import Path

sys.path.insert(0, ".")
import numpy as np  # noqa: E402

from propsim import data  # noqa: E402
from propsim.core.nozzle import CF as CF_ideal, exit_state  # noqa: E402
from propsim.core.thermo.cea import Thermo  # noqa: E402

LIB = Path(sys.argv[1])
(LIB / "examples").mkdir(exist_ok=True)
(LIB / "dashboards").mkdir(exist_ok=True)

G0 = 9.80665
PSI = 6894.757293168
IN = 0.0254
LB = 0.45359237
FT = 0.3048
BHP = 745.69987158227
RPM = 2 * math.pi / 60
FT3LB = 16.018463373960
EST = "PROPSIM-EST"


def r6(x):
    return float(f"{x:.6g}")


def pub(v, unit, loc, formula=None, src="HH-SP125"):
    d = {"value": r6(v), "unit": unit, "quality": "published", "source_id": src, "location": loc}
    if formula:
        d["formula"] = formula
    return d


def der(v, unit, loc, formula, src="HH-SP125"):
    return {
        "value": r6(v),
        "unit": unit,
        "quality": "derived",
        "source_id": src,
        "location": loc,
        "formula": formula,
    }


def est(v, unit, gid, formula):
    return {
        "value": r6(v),
        "unit": unit,
        "quality": "estimated",
        "source_id": EST,
        "location": f"examples/{gid}.graph.json",
        "formula": formula,
    }


def ev(eid, sub, prop, cond=None):
    return data.engine_value(eid, sub, prop, cond)


def eloc(eid, sub, prop):
    r = next(
        r
        for r in data.engine_params(eid, preferred_only=True)
        if r["subsystem"] == sub and r["property"] == prop
    )
    return r["location"], r["source_id"]


def epub(eid, sub, prop, unit):
    loc, src = eloc(eid, sub, prop)
    return pub(ev(eid, sub, prop), unit, loc, f"engines/engine_params.csv {eid}.{sub}.{prop}", src)


def area_valve(mdot, rho, dp, K):
    return mdot * math.sqrt(K / (2 * rho * dp))


def area_orifice(mdot, rho, dp, Cd):
    return mdot / (Cd * math.sqrt(2 * rho * dp))


def dia(A):
    return math.sqrt(4 * A / math.pi)


def haaland(Re, rel):
    return (-1.8 * math.log10((rel / 3.7) ** 1.11 + 6.9 / Re)) ** -2


def k_fittings(mdot, rho, mu, D, L, rough, dp):
    A = math.pi * D**2 / 4
    v = mdot / (rho * A)
    f = haaland(rho * v * D / mu, rough / D)
    return max(dp / (0.5 * rho * v**2) - f * L / D, 0.0), v, f


def mu_liq(fid, T, p):
    t = data.fluid_table(fid)
    sel = t["p"] == p
    i = int(np.argmin(np.abs(t["T"][sel] - T)))
    return float(t["mu"][sel][i]), float(t["a_sound"][sel][i]), float(t["T"][sel][i])


def disc_inertia(D, t, rho=8190.0):
    return rho * math.pi * (D / 2) ** 4 * t / 2


ROUGH = data.components()["line"]["parameters"][2]["default"]


def node(nid, ntype, label, params, x, y, options=None):
    n = {"id": nid, "type": ntype, "label": label, "params": params, "ui": {"x": x, "y": y}}
    if options:
        n["options"] = options
    return n


def edge(a, b, kind="fluid", medium=None):
    (na, pa), (nb, pb) = a.split("."), b.split(".")
    e = {
        "id": f"{na}.{pa}->{nb}.{pb}",
        "from": {"node": na, "port": pa},
        "to": {"node": nb, "port": pb},
        "kind": kind,
    }
    e["medium"] = medium or {"shaft": "torque", "thermal": "heat", "signal": "command"}[kind]
    return e


# =====================================================================================================
# 1. Gas-generator kerolox: Huzel & Huang A-1 (sea level)
# =====================================================================================================
def gg_kerolox():
    gid = "gg_kerolox"
    E = "HH_A1"
    rho_o, rho_f = ev(E, "engine", "rho_ox"), ev(E, "engine", "rho_fuel")
    m_o, m_f = ev(E, "thrust_chamber", "mdot_ox"), ev(E, "thrust_chamber", "mdot_fuel")
    m_go, m_gf = ev(E, "gg_ox_feed", "mdot"), ev(E, "gg_fuel_feed", "mdot")
    m_po, m_pf = ev(E, "ox_pump", "mdot"), ev(E, "fuel_pump", "mdot")
    pc = ev(E, "thrust_chamber", "pc_ns")
    Cd = 0.75
    Cd_v = data.components()["injector"]["parameters"][1]["default"]
    nodes, edges = [], []

    # tanks: pump inlet pressure (suction line and vehicle tank not in the source)
    for side, fid, p_in, T, mtot, rho in (
        ("ox", "LOX", ev(E, "ox_pump", "p_inlet"), 90.0, m_po, rho_o),
        ("fuel", "RP1", ev(E, "fuel_pump", "p_inlet"), 293.15, m_pf, rho_f),
    ):
        burn = ev(E, "engine", "burn_time")
        V = mtot * burn / rho / 0.95
        nodes.append(
            node(
                f"{side}_tank",
                "tank_regulated",
                f"{'LOX' if side == 'ox' else 'RP-1'} tank",
                {
                    "p_set": est(
                        p_in,
                        "Pa",
                        gid,
                        f"set to the published {side} pump inlet pressure "
                        f"({p_in / PSI:g} psia, Table 3-2 p.65); suction line not modelled",
                    ),
                    "V_tank": est(
                        V,
                        "m3",
                        gid,
                        f"pump flow {mtot:.4g} kg/s x burn {burn:g} s (Table 3-2) / rho / 0.95 "
                        "fill fraction; one engine's share, vehicle tank not in the source",
                    ),
                    "fill_fraction": est(0.95, "1", gid, "5 % ullage assumed"),
                    "T_prop": est(
                        T,
                        "K",
                        gid,
                        "LOX near normal boiling point; RP-1 at 20 degC (storage temperature "
                        "not in the source)",
                    ),
                },
                0,
                0 if side == "ox" else 400,
            )
        )

    # pumps
    for side, rho, mdot in (("ox", rho_o, m_po), ("fuel", rho_f, m_pf)):
        H = ev(E, f"{side}_pump", "head")
        N = ev(E, f"{side}_pump", "speed")
        if side == "ox":
            psi = pub(0.46, "1", "Sample Calculation (6-7) p.217")
            D = pub(
                14.8 * IN, "m", "Sample Calculation (6-7) p.218, impeller outside diameter d2", "14.8 in to m"
            )
        else:
            psi = est(
                0.46, "1", gid, "same head coefficient as the A-1 oxidizer pump (Sample Calculation (6-7))"
            )
            D = der(
                2 * math.sqrt(G0 * H / 0.46) / N,
                "m",
                "Table 3-2 p.65 head and speed",
                "D = 2 sqrt(g0 H / psi) / omega with psi = 0.46 (estimated)",
            )
        nodes.append(
            node(
                f"{side}_pump",
                "pump_centrifugal",
                f"{side} pump",
                {
                    "D_tip": D,
                    "psi_design": psi,
                    "eta_design": epub(E, f"{side}_pump", "efficiency", "1"),
                    "N_design": epub(E, f"{side}_pump", "speed", "rad/s"),
                    "Q_design": der(
                        mdot / rho,
                        "m3/s",
                        "Table 3-2 p.65",
                        f"pump mdot {mdot:.5g} kg/s / rho {rho:.5g} kg/m3",
                    ),
                },
                150,
                0 if side == "ox" else 400,
            )
        )

    # turbine, shaft, gas generator
    nodes.append(
        node(
            "turbine",
            "turbine_impulse",
            "turbine",
            {
                "D_mean": pub(
                    29.1 * IN,
                    "m",
                    "Sample Calculation (6-11) p.255, mean diameter of nozzles and blades",
                    "29.1 in to m",
                ),
                "eta_design": epub(E, "turbine", "efficiency", "1"),
                "pressure_ratio": epub(E, "turbine", "pressure_ratio", "1"),
                "T_inlet": epub(E, "turbine", "T_inlet", "K"),
            },
            300,
            200,
            options={"arrangement": "impulse_velocity_compounded_2row"},
        )
    )
    inertia = disc_inertia(29.1 * IN, 0.05) + 2 * disc_inertia(14.8 * IN, 0.05)
    nodes.append(
        node(
            "shaft",
            "shaft",
            "turbopump shaft",
            {
                "inertia": est(
                    inertia,
                    "kg m2",
                    gid,
                    "three solid steel discs 50 mm thick (rho 8190 kg/m3): turbine wheel "
                    "d = 0.739 m (Sample Calculation (6-11)), oxidizer impeller d = 0.376 m "
                    "(Sample Calculation (6-7)), fuel impeller taken equal; I = sum rho pi r^4 t / 2",
                ),
                "aux_power": epub(E, "shaft", "aux_power", "W"),
            },
            300,
            300,
        )
    )
    nodes.append(
        node(
            "gas_generator",
            "gas_generator",
            "gas generator",
            {
                "mixture_ratio": epub(E, "gas_generator", "MR", "1"),
                "p_injector_end": epub(E, "gas_generator", "p_injector_end", "Pa"),
                "T_max": est(
                    ev(E, "turbine", "T_inlet"),
                    "K",
                    gid,
                    "turbine inlet temperature 1400 degF (Table 3-2) used "
                    "as the limit until the materials block lands",
                ),
            },
            300,
            100,
        )
    )

    # gas-generator feed: line+entrance (K), valve, control orifice, injector
    for side, fid, m, rho in (("ox", "LOX", m_go, rho_o), ("fuel", "RP1", m_gf, rho_f)):
        sub = f"gg_{side}_feed"
        dp_line = ev(E, sub, "dp_line") + ev(E, sub, "dp_entrance")
        D = 0.05 if side == "ox" else 0.075
        mu, a, _ = mu_liq("LOX" if side == "ox" else "RP1_SURR", 90.0 if side == "ox" else 293.15, 5e6)
        K, v, f = k_fittings(m, rho, mu, D, 1.0, ROUGH["value"], dp_line)
        nodes.append(
            node(
                f"gg_{side}_line",
                "line",
                f"GG {side} line",
                {
                    "length": est(1.0, "m", gid, "tap-off to gas generator, layout not published"),
                    "diameter": est(D, "m", gid, f"chosen for about {v:.0f} m/s at {m:.3g} kg/s"),
                    "roughness": ROUGH,
                    "K_fittings": der(
                        K,
                        "1",
                        "Table 3-2 p.65 (entrance loss + line pressure drop)",
                        f"(dp_entrance + dp_line) / (rho v^2 / 2) - f L / D with f = {f:.4f} (Haaland)",
                    ),
                },
                200,
                120 if side == "ox" else 280,
            )
        )
        dpv = ev(E, sub, "dp_valve")
        Kp = data.components()["valve_poppet"]["parameters"][1]["default"]
        Av = area_valve(m, rho, dpv, Kp["value"])
        nodes.append(
            node(
                f"gg_{side}_valve",
                "valve_poppet",
                f"GG {side} valve",
                {
                    "diameter": der(
                        dia(Av),
                        "m",
                        "Table 3-2 p.65 (valve pressure drop)",
                        f"A = mdot sqrt(K / (2 rho dp)), K = {Kp['value']} ({Kp['location']})",
                    ),
                    "K_open": Kp,
                },
                230,
                120 if side == "ox" else 280,
            )
        )
        dpo = ev(E, sub, "dp_control_orifice")
        Ao = area_orifice(m, rho, dpo, Cd)
        nodes.append(
            node(
                f"gg_{side}_orifice",
                "orifice",
                f"GG {side} control orifice",
                {
                    "diameter": der(
                        dia(Ao),
                        "m",
                        "Table 3-2 p.65 (control-orifice pressure drop)",
                        "A = mdot / (Cd sqrt(2 rho dp)), Cd = 0.75 (p.130)",
                    ),
                    "Cd": Cd_v,
                },
                260,
                120 if side == "ox" else 280,
            )
        )
        dpi = ev(E, sub, "dp_injector")
        Ai = area_orifice(m, rho, dpi, Cd)
        nodes.append(
            node(
                f"gg_{side}_injector",
                "injector",
                f"GG {side} injector",
                {
                    "A_total": der(
                        Ai,
                        "m2",
                        "Table 3-2 p.65 (GG injector pressure drop)",
                        "A = mdot / (Cd sqrt(2 rho dp)), Cd = 0.75 (p.130)",
                    ),
                    "Cd": Cd_v,
                    "dp_over_pc_design": der(
                        dpi / ev(E, "gas_generator", "p_injector_end"),
                        "1",
                        "Table 3-2 p.65",
                        "GG injector dp / GG injector-end pressure",
                    ),
                },
                280,
                120 if side == "ox" else 280,
            )
        )

    # main feeds
    feeds = {
        "ox": dict(
            fid="LOX",
            m=m_o,
            rho=rho_o,
            D=pub(
                8 * IN,
                "m",
                "Sample Calculation (7-2) p.286, oxidizer " "pump discharge duct inside diameter",
                "8 in to m",
            ),
            T=90.0,
            tbl="LOX",
        ),
        "fuel": dict(
            fid="RP1",
            m=m_f,
            rho=rho_f,
            D=pub(
                7 * IN,
                "m",
                "Sample Calculation (7-7) p.300, fuel pump " "discharge duct diameter",
                "7 in to m",
            ),
            T=293.15,
            tbl="RP1_SURR",
        ),
    }
    for side, F in feeds.items():
        sub = f"{side}_feed"
        mu, a, _ = mu_liq(F["tbl"], F["T"], 10e6)
        mline = m_po if side == "ox" else m_pf
        mline -= m_go if side == "ox" else m_gf
        K, v, f = k_fittings(mline, F["rho"], mu, F["D"]["value"], 2.0, ROUGH["value"], ev(E, sub, "dp_line"))
        nodes.append(
            node(
                f"{side}_line",
                "line",
                f"{side} discharge line",
                {
                    "length": est(
                        2.0, "m", gid, "pump discharge to main valve, layout (Fig. 3-2) not dimensioned"
                    ),
                    "diameter": F["D"],
                    "roughness": ROUGH,
                    "K_fittings": der(
                        K,
                        "1",
                        "Table 3-2 p.65 (line pressure drop)",
                        f"dp_line / (rho v^2 / 2) - f L / D with v = {v:.1f} m/s, f = {f:.4f} (Haaland)",
                    ),
                },
                450,
                0 if side == "ox" else 400,
            )
        )
    # main oxidizer valve: geometry from Sample Calculation (7-2)
    nodes.append(
        node(
            "mov",
            "valve_butterfly",
            "main oxidizer valve",
            {
                "diameter": pub(
                    8 * IN * math.sqrt(0.78),
                    "m",
                    "Sample Calculation (7-2) p.286: characteristic flow area " "78 % of the 8 in duct",
                    "8 in x sqrt(0.78) to m",
                ),
                "K_open": pub(0.31, "1", "Sample Calculation (7-2) p.287"),
            },
            550,
            0,
        )
    )
    Kb = data.components()["valve_butterfly"]["parameters"][1]["default"]
    A = area_valve(m_f, rho_f, ev(E, "fuel_feed", "dp_main_valve"), Kb["value"])
    nodes.append(
        node(
            "mfv",
            "valve_butterfly",
            "main fuel valve",
            {
                "diameter": der(
                    dia(A),
                    "m",
                    "Table 3-2 p.65 (main valve pressure drop 15 psi)",
                    f"A = mdot sqrt(K / (2 rho dp)), K = {Kb['value']} ({Kb['location']})",
                ),
                "K_open": Kb,
            },
            550,
            400,
        )
    )
    Aco = area_orifice(m_f, rho_f, ev(E, "fuel_feed", "dp_calibration_orifice"), Cd)
    nodes.append(
        node(
            "fuel_orifice",
            "orifice",
            "calibration orifice",
            {
                "diameter": der(
                    dia(Aco),
                    "m",
                    "Table 3-2 p.65 (calibration orifice pressure drop 110 psi)",
                    "A = mdot / (Cd sqrt(2 rho dp)), Cd = 0.75 (p.130); Sample Calculation (7-7) sizes a VDI "
                    "orifice of 4.47 in for 100 psi with its own flow coefficient",
                ),
                "Cd": Cd_v,
            },
            600,
            400,
        )
    )
    # regen jacket: tube data at the throat, Sample Calculation (4-4)
    nodes.append(
        node(
            "regen_jacket",
            "regen_jacket",
            "regenerative jacket",
            {
                "n_channels": pub(94, "1", "Sample Calculation (4-4) p.110, number of tubes N"),
                "channel_width": pub(
                    0.855 * IN,
                    "m",
                    "Sample Calculation (4-4) p.110, tube diameter at the throat",
                    "0.855 in to m; round tube mapped to a square channel",
                ),
                "channel_height": pub(
                    0.855 * IN,
                    "m",
                    "Sample Calculation (4-4) p.110, tube diameter at the throat",
                    "0.855 in to m; round tube mapped to a square channel",
                ),
                "wall_thickness": pub(
                    0.020 * IN, "m", "Sample Calculation (4-4) p.110, tube wall thickness", "0.020 in to m"
                ),
                "k_wall": pub(
                    3.19e-4 * 1055.05585 / (0.0254 * 5 / 9),
                    "W/(m K)",
                    "Sample Calculation (4-4) p.110, Inconel X average at 1000-1200 degR",
                    "3.19e-4 Btu/(in s degF) to W/(m K); also materials/material_points.csv",
                ),
            },
            700,
            400,
            options={"hydraulics": "none"},  # the jacket_loss orifice carries the published jacket drop
        )
    )
    jdp = ev(E, "fuel_feed", "dp_jacket_manifold")
    Aj = area_orifice(m_f, rho_f, jdp, Cd)
    nodes.append(
        node(
            "jacket_loss",
            "orifice",
            "jacket and manifold loss",
            {
                "diameter": der(
                    dia(Aj),
                    "m",
                    "Table 3-2 p.65 (jacket and manifold pressure drop 290 psi)",
                    "equivalent orifice A = mdot / (Cd sqrt(2 rho dp)), Cd = 0.75; lumps the jacket hydraulic "
                    "loss until the jacket model carries its own",
                ),
                "Cd": Cd_v,
            },
            750,
            400,
        )
    )
    for side, F in feeds.items():
        mu, a, Tg = mu_liq(F["tbl"], F["T"], 10e6)
        V = est(
            0.02 if side == "ox" else 0.01, "m3", gid, "injector dome / fuel manifold volume, not published"
        )
        nodes.append(
            node(
                f"{side}_manifold",
                "manifold",
                "oxidizer dome" if side == "ox" else "fuel manifold",
                {
                    "volume": V,
                },
                800,
                0 if side == "ox" else 400,
            )
        )
        dpi = ev(E, f"{side}_feed", "dp_injector")
        Ai = area_orifice(F["m"], F["rho"], dpi, Cd)
        nodes.append(
            node(
                f"{side}_injector",
                "injector",
                f"{side} injector",
                {
                    "A_total": der(
                        Ai,
                        "m2",
                        "Table 3-2 p.65 (injector pressure drop)",
                        "A = mdot / (Cd sqrt(2 rho dp)), Cd = 0.75 (p.130)",
                    ),
                    "Cd": Cd_v,
                    "dp_over_pc_design": der(
                        dpi / pc, "1", "Table 3-2 p.65", "injector dp / nozzle stagnation pressure"
                    ),
                },
                900,
                0 if side == "ox" else 400,
            )
        )
    # the oxidizer dome drop (150 psi) as an orifice between valve and dome
    Ad = area_orifice(m_o, rho_o, ev(E, "ox_feed", "dp_dome"), Cd)
    nodes.append(
        node(
            "ox_dome_loss",
            "orifice",
            "torus dome loss",
            {
                "diameter": der(
                    dia(Ad),
                    "m",
                    "Table 3-2 p.65 (torus dome pressure drop 150 psi)",
                    "equivalent orifice A = mdot / (Cd sqrt(2 rho dp)), Cd = 0.75",
                ),
                "Cd": Cd_v,
            },
            700,
            0,
        )
    )
    nodes.append(
        node(
            "chamber",
            "chamber",
            "thrust chamber",
            {
                "Lstar": epub(E, "thrust_chamber", "Lstar", "m"),
                "A_t": epub(E, "thrust_chamber", "At", "m2"),
                "contraction_ratio": epub(E, "thrust_chamber", "contraction_ratio", "1"),
                "eta_cstar": epub(E, "thrust_chamber", "eta_cstar", "1"),
            },
            1000,
            200,
        )
    )
    nodes.append(
        node(
            "nozzle",
            "nozzle_bell",
            "nozzle",
            {
                "expansion_ratio": epub(E, "thrust_chamber", "expansion_ratio", "1"),
                "eta_CF": epub(E, "thrust_chamber", "eta_CF", "1"),
                "bell_fraction": epub(E, "thrust_chamber", "bell_fraction", "1"),
            },
            1100,
            200,
        )
    )

    edges += [
        edge("ox_tank.outlet", "ox_pump.inlet", medium="LOX"),
        edge("ox_pump.outlet", "ox_line.inlet", medium="LOX"),
        edge("ox_pump.outlet", "gg_ox_line.inlet", medium="LOX"),
        edge("ox_line.outlet", "mov.inlet", medium="LOX"),
        edge("mov.outlet", "ox_dome_loss.inlet", medium="LOX"),
        edge("ox_dome_loss.outlet", "ox_manifold.inlet", medium="LOX"),
        edge("ox_manifold.outlet", "ox_injector.inlet", medium="LOX"),
        edge("ox_injector.outlet", "chamber.ox_in", medium="LOX"),
        edge("fuel_tank.outlet", "fuel_pump.inlet", medium="RP1"),
        edge("fuel_pump.outlet", "fuel_line.inlet", medium="RP1"),
        edge("fuel_pump.outlet", "gg_fuel_line.inlet", medium="RP1"),
        edge("fuel_line.outlet", "mfv.inlet", medium="RP1"),
        edge("mfv.outlet", "fuel_orifice.inlet", medium="RP1"),
        edge("fuel_orifice.outlet", "regen_jacket.coolant_in", medium="RP1"),
        edge("regen_jacket.coolant_out", "jacket_loss.inlet", medium="RP1"),
        edge("jacket_loss.outlet", "fuel_manifold.inlet", medium="RP1"),
        edge("fuel_manifold.outlet", "fuel_injector.inlet", medium="RP1"),
        edge("fuel_injector.outlet", "chamber.fuel_in", medium="RP1"),
        edge("chamber.wall", "regen_jacket.wall", kind="thermal"),
        edge("chamber.throat", "nozzle.inlet", medium="combustion_gas"),
    ]
    for side, fid in (("ox", "LOX"), ("fuel", "RP1")):
        edges += [
            edge(f"gg_{side}_line.outlet", f"gg_{side}_valve.inlet", medium=fid),
            edge(f"gg_{side}_valve.outlet", f"gg_{side}_orifice.inlet", medium=fid),
            edge(f"gg_{side}_orifice.outlet", f"gg_{side}_injector.inlet", medium=fid),
            edge(f"gg_{side}_injector.outlet", f"gas_generator.{side}_in", medium=fid),
        ]
    edges += [
        edge("gas_generator.outlet", "turbine.inlet", medium="combustion_gas"),
        edge("turbine.shaft", "shaft.drive", kind="shaft"),
        edge("shaft.load", "ox_pump.shaft", kind="shaft"),
        edge("shaft.load", "fuel_pump.shaft", kind="shaft"),
    ]

    bcs = [
        {
            "node": "turbine",
            "port": "outlet",
            "kind": "fixed_pressure",
            "value": epub(E, "turbine", "p_exhaust", "Pa"),
            "note": "turbine exhaust static pressure; exhaust duct and its 2700 lb thrust not modelled",
        },
        {
            "node": "nozzle",
            "port": "inlet",
            "kind": "ambient_pressure",
            "value": est(
                101325.0, "Pa", gid, "sea-level standard pressure; Table 3-2 is at sea-level conditions"
            ),
        },
    ]
    t_cut = ev(E, "engine", "burn_time")
    seq = [
        {
            "t": est(0.0, "s", gid, "start signal defines t = 0"),
            "node": "gas_generator",
            "command": "ignite",
            "note": "GG dual spark plugs activated (pp.67-68)",
        },
        {
            "t": est(0.0, "s", gid, "start signal defines t = 0"),
            "node": "gg_ox_valve",
            "command": "open",
            "ramp": est(0.1, "s", gid, "poppet stroke time, not published"),
            "note": "four-way control valve admits fuel pump outlet pressure to the GG valve actuators (p.67)",
        },
        {
            "t": est(0.0, "s", gid, "start signal defines t = 0"),
            "node": "gg_fuel_valve",
            "command": "open",
            "ramp": est(0.1, "s", gid, "poppet stroke time, not published"),
        },
        {
            "when": {
                "node": "fuel_pump",
                "output": "dp",
                "op": ">=",
                "value": est(
                    0.5 * (ev(E, "fuel_pump", "p_discharge") - ev(E, "fuel_pump", "p_inlet")),
                    "Pa",
                    gid,
                    "half the design fuel pump pressure rise; the valve-spring setting is not published",
                ),
            },
            "node": "mov",
            "command": "open",
            "ramp": est(0.5, "s", gid, "butterfly stroke time, not published"),
            "note": "main oxidizer valve opens at a fuel pump outlet pressure set by its spring (p.67)",
        },
        {
            "when": {
                "node": "mov",
                "output": "opening",
                "op": ">=",
                "value": est(
                    0.05, "1", gid, "igniter sequence valve linked to the first part of the MOV stroke"
                ),
            },
            "node": "chamber",
            "command": "ignite",
            "note": "hypergol igniter cartridge (p.67)",
        },
        {
            "when": {
                "node": "chamber",
                "output": "p_c",
                "op": ">=",
                "value": est(0.05 * pc, "Pa", gid, "ignition monitor valve threshold, not published"),
            },
            "node": "mfv",
            "command": "open",
            "ramp": est(0.5, "s", gid, "butterfly stroke time, not published"),
            "note": "ignition monitor valve admits fuel pressure to the main fuel valve; last start step (p.68)",
        },
        {
            "t": epub(E, "engine", "burn_time", "s"),
            "node": "*",
            "command": "cutoff",
            "note": "four-way valve de-energized; all valves close (p.68)",
        },
    ]
    seq[-1]["t"] = pub(t_cut, "s", eloc(E, "engine", "burn_time")[0], "nominal single-firing duration")
    return {
        "schema_version": "1.0",
        "graph_id": gid,
        "name": "Gas-generator kerolox engine (Huzel & Huang A-1, 750 klbf sea level)",
        "description": "LOX/RP-1 gas-generator cycle on one turbopump shaft, fuel-cooled tubular chamber. Built from "
        "Huzel & Huang Table 3-2 and the A-1 sample calculations; geometry the book does not give is "
        "derived from its published pressure drops or flagged estimated.",
        "cycle": "gas_generator",
        "reference_engine": E,
        "propellants": {"oxidizer": "LOX", "fuel": "RP1"},
        "nodes": nodes,
        "edges": edges,
        "boundary_conditions": bcs,
        "sequence": seq,
        "dashboard": "dashboards/default_panels.json",
        "validation_targets": [
            {"node": "chamber", "output": "p_c", "value": epub(E, "thrust_chamber", "pc_ns", "Pa")},
            {"node": "nozzle", "output": "thrust", "value": epub(E, "thrust_chamber", "thrust", "N")},
            {"node": "ox_pump", "output": "power", "value": epub(E, "ox_pump", "power", "W")},
            {"node": "fuel_pump", "output": "power", "value": epub(E, "fuel_pump", "power", "W")},
            {"node": "turbine", "output": "power", "value": epub(E, "turbine", "power", "W")},
            {"node": "shaft", "output": "speed", "value": epub(E, "ox_pump", "speed", "rad/s")},
        ],
        "known_limitations": [
            "Suction lines and vehicle tanks are not modelled: tank pressure is the published pump inlet pressure.",
            "Jacket and dome hydraulic losses are lumped into equivalent orifices sized from the published drops.",
            "Turbine exhaust duct and its thrust (2700 lb, Sample Calculation (6-6)) are not modelled.",
            "Valve stroke times and sequence thresholds are estimated; the book gives the order, not the timing.",
            "Table 3-2 main oxidizer valve drop is 35 psi; the valve geometry of Sample Calculation (7-2) gives "
            "24.6 psi. The sample-calculation geometry is used.",
        ],
    }


# =====================================================================================================
# 2. Pressure-fed kerolox: A-4 feed architecture with LOX/RP-1
# =====================================================================================================
def pressure_fed_kerolox():
    gid = "pressure_fed_kerolox"
    E = "HH_A4"
    F = ev(E, "engine", "thrust")
    pc = ev(E, "thrust_chamber", "pc_ns")
    MR = ev("HH_A1", "thrust_chamber", "MR")
    eps = ev(E, "thrust_chamber", "expansion_ratio")
    eta_c = ev(E, "thrust_chamber", "eta_cstar")
    eta_f = ev(E, "thrust_chamber", "eta_CF")
    ch = Thermo("LOX/RP-1", source="tables").chamber(MR, pc)
    g = float(ch.gamma)
    pe_pc = exit_state(eps, g)[1]
    cf_vac = float(CF_ideal(g, pe_pc, 0.0, eps))
    At = F / (eta_f * cf_vac * pc)
    mdot = pc * At / (eta_c * float(ch.cstar))
    m_o, m_f = mdot * MR / (1 + MR), mdot / (1 + MR)
    rho_o = ev("HH_A1", "engine", "rho_ox")
    rho_f = ev("HH_A1", "engine", "rho_fuel")
    Cd = 0.75
    Cd_v = data.components()["injector"]["parameters"][1]["default"]
    th_formula = (
        f"Thermo('LOX/RP-1', source='tables').chamber(MR={MR}, pc={pc:.6g}) gives c* {float(ch.cstar):.1f} "
        f"m/s, gamma {g:.4f}; vacuum C_F {cf_vac:.4f} at eps {eps:g}"
    )
    nodes = []
    for side, fid, rho, m, T in (("ox", "LOX", rho_o, m_o, 90.0), ("fuel", "RP1", rho_f, m_f, 293.15)):
        burn = ev(E, "engine", "burn_time")
        nodes.append(
            node(
                f"{side}_tank",
                "tank_regulated",
                f"{fid} tank",
                {
                    "p_set": epub(E, f"{side}_tank", "p", "Pa"),
                    "V_tank": est(
                        m * burn / rho / 0.975,
                        "m3",
                        gid,
                        f"mdot {m:.4g} kg/s x {burn:g} s (A-4 duration, Table "
                        "3-5) / rho / 0.975 (A-4 ullage 2.5 %)",
                    ),
                    "fill_fraction": pub(
                        0.975, "1", "Table 3-5 p.77, 2.5 percent ullage volume", "1 - 0.025"
                    ),
                    "T_prop": est(T, "K", gid, "LOX near normal boiling point; RP-1 at 20 degC"),
                },
                0,
                0 if side == "ox" else 300,
            )
        )
        dp_line = ev(E, f"{side}_feed", "dp_line")
        mu, a, _ = mu_liq("LOX" if side == "ox" else "RP1_SURR", T, 2e6)
        D = 0.04 if side == "ox" else 0.03
        K, v, f = k_fittings(m, rho, mu, D, 1.5, ROUGH["value"], dp_line)
        nodes.append(
            node(
                f"{side}_line",
                "line",
                f"{side} feed line",
                {
                    "length": est(1.5, "m", gid, "tank to valve, layout not published"),
                    "diameter": est(D, "m", gid, f"chosen for about {v:.1f} m/s"),
                    "roughness": ROUGH,
                    "K_fittings": der(
                        K,
                        "1",
                        "Table 3-5 p.77 (line pressure drop)",
                        f"dp_line / (rho v^2 / 2) - f L / D with f = {f:.4f} (Haaland)",
                    ),
                },
                150,
                0 if side == "ox" else 300,
            )
        )
        Kb = data.components()["valve_ball"]["parameters"][1]["default"]
        A = area_valve(
            m,
            rho,
            ev(E, f"{side}_feed", "dp_main_valve") if side == "fuel" else ev(E, "ox_feed", "dp_main_valve"),
            Kb["value"],
        )
        nodes.append(
            node(
                f"{side}_valve",
                "valve_ball",
                f"main {side} valve",
                {
                    "diameter": der(
                        dia(A),
                        "m",
                        "Table 3-5 p.77 (main valve pressure drop)",
                        f"A = mdot sqrt(K / (2 rho dp)), K = {Kb['value']} ({Kb['location']})",
                    ),
                    "K_open": Kb,
                },
                300,
                0 if side == "ox" else 300,
            )
        )
        dpo = ev(E, f"{side}_feed", "dp_calibration_orifice")
        nodes.append(
            node(
                f"{side}_orifice",
                "orifice",
                f"{side} calibration orifice",
                {
                    "diameter": der(
                        dia(area_orifice(m, rho, dpo, Cd)),
                        "m",
                        "Table 3-5 p.77 (calibration orifice drop)",
                        "A = mdot / (Cd sqrt(2 rho dp)), Cd = 0.75",
                    ),
                    "Cd": Cd_v,
                },
                400,
                0 if side == "ox" else 300,
            )
        )
        nodes.append(
            node(
                f"{side}_manifold",
                "manifold",
                f"{side} manifold",
                {
                    "volume": est(
                        0.0005 if side == "ox" else 0.0004, "m3", gid, "manifold volume, not published"
                    ),
                },
                500,
                0 if side == "ox" else 300,
            )
        )
        dpi = ev(E, f"{side}_feed", "dp_injector")
        nodes.append(
            node(
                f"{side}_injector",
                "injector",
                f"{side} injector",
                {
                    "A_total": der(
                        area_orifice(m, rho, dpi, Cd),
                        "m2",
                        "Table 3-5 p.77 (injector pressure drop) with " "the derived flow",
                        "A = mdot / (Cd sqrt(2 rho dp))",
                    ),
                    "Cd": Cd_v,
                    "dp_over_pc_design": der(
                        dpi / pc, "1", "Table 3-5 p.77", "injector dp / chamber pressure"
                    ),
                },
                600,
                0 if side == "ox" else 300,
            )
        )
    nodes.append(
        node(
            "chamber",
            "chamber",
            "thrust chamber",
            {
                "Lstar": der(
                    45 * IN,
                    "m",
                    "Table 4-1 p.87, LOX/RP-1 L* 40 to 50 in",
                    "mid value of the range, 45 in to m",
                ),
                "A_t": der(
                    At,
                    "m2",
                    "Table 3-5 p.77 (thrust, chamber pressure, C_F efficiency)",
                    f"A_t = F / (eta_CF C_F,vac P_c); {th_formula}",
                    src="HH-SP125",
                ),
                "contraction_ratio": epub(E, "thrust_chamber", "contraction_ratio", "1"),
                "eta_cstar": epub(E, "thrust_chamber", "eta_cstar", "1"),
            },
            700,
            150,
        )
    )
    nodes.append(
        node(
            "nozzle",
            "nozzle_bell",
            "nozzle",
            {
                "expansion_ratio": epub(E, "thrust_chamber", "expansion_ratio", "1"),
                "eta_CF": epub(E, "thrust_chamber", "eta_CF", "1"),
                "bell_fraction": epub(E, "thrust_chamber", "bell_fraction", "1"),
            },
            800,
            150,
        )
    )
    edges = []
    for side, fid in (("ox", "LOX"), ("fuel", "RP1")):
        edges += [
            edge(f"{side}_tank.outlet", f"{side}_line.inlet", medium=fid),
            edge(f"{side}_line.outlet", f"{side}_valve.inlet", medium=fid),
            edge(f"{side}_valve.outlet", f"{side}_orifice.inlet", medium=fid),
            edge(f"{side}_orifice.outlet", f"{side}_manifold.inlet", medium=fid),
            edge(f"{side}_manifold.outlet", f"{side}_injector.inlet", medium=fid),
            edge(f"{side}_injector.outlet", f"chamber.{side}_in", medium=fid),
        ]
    edges.append(edge("chamber.throat", "nozzle.inlet", medium="combustion_gas"))
    seq = [
        {
            "t": est(0.0, "s", gid, "start signal defines t = 0"),
            "node": "ox_valve",
            "command": "open",
            "ramp": est(0.05, "s", gid, "ball valve stroke, not published"),
            "note": "oxidizer lead; the A-4 hypergols need no igniter, a LOX/RP-1 version does",
        },
        {
            "t": est(0.0, "s", gid, "igniter on with the oxidizer lead"),
            "node": "chamber",
            "command": "ignite",
        },
        {
            "t": est(0.1, "s", gid, "fuel follows the oxidizer by 0.1 s, not published"),
            "node": "fuel_valve",
            "command": "open",
            "ramp": est(0.05, "s", gid, "ball valve stroke, not published"),
        },
        {"t": est(20.0, "s", gid, "short firing for the example"), "node": "*", "command": "cutoff"},
    ]
    return {
        "schema_version": "1.0",
        "graph_id": gid,
        "name": "Pressure-fed kerolox engine (A-4 feed system, LOX/RP-1, 7.5 klbf vacuum)",
        "description": "Regulated pressure-fed LOX/RP-1 engine. Feed pressures, pressure drops, chamber pressure, "
        "thrust level and nozzle come from the Huzel & Huang A-4 stage engine (Table 3-5); the "
        "propellants are changed to LOX/RP-1, so throat area and flows are derived from the shipped "
        "CEA tables. No published pressure-fed kerolox engine is in the library yet.",
        "cycle": "pressure_fed",
        "reference_engine": None,
        "propellants": {"oxidizer": "LOX", "fuel": "RP1"},
        "nodes": nodes,
        "edges": edges,
        "boundary_conditions": [
            {
                "node": "nozzle",
                "port": "inlet",
                "kind": "ambient_pressure",
                "value": pub(0.0, "Pa", "Table 3-5 p.77, vacuum conditions"),
            },
        ],
        "sequence": seq,
        "dashboard": "dashboards/default_panels.json",
        "validation_targets": [
            {"node": "chamber", "output": "p_c", "value": epub(E, "thrust_chamber", "pc_ns", "Pa")},
            {"node": "nozzle", "output": "thrust", "value": epub(E, "thrust_chamber", "thrust", "N")},
        ],
        "known_limitations": [
            "A-4 is N2O4/N2H4; its feed-system pressure drops are reused for LOX/RP-1 at the derived flow rates.",
            "The mixture ratio 2.35 is the A-1 thrust-chamber value, not an optimum for 100 psia.",
            "LOX/RP-1 is not hypergolic: the sequence adds an igniter, timing estimated.",
        ],
    }


# =====================================================================================================
# 3. Expander hydrolox: A-2 chamber and pumps with a closed expander turbine drive
# =====================================================================================================
def expander_hydrolox():
    gid = "expander_hydrolox"
    E = "HH_A2"
    pc = ev(E, "thrust_chamber", "pc_ns")
    m_o, m_f = ev(E, "thrust_chamber", "mdot_ox"), ev(E, "thrust_chamber", "mdot_fuel")
    rho_o, rho_f = ev(E, "engine", "rho_ox"), ev(E, "engine", "rho_fuel")
    Cd = 0.75
    Cd_v = data.components()["injector"]["parameters"][1]["default"]
    eta_po, eta_pf = ev(E, "ox_pump", "efficiency"), ev(E, "fuel_pump", "efficiency")
    eta_t = ev(E, "fuel_turbine", "efficiency")
    T_t = 250.0
    # H2 at turbine inlet from the CoolProp table (nearest grid point)
    t = data.fluid_table("LH2")
    sel = t["p"] == 10e6
    i = int(np.argmin(np.abs(t["T"][sel] - T_t)))
    cp, cv = float(t["cp"][sel][i]), float(t["cv"][sel][i])
    gam = cp / cv
    Tg = float(t["T"][sel][i])
    rho_jx = float(t["rho"][sel][i])
    # pressures along the fuel side (Pa)
    p_inj_f = pc + ev(E, "fuel_feed", "dp_injector")
    p_t_out = p_inj_f + ev(E, "fuel_feed", "dp_line")
    P_po = m_o * G0 * ev(E, "ox_pump", "head") / eta_po
    p_in_f = ev(E, "fuel_pump", "p_inlet")
    dp_up = (
        ev(E, "fuel_feed", "dp_main_valve")
        + ev(E, "fuel_feed", "dp_jacket_manifold")
        + ev(E, "fuel_feed", "dp_line")
    )
    # fixed point: fuel pump discharge sets its power; turbines in series (fuel turbine first) supply both pumps
    p_dis = ev(E, "fuel_pump", "p_discharge")
    for _ in range(200):
        P_pf = m_f * (p_dis - p_in_f) / (rho_f * eta_pf)
        dh = (P_pf + P_po) / (m_f * eta_t)
        PR = (1 - dh / (cp * Tg)) ** (-gam / (gam - 1))
        p_dis = p_t_out * PR + dp_up
    PR_f = (1 - P_pf / (m_f * eta_t * cp * Tg)) ** (-gam / (gam - 1))
    T2 = Tg * (1 - eta_t * (1 - PR_f ** (-(gam - 1) / gam)))
    PR_o = PR / PR_f
    H_f = (p_dis - p_in_f) / (rho_f * G0)
    form = (
        f"closed expander balance: fuel turbine then oxidizer turbine in series, inlet {Tg:.0f} K H2 (cp "
        f"{cp:.0f}, gamma {gam:.3f} from fluids/tables/LH2.csv at 10 MPa); eta_t 0.625 (A-2 fuel turbine); "
        "turbine exit to fuel injector with the A-2 line and injector drops"
    )
    nodes = []
    for side, fid, rho, mdot, T in (("ox", "LOX", rho_o, m_o, 90.0), ("fuel", "LH2", rho_f, m_f, 20.3)):
        burn = ev(E, "engine", "burn_time")
        nodes.append(
            node(
                f"{side}_tank",
                "tank_regulated",
                f"{fid} tank",
                {
                    "p_set": est(
                        ev(E, f"{side}_pump", "p_inlet"),
                        "Pa",
                        gid,
                        "set to the published A-2 pump inlet pressure "
                        "(Table 3-3); suction line not modelled",
                    ),
                    "V_tank": est(
                        mdot * burn / rho / 0.95, "m3", gid, f"mdot x {burn:g} s (Table 3-3) / rho / 0.95"
                    ),
                    "fill_fraction": est(0.95, "1", gid, "5 % ullage assumed"),
                    "T_prop": est(T, "K", gid, "near the normal boiling point"),
                },
                0,
                0 if side == "ox" else 400,
            )
        )
    nodes.append(
        node(
            "ox_pump",
            "pump_centrifugal",
            "oxidizer pump",
            {
                "D_tip": der(
                    2 * math.sqrt(G0 * ev(E, "ox_pump", "head") / 0.46) / ev(E, "ox_pump", "speed"),
                    "m",
                    "Table 3-3 p.69 head and speed",
                    "D = 2 sqrt(g0 H / psi) / omega, psi = 0.46 (estimated)",
                ),
                "psi_design": est(
                    0.46, "1", gid, "A-1 oxidizer pump head coefficient (Sample Calculation (6-7))"
                ),
                "eta_design": epub(E, "ox_pump", "efficiency", "1"),
                "N_design": epub(E, "ox_pump", "speed", "rad/s"),
                "Q_design": der(m_o / rho_o, "m3/s", "Table 3-3 p.69", "thrust-chamber oxidizer flow / rho"),
            },
            150,
            0,
        )
    )
    nodes.append(
        node(
            "fuel_pump",
            "pump_centrifugal",
            "fuel pump",
            {
                "D_tip": der(
                    2 * math.sqrt(G0 * H_f / 0.46) / ev(E, "fuel_pump", "speed"),
                    "m",
                    "Table 3-3 p.69 speed; head from the expander balance",
                    f"D = 2 sqrt(g0 H / psi) / omega, H = {H_f:.0f} m, psi = 0.46 (estimated); a real pump this "
                    "size would use two or more stages",
                ),
                "psi_design": est(
                    0.46, "1", gid, "A-1 oxidizer pump head coefficient (Sample Calculation (6-7))"
                ),
                "eta_design": epub(E, "fuel_pump", "efficiency", "1"),
                "N_design": epub(E, "fuel_pump", "speed", "rad/s"),
                "Q_design": der(m_f / rho_f, "m3/s", "Table 3-3 p.69", "thrust-chamber fuel flow / rho"),
            },
            150,
            400,
        )
    )
    for tn, PRx, Tin, pump in (("fuel_turbine", PR_f, Tg, "fuel_pump"), ("ox_turbine", PR_o, T2, "ox_pump")):
        Pt = P_pf if tn == "fuel_turbine" else P_po
        C0 = math.sqrt(2 * cp * Tin * (1 - PRx ** (-(gam - 1) / gam)))
        U = 0.3 * C0
        nodes.append(
            node(
                tn,
                "turbine_impulse",
                tn.replace("_", " "),
                {
                    "D_mean": est(
                        2 * U / ev(E, pump, "speed"),
                        "m",
                        gid,
                        f"U/C0 = 0.3 (impulse turbine range, Fig. 6-27) " f"with C0 = {C0:.0f} m/s",
                    ),
                    "eta_design": epub(E, "fuel_turbine", "efficiency", "1"),
                    "pressure_ratio": est(PRx, "1", gid, form + f"; this turbine supplies {Pt / 1e6:.3f} MW"),
                    "T_inlet": est(
                        Tin,
                        "K",
                        gid,
                        (
                            f"jacket exit temperature assumed {Tg:.0f} K, nearest LH2 table grid "
                            "point to 250 K (no source in the library); "
                            if tn == "fuel_turbine"
                            else "fuel turbine exit temperature; "
                        )
                        + form,
                    ),
                },
                300,
                150 if tn == "fuel_turbine" else 250,
                options={"arrangement": "impulse_single_stage"},
            )
        )
    for s, pump, tn in (("ox", "ox_pump", "ox_turbine"), ("fuel", "fuel_pump", "fuel_turbine")):
        nodes.append(
            node(
                f"{s}_shaft",
                "shaft",
                f"{s} turbopump shaft",
                {
                    "inertia": est(
                        0.05 if s == "ox" else 0.02,
                        "kg m2",
                        gid,
                        "small upper-stage turbopump rotor, not " "published; PropSim cycle default order",
                    ),
                    "aux_power": est(0.0, "W", gid, "no auxiliary drive listed in Table 3-3"),
                },
                350,
                50 if s == "ox" else 350,
            )
        )
    for side, fid, m, rho in (("ox", "LOX", m_o, rho_o), ("fuel", "LH2", m_f, rho_f)):
        Kb = data.components()["valve_butterfly"]["parameters"][1]["default"]
        nodes.append(
            node(
                f"{side}_valve",
                "valve_butterfly",
                f"main {side} valve",
                {
                    "diameter": der(
                        dia(area_valve(m, rho, ev(E, f"{side}_feed", "dp_main_valve"), Kb["value"])),
                        "m",
                        "Table 3-3 p.69 (main valve pressure drop)",
                        f"A = mdot sqrt(K / (2 rho dp)), K = {Kb['value']} ({Kb['location']})",
                    ),
                    "K_open": Kb,
                },
                450,
                0 if side == "ox" else 400,
            )
        )
    dp_ox_rest = (
        ev(E, "ox_feed", "dp_dome") + ev(E, "ox_feed", "dp_calibration_orifice") + ev(E, "ox_feed", "dp_line")
    )
    nodes.append(
        node(
            "ox_loss",
            "orifice",
            "oxidizer line, orifice and dome loss",
            {
                "diameter": der(
                    dia(area_orifice(m_o, rho_o, dp_ox_rest, Cd)),
                    "m",
                    "Table 3-3 p.69 (line 20 + calibration orifice 60 + dome 40 psi)",
                    "equivalent orifice A = mdot / (Cd sqrt(2 rho dp)), Cd = 0.75",
                ),
                "Cd": Cd_v,
            },
            550,
            0,
        )
    )
    nodes.append(
        node(
            "regen_jacket",
            "regen_jacket",
            "regenerative jacket",
            {
                "n_channels": pub(178, "1", "Sample Calculation (4-4) p.112, number of tubes N"),
                "channel_width": pub(
                    0.185 * IN,
                    "m",
                    "Sample Calculation (4-4) p.112, tube diameter at the throat",
                    "0.185 in to m; round tube mapped to a square channel",
                ),
                "channel_height": pub(
                    0.185 * IN,
                    "m",
                    "Sample Calculation (4-4) p.112, tube diameter at the throat",
                    "0.185 in to m; round tube mapped to a square channel",
                ),
                "wall_thickness": pub(
                    0.008 * IN, "m", "Sample Calculation (4-4) p.112, tube wall thickness", "0.008 in to m"
                ),
                "k_wall": pub(
                    28.86,
                    "W/(m K)",
                    "Sample Calculation (4-4) p.112, Inconel X at 1600 degR",
                    "3.86e-4 Btu/(in s degF) x 1055.056 / (0.0254 x 5/9)",
                ),
            },
            550,
            400,
            options={"hydraulics": "none"},  # the jacket_loss orifice carries the published jacket drop
        )
    )
    nodes.append(
        node(
            "jacket_loss",
            "orifice",
            "jacket hydraulic loss",
            {
                "diameter": der(
                    dia(area_orifice(m_f, rho_jx, ev(E, "fuel_feed", "dp_jacket_manifold"), Cd)),
                    "m",
                    "Table 3-3 p.69 (jacket and manifold pressure drop 325 psi)",
                    f"equivalent orifice A = mdot / (Cd sqrt(2 rho dp)), Cd = 0.75, with the density of the heated "
                    f"hydrogen leaving the jacket ({rho_jx:.3f} kg/m3 at {Tg:.0f} K, 10 MPa, CoolProp table); "
                    "lumps the jacket and manifold loss (the jacket node carries heat only)",
                ),
                "Cd": Cd_v,
            },
            600,
            400,
        )
    )
    for side, fid, m, rho in (("ox", "LOX", m_o, rho_o), ("fuel", "LH2", m_f, rho_f)):
        nodes.append(
            node(
                f"{side}_manifold",
                "manifold",
                f"{side} manifold",
                {
                    "volume": est(
                        0.005 if side == "ox" else 0.008, "m3", gid, "manifold volume, not published"
                    ),
                },
                700,
                0 if side == "ox" else 400,
            )
        )
        dpi = ev(E, f"{side}_feed", "dp_injector")
        rho_i = rho if side == "ox" else float(t["rho"][sel][int(np.argmin(np.abs(t["T"][sel] - T2)))])
        nodes.append(
            node(
                f"{side}_injector",
                "injector",
                f"{side} injector",
                {
                    "A_total": der(
                        area_orifice(m, rho_i, dpi, Cd),
                        "m2",
                        "Table 3-3 p.69 (injector pressure drop)",
                        "A = mdot / (Cd sqrt(2 rho dp)), Cd = 0.75"
                        + (
                            ""
                            if side == "ox"
                            else f"; gaseous H2 density {rho_i:.1f} kg/m3 at the turbine exit"
                        ),
                    ),
                    "Cd": Cd_v,
                    "dp_over_pc_design": der(
                        dpi / pc, "1", "Table 3-3 p.69", "injector dp / chamber pressure"
                    ),
                },
                800,
                0 if side == "ox" else 400,
            )
        )
    nodes.append(
        node(
            "chamber",
            "chamber",
            "thrust chamber",
            {
                "Lstar": epub(E, "thrust_chamber", "Lstar", "m"),
                "A_t": epub(E, "thrust_chamber", "At", "m2"),
                "contraction_ratio": epub(E, "thrust_chamber", "contraction_ratio", "1"),
                "eta_cstar": epub(E, "thrust_chamber", "eta_cstar", "1"),
            },
            900,
            200,
        )
    )
    nodes.append(
        node(
            "nozzle",
            "nozzle_bell",
            "nozzle",
            {
                "expansion_ratio": epub(E, "thrust_chamber", "expansion_ratio", "1"),
                "eta_CF": epub(E, "thrust_chamber", "eta_CF", "1"),
                "bell_fraction": epub(E, "thrust_chamber", "bell_fraction", "1"),
            },
            1000,
            200,
        )
    )
    edges = [
        edge("ox_tank.outlet", "ox_pump.inlet", medium="LOX"),
        edge("ox_pump.outlet", "ox_valve.inlet", medium="LOX"),
        edge("ox_valve.outlet", "ox_loss.inlet", medium="LOX"),
        edge("ox_loss.outlet", "ox_manifold.inlet", medium="LOX"),
        edge("ox_manifold.outlet", "ox_injector.inlet", medium="LOX"),
        edge("ox_injector.outlet", "chamber.ox_in", medium="LOX"),
        edge("fuel_tank.outlet", "fuel_pump.inlet", medium="LH2"),
        edge("fuel_pump.outlet", "fuel_valve.inlet", medium="LH2"),
        edge("fuel_valve.outlet", "regen_jacket.coolant_in", medium="LH2"),
        edge("regen_jacket.coolant_out", "jacket_loss.inlet", medium="LH2"),
        edge("jacket_loss.outlet", "fuel_turbine.inlet", medium="LH2"),
        edge("fuel_turbine.outlet", "ox_turbine.inlet", medium="LH2"),
        edge("ox_turbine.outlet", "fuel_manifold.inlet", medium="LH2"),
        edge("fuel_manifold.outlet", "fuel_injector.inlet", medium="LH2"),
        edge("fuel_injector.outlet", "chamber.fuel_in", medium="LH2"),
        edge("chamber.wall", "regen_jacket.wall", kind="thermal"),
        edge("chamber.throat", "nozzle.inlet", medium="combustion_gas"),
        edge("fuel_turbine.shaft", "fuel_shaft.drive", kind="shaft"),
        edge("fuel_shaft.load", "fuel_pump.shaft", kind="shaft"),
        edge("ox_turbine.shaft", "ox_shaft.drive", kind="shaft"),
        edge("ox_shaft.load", "ox_pump.shaft", kind="shaft"),
    ]
    seq = [
        {
            "t": est(0.0, "s", gid, "start signal defines t = 0"),
            "node": "fuel_valve",
            "command": "open",
            "ramp": est(0.3, "s", gid, "valve stroke, not published"),
            "note": "tank-head fuel flow through jacket and turbines starts the bootstrap; the A-2 itself starts "
            "with a solid-grain spinner (p.70)",
        },
        {
            "t": est(0.5, "s", gid, "after fuel lead, not published"),
            "node": "ox_valve",
            "command": "open",
            "ramp": est(0.3, "s", gid, "valve stroke, not published"),
        },
        {"t": est(0.5, "s", gid, "with the oxidizer valve"), "node": "chamber", "command": "ignite"},
        {"t": est(60.0, "s", gid, "short firing for the example"), "node": "*", "command": "cutoff"},
    ]
    return {
        "schema_version": "1.0",
        "graph_id": gid,
        "name": "Expander hydrolox engine (A-2 chamber and pumps, closed expander drive, 150 klbf vacuum)",
        "description": "LOX/LH2 closed expander cycle: fuel heated in the regenerative jacket drives the fuel and "
        "oxidizer turbines in series and is then injected. The chamber, injector drops, pumps and "
        "speeds are the Huzel & Huang A-2 values (Table 3-3); the A-2 itself is a tap-off engine, so "
        "the turbine drive is derived from a power balance with an estimated jacket exit temperature.",
        "cycle": "expander_closed",
        "reference_engine": None,
        "propellants": {"oxidizer": "LOX", "fuel": "LH2"},
        "nodes": nodes,
        "edges": edges,
        "boundary_conditions": [
            {
                "node": "nozzle",
                "port": "inlet",
                "kind": "ambient_pressure",
                "value": pub(0.0, "Pa", "Table 3-3 p.69, vacuum conditions"),
            },
        ],
        "sequence": seq,
        "dashboard": "dashboards/default_panels.json",
        "validation_targets": [
            {"node": "chamber", "output": "p_c", "value": epub(E, "thrust_chamber", "pc_ns", "Pa")},
            {"node": "nozzle", "output": "thrust", "value": epub(E, "thrust_chamber", "thrust", "N")},
        ],
        "design_point": {
            "fuel_pump_discharge": est(p_dis, "Pa", gid, form),
            "fuel_pump_power": est(P_pf, "W", gid, form),
            "ox_pump_power": der(
                P_po, "W", "Table 3-3 p.69", "mdot g0 H / eta, thrust-chamber oxidizer flow"
            ),
        },
        "known_limitations": [
            "Hybrid of published A-2 data and an estimated expander drive; not a flight engine.",
            f"The balance needs a fuel pump discharge of {p_dis / PSI:.0f} psia against 1400 psia for the A-2 "
            "tap-off engine, so the fuel pump head is higher than published.",
            f"Jacket exit temperature {Tg:.0f} K is an assumption; the jacket model will compute it once wired.",
            "Fuel pump tip speed from psi = 0.46 is about 600 m/s; a real pump of this head would use two or "
            "more stages.",
        ],
    }


# =====================================================================================================
# 4. SSME staged combustion at rated power level, public NASA sources only
# =====================================================================================================
SNR = "NASA-SNR-1981"
TP37 = "NASA-TP-3700"
CPS = "COOLPROP-8.0.0"


def rho_grid(fid, T, p):
    t = data.fluid_table(fid)
    sel = t["p"] == p
    i = int(np.argmin(np.abs(t["T"][sel] - T)))
    return float(t["rho"][sel][i]), float(t["T"][sel][i])


def h2_rich_gas(MR, T):
    """Frozen H2/H2O preburner gas at O/F mass ratio MR (all O2 burned), ideal-gas cp0 from CoolProp."""
    import CoolProp.CoolProp as CP

    n_h2o = MR / 31.9988 * 2  # kmol H2O per kg H2 (O2 + 2 H2 -> 2 H2O)
    n_h2 = 1 / 2.01588 - n_h2o
    x_w = n_h2o / (n_h2 + n_h2o)
    M = (1 + MR) / (n_h2 + n_h2o) * 1e-3  # kg/mol
    cpm = (1 - x_w) * CP.PropsSI("CP0MOLAR", "T", T, "P", 1e5, "Hydrogen") + x_w * CP.PropsSI(
        "CP0MOLAR", "T", T, "P", 1e5, "Water"
    )
    cp = cpm / M
    gam = cpm / (cpm - 8.314462618)
    return cp, gam, x_w, M


def ssme_rpl():
    gid = "ssme_staged_combustion"
    sv = lambda v, unit, loc, formula=None: pub(v, unit, loc, formula, src=SNR)  # noqa: E731
    dv = lambda v, unit, loc, formula, src=SNR: der(v, unit, loc, formula, src=src)  # noqa: E731
    # --- published operating point (Space Shuttle News Reference, rated power level) ---
    F_vac, isp_vac = 470000 * LB * G0, 455.2
    pc = 2970 * PSI
    MR, eps, cr = 6.0, 77.5, 2.96
    mdot = F_vac / (isp_vac * G0)
    m_o, m_f = mdot * MR / (1 + MR), mdot / (1 + MR)
    pumps = {  # id: (fluid, rpm, power bhp, p_in psia, p_out psia, flow lb/s, location)
        "lpftp": ("LH2", 14700, 2400, 30, 232, 147, "p.2-12, low-pressure fuel turbopump"),
        "hpftp": ("LH2", 35000, 62270, 176, 6210, 147, "pp.2-12 to 2-13, high-pressure fuel turbopump"),
        "lpotp": ("LOX", 5150, 1470, 100, 415, 883, "p.2-13, low-pressure oxidizer turbopump"),
        "hpotp": (
            "LOX",
            29057,
            20977,
            360,
            4632,
            1066,
            "p.2-13, high-pressure oxidizer turbopump (main pump)",
        ),
        "pbp": (
            "LOX",
            29057,
            1472,
            4437,
            7635,
            86,
            "p.2-13, high-pressure oxidizer turbopump (preburner pump)",
        ),
    }
    rho_o, T_o = rho_grid("LOX", 90.0, 1e6)
    rho_f, T_f = rho_grid("LH2", 20.3, 1e6)
    rho = {"LOX": rho_o, "LH2": rho_f}
    # --- preburner split (NASA TP-3700 Table A.6, heat-balance column, 100 % power) ---
    fpb_f, fpb_o, opb_f, opb_o = 78.18 * LB, 67.78 * LB, 35.1 * LB, 23.67 * LB
    MR_fpb, MR_opb = fpb_o / fpb_f, opb_o / opb_f
    m_fpb, m_opb = fpb_f + fpb_o, opb_f + opb_o
    T_fpb, T_opb = 961.0, 817.0  # turbine inlet temperatures, SNR p.2-13
    p_fpb, p_opb = 5164 * PSI, 5228 * PSI
    eta_t = 0.75
    eta_note = (
        "eta_t 0.75 estimated for the two-stage reaction turbines (no public value); pressure ratio from "
        "P = mdot eta_t cp T_in (1 - PR^-((gamma-1)/gamma)) with the published turbopump power and inlet "
        "temperature and the frozen H2/H2O gas of the preburner mixture ratio (cp0 from CoolProp)"
    )
    turb = {}
    for tid, m, T, MRp, P, p_in in (
        ("hpft", m_fpb, T_fpb, MR_fpb, pumps["hpftp"][2] * BHP, p_fpb),
        ("hpot", m_opb, T_opb, MR_opb, (pumps["hpotp"][2] + pumps["pbp"][2]) * BHP, p_opb),
    ):
        cp, gam, x_w, M = h2_rich_gas(MRp, T)
        ratio = P / (m * eta_t * cp * T)
        PR = (1 - ratio) ** (-gam / (gam - 1))
        turb[tid] = dict(
            m=m, T=T, MR=MRp, P=P, cp=cp, gam=gam, x_w=x_w, M=M, PR=PR, p_in=p_in, p_out=p_in / PR
        )
    p_hgm = min(turb["hpft"]["p_out"], turb["hpot"]["p_out"])
    # LPFT: driven by the hydrogen that cooled the main chamber (about 25 % of the fuel, SNR p.2-9)
    import CoolProp.CoolProp as CP

    m_lpft = 0.25 * m_f
    T_lpft = 469.1 * 5 / 9  # main-chamber coolant temperature, TP-3700 Table A.6 heat balance
    P_lpft = pumps["lpftp"][2] * BHP
    eta_lpft = 0.6
    p_lpft_in = turb["hpft"]["p_out"] * 1.2
    for _ in range(100):
        cp_l = CP.PropsSI("C", "T", T_lpft, "P", p_lpft_in, "Hydrogen")
        cv_l = CP.PropsSI("O", "T", T_lpft, "P", p_lpft_in, "Hydrogen")
        g_l = cp_l / cv_l
        PR_l = (1 - P_lpft / (m_lpft * eta_lpft * cp_l * T_lpft)) ** (-g_l / (g_l - 1))
        p_lpft_in = p_hgm * PR_l
    # LPOT: hydraulic turbine driven by HPOTP discharge oxygen, flow = HPOTP flow - LPOTP flow
    m_lpot = (pumps["hpotp"][5] - pumps["lpotp"][5]) * LB
    p_lpot_in, p_lpot_out = 4488 * PSI, pumps["lpotp"][4] * PSI
    P_lpot = pumps["lpotp"][2] * BHP
    eta_lpot = P_lpot / (m_lpot * (p_lpot_in - p_lpot_out) / rho_o)
    # chamber: c* from the shipped CEA tables, eta_c* library default, eta_CF closes the published Isp
    ch = Thermo("LOX/LH2", source="tables").chamber(MR, pc)
    g = float(ch.gamma)
    cf_vac = float(CF_ideal(g, exit_state(eps, g)[1], 0.0, eps))
    Cdef = data.components()["chamber"]["parameters"][3]["default"]
    eta_c = Cdef["value"]
    At = mdot * eta_c * float(ch.cstar) / pc
    eta_CF = isp_vac * G0 / (eta_c * float(ch.cstar) * cf_vac)
    th = (
        f"Thermo('LOX/LH2', source='tables').chamber(MR=6, pc=2970 psia) gives c* {float(ch.cstar):.1f} m/s, "
        f"T_c {float(ch.Tc):.0f} K (published about 3590 K, p.2-9), gamma {g:.4f}; vacuum C_F {cf_vac:.4f} at "
        f"eps 77.5; mdot = F_vac / (Isp_vac g0) = {mdot:.1f} kg/s"
    )
    # valves: published flow-passage sizes, ball-valve K from the library
    Kb = data.components()["valve_ball"]["parameters"][1]["default"]
    Kp = data.components()["valve_poppet"]["parameters"][1]["default"]
    A_main = math.pi * (2.5 * IN) ** 2 / 4
    dp_mov = Kb["value"] * 0.5 * rho_o * (m_o / (rho_o * A_main)) ** 2
    dp_mfv = Kb["value"] * 0.5 * rho_f * (m_f / (rho_f * A_main)) ** 2
    dp_inj_o = pumps["hpotp"][4] * PSI - dp_mov - pc
    dp_inj_f = p_hgm - pc
    Cd = 0.75
    Cd_v = data.components()["injector"]["parameters"][1]["default"]
    # hot-gas density at the injector: ideal gas at the HPFT exit temperature
    T_hg = T_fpb * (
        1 - eta_t * (1 - turb["hpft"]["PR"] ** (-(turb["hpft"]["gam"] - 1) / turb["hpft"]["gam"]))
    )
    rho_hg = p_hgm * turb["hpft"]["M"] / (8.314462618 * T_hg)
    m_hg = m_f + fpb_o + opb_o
    m_o_mcc = m_o - fpb_o - opb_o

    nodes = []
    for side, fid, T, p_in, y in (("ox", "LOX", 90.0, 100, 0), ("fuel", "LH2", 20.3, 30, 500)):
        m = m_o if side == "ox" else m_f
        burn = 8.5 * 60
        nodes.append(
            node(
                f"{side}_tank",
                "tank_regulated",
                f"{fid} tank",
                {
                    "p_set": sv(
                        p_in * PSI,
                        "Pa",
                        pumps["lpotp" if side == "ox" else "lpftp"][6],
                        f"{p_in} psia low-pressure pump inlet; suction line not modelled",
                    ),
                    "V_tank": est(
                        m * burn / rho[fid] / 0.95,
                        "m3",
                        gid,
                        "rated-power flow x 8.5 min (nominal operating time, SNR p.2-4) / rho / 0.95; "
                        "one engine's share, not the External Tank",
                    ),
                    "fill_fraction": est(0.95, "1", gid, "5 % ullage assumed"),
                    "T_prop": est(T, "K", gid, "near the normal boiling point"),
                },
                0,
                y,
            )
        )
    pump_x = {
        "lpftp": (150, 500),
        "hpftp": (300, 500),
        "lpotp": (150, 0),
        "hpotp": (300, 0),
        "pbp": (450, 100),
    }
    for pid, (fid, rpm, bhp, pi, po, flow, loc) in pumps.items():
        m, dp, w = flow * LB, (po - pi) * PSI, rpm * RPM
        H = dp / (rho[fid] * G0)
        eta = m * dp / (rho[fid] * bhp * BHP)
        psi = 0.46
        nodes.append(
            node(
                f"{pid}_pump",
                "pump_centrifugal",
                {
                    "lpftp": "low-pressure fuel pump",
                    "hpftp": "high-pressure fuel pump (3 stages)",
                    "lpotp": "low-pressure oxidizer pump",
                    "hpotp": "high-pressure oxidizer main pump",
                    "pbp": "preburner boost pump",
                }[pid],
                {
                    "D_tip": est(
                        2 * math.sqrt(G0 * H / psi) / w,
                        "m",
                        gid,
                        f"D = 2 sqrt(g0 H / psi) / omega with H = {H:.0f} m from the published pressure "
                        f"rise, psi = 0.46 (A-1 head coefficient)"
                        + ("; per-stage head is a third of this" if pid == "hpftp" else ""),
                    ),
                    "psi_design": est(
                        psi, "1", gid, "A-1 oxidizer pump head coefficient (Sample Calculation (6-7))"
                    ),
                    "eta_design": dv(
                        eta,
                        "1",
                        loc,
                        f"eta = mdot dp / (rho P) = {m:.1f} kg/s x {dp / 1e6:.3f} MPa / ({rho[fid]:.1f} "
                        f"kg/m3 x {bhp} bhp); incompressible, inlet density from fluids/tables/{fid}.csv"
                        + ("; LH2 compressibility raises the true value" if fid == "LH2" else ""),
                    ),
                    "N_design": sv(w, "rad/s", loc, f"{rpm} rpm"),
                    "Q_design": sv(
                        m / rho[fid],
                        "m3/s",
                        loc,
                        f"{flow} lb/s / {rho[fid]:.1f} kg/m3"
                        + ("; printed '~9 kg/s (86 lb/s)', the lb/s figure is used" if pid == "pbp" else ""),
                    ),
                },
                *pump_x[pid],
                options={
                    "stages": "3" if pid == "hpftp" else "1",
                    "impeller": "axial" if pid in ("lpftp", "lpotp") else "centrifugal",
                },
            )
        )
    tloc = "p.2-13 turbine inlet pressure and temperature"
    for tid, label, pump_id, x, y in (
        ("hpft", "high-pressure fuel turbine", "hpftp", 450, 400),
        ("hpot", "high-pressure oxidizer turbine", "hpotp", 450, 200),
    ):
        t = turb[tid]
        C0 = math.sqrt(2 * t["cp"] * t["T"] * (1 - t["PR"] ** (-(t["gam"] - 1) / t["gam"])))
        w = pumps[pump_id][1] * RPM
        nodes.append(
            node(
                f"{tid}_turbine",
                "turbine_impulse",
                label,
                {
                    "D_mean": est(
                        2 * 0.5 * C0 / w,
                        "m",
                        gid,
                        f"U/C0 = 0.5 assumed for a two-stage reaction turbine, C0 = {C0:.0f} m/s",
                    ),
                    "eta_design": est(eta_t, "1", gid, eta_note),
                    "pressure_ratio": est(
                        t["PR"],
                        "1",
                        gid,
                        eta_note + f"; mdot {t['m']:.2f} kg/s (TP-3700 preburner flows), power "
                        f"{t['P'] / 1e6:.3f} MW (SNR p.2-12/2-13), cp {t['cp']:.0f} J/(kg K), "
                        f"gamma {t['gam']:.3f}; exit {t['p_out'] / PSI:.0f} psia",
                    ),
                    "T_inlet": sv(t["T"], "K", tloc, f"{t['T']:.0f} K as printed"),
                },
                x,
                y,
                options={"arrangement": "reaction_2stage"},
            )
        )
    nodes.append(
        node(
            "lpft_turbine",
            "turbine_impulse",
            "low-pressure fuel turbine",
            {
                "D_mean": est(0.2, "m", gid, "not published; order of the LPFTP impeller size"),
                "eta_design": est(eta_lpft, "1", gid, "two-stage hydrogen turbine, no public value"),
                "pressure_ratio": est(
                    PR_l,
                    "1",
                    gid,
                    f"power 2400 bhp (p.2-12) with mdot = 25 % of the fuel flow (p.2-9) = "
                    f"{m_lpft:.2f} kg/s, inlet {T_lpft:.1f} K (TP-3700 Table A.6 main-chamber "
                    f"coolant temperature, 469.1 degR), real-gas cp/cv from CoolProp, eta 0.6; "
                    f"exit at the hot-gas manifold {p_hgm / PSI:.0f} psia",
                ),
                "T_inlet": dv(
                    T_lpft, "K", "Table A.6 p.26, TCMBF heat-balance column", "469.1 degR x 5/9", src=TP37
                ),
            },
            300,
            650,
            options={"arrangement": "reaction_2stage"},
        )
    )
    nodes.append(
        node(
            "lpot_turbine",
            "turbine_impulse",
            "low-pressure oxidizer turbine (hydraulic)",
            {
                "D_mean": est(0.2, "m", gid, "not published; order of the LPOTP size"),
                "eta_design": dv(
                    eta_lpot,
                    "1",
                    "p.2-13, LPOTP power and turbine drive pressure",
                    f"eta = P / (mdot dp / rho) with P = 1470 bhp, mdot = HPOTP - LPOTP flow = "
                    f"{m_lpot:.1f} kg/s, dp = 4488 - 415 psia, rho {rho_o:.1f} kg/m3",
                ),
                "pressure_ratio": dv(
                    p_lpot_in / p_lpot_out,
                    "1",
                    "p.2-13",
                    "4488 psia drive / 415 psia LPOTP " "discharge, where the drive flow rejoins",
                ),
            },
            150,
            150,
            options={"arrangement": "hydraulic_6stage"},
        )
    )
    for sid, d, x, y in (
        ("lpftp", "LPFTP", 200, 600),
        ("hpftp", "HPFTP", 380, 450),
        ("lpotp", "LPOTP", 200, 80),
        ("hpotp", "HPOTP", 380, 120),
    ):
        nodes.append(
            node(
                f"{sid}_shaft",
                "shaft",
                f"{d} shaft",
                {"aux_power": est(0.0, "W", gid, "no auxiliary drive")},
                x,
                y,
            )
        )
    for gg, label, MRp, m, p, x, y in (
        ("fpb", "fuel preburner", MR_fpb, m_fpb, p_fpb, 600, 450),
        ("opb", "oxidizer preburner", MR_opb, m_opb, p_opb, 600, 250),
    ):
        o, f = (fpb_o, fpb_f) if gg == "fpb" else (opb_o, opb_f)
        nodes.append(
            node(
                gg,
                "gas_generator",
                label,
                {
                    "mixture_ratio": dv(
                        MRp,
                        "1",
                        "Table A.6 p.26, heat-balance column",
                        f"oxygen / hydrogen flow = {o / LB:.2f} / {f / LB:.2f} lb/s",
                        src=TP37,
                    ),
                    "p_injector_end": dv(
                        p,
                        "Pa",
                        tloc,
                        f"set to the {'HPFT' if gg == 'fpb' else 'HPOT'} inlet "
                        f"pressure {p / PSI:.0f} psia (preburner and turbine inlet share a duct)",
                    ),
                },
                x,
                y,
            )
        )
    # Preburner fuel injectors: both preburners draw hydrogen from the nozzle-coolant outlet, yet run at different
    # published pressures (5164 / 5228 psia), so each needs its own injector drop; without them the two
    # fixed-pressure-ratio turbine drives cannot share one hot-gas manifold.
    p_pb_up = 1.15 * max(p_fpb, p_opb)
    rho_pb, T_pb = rho_grid("LH2", 250.0, 40e6)
    Cd_inj = data.components()["injector"]["parameters"][1]["default"]
    for gg, f_pb, p_pb, x, y in (("fpb", fpb_f, p_fpb, 520, 470), ("opb", opb_f, p_opb, 520, 270)):
        dp_pb = p_pb_up - p_pb
        A_pb = f_pb / (Cd_inj["value"] * math.sqrt(2.0 * rho_pb * dp_pb))
        nodes.append(
            node(
                f"{gg}_fuel_injector",
                "injector",
                f"{'fuel' if gg == 'fpb' else 'oxidizer'} preburner fuel injector",
                {
                    "A_total": est(
                        A_pb,
                        "m2",
                        gid,
                        f"A = mdot_f / (Cd sqrt(2 rho dp)) with the TP-3700 preburner hydrogen flow {f_pb / LB:.2f} "
                        f"lb/s, Cd {Cd_inj['value']}, upstream pressure 1.15 x the higher preburner pressure "
                        f"(15 % stiffness, H&H p.128) and hydrogen at {T_pb:.0f} K, 40 MPa ({rho_pb:.2f} kg/m3, "
                        "CoolProp table); injector areas are not in a public NASA source",
                    ),
                    "Cd": Cd_inj,
                    "dp_over_pc_design": est(
                        dp_pb / p_pb, "1", gid, "drop over preburner pressure for the area above"
                    ),
                },
                x,
                y,
                options={"type": "coaxial"},
            )
        )
    for vid, label, d_in, A, K, m, rr, x, y in (
        ("mov", "main oxidizer valve", None, A_main, Kb, m_o_mcc, rho_o, 600, 0),
        ("mfv", "main fuel valve", None, A_main, Kb, m_f, rho_f, 450, 550),
        (
            "fpov",
            "fuel preburner oxidizer valve",
            None,
            math.pi * (1.1 * IN) ** 2 / 4,
            Kp,
            fpb_o,
            rho_o,
            550,
            350,
        ),
        ("opov", "oxidizer preburner oxidizer valve", None, 1.1 * 0.285 * IN**2, Kp, opb_o, rho_o, 550, 200),
    ):
        loc = {
            "mov": "p.2-13, main oxidizer valve 2.5 in flow passage",
            "mfv": "p.2-13, main fuel valve 2.5 in " "flow passage",
            "fpov": "p.2-13, fuel preburner oxidizer valve 1.1 in flow passage",
            "opov": "p.2-13, oxidizer preburner oxidizer valve 1.1 x 0.285 in slot",
        }[vid]
        nodes.append(
            node(
                vid,
                "valve_ball" if vid in ("mov", "mfv") else "valve_poppet",
                label,
                {
                    "diameter": sv(
                        dia(A),
                        "m",
                        loc,
                        (
                            "slot area 1.1 x 0.285 in2 as an equivalent round passage"
                            if vid == "opov"
                            else "in x 0.0254"
                        ),
                    ),
                    "K_open": Kb if vid in ("mov", "mfv") else Kp,
                },
                x,
                y,
            )
        )
    for jid, label, n, loc, x, y in (
        (
            "mcc_jacket",
            "main combustion chamber coolant channels",
            390,
            "p.2-9, 390 coolant channels",
            750,
            600,
        ),
        ("nozzle_jacket", "nozzle coolant tubes", 1080, "p.2-10, 1080 tubes", 600, 600),
    ):
        nodes.append(
            node(
                jid,
                "regen_jacket",
                label,
                {
                    "n_channels": sv(n, "1", loc),
                },
                x,
                y,
            )
        )
        if jid == "mcc_jacket":
            D_t = dia(At)
            w_ch = 0.5 * math.pi * D_t / n
            geo = {
                "channel_width": est(
                    w_ch,
                    "m",
                    gid,
                    f"half of the throat circumference pi D_t / 390 (D_t {D_t:.4f} m "
                    "from A_t), the rest ribs; not in a public NASA source",
                ),
                "channel_height": est(2.5 * w_ch, "m", gid, "aspect ratio 2.5 assumed for milled channels"),
                "wall_thickness": est(
                    0.7e-3, "m", gid, "0.7 mm hot-gas wall assumed for a milled copper liner"
                ),
            }
        else:
            w_t = math.pi * 2.4 / n
            geo = {
                "channel_width": dv(
                    w_t,
                    "m",
                    "p.2-10, 1080 tubes, 2.4 m exit diameter",
                    "tube width at the exit = pi x 2.4 m / 1080; round tube mapped to a square channel",
                ),
                "channel_height": dv(
                    w_t, "m", "p.2-10, 1080 tubes, 2.4 m exit diameter", "equal to the width (round tube)"
                ),
                "wall_thickness": est(
                    0.3e-3, "m", gid, "0.3 mm tube wall assumed; not in a public NASA source"
                ),
            }
        nodes[-1]["params"].update(geo)
        # main combustion chamber liner to the nozzle joint, coolant tubes from there to the exit; the joint area
        # ratio of 5 is an engineering estimate (the cycle solver splits the wall heat load there)
        nodes[-1]["options"] = {"section": "chamber" if jid == "mcc_jacket" else "nozzle", "split_area_ratio": "5"}
    nodes.append(
        node(
            "hgm",
            "manifold",
            "hot-gas manifold",
            {
                "volume": est(0.1, "m3", gid, "not published; order of the twin-duct hot-gas manifold"),
            },
            800,
            350,
        )
    )
    nodes.append(
        node(
            "ox_manifold",
            "manifold",
            "oxidizer dome",
            {
                "volume": est(0.02, "m3", gid, "not published"),
            },
            750,
            0,
        )
    )
    for side, A_m, dp, rr, m, x, y in (
        ("ox", None, dp_inj_o, rho_o, m_o_mcc, 850, 100),
        ("fuel", None, dp_inj_f, rho_hg, m_hg, 850, 300),
    ):
        nodes.append(
            node(
                f"{side}_injector",
                "injector",
                f"main injector, {'oxidizer' if side == 'ox' else 'hot-gas fuel'} side",
                {
                    "A_total": est(
                        area_orifice(m, rr, dp, Cd),
                        "m2",
                        gid,
                        f"A = mdot / (Cd sqrt(2 rho dp)), Cd 0.75, mdot {m:.1f} kg/s, dp {dp / PSI:.0f} psi "
                        + (
                            "= HPOTP discharge 4632 psia - main valve loss - P_c"
                            if side == "ox"
                            else f"= hot-gas manifold (turbine exit estimate) - P_c, gas density {rr:.1f} kg/m3"
                        ),
                    ),
                    "Cd": Cd_v,
                    "dp_over_pc_design": est(
                        dp / pc, "1", gid, "injector dp / chamber pressure from the balance"
                    ),
                },
                x,
                y,
                options={"type": "coaxial"},
            )
        )
    nodes.append(
        node(
            "chamber",
            "chamber",
            "main combustion chamber",
            {
                "A_t": dv(
                    At,
                    "m2",
                    "p.2-4 vacuum thrust, Isp and chamber pressure",
                    f"A_t = mdot eta_c* c* / P_c, eta_c* {eta_c} (library default); {th}",
                ),
                "contraction_ratio": sv(cr, "1", "p.2-9, contraction area ratio 2.96"),
                "eta_cstar": Cdef,
            },
            950,
            200,
        )
    )
    nodes.append(
        node(
            "nozzle",
            "nozzle_bell",
            "nozzle",
            {
                "expansion_ratio": sv(eps, "1", "p.2-4 and p.2-10, area ratio 77.5"),
                "eta_CF": dv(
                    eta_CF,
                    "1",
                    "p.2-4 vacuum Isp 455.2 s",
                    f"eta_CF = Isp_vac g0 / (eta_c* c* C_F,vac); {th}",
                ),
            },
            1050,
            200,
        )
    )
    L, H, G = "LOX", "LH2", "combustion_gas"
    edges = [
        edge("ox_tank.outlet", "lpotp_pump.inlet", medium=L),
        edge("lpotp_pump.outlet", "hpotp_pump.inlet", medium=L),
        edge("hpotp_pump.outlet", "mov.inlet", medium=L),
        edge("hpotp_pump.outlet", "lpot_turbine.inlet", medium=L),
        edge("hpotp_pump.outlet", "pbp_pump.inlet", medium=L),
        edge("lpot_turbine.outlet", "hpotp_pump.inlet", medium=L),
        edge("mov.outlet", "ox_manifold.inlet", medium=L),
        edge("ox_manifold.outlet", "ox_injector.inlet", medium=L),
        edge("ox_injector.outlet", "chamber.ox_in", medium=L),
        edge("pbp_pump.outlet", "fpov.inlet", medium=L),
        edge("pbp_pump.outlet", "opov.inlet", medium=L),
        edge("fpov.outlet", "fpb.ox_in", medium=L),
        edge("opov.outlet", "opb.ox_in", medium=L),
        edge("fuel_tank.outlet", "lpftp_pump.inlet", medium=H),
        edge("lpftp_pump.outlet", "hpftp_pump.inlet", medium=H),
        edge("hpftp_pump.outlet", "mfv.inlet", medium=H),
        edge("mfv.outlet", "mcc_jacket.coolant_in", medium=H),
        edge("mfv.outlet", "nozzle_jacket.coolant_in", medium=H),
        edge("mcc_jacket.coolant_out", "lpft_turbine.inlet", medium=H),
        edge("lpft_turbine.outlet", "hgm.inlet", medium=H),
        edge("nozzle_jacket.coolant_out", "fpb_fuel_injector.inlet", medium=H),
        edge("nozzle_jacket.coolant_out", "opb_fuel_injector.inlet", medium=H),
        edge("fpb_fuel_injector.outlet", "fpb.fuel_in", medium=H),
        edge("opb_fuel_injector.outlet", "opb.fuel_in", medium=H),
        edge("fpb.outlet", "hpft_turbine.inlet", medium=G),
        edge("opb.outlet", "hpot_turbine.inlet", medium=G),
        edge("hpft_turbine.outlet", "hgm.inlet", medium=G),
        edge("hpot_turbine.outlet", "hgm.inlet", medium=G),
        edge("hgm.outlet", "fuel_injector.inlet", medium=G),
        edge("fuel_injector.outlet", "chamber.fuel_in", medium=G),
        edge("chamber.throat", "nozzle.inlet", medium=G),
        edge("chamber.wall", "mcc_jacket.wall", kind="thermal"),
        edge("chamber.wall", "nozzle_jacket.wall", kind="thermal"),
    ]
    for sid, turb_id, pump_ids in (
        ("lpftp", "lpft", ["lpftp"]),
        ("hpftp", "hpft", ["hpftp"]),
        ("lpotp", "lpot", ["lpotp"]),
        ("hpotp", "hpot", ["hpotp", "pbp"]),
    ):
        edges.append(edge(f"{turb_id}_turbine.shaft", f"{sid}_shaft.drive", kind="shaft"))
        for p in pump_ids:
            edges.append(edge(f"{sid}_shaft.load", f"{p}_pump.shaft", kind="shaft"))
    seq_loc = "start command to 90 % thrust in about 3.6 s (NSTS 1988 News Reference, not in the library)"
    seq = [
        {
            "t": est(0.0, "s", gid, "start command defines t = 0"),
            "node": "mfv",
            "command": "open",
            "ramp": est(0.5, "s", gid, "fuel lead; valve stroke not published"),
            "note": "the real start is a closed-loop schedule by the engine controller",
        },
        {
            "t": est(0.3, "s", gid, "after fuel lead, not published"),
            "node": "fpov",
            "command": "open",
            "ramp": est(1.0, "s", gid, seq_loc),
        },
        {
            "t": est(0.4, "s", gid, "after fuel lead, not published"),
            "node": "opov",
            "command": "open",
            "ramp": est(1.0, "s", gid, seq_loc),
        },
        {"t": est(0.4, "s", gid, "with the preburner valves"), "node": "fpb", "command": "ignite"},
        {"t": est(0.4, "s", gid, "with the preburner valves"), "node": "opb", "command": "ignite"},
        {
            "t": est(0.6, "s", gid, "not published"),
            "node": "mov",
            "command": "open",
            "ramp": est(2.0, "s", gid, seq_loc),
        },
        {"t": est(0.6, "s", gid, "with the main oxidizer valve"), "node": "chamber", "command": "ignite"},
        {
            "t": est(510.0, "s", gid, "about 8.5 min nominal operating time (SNR p.2-4)"),
            "node": "*",
            "command": "cutoff",
        },
    ]
    dp_balance = (
        f"Main valve losses with K {Kb['value']} through the published 2.5 in passages: oxidizer {dp_mov / PSI:.0f} "
        f"psi, fuel {dp_mfv / PSI:.0f} psi."
    )
    return {
        "schema_version": "1.0",
        "graph_id": gid,
        "name": "SSME staged combustion at rated power level (public NASA data)",
        "description": "LOX/LH2 fuel-rich staged combustion with two preburners and four turbopumps, built only from "
        "public NASA documents: station pressures, speeds, powers, flows and turbine inlet temperatures from the "
        "1981 Space Shuttle News Reference (rated power level), preburner flow split from the heat-balance column "
        "of NASA TP-3700 Table A.6, chamber c* from the shipped CEA tables. Turbine efficiencies, line volumes and "
        "the start sequence are estimates.",
        "cycle": "staged_combustion_fuel_rich",
        "reference_engine": "RS25",
        "propellants": {"oxidizer": "LOX", "fuel": "LH2"},
        "nodes": nodes,
        "edges": edges,
        "boundary_conditions": [
            {
                "node": "nozzle",
                "port": "inlet",
                "kind": "ambient_pressure",
                "value": sv(0.0, "Pa", "p.2-4, vacuum thrust and Isp"),
            },
        ],
        "sequence": seq,
        "dashboard": "dashboards/default_panels.json",
        "validation_targets": [
            {"node": "chamber", "output": "p_c", "value": sv(pc, "Pa", "p.2-4 summary, 2970 psia")},
            {
                "node": "nozzle",
                "output": "thrust",
                "value": sv(F_vac, "N", "p.2-4 summary, 470 000 lbf vacuum", "lbf x 4.448222"),
            },
            {"node": "nozzle", "output": "isp", "value": sv(isp_vac, "s", "p.2-4 summary, vacuum")},
            {"node": "chamber", "output": "T_c", "value": sv(3590.0, "K", "p.2-9, about 3590 K")},
            {
                "node": "hpftp_pump",
                "output": "speed",
                "value": sv(35000 * RPM, "rad/s", pumps["hpftp"][6], "35 000 rpm"),
            },
            {
                "node": "hpotp_pump",
                "output": "speed",
                "value": sv(29057 * RPM, "rad/s", pumps["hpotp"][6], "29 057 rpm"),
            },
        ],
        "design_point": {
            "mdot_total": dv(mdot, "kg/s", "p.2-4", "F_vac / (Isp_vac g0)"),
            "mdot_fuel": dv(m_f, "kg/s", "p.2-4", "mdot / (1 + MR); LPFTP/HPFTP flow printed as 67 kg/s"),
            "mdot_ox": dv(m_o, "kg/s", "p.2-4", "mdot MR / (1 + MR); LPOTP flow printed as 401 kg/s"),
            "hpft_exit_pressure": est(turb["hpft"]["p_out"], "Pa", gid, eta_note),
            "hpot_exit_pressure": est(turb["hpot"]["p_out"], "Pa", gid, eta_note),
            "lpft_inlet_pressure": est(p_lpft_in, "Pa", gid, "LPFT power balance, see lpft_turbine"),
            "mov_dp": der(dp_mov, "Pa", "p.2-13 passage", dp_balance, src=SNR),
            "mfv_dp": der(dp_mfv, "Pa", "p.2-13 passage", dp_balance, src=SNR),
        },
        "known_limitations": [
            "Station data mix two public sources: the 1981 News Reference engine (rated power, 2970 psia) and the "
            "NASA TP-3700 heat balance (3006 psia) for the preburner split; the 104.5 % power level flown later "
            "has no complete public station set.",
            "The low-pressure oxidizer turbine is a liquid-oxygen hydraulic turbine; the catalog has only a gas "
            "turbine, so its pressure ratio and efficiency are carried but its gas equations do not apply.",
            "The nozzle coolant tubes take heat from the chamber wall port; the catalog has no nozzle wall port.",
            "Chamber coolant valve, pogo accumulator, tank pressurization taps, igniters and purges are not modelled.",
            f"Turbine efficiencies are estimates; with them the turbine exits sit at "
            f"{turb['hpft']['p_out'] / PSI:.0f} psia (HPFT) and {turb['hpot']['p_out'] / PSI:.0f} psia (HPOT) "
            "against 2970 psia in the chamber.",
            "Regenerative channel geometry and shaft inertias are not in a public NASA source and are left null.",
        ],
    }


# =====================================================================================================
# Default panels per component type
# =====================================================================================================
PALETTE = ["#4269d0", "#efb118", "#ff725c", "#6cc5b0", "#3ca951", "#ff8ab7", "#a463f2", "#97bbf5"]


def default_panels():
    comps = data.components()
    out = {}

    def s(t, name, i):
        o = next(o for o in comps[t]["outputs_for_plots"] if o["name"] == name)
        return {
            "node": "$node",
            "output": name,
            "unit": o["unit"],
            "label": o["meaning"],
            "color": PALETTE[i],
        }

    spec = {
        "tank_regulated": [
            ("Tank pressure and outflow", ["p_ullage"], ["mdot_out"]),
            ("Propellant left", ["m_liq"], []),
        ],
        "tank_blowdown": [
            ("Tank pressure and outflow", ["p_ullage"], ["mdot_out"]),
            ("Propellant left", ["m_liq"], []),
        ],
        "line": [("Line flow", ["mdot"], ["dp_friction"])],
        "valve_ball": [("Valve", ["opening"], ["mdot"])],
        "valve_butterfly": [("Valve", ["opening"], ["mdot"])],
        "valve_poppet": [("Valve", ["opening"], ["mdot"])],
        "check_valve": [("Check valve", ["mdot"], ["dp"])],
        "orifice": [("Orifice", ["mdot"], ["dp"])],
        "cavitating_venturi": [("Venturi", ["mdot"], ["cavitating"])],
        "manifold": [("Manifold pressure", ["p"], [])],
        "injector": [("Injector", ["mdot"], ["dp_over_pc"])],
        "chamber": [
            ("Chamber pressure and temperature", ["p_c"], ["T_c"]),
            ("Mixture ratio and c*", ["mixture_ratio"], ["cstar"]),
        ],
        "nozzle_bell": [("Thrust and Isp", ["thrust"], ["isp"])],
        "pump_centrifugal": [
            ("Pump speed and head", ["speed"], ["head"]),
            ("Pump power and NPSH", ["power"], ["npsh_available"]),
        ],
        "inducer": [("Inducer", ["head"], ["npsh_available"])],
        "turbine_impulse": [
            ("Turbine power and torque", ["power"], ["torque"]),
            ("Turbine efficiency", ["efficiency"], ["u_over_c0"]),
        ],
        "gas_generator": [("Gas generator", ["p_gg"], ["T_gg"])],
        "shaft": [("Shaft", ["speed"], ["torque_net"])],
        "regen_jacket": [
            ("Wall temperatures", ["T_wall_gas", "T_wall_cool"], ["q_wall"]),
            ("Coolant", ["T_cool"], ["p_cool"]),
        ],
    }
    for t, panels in spec.items():
        out[t] = []
        for k, (title, left, right) in enumerate(panels):
            out[t].append(
                {
                    "id": f"{t}_{k}",
                    "title": title,
                    "kind": "time_series",
                    "x": {"var": "t", "unit": "s", "label": "time"},
                    "y_left": [s(t, n, i) for i, n in enumerate(left)],
                    "y_right": [s(t, n, len(left) + i) for i, n in enumerate(right)],
                }
            )
    out["nozzle_bell"].append(
        {"id": "nozzle_bell_field", "title": "Nozzle Mach field", "kind": "nozzle_field", "node": "$node"}
    )
    out["chamber"].append({"id": "chamber_3d", "title": "Engine 3D view", "kind": "view3d", "node": "$node"})
    missing = sorted(set(comps) - set(out))
    assert not missing, missing
    return {
        "schema_version": "1.0",
        "dashboard_id": "default_panels",
        "description": "Standard panels per component type; '$node' is replaced by each node id of that type. "
        "Output names and units are those of components/<type>.json outputs_for_plots.",
        "panels_by_type": out,
    }


GRAPH_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "https://propsim/library/schemas/graph.schema.json",
    "title": "Engine graph",
    "description": "A propulsion system as nodes (component instances of components/<type>.json) joined by typed "
    "edges between ports. SI units only: every parameter is a sourced value whose unit must equal "
    "the unit of that parameter in the component type. A fluid out-port may feed several edges (a "
    "tee at equal pressure); a shaft port may join several machines on one shaft.",
    "type": "object",
    "additionalProperties": False,
    "required": [
        "schema_version",
        "graph_id",
        "name",
        "description",
        "cycle",
        "propellants",
        "nodes",
        "edges",
        "boundary_conditions",
        "sequence",
    ],
    "$defs": {
        "sv": {"$ref": "sourced_value.schema.json"},
        "port_ref": {
            "type": "object",
            "required": ["node", "port"],
            "additionalProperties": False,
            "properties": {"node": {"type": "string"}, "port": {"type": "string"}},
        },
    },
    "properties": {
        "schema_version": {"const": "1.0"},
        "graph_id": {"type": "string", "pattern": "^[a-z][a-z0-9_]*$"},
        "name": {"type": "string"},
        "description": {"type": "string"},
        "cycle": {"type": "string", "description": "cycle_id of cycles/cycles.json"},
        "reference_engine": {"type": ["string", "null"], "description": "engine_id of engines/engines.csv"},
        "propellants": {
            "type": "object",
            "required": ["oxidizer", "fuel"],
            "additionalProperties": False,
            "properties": {"oxidizer": {"type": "string"}, "fuel": {"type": "string"}},
            "description": "fluid_id values of fluids/fluids.csv",
        },
        "nodes": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "required": ["id", "type", "params"],
                "additionalProperties": False,
                "properties": {
                    "id": {"type": "string", "pattern": "^[a-z][a-z0-9_]*$"},
                    "type": {"type": "string", "description": "component type (components/<type>.json)"},
                    "label": {"type": "string"},
                    "params": {
                        "type": "object",
                        "additionalProperties": {"$ref": "#/$defs/sv"},
                        "description": "parameter name -> sourced value; omitted parameters take the type "
                        "default; parameters with user_input must be given",
                    },
                    "options": {
                        "type": "object",
                        "additionalProperties": {"type": "string"},
                        "description": "categorical choices, e.g. turbine arrangement or injector type",
                    },
                    "ui": {
                        "type": "object",
                        "properties": {"x": {"type": "number"}, "y": {"type": "number"}},
                    },
                },
            },
        },
        "edges": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["id", "from", "to", "kind", "medium"],
                "additionalProperties": False,
                "properties": {
                    "id": {"type": "string"},
                    "from": {"$ref": "#/$defs/port_ref"},
                    "to": {"$ref": "#/$defs/port_ref"},
                    "kind": {"enum": ["fluid", "shaft", "thermal", "signal"]},
                    "medium": {
                        "type": "string",
                        "description": "fluid edges: a fluid_id or combustion_gas; shaft: "
                        "torque; thermal: heat; signal: command",
                    },
                },
            },
        },
        "boundary_conditions": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["node", "port", "kind", "value"],
                "additionalProperties": False,
                "properties": {
                    "node": {"type": "string"},
                    "port": {"type": "string"},
                    "kind": {
                        "enum": ["ambient_pressure", "fixed_pressure", "fixed_temperature", "fixed_mass_flow"]
                    },
                    "value": {"$ref": "#/$defs/sv"},
                    "note": {"type": "string"},
                },
            },
        },
        "sequence": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["node", "command"],
                "additionalProperties": False,
                "oneOf": [{"required": ["t"]}, {"required": ["when"]}],
                "properties": {
                    "t": {"$ref": "#/$defs/sv", "description": "time from start signal (s)"},
                    "when": {
                        "type": "object",
                        "required": ["node", "output", "op", "value"],
                        "additionalProperties": False,
                        "properties": {
                            "node": {"type": "string"},
                            "output": {"type": "string"},
                            "op": {"enum": [">=", "<="]},
                            "value": {"$ref": "#/$defs/sv"},
                        },
                        "description": "event trigger on a node output (outputs_for_plots name)",
                    },
                    "node": {"type": "string", "description": "node id, or * for every valve (cutoff)"},
                    "command": {"enum": ["open", "close", "set", "ignite", "cutoff"]},
                    "value": {
                        "$ref": "#/$defs/sv",
                        "description": "target for set (e.g. valve opening fraction)",
                    },
                    "ramp": {"$ref": "#/$defs/sv", "description": "stroke time for open, close or set (s)"},
                    "note": {"type": "string"},
                },
            },
        },
        "dashboard": {"type": "string", "description": "library path of the dashboard file"},
        "validation_targets": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["node", "output", "value"],
                "additionalProperties": False,
                "properties": {
                    "node": {"type": "string"},
                    "output": {"type": "string"},
                    "value": {"$ref": "#/$defs/sv"},
                },
                "description": "published steady-state values the simulation should reproduce",
            },
        },
        "design_point": {"type": "object", "additionalProperties": {"$ref": "#/$defs/sv"}},
        "known_limitations": {"type": "array", "items": {"type": "string"}},
    },
}

SERIES = {
    "type": "object",
    "required": ["node", "output", "unit"],
    "additionalProperties": False,
    "properties": {
        "node": {"type": "string", "description": "node id, or $node in default panels"},
        "output": {"type": "string"},
        "unit": {"type": "string"},
        "label": {"type": "string"},
        "color": {"type": "string", "pattern": "^#[0-9a-fA-F]{6}$"},
    },
}
PANEL = {
    "type": "object",
    "required": ["id", "title", "kind"],
    "additionalProperties": False,
    "properties": {
        "id": {"type": "string"},
        "title": {"type": "string"},
        "kind": {"enum": ["time_series", "xy", "nozzle_field", "view3d"]},
        "x": {
            "type": "object",
            "required": ["var", "unit"],
            "additionalProperties": False,
            "properties": {
                "var": {"type": "string"},
                "unit": {"type": "string"},
                "label": {"type": "string"},
                "node": {"type": "string"},
            },
        },
        "y_left": {"type": "array", "items": SERIES},
        "y_right": {"type": "array", "items": SERIES},
        "node": {"type": "string", "description": "node shown by nozzle_field and view3d panels"},
        "layout": {
            "type": "object",
            "properties": {
                "row": {"type": "integer"},
                "col": {"type": "integer"},
                "w": {"type": "integer"},
                "h": {"type": "integer"},
            },
        },
    },
    "allOf": [
        {
            "if": {"properties": {"kind": {"enum": ["time_series", "xy"]}}},
            "then": {"required": ["x", "y_left"]},
        },
        {
            "if": {"properties": {"kind": {"enum": ["nozzle_field", "view3d"]}}},
            "then": {"required": ["node"]},
        },
    ],
}
DASH_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "https://propsim/library/schemas/dashboard.schema.json",
    "title": "Dashboard",
    "description": "Plot panels for a simulation run. Either a concrete dashboard (panels) or the default panel "
    "set per component type (panels_by_type, with $node placeholders). Units are SI strings and "
    "must equal the unit of the output in the component type.",
    "type": "object",
    "additionalProperties": False,
    "required": ["schema_version", "dashboard_id"],
    "oneOf": [{"required": ["panels"]}, {"required": ["panels_by_type"]}],
    "properties": {
        "schema_version": {"const": "1.0"},
        "dashboard_id": {"type": "string"},
        "description": {"type": "string"},
        "panels": {"type": "array", "items": PANEL},
        "panels_by_type": {"type": "object", "additionalProperties": {"type": "array", "items": PANEL}},
    },
}


def dump(obj, path):
    path.write_text(json.dumps(obj, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


dump(GRAPH_SCHEMA, LIB / "schemas" / "graph.schema.json")
dump(DASH_SCHEMA, LIB / "schemas" / "dashboard.schema.json")
dump(default_panels(), LIB / "dashboards" / "default_panels.json")
for fn in (pressure_fed_kerolox, gg_kerolox, expander_hydrolox, ssme_rpl):
    g = fn()
    dump(g, LIB / "examples" / f"{g['graph_id']}.graph.json")
    print(g["graph_id"], len(g["nodes"]), "nodes", len(g["edges"]), "edges")
