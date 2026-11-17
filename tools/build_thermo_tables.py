"""Build propsim/core/thermo/data/cea_tables.npz from RocketCEA. Run: python tools/build_thermo_tables.py

Grid per propellant pair: O/F (41 pts, pair range) x P_c (12 log pts, 0.5-20 MPa) x area ratio (16 log pts,
1.5-100), equilibrium and frozen-from-chamber flow. All values stored in SI.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from propsim.core.thermo.cea import DATA, EPS_RANGE, FT, PAIRS, PC_RANGE, PSI, RANKINE  # noqa: E402

N_MR, N_PC, N_EPS = 41, 12, 16


def build_pair(key: str) -> dict[str, np.ndarray]:
    from rocketcea.cea_obj import CEA_Obj
    p = PAIRS[key]
    obj = CEA_Obj(oxName=p.ox, fuelName=p.fuel)
    mr = np.linspace(p.mr_lo, p.mr_hi, N_MR)
    lnpc = np.linspace(np.log(PC_RANGE[0]), np.log(PC_RANGE[1]), N_PC)
    lneps = np.linspace(np.log(EPS_RANGE[0]), np.log(EPS_RANGE[1]), N_EPS)
    shape2, shape3 = (N_MR, N_PC), (N_MR, N_PC, N_EPS)
    out = {n: np.full(shape2, np.nan) for n in ("cstar", "cstar_fr", "Tc", "gamma", "Mw")}
    out["ivac_eq"], out["ivac_fr"] = np.full(shape3, np.nan), np.full(shape3, np.nan)
    for i, m in enumerate(mr):
        for j, lp in enumerate(lnpc):
            pc = np.exp(lp) / PSI
            for k, le in enumerate(lneps):
                e = float(np.exp(le))
                iv, cs, tc = obj.get_IvacCstrTc(Pc=pc, MR=m, eps=e, frozen=0)
                out["ivac_eq"][i, j, k] = iv
                ivf, csf, _ = obj.get_IvacCstrTc(Pc=pc, MR=m, eps=e, frozen=1)
                out["ivac_fr"][i, j, k] = ivf
                if k == 0:
                    out["cstar_fr"][i, j] = csf * FT
                    mw, g = obj.get_Throat_MolWt_gamma(Pc=pc, MR=m, eps=e)
                    out["cstar"][i, j], out["Tc"][i, j] = cs * FT, tc * RANKINE
                    out["gamma"][i, j], out["Mw"][i, j] = g, mw * 1e-3
    out["mr"], out["lnpc"], out["lneps"] = mr, lnpc, lneps
    for n, a in out.items():
        if not np.all(np.isfinite(a)) or (n not in ("mr", "lnpc", "lneps") and not np.all(a > 0)):
            raise RuntimeError(f"{key}: non-finite or non-positive values in {n} (CEA solve failed)")
    if np.any(out["ivac_fr"] > out["ivac_eq"] * 1.001):
        raise RuntimeError(f"{key}: frozen Isp exceeds equilibrium Isp (unphysical)")
    return out


def main() -> int:
    import rocketcea
    t0 = time.time()
    store: dict[str, np.ndarray] = {}
    for key in PAIRS:
        t = time.time()
        for n, a in build_pair(key).items():
            store[f"{key}__{n}"] = a
        print(f"{key:12s} done in {time.time() - t:5.1f} s", flush=True)
    store["meta__rocketcea_version"] = np.array(getattr(rocketcea, "__version__", "unknown"))
    DATA.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(DATA, **store)
    print(f"wrote {DATA} ({DATA.stat().st_size / 1e3:.0f} kB) in {time.time() - t0:.0f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
