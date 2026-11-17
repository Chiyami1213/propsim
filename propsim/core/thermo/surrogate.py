"""PHASE 1 PLACEHOLDER combustion properties for GOX/HTPB.

NOT CEA. A smooth analytic stand-in so the Run demo has a plausible c*(O/F) and T_c(O/F);
Phase 2 replaces this module with the RocketCEA wrapper + tabulated fallback.
Peak c* ~ 1800 m/s near O/F ~ 2.1 with a steeper decline on the rich side.
"""
from __future__ import annotations

import numpy as np

OF_OPT = 2.1
GAMMA = 1.2


def cstar_gox_htpb(OF):
    r"""$c^*(\frac{O}{F})$ in (m·s⁻¹)."""
    OF = np.asarray(OF, float)
    k = np.where(OF > OF_OPT, 130.0, 250.0)
    return 1800.0 - k * (OF - OF_OPT) ** 2


def tc_gox_htpb(OF):
    r"""Chamber temperature $T_c(\frac{O}{F})$ in (K)."""
    OF = np.asarray(OF, float)
    return 3450.0 - 230.0 * (OF - 2.4) ** 2
