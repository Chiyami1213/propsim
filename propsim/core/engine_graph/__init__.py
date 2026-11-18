"""Liquid-engine cycle sandbox core: steady operating point and design sizing of any engine graph.

``solve(graph)`` returns a :class:`~.network.Solution` with every station (pressure, temperature, flow, phase), every
component's outputs (pump head and power, turbine power and pressure ratio, chamber pressure, thrust, jacket heat load)
and the shaft balances.  Graphs follow ``propsim/data/library/schemas/graph.schema.json``.
"""
from __future__ import annotations

from .network import GraphError, Network, Solution


def solve(graph: dict, targets: list[dict] | None = None, free: list[dict] | None = None, **kw) -> Solution:
    """Steady operating point of ``graph``; with ``targets`` and ``free`` the free parameters are sized to the targets."""
    return Network(graph, targets, free).solve(**kw)


__all__ = ["GraphError", "Network", "Solution", "solve"]
