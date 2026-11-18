"""Run-mode result container and plot descriptions (UI-agnostic)."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Result:
    """Time array plus named channels on a uniform time grid (frame index == sample index)."""

    t: np.ndarray
    channels: dict[str, np.ndarray]
    meta: dict = field(default_factory=dict)      # scalars / strings (summary, params, model key)
    log: list[str] = field(default_factory=list)
    arrays: dict[str, np.ndarray] = field(default_factory=dict)      # bulk data (e.g. 3-D snapshots), not per-frame channels

    @property
    def n(self) -> int:
        return len(self.t)

    def at(self, name: str, i: int) -> float:
        return float(self.channels[name][i])


@dataclass
class Curve:
    x: np.ndarray
    y: np.ndarray
    color: str                      # theme quantity key ("pressure", ...) or a hex color
    label: str = ""
    style: str = "solid"            # "solid" | "dash"


@dataclass
class PlotSpec:
    title: str                      # LaTeX/mathtext string
    xlabel: str
    ylabel: str
    curves: list[Curve] = field(default_factory=list)
    markers: list[tuple[float, float, str]] = field(default_factory=list)   # (x, y, color)
    cursor: bool = False            # show the playback cursor (Run mode)
    cursor_channel: str | None = None   # channel whose value marks the dot at the cursor time
    cursor_color: str = "text"


@dataclass
class TabSpec:
    name: str                       # LaTeX/mathtext string
    plots: list[PlotSpec]
