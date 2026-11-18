"""Model controller interface: binds a parameter schema + physics to plots, a 3D scene and readouts.

The controller holds no Qt widgets other than through the ``Viewport3D`` it is handed (scene building).
Live models implement ``evaluate``; Run models provide a ``Simulation`` and are replayed frame by frame.
"""
from __future__ import annotations

import numpy as np

from ..core.result import Result, TabSpec
from ..core.schema import ParamModel

RUN_COLORS = ["#4ea1ff", "#ff9d3c", "#5cc97a", "#b07cff", "#ff5a3c", "#e6c34a"]


class ModelController:
    key: str = ""
    title: str = ""
    module: str = ""
    mode: str = "live"                      # "live" | "run"
    Params: type[ParamModel]
    fields: list[str] = []                  # LaTeX captions of the selectable 3D colour fields
    sim = None                              # Simulation for Run models

    # --- Live ---------------------------------------------------------------------------------
    def evaluate(self, p) -> dict:
        raise NotImplementedError

    def prepare_params(self, p, home) -> None:
        """Last step before a Run: models that need data from the data folder (e.g. imported thrust profiles) fill it
        into ``p``; raise ``ValueError`` with a helpful message when it is missing."""

    # --- views --------------------------------------------------------------------------------
    def reset_scene(self) -> None:
        """Forget cached mesh handles; called when the viewport is cleared on a model switch."""
        for name, empty in (("_m", {}), ("_built", None), ("_cut_built", None), ("_field_built", None)):
            if hasattr(self, name):
                setattr(self, name, empty)

    def build_scene(self, vp, p, r=None) -> None:
        """(Re)build static geometry. Called when the model, cutaway angle, or Live geometry changes."""

    def update_scene(self, vp, p, r, frame: int | None) -> None:
        """Cheap per-frame / per-parameter update of the existing meshes (points and scalars only)."""

    def tabs(self, p, r, compare: list[tuple[str, Result]] | None = None) -> list[TabSpec]:
        raise NotImplementedError

    def readout(self, p, r, frame: int | None) -> list[str]:
        return []

    def caption(self, p, r, frame: int | None) -> str:
        return ""

    def summary_line(self, p, r) -> str:
        return ""

    def export_data(self, p, r) -> dict[str, np.ndarray]:
        return {}
