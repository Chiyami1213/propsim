"""Comparison figures for the validation suite: gridded, mathtext labels, saved locally under <data folder>/dev/validation (not part of the repository)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from propsim.storage import platform_default

OUT = platform_default() / "dev" / "validation"


def new_fig(ncols: int = 1, w: float = 5.0, h: float = 3.8):
    with plt.rc_context({"mathtext.fontset": "cm"}):
        fig, axes = plt.subplots(1, ncols, figsize=(w * ncols, h), dpi=130, squeeze=False)
    for ax in axes[0]:
        ax.grid(True, which="major", lw=0.7, alpha=0.6)
        ax.minorticks_on()
        ax.grid(True, which="minor", lw=0.3, alpha=0.4)
    return fig, axes[0]


def save(fig, name: str) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    path = OUT / f"{name}.png"
    fig.savefig(path)
    plt.close(fig)
    return path
