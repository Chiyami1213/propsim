"""Publication-style Matplotlib export of the current plot tab: same palette, major + minor grid, mathtext labels."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg", force=False)
import matplotlib.pyplot as plt  # noqa: E402

from ..core.result import TabSpec  # noqa: E402
from .text import apa, to_math  # noqa: E402
from .theme import theme  # noqa: E402



def _col(key: str, t: dict) -> str:
    from .theme import QUANTITY
    return key if key.startswith("#") else (QUANTITY.get(key) or t[key])


def tab_to_figure(tab: TabSpec, cursor_t: float | None = None, dpi: int = 160, tokens: dict | None = None):
    """Matplotlib figure of one plot tab. ``tokens`` is a theme token set (default: the current theme); reports pass the
    light theme so figures print well."""
    t = tokens or theme.t
    n = max(len(tab.plots), 1)
    with plt.rc_context({"mathtext.fontset": "cm", "font.family": "DejaVu Sans", "text.color": t["text"],
                         "axes.labelcolor": t["text"], "xtick.color": t["dim"], "ytick.color": t["dim"]}):
        fig, axes = plt.subplots(1, n, figsize=(4.6 * n, 3.6), dpi=dpi, facecolor=t["bg"], squeeze=False)
        for ax, ps in zip(axes[0], tab.plots):
            ax.set_facecolor(t["plot_bg"])
            for c in ps.curves:
                ax.plot(c.x, c.y, color=_col(c.color, t), lw=1.8, ls="--" if c.style == "dash" else "-",
                        label=to_math(c.label) if c.label else None)
            for x, y, col in ps.markers:
                ax.plot([x], [y], "o", color=_col(col, t), ms=6)
            if cursor_t is not None and ps.cursor:
                ax.axvline(cursor_t, color=t["text"], lw=1, ls="--", alpha=0.7)
            ax.grid(True, which="major", color=t["grid_major"], lw=0.7)
            ax.minorticks_on()
            ax.grid(True, which="minor", color=t["grid_minor"], lw=0.4)
            ax.set_xlabel(to_math(ps.xlabel), fontsize=9)
            ax.set_ylabel(to_math(ps.ylabel), fontsize=9)
            ax.set_title(to_math(apa(ps.title)), fontsize=9.5, color=t["text"])
            if any(c.label for c in ps.curves):
                ax.legend(fontsize=7, facecolor=t["panel"], edgecolor=t["edge"], labelcolor=t["text"])
            for sp in ax.spines.values():
                sp.set_color(t["grid_major"])
        fig.tight_layout()
    return fig


def save_tab_png(path: str, tab: TabSpec, cursor_t: float | None = None) -> None:
    fig = tab_to_figure(tab, cursor_t)
    fig.savefig(path, facecolor=fig.get_facecolor())
    plt.close(fig)
