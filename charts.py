"""Render README charts (light + dark PNGs) from output/. Run after main.py.

    python charts.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from backtest import drawdown  # noqa: E402

ROOT = Path(__file__).parent
OUT = ROOT / "output"
CHARTS = ROOT / "docs" / "charts"

THEMES = {
    "light": {"surface": "#fcfcfb", "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781",
              "grid": "#e1e0d9", "axis": "#c3c2b7", "s1": "#2a78d6", "s2": "#eb6834"},
    "dark": {"surface": "#1a1a19", "ink": "#f0efec", "ink2": "#c3c2b7", "muted": "#898781",
             "grid": "#2c2c2a", "axis": "#383835", "s1": "#3987e5", "s2": "#d95926"},
}
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

plt.rcParams.update({"font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"], "font.size": 10})


def series(t: dict) -> list[tuple[str, str, dict]]:
    """(column, label, line style); strategies get colour, benchmarks stay neutral."""
    return [("xs", "Cross-sectional momentum", {"color": t["s1"], "lw": 2}),
            ("ts", "Time-series momentum", {"color": t["s2"], "lw": 2}),
            ("ew", "Equal-weight universe", {"color": t["ink2"], "lw": 1.3}),
            ("QQQ", "Nasdaq-100 (QQQ)", {"color": t["muted"], "lw": 1.3, "ls": (0, (4, 3))})]


def style_axes(ax, t: dict, grid_axis: str = "y") -> None:
    ax.set_facecolor(t["surface"])
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(t["axis"])
    ax.tick_params(colors=t["muted"], length=0)
    ax.grid(axis=grid_axis, color=t["grid"], linewidth=0.8)
    ax.set_axisbelow(True)


def title(fig, ax, t: dict, text: str, sub: str) -> None:
    ax.set_title(text, loc="left", color=t["ink"], fontsize=12, fontweight="semibold", pad=24)
    ax.text(0, 1.02, sub, transform=ax.transAxes, color=t["ink2"], fontsize=9.5, va="bottom")


def legend(ax, t: dict, **kw) -> None:
    leg = ax.legend(frameon=False, labelcolor=t["ink2"], fontsize=9, **kw)
    leg.set_zorder(5)


def save(fig, name: str, theme: str) -> None:
    CHARTS.mkdir(parents=True, exist_ok=True)
    suffix = "" if theme == "light" else "-dark"
    fig.savefig(CHARTS / f"{name}{suffix}.png", dpi=200, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def load_returns() -> pd.DataFrame:
    r = pd.read_csv(OUT / "monthly_returns.csv", index_col=0)
    r.index = pd.PeriodIndex(r.index, freq="M").to_timestamp(how="end")
    return r


# ---------------------------------------------------------------------------
def equity_curve(r: pd.DataFrame, theme: str) -> None:
    t = THEMES[theme]
    fig, ax = plt.subplots(figsize=(8.6, 4.4), facecolor=t["surface"])
    style_axes(ax, t)
    for col, label, kw in reversed(series(t)):
        wealth = (1 + r[col]).cumprod()
        ax.plot(wealth.index, wealth, label=label, **kw)
        ax.annotate(f"${wealth.iloc[-1]:,.0f}", (wealth.index[-1], wealth.iloc[-1]), xytext=(6, 0),
                    textcoords="offset points", va="center", color=t["ink2"], fontsize=9)
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"${v:,.0f}"))
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], frameon=False, labelcolor=t["ink2"], fontsize=9, loc="upper left")
    title(fig, ax, t, "Growth of $1, 2010–2025",
          "Net of 10 bps trading costs, monthly rebalancing, log scale")
    save(fig, "equity_curve", theme)


def drawdowns(r: pd.DataFrame, theme: str) -> None:
    t = THEMES[theme]
    fig, ax = plt.subplots(figsize=(8.6, 3.6), facecolor=t["surface"])
    style_axes(ax, t)
    for col, label, kw in reversed(series(t)):
        ax.plot(r.index, drawdown(r[col]) * 100, label=label, **kw)
    ax.axhline(0, color=t["axis"], lw=0.8)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], frameon=False, labelcolor=t["ink2"], fontsize=9,
              loc="lower left", ncol=2)
    title(fig, ax, t, "Drawdown from previous peak",
          "2022 is the stress test: rate rises hit long-duration tech and fintech hardest")
    save(fig, "drawdowns", theme)


def robustness(grid: pd.DataFrame, theme: str) -> None:
    t = THEMES[theme]
    cmap = LinearSegmentedColormap.from_list("blue", BLUE_RAMP if theme == "light" else BLUE_RAMP[::-1])
    fig, ax = plt.subplots(figsize=(5.2, 3.6), facecolor=t["surface"])
    ax.set_facecolor(t["surface"])
    vals = grid.to_numpy(dtype=float)
    ax.imshow(vals, cmap=cmap, aspect="auto", vmin=vals.min() - 0.1, vmax=vals.max())
    ax.set_xticks(range(grid.shape[1]), [f"Top {c}" for c in grid.columns])
    ax.set_yticks(range(grid.shape[0]), [f"{i}-month" for i in grid.index])
    ax.tick_params(colors=t["ink2"], length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks(np.arange(-0.5, grid.shape[1]), minor=True)
    ax.set_yticks(np.arange(-0.5, grid.shape[0]), minor=True)
    ax.grid(which="minor", color=t["surface"], linewidth=2)
    ax.tick_params(which="minor", length=0)
    mid = (vals.min() + vals.max()) / 2
    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            dark_cell = (vals[i, j] > mid) == (theme == "light")
            ax.text(j, i, f"{vals[i, j]:.2f}", ha="center", va="center", fontsize=10,
                    color="#ffffff" if dark_cell else "#0b0b0b")
    title(fig, ax, t, "Sharpe ratio by signal and portfolio size",
          "Cross-sectional strategy, lookback × names held")
    save(fig, "robustness", theme)


def placebo(sharpes: np.ndarray, res: dict, theme: str) -> None:
    t = THEMES[theme]
    fig, ax = plt.subplots(figsize=(8.6, 3.4), facecolor=t["surface"])
    style_axes(ax, t)
    ax.hist(sharpes, bins=40, color=t["muted"], alpha=0.55, edgecolor=t["surface"], linewidth=1)
    xs, ew = res["summary"]["xs"]["sharpe"], res["summary"]["ew"]["sharpe"]
    ax.axvline(ew, color=t["ink2"], lw=1.3, ls=(0, (4, 3)))
    ax.axvline(xs, color=t["s1"], lw=2)
    top = ax.get_ylim()[1]
    ax.text(xs, top * 0.95, f"  Momentum {xs:.2f}", color=t["ink"], va="top", fontsize=9)
    ax.text(ew, top * 0.80, f"Equal weight {ew:.2f}  ", color=t["ink2"], va="top", ha="right", fontsize=9)
    ax.set_yticks([])
    n = res["config"]["top_n"]
    title(fig, ax, t, f"Momentum vs {res['placebo']['runs']:,} random {n}-stock portfolios",
          f"Sharpe ratios from the same universe; momentum beats {res['placebo']['pct_below_xs']:.0%} of them")
    save(fig, "placebo", theme)


def main() -> None:
    r = load_returns()
    grid = pd.read_csv(OUT / "robustness_grid.csv", index_col=0)
    sharpes = np.loadtxt(OUT / "placebo_sharpes.csv")
    res = json.loads((OUT / "results.json").read_text(encoding="utf-8"))
    for theme in THEMES:
        equity_curve(r, theme)
        drawdowns(r, theme)
        robustness(grid, theme)
        placebo(sharpes, res, theme)
    print(f"Charts written to {CHARTS}")


if __name__ == "__main__":
    main()
