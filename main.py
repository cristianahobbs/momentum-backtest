"""Run the backtest, write output/ and refresh the README findings.

    python main.py                  # use cached prices in data/
    python main.py --refresh        # re-download from Yahoo Finance
    python main.py --lookback 6 --top-n 3 --cost-bps 25
"""
from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

import backtest as bt
from data import BENCHMARKS, RISK_FREE, UNIVERSE, load_prices

ROOT = Path(__file__).parent
OUT = ROOT / "output"
README = ROOT / "README.md"
START, END = "2010-01", "2025-12"
GRID_LOOKBACK = [3, 6, 9, 12]
GRID_TOP_N = [3, 5, 8]
COSTS = [0, 10, 25, 50]
PLACEBO_RUNS = 1000

NAMES = {"xs": "Cross-sectional momentum", "ts": "Time-series momentum",
         "ew": "Equal-weight universe", **{k: v for k, v in BENCHMARKS.items()}}


def setup(prices: pd.DataFrame):
    rf = (prices[RISK_FREE] / 100 / 12).shift(1)  # yield known at the start of the month
    stocks = prices[list(UNIVERSE)]
    returns = stocks.pct_change(fill_method=None)
    return stocks, returns, rf


def window(df):
    return df.loc[START:END]


def run_strategies(stocks, returns, rf, cfg: bt.Config) -> dict[str, pd.DataFrame]:
    mom = bt.momentum(stocks, cfg.lookback, cfg.skip)
    weights = {
        "xs": bt.xs_weights(mom, cfg.top_n),
        "ts": bt.ts_weights(mom, bt.cumulative_rf(rf.fillna(0), cfg.lookback, cfg.skip)),
        "ew": bt.ew_weights(mom),
    }
    return {k: window(bt.simulate(w, returns, rf.fillna(0), cfg.cost_bps)) for k, w in weights.items()}


def xs_sharpe(stocks, returns, rf, rf_w, cfg: bt.Config) -> float:
    return bt.stats(run_strategies(stocks, returns, rf, cfg)["xs"]["net"], rf_w)["sharpe"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--refresh", action="store_true", help="re-download prices")
    ap.add_argument("--lookback", type=int, default=12)
    ap.add_argument("--skip", type=int, default=1)
    ap.add_argument("--top-n", type=int, default=5)
    ap.add_argument("--cost-bps", type=float, default=10.0)
    ap.add_argument("--no-readme", action="store_true", help="don't rewrite the README findings")
    args = ap.parse_args()
    cfg = bt.Config(args.lookback, args.skip, args.top_n, args.cost_bps)

    prices = load_prices(refresh=args.refresh)
    stocks, returns, rf = setup(prices)
    rf_w = window(rf).fillna(0)
    sims = run_strategies(stocks, returns, rf, cfg)

    rets = pd.DataFrame({k: s["net"] for k, s in sims.items()})
    for b in BENCHMARKS:
        rets[b] = window(prices[b].pct_change(fill_method=None))

    summary = {}
    for k in rets:
        s = bt.stats(rets[k], rf_w, sims[k]["turnover"] if k in sims else None)
        s["vs_ew"] = bt.relative(rets[k], rets["ew"], rf_w) if k in ("xs", "ts") else None
        s["vs_qqq"] = bt.relative(rets[k], rets["QQQ"], rf_w) if k != "QQQ" else None
        s["first_half_sharpe"] = bt.stats(rets[k].loc[:"2017-12"], rf_w.loc[:"2017-12"])["sharpe"]
        s["second_half_sharpe"] = bt.stats(rets[k].loc["2018-01":], rf_w.loc["2018-01":])["sharpe"]
        summary[k] = s

    calendar = (1 + rets).groupby(rets.index.year).prod() - 1

    # Robustness: every lookback x portfolio size, and the default at several cost levels
    grid = pd.DataFrame(index=GRID_LOOKBACK, columns=GRID_TOP_N, dtype=float)
    for lb in GRID_LOOKBACK:
        for n in GRID_TOP_N:
            grid.loc[lb, n] = xs_sharpe(stocks, returns, rf, rf_w, bt.Config(lb, cfg.skip, n, cfg.cost_bps))
    costs = {c: xs_sharpe(stocks, returns, rf, rf_w, bt.Config(cfg.lookback, cfg.skip, cfg.top_n, c))
             for c in COSTS}

    # Placebo: same universe, same number of names, picked at random
    mom = bt.momentum(stocks, cfg.lookback, cfg.skip)
    rng = np.random.default_rng(7)
    placebo = np.array([
        bt.stats(window(bt.simulate(bt.random_weights(mom, cfg.top_n, rng), returns, rf.fillna(0),
                                    cfg.cost_bps))["net"], rf_w)["sharpe"]
        for _ in range(PLACEBO_RUNS)])

    OUT.mkdir(exist_ok=True)
    rets.index = rets.index.astype(str)
    rets.to_csv(OUT / "monthly_returns.csv", float_format="%.6f")
    calendar.to_csv(OUT / "calendar_returns.csv", float_format="%.4f")
    grid.to_csv(OUT / "robustness_grid.csv", float_format="%.3f")
    np.savetxt(OUT / "placebo_sharpes.csv", placebo, fmt="%.4f")
    universe_start = {t: str(stocks[t].first_valid_index()) for t in UNIVERSE}
    result = {
        "config": asdict(cfg), "start": START, "end": END, "summary": summary,
        "costs": costs, "placebo": {"runs": PLACEBO_RUNS, "mean": placebo.mean(),
                                    "pct_below_xs": float((placebo < summary["xs"]["sharpe"]).mean())},
        "universe_first_price": universe_start,
        "calendar": {str(y): row.to_dict() for y, row in calendar.iterrows()},
    }
    (OUT / "results.json").write_text(json.dumps(result, indent=2, default=float), encoding="utf-8")

    print_table(summary)
    print(f"\nPlacebo: momentum Sharpe beats {result['placebo']['pct_below_xs']:.0%} of "
          f"{PLACEBO_RUNS} random {cfg.top_n}-stock portfolios (mean random Sharpe {placebo.mean():.2f})")
    print("Sharpe by cost (bps):", {c: round(v, 2) for c, v in costs.items()})
    print("\nRobustness grid (rows lookback, cols top N):\n", grid.round(2))
    if not args.no_readme and README.exists():
        update_readme(result)


def print_table(summary: dict) -> None:
    print(f"{'':28}{'CAGR':>8}{'Vol':>8}{'Sharpe':>8}{'MaxDD':>8}{'Calmar':>8}")
    for k, s in summary.items():
        print(f"{NAMES[k]:28}{s['cagr']:8.1%}{s['vol']:8.1%}{s['sharpe']:8.2f}{s['max_dd']:8.1%}{s['calmar']:8.2f}")


def pct(x: float) -> str:
    return f"{x:+.1%}" if x < 0 else f"{x:.1%}"


def findings_block(res: dict) -> str:
    s, cfg = res["summary"], res["config"]
    order = ["xs", "ts", "ew", "QQQ", "SPY"]
    head = "| | " + " | ".join(NAMES[k] for k in order) + " |\n|---|" + "---|" * len(order) + "\n"
    rows = [
        ("CAGR", lambda d: pct(d["cagr"])),
        ("Volatility", lambda d: pct(d["vol"])),
        ("Sharpe", lambda d: f"{d['sharpe']:.2f}"),
        ("Sortino", lambda d: f"{d['sortino']:.2f}"),
        ("Max drawdown", lambda d: pct(d["max_dd"])),
        ("Max drawdown window", lambda d: f"{d['max_dd_peak']} to {d['max_dd_trough']}"),
        ("Worst month", lambda d: pct(d["worst_month"])),
        ("Sharpe 2010-17 / 2018-25", lambda d: f"{d['first_half_sharpe']:.2f} / {d['second_half_sharpe']:.2f}"),
        ("Annual turnover", lambda d: f"{d['turnover']:.1f}x" if "turnover" in d else "–"),
    ]
    table = head + "\n".join(f"| {name} | " + " | ".join(f(s[k]) for k in order) + " |" for name, f in rows)
    xs, ts = s["xs"], s["ts"]
    cal = res["calendar"]
    bullets = [
        f"**Cross-sectional** ({cfg['lookback']}-{cfg['skip']} momentum, top {cfg['top_n']}, "
        f"{cfg['cost_bps']:.0f} bps costs): Sharpe {xs['sharpe']:.2f} vs {s['ew']['sharpe']:.2f} for "
        f"equal-weighting the same stocks. Alpha against that equal-weight universe is "
        f"{pct(xs['vs_ew']['alpha'])} a year (Newey-West t = {xs['vs_ew']['alpha_t']:.2f}, beta "
        f"{xs['vs_ew']['beta']:.2f}).",
        f"**Placebo test**: the momentum Sharpe beats {res['placebo']['pct_below_xs']:.0%} of "
        f"{res['placebo']['runs']:,} portfolios that hold {cfg['top_n']} random stocks from the same "
        f"universe (average random Sharpe {res['placebo']['mean']:.2f}).",
        f"**Time-series** (hold each stock only while its trailing return beats T-bills): max drawdown "
        f"{pct(ts['max_dd'])} vs {pct(s['ew']['max_dd'])} for the equal-weight universe; Sharpe "
        f"{ts['sharpe']:.2f}.",
        f"**2020 and 2022**: cross-sectional returned {pct(cal['2020']['xs'])} in 2020 and "
        f"{pct(cal['2022']['xs'])} in 2022; QQQ returned {pct(cal['2020']['QQQ'])} and "
        f"{pct(cal['2022']['QQQ'])}.",
        "**Costs**: Sharpe at " + ", ".join(f"{int(c)} bps: {v:.2f}" for c, v in res["costs"].items()) + ".",
    ]
    return (f"Backtest {res['start']} to {res['end']}, monthly rebalancing, net of costs. "
            f"Generated by `python main.py`; raw numbers are in [`output/`](output/).\n\n"
            + table + "\n\n" + "\n".join(f"- {b}" for b in bullets)
            + "\n\nHistorical simulation, not investment advice.")


def update_readme(res: dict) -> None:
    text = README.read_text(encoding="utf-8")
    new = re.sub(r"(<!-- findings:start -->\n).*?(<!-- findings:end -->)",
                 lambda m: m.group(1) + findings_block(res) + "\n" + m.group(2), text, flags=re.S)
    README.write_text(new, encoding="utf-8")


if __name__ == "__main__":
    main()
