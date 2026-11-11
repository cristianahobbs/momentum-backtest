"""Momentum signals, portfolio simulation and performance statistics.

Timing convention: weights decided with prices up to the close of month t are held
through month t+1. Nothing in a weight at t can see a price after t; the tests check this.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Config:
    lookback: int = 12   # months of return in the signal
    skip: int = 1        # most recent months excluded (short-term reversal)
    top_n: int = 5       # names held by the cross-sectional strategy
    cost_bps: float = 10.0  # one-way cost per unit of weight traded


# --- signals ---------------------------------------------------------------
def momentum(prices: pd.DataFrame, lookback: int, skip: int) -> pd.DataFrame:
    """Return from t-lookback to t-skip. NaN unless the stock has a price at t and t-lookback."""
    mom = prices.shift(skip) / prices.shift(lookback) - 1
    return mom.where(prices.notna())


def cumulative_rf(rf: pd.Series, lookback: int, skip: int) -> pd.Series:
    """T-bill return over the same window as the momentum signal."""
    growth = (1 + rf).cumprod()
    return growth.shift(skip) / growth.shift(lookback) - 1


def xs_weights(mom: pd.DataFrame, top_n: int) -> pd.DataFrame:
    """Cross-sectional: equal weight in the top_n names by momentum."""
    rank = mom.rank(axis=1, ascending=False, method="first")
    picks = rank <= top_n
    return picks.div(picks.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)


def ts_weights(mom: pd.DataFrame, rf_window: pd.Series) -> pd.DataFrame:
    """Time-series (absolute): each eligible name gets 1/N if it beat T-bills, else its slot sits in cash."""
    eligible = mom.notna()
    n = eligible.sum(axis=1).replace(0, np.nan)
    winners = mom.gt(rf_window.fillna(0), axis=0) & eligible
    return winners.div(n, axis=0).fillna(0.0)


def ew_weights(mom: pd.DataFrame) -> pd.DataFrame:
    """Equal weight in every eligible name: the same universe with no signal."""
    eligible = mom.notna()
    return eligible.div(eligible.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)


def random_weights(mom: pd.DataFrame, top_n: int, rng: np.random.Generator) -> pd.DataFrame:
    """Placebo: top_n names picked at random from the eligible set each month."""
    noise = pd.DataFrame(rng.random(mom.shape), index=mom.index, columns=mom.columns).where(mom.notna())
    return xs_weights(noise, top_n)


# --- simulation ------------------------------------------------------------
def simulate(weights: pd.DataFrame, returns: pd.DataFrame, rf: pd.Series, cost_bps: float) -> pd.DataFrame:
    """Monthly gross/net returns and turnover. Uninvested weight earns the T-bill rate."""
    w = weights.reindex(returns.index).fillna(0.0)
    held = w.shift(1).fillna(0.0)
    r = returns.fillna(0.0)
    gross = (held * r).sum(axis=1) + (1 - held.sum(axis=1)) * rf
    drifted = held.mul(1 + r).div(1 + gross, axis=0)
    turnover = (w - drifted).abs().sum(axis=1)  # traded at the close of month t
    cost = turnover.shift(1).fillna(0.0) * cost_bps / 1e4
    return pd.DataFrame({"gross": gross, "net": gross - cost, "turnover": turnover})


# --- statistics ------------------------------------------------------------
def drawdown(r: pd.Series) -> pd.Series:
    wealth = (1 + r).cumprod()
    return wealth / wealth.cummax().clip(lower=1.0) - 1


def newey_west_t(x: np.ndarray, y: np.ndarray, lags: int = 3) -> tuple[float, float, float]:
    """OLS y = a + b x; returns (alpha, beta, t-stat of alpha) with Newey-West errors."""
    X = np.column_stack([np.ones_like(x), x])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    u = y - X @ coef
    Xu = X * u[:, None]
    S = Xu.T @ Xu
    for k in range(1, lags + 1):
        g = Xu[k:].T @ Xu[:-k]
        S += (1 - k / (lags + 1)) * (g + g.T)
    XtX_inv = np.linalg.inv(X.T @ X)
    cov = XtX_inv @ S @ XtX_inv
    return float(coef[0]), float(coef[1]), float(coef[0] / np.sqrt(cov[0, 0]))


def stats(r: pd.Series, rf: pd.Series, turnover: pd.Series | None = None) -> dict:
    ex = r - rf
    years = len(r) / 12
    dd = drawdown(r)
    trough = dd.idxmin()
    peak = (1 + r).cumprod().loc[:trough].idxmax()
    downside = np.sqrt((ex.clip(upper=0) ** 2).mean()) * np.sqrt(12)
    cagr = (1 + r).prod() ** (1 / years) - 1
    out = {
        "cagr": cagr,
        "vol": r.std() * np.sqrt(12),
        "sharpe": ex.mean() * 12 / (ex.std() * np.sqrt(12)),
        "sortino": ex.mean() * 12 / downside if downside > 0 else np.nan,
        "max_dd": dd.min(),
        "max_dd_peak": str(peak),
        "max_dd_trough": str(trough),
        "calmar": cagr / abs(dd.min()) if dd.min() < 0 else np.nan,
        "worst_month": r.min(),
        "hit_rate": (r > 0).mean(),
    }
    if turnover is not None:
        out["turnover"] = turnover.mean() * 12  # annual one-way, as a multiple of NAV
    return out


def relative(r: pd.Series, bench: pd.Series, rf: pd.Series) -> dict:
    """Alpha, beta and information ratio against a benchmark (monthly excess returns)."""
    a, b, t = newey_west_t((bench - rf).to_numpy(), (r - rf).to_numpy())
    active = r - bench
    return {"alpha": a * 12, "beta": b, "alpha_t": t,
            "info_ratio": active.mean() * 12 / (active.std() * np.sqrt(12))}
