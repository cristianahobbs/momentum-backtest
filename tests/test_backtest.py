"""Offline tests on synthetic prices: run with `python -m unittest` from the repo root."""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import backtest as bt  # noqa: E402

MONTHS = pd.period_range("2015-01", periods=36, freq="M")


def random_prices(n_assets: int = 6, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    r = rng.normal(0.01, 0.08, size=(len(MONTHS), n_assets))
    return pd.DataFrame(100 * np.cumprod(1 + r, axis=0), index=MONTHS,
                        columns=[f"S{i}" for i in range(n_assets)])


class SignalTests(unittest.TestCase):
    def test_momentum_is_twelve_minus_one_return(self):
        p = random_prices()
        mom = bt.momentum(p, 12, 1)
        t = MONTHS[20]
        expected = p.loc[MONTHS[19], "S0"] / p.loc[MONTHS[8], "S0"] - 1
        self.assertAlmostEqual(mom.loc[t, "S0"], expected)

    def test_no_look_ahead(self):
        """Changing prices after month t must not change any weight decided at or before t."""
        p = random_prices()
        cut = MONTHS[24]
        shocked = p.copy()
        shocked.loc[shocked.index > cut] *= np.random.default_rng(1).uniform(0.2, 5.0, size=(11, p.shape[1]))
        for build in (lambda m: bt.xs_weights(m, 2), bt.ew_weights,
                      lambda m: bt.ts_weights(m, pd.Series(0.0, index=m.index))):
            a = build(bt.momentum(p, 12, 1)).loc[:cut]
            b = build(bt.momentum(shocked, 12, 1)).loc[:cut]
            pd.testing.assert_frame_equal(a, b)

    def test_new_listing_needs_full_lookback(self):
        p = random_prices()
        p.loc[:MONTHS[9], "S0"] = np.nan  # first price in month 10
        mom = bt.momentum(p, 12, 1)
        self.assertTrue(mom.loc[:MONTHS[21], "S0"].isna().all())
        self.assertFalse(np.isnan(mom.loc[MONTHS[22], "S0"]))
        self.assertEqual(bt.xs_weights(mom, 6).loc[MONTHS[21], "S0"], 0.0)

    def test_xs_picks_the_winners(self):
        mom = pd.DataFrame([[0.5, -0.1, 0.2, np.nan]], index=MONTHS[:1], columns=list("ABCD"))
        w = bt.xs_weights(mom, 2).iloc[0]
        self.assertEqual(w.to_dict(), {"A": 0.5, "B": 0.0, "C": 0.5, "D": 0.0})

    def test_ts_leaves_losers_in_cash(self):
        mom = pd.DataFrame([[0.5, -0.1, 0.01, 0.2]], index=MONTHS[:1], columns=list("ABCD"))
        w = bt.ts_weights(mom, pd.Series([0.02], index=MONTHS[:1])).iloc[0]
        self.assertEqual(w.to_dict(), {"A": 0.25, "B": 0.0, "C": 0.0, "D": 0.25})


class SimulationTests(unittest.TestCase):
    def test_single_asset_tracks_its_return(self):
        p = random_prices(1)
        r = p.pct_change()
        w = pd.DataFrame(1.0, index=MONTHS, columns=p.columns)
        sim = bt.simulate(w, r, pd.Series(0.0, index=MONTHS), cost_bps=0)
        pd.testing.assert_series_equal(sim["gross"].iloc[1:], r["S0"].iloc[1:], check_names=False)

    def test_cash_earns_risk_free(self):
        p = random_prices(2)
        rf = pd.Series(0.003, index=MONTHS)
        w = pd.DataFrame(0.0, index=MONTHS, columns=p.columns)
        sim = bt.simulate(w, p.pct_change(), rf, cost_bps=50)
        self.assertTrue(np.allclose(sim["net"], 0.003))

    def test_turnover_and_costs(self):
        """Buy-and-hold one asset: 100% turnover on day one, none after, cost charged once."""
        p = random_prices(1)
        w = pd.DataFrame(1.0, index=MONTHS, columns=p.columns)
        sim = bt.simulate(w, p.pct_change(), pd.Series(0.0, index=MONTHS), cost_bps=20)
        self.assertAlmostEqual(sim["turnover"].iloc[0], 1.0)
        self.assertTrue(np.allclose(sim["turnover"].iloc[1:], 0.0))
        self.assertAlmostEqual(sim["gross"].iloc[1] - sim["net"].iloc[1], 0.002)

    def test_rebalancing_drifted_weights_creates_turnover(self):
        r = pd.DataFrame({"A": [0.0, 0.10], "B": [0.0, -0.10]}, index=MONTHS[:2])
        w = pd.DataFrame(0.5, index=MONTHS[:2], columns=["A", "B"])
        sim = bt.simulate(w, r, pd.Series(0.0, index=MONTHS[:2]), cost_bps=0)
        # after the move A is 55% and B 45%; back to 50/50 trades 10% of NAV
        self.assertAlmostEqual(sim["turnover"].iloc[1], 0.10)


class StatsTests(unittest.TestCase):
    def test_max_drawdown(self):
        r = pd.Series([0.10, -0.50, 0.20, 0.50], index=MONTHS[:4])
        self.assertAlmostEqual(bt.drawdown(r).min(), -0.50)

    def test_drawdown_counts_losses_from_the_start(self):
        r = pd.Series([-0.20, 0.10], index=MONTHS[:2])
        self.assertAlmostEqual(bt.drawdown(r).iloc[0], -0.20)

    def test_regression_recovers_alpha_and_beta(self):
        rng = np.random.default_rng(3)
        x = rng.normal(0, 0.05, 500)
        y = 0.004 + 1.5 * x + rng.normal(0, 0.001, 500)
        a, b, t = bt.newey_west_t(x, y)
        self.assertAlmostEqual(a, 0.004, places=3)
        self.assertAlmostEqual(b, 1.5, places=2)
        self.assertGreater(t, 10)


if __name__ == "__main__":
    unittest.main()
