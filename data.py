"""Monthly total-return prices from Yahoo Finance's public chart endpoint.

Prices are dividend- and split-adjusted month-end closes. Everything is cached to
data/prices.csv so a run is reproducible offline; pass refresh=True to re-download.
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent
CACHE = ROOT / "data" / "prices.csv"

# Listed US tickers only. Private companies (Anthropic, OpenAI, Stripe, ...) have no
# public price history, so they cannot be backtested.
UNIVERSE = {
    # big tech
    "AAPL": "Apple", "MSFT": "Microsoft", "AMZN": "Amazon", "GOOGL": "Alphabet", "META": "Meta",
    "NVDA": "Nvidia", "TSLA": "Tesla", "NFLX": "Netflix", "ORCL": "Oracle", "ADBE": "Adobe",
    "CRM": "Salesforce", "PLTR": "Palantir", "BABA": "Alibaba",
    # payments and fintech
    "V": "Visa", "MA": "Mastercard", "PYPL": "PayPal", "XYZ": "Block", "INTU": "Intuit",
    "SHOP": "Shopify", "COIN": "Coinbase", "HOOD": "Robinhood", "SOFI": "SoFi", "AFRM": "Affirm",
    "NU": "Nu Holdings", "UPST": "Upstart",
}
BENCHMARKS = {"SPY": "S&P 500 (SPY)", "QQQ": "Nasdaq-100 (QQQ)"}
RISK_FREE = "^IRX"  # 13-week T-bill yield, annualised percent


def _epoch(date: str) -> int:
    return int(datetime.fromisoformat(date).replace(tzinfo=timezone.utc).timestamp())


def fetch(symbol: str, start: str, end: str) -> pd.Series:
    """Month-end adjusted closes for one symbol, indexed by monthly Period."""
    url = (f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
           f"?period1={_epoch(start)}&period2={_epoch(end)}&interval=1mo&events=div,splits")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                result = json.loads(resp.read())["chart"]["result"][0]
            break
        except urllib.error.URLError:  # Yahoo drops connections when rate limiting
            if attempt == 3:
                raise
            time.sleep(5 * 2 ** attempt)
    ind = result["indicators"]
    closes = (ind.get("adjclose") or [{}])[0].get("adjclose") or ind["quote"][0]["close"]
    months = pd.to_datetime(result["timestamp"], unit="s").to_period("M")
    s = pd.Series(closes, index=months, dtype=float, name=symbol).dropna()
    return s[~s.index.duplicated(keep="last")]


def load_prices(start: str = "2008-01-01", end: str = "2026-01-01", refresh: bool = False) -> pd.DataFrame:
    """Prices for the universe, benchmarks and T-bill yield; columns are tickers."""
    if CACHE.exists() and not refresh:
        df = pd.read_csv(CACHE, index_col=0)
        df.index = pd.PeriodIndex(df.index, freq="M")
        return df
    cols = {}
    for sym in [*UNIVERSE, *BENCHMARKS, RISK_FREE]:
        cols[sym] = fetch(sym, start, end)
        time.sleep(1.0)
    df = pd.DataFrame(cols).sort_index()
    df = df[df.index < pd.Period(end, "M")]
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CACHE, float_format="%.6f")
    return df
