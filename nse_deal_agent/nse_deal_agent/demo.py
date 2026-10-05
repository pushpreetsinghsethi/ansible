"""Synthetic NSE data so the pipeline can be exercised offline (`--demo`)."""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

from .nse_client import normalize_deal_rows


def _bdays(end: dt.date, n: int) -> list[dt.date]:
    return [d.date() for d in pd.bdate_range(end=end, periods=n)]


def _walk(seed: int, n: int, start: float, drift: float) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rets = rng.normal(drift, 0.016, n)
    close = start * np.exp(np.cumsum(rets))
    high = close * (1 + rng.uniform(0.002, 0.018, n))
    low = close * (1 - rng.uniform(0.002, 0.018, n))
    open_ = np.clip(close * (1 + rng.normal(0, 0.006, n)), low, high)
    vol = rng.integers(400_000, 1_600_000, n).astype(float)
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": vol,
                         "delivery_pct": rng.uniform(30, 65, n)})


class DemoNSE:
    """Same interface as NSEClient, backed by deterministic random walks."""

    STOCKS = {"ALPHAINFRA": (11, 820.0, 0.0018), "BETAPHARMA": (23, 1460.0, -0.0016),
              "GAMMASMALL": (37, 96.0, 0.0002)}

    def __init__(self, today: dt.date | None = None):
        self.today = today or dt.date.today()
        self.dates = _bdays(self.today, 320)

    def deals(self, start, end, kind="bulk"):
        if kind != "bulk":
            return normalize_deal_rows([], "BLOCK")
        d = self.dates[-8].strftime("%d-%b-%Y")
        rows = [
            ("ALPHAINFRA", "Alpha Infra Ltd", "GOLDMAN SACHS (SINGAPORE) PTE", "BUY", 1_800_000, self._px("ALPHAINFRA", -8)),
            ("ALPHAINFRA", "Alpha Infra Ltd", "PROMOTER HOLDCO LLP", "SELL", 400_000, self._px("ALPHAINFRA", -8)),
            ("BETAPHARMA", "Beta Pharma Ltd", "XYZ PRIVATE EQUITY FUND IV", "SELL", 2_100_000, self._px("BETAPHARMA", -8)),
            ("BETAPHARMA", "Beta Pharma Ltd", "SBI MUTUAL FUND", "BUY", 600_000, self._px("BETAPHARMA", -8)),
            ("GAMMASMALL", "Gamma Small Caps Ltd", "RANDOM TRADER", "BUY", 3_000_000, self._px("GAMMASMALL", -8)),
            ("GAMMASMALL", "Gamma Small Caps Ltd", "RANDOM TRADER", "SELL", 2_950_000, self._px("GAMMASMALL", -8)),
        ]
        return normalize_deal_rows(
            [{"BD_DT_DATE": d, "BD_SYMBOL": s, "BD_SCRIP_NAME": n, "BD_CLIENT_NAME": c, "BD_BUY_SELL": side,
              "BD_QTY_TRD": q, "BD_TP_WATP": round(p, 2)} for s, n, c, side, q, p in rows], "BULK")

    def latest_large_deals(self):
        return self.deals(None, None)

    def _frame(self, sym):
        seed, start, drift = self.STOCKS[sym]
        df = _walk(seed, len(self.dates), start, drift)
        df.index = self.dates
        df.index.name = "date"
        df.loc[df.index[-8], "volume"] *= 4  # deal-day volume spike
        return df

    def _px(self, sym, i):
        return float(self._frame(sym)["close"].iloc[i])

    def price_history(self, symbol, start, end):
        return self._frame(symbol) if symbol in self.STOCKS else pd.DataFrame()

    def index_history(self, start, end):
        df = _walk(5, len(self.dates), 25000.0, 0.0004)[["close"]]
        df.index = self.dates
        return df[df.index >= start]

    def fno_symbols(self):
        return {"ALPHAINFRA", "BETAPHARMA"}

    def lot_size(self, symbol):
        return {"ALPHAINFRA": 700, "BETAPHARMA": 350}.get(symbol)

    def option_chain(self, symbol):
        spot = float(self._frame(symbol)["close"].iloc[-1])
        step = 20.0 if spot < 1000 else 50.0
        atm = round(spot / step) * step
        expiry = (self.today + dt.timedelta(days=(3 - self.today.weekday()) % 7 + 14)).strftime("%d-%b-%Y")
        bullish = symbol == "ALPHAINFRA"
        data = []
        for k in range(-8, 9):
            strike = atm + k * step
            ce_oi = 2000 + (6000 if k == (3 if bullish else 1) else 0) + max(0, k) * 300
            pe_oi = 2000 + (7000 if k == (-1 if bullish else -3) else 0) + max(0, -k) * 300
            data.append({"strikePrice": strike, "expiryDate": expiry,
                         "CE": {"openInterest": ce_oi, "changeinOpenInterest": ce_oi * 0.1,
                                "impliedVolatility": 28 + abs(k), "lastPrice": 0},
                         "PE": {"openInterest": pe_oi, "changeinOpenInterest": pe_oi * 0.1,
                                "impliedVolatility": 30 + abs(k), "lastPrice": 0}})
        return {"records": {"data": data, "expiryDates": [expiry], "underlyingValue": spot}}
