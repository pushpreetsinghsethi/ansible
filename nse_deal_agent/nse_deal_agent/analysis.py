"""Deal aggregation, post-deal price behaviour and technical trend scoring."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

CRORE = 1e7


# ---------------------------------------------------------------- deals
def summarize_deals(deals: pd.DataFrame, min_value_cr: float = 0.0) -> pd.DataFrame:
    """One row per symbol: buy/sell value, net value, deal count, top clients.

    `net_value_cr` > 0 means the large players were net buyers over the window.
    """
    if deals.empty:
        return pd.DataFrame()
    d = deals.copy()
    d["qty"] = pd.to_numeric(d["qty"], errors="coerce")
    d["price"] = pd.to_numeric(d["price"], errors="coerce")
    d = d.dropna(subset=["qty", "price"])
    d["value_cr"] = d["qty"] * d["price"] / CRORE
    d["signed_value_cr"] = np.where(d["side"] == "BUY", d["value_cr"], -d["value_cr"])
    d["signed_qty"] = np.where(d["side"] == "BUY", d["qty"], -d["qty"])

    def top_clients(g: pd.DataFrame, side: str) -> str:
        s = g[g["side"] == side].groupby("client")["value_cr"].sum().nlargest(3)
        return "; ".join(f"{c} (₹{v:,.1f} Cr)" for c, v in s.items())

    rows = []
    for sym, g in d.groupby("symbol"):
        buy = g.loc[g["side"] == "BUY", "value_cr"].sum()
        sell = g.loc[g["side"] == "SELL", "value_cr"].sum()
        buy_qty = g.loc[g["side"] == "BUY", "qty"].sum()
        sell_qty = g.loc[g["side"] == "SELL", "qty"].sum()
        rows.append(
            {
                "symbol": sym,
                "name": g["name"].iloc[0],
                "first_deal": g["date"].min(),
                "last_deal": g["date"].max(),
                "deals": len(g),
                "deal_types": ",".join(sorted(g["deal_type"].unique())),
                "buy_value_cr": buy,
                "sell_value_cr": sell,
                "net_value_cr": buy - sell,
                "gross_value_cr": buy + sell,
                "net_qty": g["signed_qty"].sum(),
                "avg_buy_price": (g.loc[g["side"] == "BUY", "qty"] * g.loc[g["side"] == "BUY", "price"]).sum() / buy_qty
                if buy_qty
                else np.nan,
                "avg_sell_price": (g.loc[g["side"] == "SELL", "qty"] * g.loc[g["side"] == "SELL", "price"]).sum() / sell_qty
                if sell_qty
                else np.nan,
                "top_buyers": top_clients(g, "BUY"),
                "top_sellers": top_clients(g, "SELL"),
            }
        )
    out = pd.DataFrame(rows)
    out["deal_bias"] = np.select(
        [out["net_value_cr"] > 0.1 * out["gross_value_cr"], out["net_value_cr"] < -0.1 * out["gross_value_cr"]],
        ["NET BUY", "NET SELL"],
        "MIXED / CHURN",
    )
    out = out[out["gross_value_cr"] >= min_value_cr]
    return out.sort_values("gross_value_cr", ascending=False).reset_index(drop=True)


# ----------------------------------------------------------- indicators
def add_indicators(px: pd.DataFrame) -> pd.DataFrame:
    df = px.copy()
    c, h, l = df["close"], df["high"], df["low"]
    df["sma20"] = c.rolling(20).mean()
    df["sma50"] = c.rolling(50).mean()
    df["sma200"] = c.rolling(200).mean()
    df["ema9"] = c.ewm(span=9, adjust=False).mean()
    df["ema21"] = c.ewm(span=21, adjust=False).mean()

    delta = c.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    df["rsi14"] = (100 - 100 / (1 + rs)).fillna(100)

    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    df["atr14"] = tr.ewm(alpha=1 / 14, adjust=False).mean()
    df["vol_avg20"] = df["volume"].rolling(20).mean()
    df["vol_ratio"] = df["volume"] / df["vol_avg20"]
    df["swing_low10"] = l.rolling(10).min()
    df["swing_high10"] = h.rolling(10).max()
    df["high52w"] = h.rolling(252, min_periods=20).max()
    df["low52w"] = l.rolling(252, min_periods=20).min()
    return df


# ------------------------------------------------------ post-deal moves
def post_deal_behaviour(px: pd.DataFrame, deal_date: dt.date, deal_price: float | None,
                        benchmark: pd.DataFrame | None = None) -> dict:
    """How price and volume behaved from the deal day onward."""
    if px.empty:
        return {}
    idx = [d for d in px.index if d >= deal_date]
    if not idx:
        return {}
    d0 = idx[0]
    pos = px.index.get_loc(d0)
    base_close = px["close"].iloc[pos]
    prev_close = px["close"].iloc[pos - 1] if pos > 0 else np.nan
    last_close = px["close"].iloc[-1]
    out = {
        "deal_day": d0,
        "deal_day_close": base_close,
        "deal_day_change_pct": (base_close / prev_close - 1) * 100 if prev_close == prev_close else np.nan,
        "deal_day_vol_ratio": px["vol_ratio"].iloc[pos] if "vol_ratio" in px else np.nan,
        "deal_day_delivery_pct": px["delivery_pct"].iloc[pos] if "delivery_pct" in px else np.nan,
        "sessions_since": len(px) - 1 - pos,
        "last_close": last_close,
        "change_since_deal_pct": (last_close / base_close - 1) * 100,
        "max_up_since_pct": (px["high"].iloc[pos:].max() / base_close - 1) * 100,
        "max_down_since_pct": (px["low"].iloc[pos:].min() / base_close - 1) * 100,
    }
    if deal_price and deal_price == deal_price:
        out["vs_deal_price_pct"] = (last_close / deal_price - 1) * 100
    for n in (1, 3, 5, 10):
        if pos + n < len(px):
            out[f"ret_{n}d_pct"] = (px["close"].iloc[pos + n] / base_close - 1) * 100
    if benchmark is not None and not benchmark.empty:
        b = benchmark[benchmark.index >= d0]["close"]
        if len(b) > 1:
            out["benchmark_change_pct"] = (b.iloc[-1] / b.iloc[0] - 1) * 100
            out["relative_strength_pct"] = out["change_since_deal_pct"] - out["benchmark_change_pct"]
    return out


# --------------------------------------------------------- trend scoring
@dataclass
class TrendView:
    score: int                      # -100 (strong bearish) .. +100 (strong bullish)
    label: str
    reasons: list[str] = field(default_factory=list)


def trend_view(df: pd.DataFrame, deal_bias: str | None = None, post: dict | None = None) -> TrendView:
    """Rule-based bull/bear score from the latest bar. Transparent on purpose."""
    last = df.iloc[-1]
    pts: list[tuple[int, str]] = []

    def add(cond: bool, w: int, yes: str, no: str | None = None) -> None:
        if cond:
            pts.append((w, yes))
        elif no:
            pts.append((-w, no))

    c = last["close"]
    if not np.isnan(last["sma20"]):
        add(c > last["sma20"], 10, "Close above 20-DMA", "Close below 20-DMA")
    if not np.isnan(last["sma50"]):
        add(c > last["sma50"], 12, "Close above 50-DMA", "Close below 50-DMA")
    if not np.isnan(last["sma200"]):
        add(c > last["sma200"], 12, "Close above 200-DMA (long-term uptrend)",
            "Close below 200-DMA (long-term downtrend)")
    add(last["ema9"] > last["ema21"], 10, "9-EMA above 21-EMA (short-term momentum up)",
        "9-EMA below 21-EMA (short-term momentum down)")
    add(last["macd"] > last["macd_signal"], 10, "MACD above signal", "MACD below signal")

    rsi = last["rsi14"]
    if rsi >= 75:
        pts.append((-5, f"RSI {rsi:.0f} - overbought, chase risk"))
    elif rsi >= 55:
        pts.append((10, f"RSI {rsi:.0f} - bullish zone"))
    elif rsi <= 25:
        pts.append((5, f"RSI {rsi:.0f} - oversold, bounce possible"))
    elif rsi <= 45:
        pts.append((-10, f"RSI {rsi:.0f} - bearish zone"))
    else:
        pts.append((0, f"RSI {rsi:.0f} - neutral"))

    if deal_bias == "NET BUY":
        pts.append((12, "Large players were net BUYERS in bulk/block deals"))
    elif deal_bias == "NET SELL":
        pts.append((-12, "Large players were net SELLERS in bulk/block deals"))

    if post:
        chg = post.get("change_since_deal_pct")
        if chg is not None and chg == chg:
            add(chg > 0, 12, f"Price up {chg:.1f}% since the deal (market accepting the deal)",
                f"Price down {abs(chg):.1f}% since the deal (deal absorbed with weakness)")
        rs = post.get("relative_strength_pct")
        if rs is not None and rs == rs:
            add(rs > 0, 7, f"Outperformed NIFTY by {rs:.1f} pts since deal",
                f"Underperformed NIFTY by {abs(rs):.1f} pts since deal")
        vr = post.get("deal_day_vol_ratio")
        if vr is not None and vr == vr and vr > 2:
            pts.append((0, f"Deal-day volume {vr:.1f}x the 20-day average"))

    score = int(max(-100, min(100, sum(w for w, _ in pts))))
    if score >= 40:
        label = "BULLISH"
    elif score >= 15:
        label = "MILDLY BULLISH"
    elif score > -15:
        label = "NEUTRAL / SIDEWAYS"
    elif score > -40:
        label = "MILDLY BEARISH"
    else:
        label = "BEARISH"
    return TrendView(score, label, [r for _, r in pts])
