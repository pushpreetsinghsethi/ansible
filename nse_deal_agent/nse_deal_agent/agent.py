"""Orchestrates the full research run: deals -> price action -> sentiment -> F&O plan."""

from __future__ import annotations

import datetime as dt
import logging
from dataclasses import dataclass, field

import pandas as pd

from . import analysis, derivatives, sentiment
from .nse_client import NSEClient

log = logging.getLogger(__name__)

TECH_WEIGHT = 0.6  # blended score = 60% price/deal signals + 40% news sentiment


@dataclass
class StockReport:
    symbol: str
    deal_summary: dict
    post_deal: dict
    trend: analysis.TrendView
    latest: dict
    fno_eligible: bool
    lot_size: int | None
    chain: derivatives.ChainStats | None
    sentiment: dict
    final_score: int
    final_label: str
    plan: derivatives.TradePlan
    notes: list[str] = field(default_factory=list)


@dataclass
class Config:
    start: dt.date
    end: dt.date
    min_value_cr: float = 10.0
    top_n: int = 10
    symbols: list[str] | None = None
    capital: float = 500000.0
    risk_pct: float = 1.0
    use_claude: bool = True
    include_block: bool = True


def label_for(score: int) -> str:
    if score >= 40:
        return "BULLISH"
    if score >= 15:
        return "MILDLY BULLISH"
    if score > -15:
        return "NEUTRAL / SIDEWAYS"
    if score > -40:
        return "MILDLY BEARISH"
    return "BEARISH"


class DealResearchAgent:
    def __init__(self, cfg: Config, nse: NSEClient | None = None, claude_client=None):
        self.cfg = cfg
        self.nse = nse or NSEClient()
        self.claude_client = claude_client

    # ------------------------------------------------------------ steps
    def collect_deals(self, deals: pd.DataFrame | None = None) -> pd.DataFrame:
        if deals is None:
            frames = [self.nse.deals(self.cfg.start, self.cfg.end, "bulk")]
            if self.cfg.include_block:
                frames.append(self.nse.deals(self.cfg.start, self.cfg.end, "block"))
            deals = pd.concat(frames, ignore_index=True)
            if deals.empty and self.cfg.end >= dt.date.today() - dt.timedelta(days=1):
                deals = self.nse.latest_large_deals()
        if self.cfg.symbols:
            deals = deals[deals["symbol"].isin([s.upper() for s in self.cfg.symbols])]
        return deals

    def analyse_symbol(self, row: dict, fno: set[str], benchmark: pd.DataFrame | None) -> StockReport | None:
        sym = row["symbol"]
        notes: list[str] = []
        hist_start = row["first_deal"] - dt.timedelta(days=400)
        px = self.nse.price_history(sym, hist_start, dt.date.today())
        if px.empty or len(px) < 30:
            log.warning("%s: not enough price history, skipping", sym)
            return None
        px = analysis.add_indicators(px)
        deal_px = row["avg_buy_price"] if row["deal_bias"] == "NET BUY" else row["avg_sell_price"]
        post = analysis.post_deal_behaviour(px, row["first_deal"], deal_px, benchmark)
        trend = analysis.trend_view(px, row["deal_bias"], post)
        last = px.iloc[-1]
        latest = {k: float(last[k]) for k in ("close", "sma20", "sma50", "sma200", "rsi14", "atr14",
                                              "macd", "macd_signal", "swing_low10", "swing_high10",
                                              "high52w", "low52w")}
        latest["date"] = px.index[-1]

        eligible = sym in fno
        lot = chain = None
        if eligible:
            lot = self.nse.lot_size(sym)
            chain = derivatives.parse_option_chain(self.nse.option_chain(sym))
            if lot is None:
                notes.append("Lot size unavailable from NSE - check the F&O contract specs.")
            if chain is None:
                notes.append("Option chain unavailable - options section skipped.")
        else:
            notes.append("Not in NSE's F&O list - only cash-market (delivery/intraday) trades are possible.")

        senti: dict = {}
        if self.cfg.use_claude:
            senti = sentiment.research_sentiment(
                sym, row.get("name") or sym,
                deal_context=_deal_context(row, post),
                technical_context=_tech_context(latest, trend, chain),
                client=self.claude_client,
            )
            if "error" in senti:
                notes.append(f"Sentiment research skipped: {senti['error']}")

        s_score = senti.get("sentiment_score") if isinstance(senti.get("sentiment_score"), (int, float)) else None
        final = trend.score if s_score is None else round(TECH_WEIGHT * trend.score + (1 - TECH_WEIGHT) * s_score)
        plan = derivatives.build_trade_plan(px, final, chain, lot, eligible, self.cfg.capital, self.cfg.risk_pct)
        return StockReport(sym, row, post, trend, latest, eligible, lot, chain, senti, final,
                           label_for(final), plan, notes)

    # -------------------------------------------------------------- run
    def run(self, deals: pd.DataFrame | None = None) -> tuple[pd.DataFrame, list[StockReport]]:
        deals = self.collect_deals(deals)
        summary = analysis.summarize_deals(deals, self.cfg.min_value_cr)
        if summary.empty:
            return summary, []
        fno = self.nse.fno_symbols()
        benchmark = self.nse.index_history(self.cfg.start - dt.timedelta(days=10), dt.date.today())
        reports = []
        for row in summary.head(self.cfg.top_n).to_dict("records"):
            log.info("Analysing %s", row["symbol"])
            r = self.analyse_symbol(row, fno, benchmark)
            if r:
                reports.append(r)
        return summary, reports


def _deal_context(row: dict, post: dict) -> str:
    lines = [
        f"Window: {row['first_deal']} to {row['last_deal']}; {row['deals']} deal(s) ({row['deal_types']}).",
        f"Bought ₹{row['buy_value_cr']:,.1f} Cr, sold ₹{row['sell_value_cr']:,.1f} Cr -> {row['deal_bias']}.",
    ]
    if row.get("top_buyers"):
        lines.append(f"Top buyers: {row['top_buyers']}")
    if row.get("top_sellers"):
        lines.append(f"Top sellers: {row['top_sellers']}")
    if post:
        lines.append(
            f"Since the deal: {post.get('change_since_deal_pct', float('nan')):+.1f}% over "
            f"{post.get('sessions_since')} sessions; deal-day move {post.get('deal_day_change_pct', float('nan')):+.1f}%."
        )
    return "\n".join(lines)


def _tech_context(latest: dict, trend: analysis.TrendView, chain) -> str:
    s = (f"Close {latest['close']:.2f}, RSI {latest['rsi14']:.0f}, 20/50/200-DMA "
         f"{latest['sma20']:.2f}/{latest['sma50']:.2f}/{latest['sma200']:.2f}. "
         f"Rule-based trend: {trend.label} ({trend.score:+d}).")
    if chain:
        s += (f" Options: PCR {chain.pcr_oi:.2f}, max pain {chain.max_pain:g}, call wall {chain.call_wall:g}, "
              f"put wall {chain.put_wall:g}, ATM IV {chain.atm_iv * 100:.0f}%.")
    return s
