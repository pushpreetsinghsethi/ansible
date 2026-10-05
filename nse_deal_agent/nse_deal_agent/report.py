"""Markdown rendering of a research run."""

from __future__ import annotations

import datetime as dt
import math

import pandas as pd

from .agent import StockReport

DISCLAIMER = (
    "> **Disclaimer:** Generated automatically from exchange data, rule-based technicals and AI news "
    "research. It is NOT investment advice and the author is not a SEBI-registered adviser. F&O "
    "trading carries a high risk of loss (SEBI's study: ~9 in 10 individual F&O traders lose money). "
    "Verify every level, lot size and margin with your broker before trading."
)


def _f(v, fmt="{:,.2f}", na="—"):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return na
    return fmt.format(v)


def _v(v):
    if isinstance(v, float):
        return f"{v:,.2f}" if abs(v) < 1000 else f"{v:,.0f}"
    if isinstance(v, int) and not isinstance(v, bool):
        return f"{v:,}"
    return v


def render(summary: pd.DataFrame, reports: list[StockReport], start: dt.date, end: dt.date) -> str:
    out = [f"# NSE Bulk & Block Deal Research — {start:%d %b %Y} to {end:%d %b %Y}", "",
           f"_Generated {dt.datetime.now():%d %b %Y %H:%M}_", "", DISCLAIMER, ""]
    if summary.empty:
        out.append("No bulk/block deals matched the filters.")
        return "\n".join(out)

    out += ["## Deal scan", "",
            "| Symbol | Deals | Buy ₹Cr | Sell ₹Cr | Net ₹Cr | Bias |",
            "|---|---:|---:|---:|---:|---|"]
    for r in summary.itertuples():
        out.append(f"| {r.symbol} | {r.deals} | {r.buy_value_cr:,.1f} | {r.sell_value_cr:,.1f} | "
                   f"{r.net_value_cr:+,.1f} | {r.deal_bias} |")
    out.append("")

    if reports:
        out += ["## Verdict board", "",
                "| Symbol | Since deal | Technical | News | Final view | F&O | Trade |",
                "|---|---:|---:|---|---|---|---|"]
        for r in reports:
            news = r.sentiment.get("sentiment", "—") if r.sentiment and "error" not in r.sentiment else "—"
            trade = r.plan.direction
            if r.plan.entry:
                trade += f" @ {r.plan.entry:,.2f}, SL {r.plan.stop_loss:,.2f}, T1 {r.plan.target1:,.2f}"
            out.append(f"| {r.symbol} | {_f(r.post_deal.get('change_since_deal_pct'), '{:+.1f}%')} | "
                       f"{r.trend.score:+d} | {news} | **{r.final_label}** ({r.final_score:+d}) | "
                       f"{'Yes' if r.fno_eligible else 'No'} | {trade} |")
        out.append("")

    for r in reports:
        out += _stock_section(r)
    return "\n".join(out)


def _stock_section(r: StockReport) -> list[str]:
    d, p, L = r.deal_summary, r.post_deal, r.latest
    o = [f"---", f"## {r.symbol} — {d.get('name') or ''}", "",
         f"**Final view: {r.final_label} ({r.final_score:+d}/100)**", "", "### 1. The deal", "",
         f"- {d['deals']} {d['deal_types'].lower()} deal(s) between {d['first_deal']} and {d['last_deal']}",
         f"- Bought ₹{d['buy_value_cr']:,.1f} Cr (avg {_f(d['avg_buy_price'])}), sold ₹{d['sell_value_cr']:,.1f} Cr "
         f"(avg {_f(d['avg_sell_price'])}) → **{d['deal_bias']}**"]
    if d.get("top_buyers"):
        o.append(f"- Buyers: {d['top_buyers']}")
    if d.get("top_sellers"):
        o.append(f"- Sellers: {d['top_sellers']}")

    o += ["", "### 2. How the stock behaved after the deal", ""]
    if p:
        o += [
            f"- Deal day ({p['deal_day']}): close {_f(p['deal_day_close'])}, move {_f(p.get('deal_day_change_pct'), '{:+.2f}%')}, "
            f"volume {_f(p.get('deal_day_vol_ratio'), '{:.1f}x')} of 20-day avg, delivery {_f(p.get('deal_day_delivery_pct'), '{:.0f}%')}",
            f"- Returns after deal: 1d {_f(p.get('ret_1d_pct'), '{:+.1f}%')}, 3d {_f(p.get('ret_3d_pct'), '{:+.1f}%')}, "
            f"5d {_f(p.get('ret_5d_pct'), '{:+.1f}%')}, 10d {_f(p.get('ret_10d_pct'), '{:+.1f}%')}",
            f"- Now {_f(p['last_close'])} → **{_f(p['change_since_deal_pct'], '{:+.1f}%')}** over {p['sessions_since']} sessions "
            f"(best {_f(p['max_up_since_pct'], '{:+.1f}%')}, worst {_f(p['max_down_since_pct'], '{:+.1f}%')})",
        ]
        if "vs_deal_price_pct" in p:
            o.append(f"- Vs the deal price: {_f(p['vs_deal_price_pct'], '{:+.1f}%')} "
                     f"({'big players in profit' if p['vs_deal_price_pct'] > 0 else 'big players under water'})")
        if "relative_strength_pct" in p:
            o.append(f"- NIFTY moved {_f(p['benchmark_change_pct'], '{:+.1f}%')} in the same period → relative strength "
                     f"{_f(p['relative_strength_pct'], '{:+.1f} pts')}")
    else:
        o.append("- No sessions since the deal yet.")

    o += ["", f"### 3. Technical trend: {r.trend.label} ({r.trend.score:+d})", "",
          f"Close {_f(L['close'])} | 20-DMA {_f(L['sma20'])} | 50-DMA {_f(L['sma50'])} | 200-DMA {_f(L['sma200'])} | "
          f"RSI {_f(L['rsi14'], '{:.0f}')} | ATR {_f(L['atr14'])} | 52w {_f(L['low52w'])}–{_f(L['high52w'])}", ""]
    o += [f"- {x}" for x in r.trend.reasons]

    o += ["", "### 4. News & market sentiment", ""]
    s = r.sentiment
    if s and "error" not in s:
        o.append(f"**{s.get('sentiment', '?')} ({s.get('sentiment_score', '?')})** — {s.get('summary', '')}")
        if s.get("deal_interpretation"):
            o += ["", f"_Deal interpretation:_ {s['deal_interpretation']}"]
        for title, key in (("Drivers", "key_drivers"), ("Upcoming events", "upcoming_events"), ("Risks", "risks")):
            if s.get(key):
                o += ["", f"**{title}:**"] + [f"- {x}" for x in s[key]]
        if s.get("headlines"):
            o += ["", "**Headlines:**"]
            for h in s["headlines"][:8]:
                o.append(f"- {h.get('date', '')} — [{h.get('title', '')}]({h.get('url', '')}) ({h.get('source', '')})")
    else:
        o.append("_Not run._" if not s else f"_Unavailable: {s.get('error')}_")

    o += ["", "### 5. F&O analysis", ""]
    if not r.fno_eligible:
        o.append("Not in the F&O segment — futures/options trading is not possible on this stock.")
    else:
        o.append(f"F&O eligible. Lot size: **{r.lot_size or 'n/a'}**.")
        c = r.chain
        if c:
            o += ["", f"- Expiry {c.expiry}; spot {_f(c.spot)}; ATM {c.atm_strike:g}; ATM IV {c.atm_iv * 100:.0f}%",
                  f"- PCR (OI) {c.pcr_oi:.2f} → {c.bias}",
                  f"- Max pain {c.max_pain:g}",
                  f"- Resistance (highest call OI) {c.call_wall:g}; fresh call writing at {c.call_oi_build:g}",
                  f"- Support (highest put OI) {c.put_wall:g}; fresh put writing at {c.put_oi_build:g}"]

    pl = r.plan
    o += ["", f"### 6. Trade plan: {pl.direction}", "", pl.rationale]
    if pl.entry:
        o += ["", "| | Level |", "|---|---:|",
              f"| Entry | {pl.entry:,.2f} (zone {pl.entry_zone[0]:,.2f}–{pl.entry_zone[1]:,.2f}) |",
              f"| Stop-loss | {pl.stop_loss:,.2f} (risk {pl.risk_per_share:,.2f}/share) |",
              f"| Target 1 | {pl.target1:,.2f} (R:R {pl.reward_risk_t1}) |",
              f"| Target 2 | {pl.target2:,.2f} |", "",
              f"- {pl.trailing_rule}", f"- {pl.invalidation}"]
        if pl.futures:
            o += ["", "**Futures:**"] + [f"- {k.replace('_', ' ')}: {_v(v)}" for k, v in pl.futures.items()]
        if pl.options:
            o += ["", "**Options:**"]
            for idea in pl.options:
                o.append(f"- **{idea['strategy']}** — " + "; ".join(
                    f"{k.replace('_', ' ')}: {_v(v)}" for k, v in idea.items() if k != "strategy"))
    else:
        o += ["", f"- {pl.invalidation}"]
    if r.notes:
        o += ["", "_Notes:_"] + [f"- {n}" for n in r.notes]
    o.append("")
    return o
