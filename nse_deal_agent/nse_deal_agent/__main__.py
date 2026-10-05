"""CLI: python -m nse_deal_agent [options]"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import os
import sys

import pandas as pd

from .agent import Config, DealResearchAgent
from .nse_client import NSEClient, load_deals_csv
from .report import render


def _date(s: str) -> dt.date:
    return dt.datetime.strptime(s, "%Y-%m-%d").date()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="nse_deal_agent",
        description="Research NSE bulk/block deals, post-deal price action, news sentiment and F&O trade setups.",
    )
    p.add_argument("--from", dest="start", type=_date, help="start date YYYY-MM-DD (default: 7 days ago)")
    p.add_argument("--to", dest="end", type=_date, help="end date YYYY-MM-DD (default: today)")
    p.add_argument("--symbols", nargs="*", help="only analyse these NSE symbols")
    p.add_argument("--min-value", type=float, default=10.0, help="minimum gross deal value in ₹ crore (default 10)")
    p.add_argument("--top", type=int, default=10, help="analyse the N largest deals (default 10)")
    p.add_argument("--capital", type=float, default=500000, help="trading capital in ₹ for position sizing")
    p.add_argument("--risk-pct", type=float, default=1.0, help="%% of capital to risk per trade (default 1)")
    p.add_argument("--deals-csv", nargs="*", help="NSE bulk/block deal CSV(s) downloaded manually")
    p.add_argument("--bulk-only", action="store_true", help="skip block deals")
    p.add_argument("--no-ai", action="store_true", help="skip Claude news/sentiment research")
    p.add_argument("--demo", action="store_true", help="run on synthetic data (no network)")
    p.add_argument("--out", default="nse_deal_report.md", help="markdown report path")
    p.add_argument("-v", "--verbose", action="store_true")
    a = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO if a.verbose else logging.WARNING, format="%(levelname)s %(message)s")
    end = a.end or dt.date.today()
    start = a.start or end - dt.timedelta(days=7)
    use_ai = not a.no_ai and not a.demo
    if use_ai and not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
        print("note: no ANTHROPIC_API_KEY set - trying other Anthropic credentials; use --no-ai to skip news research",
              file=sys.stderr)

    cfg = Config(start=start, end=end, min_value_cr=a.min_value, top_n=a.top, symbols=a.symbols,
                 capital=a.capital, risk_pct=a.risk_pct, use_claude=use_ai, include_block=not a.bulk_only)
    if a.demo:
        from .demo import DemoNSE
        nse = DemoNSE()
    else:
        nse = NSEClient()

    deals = None
    if a.deals_csv:
        deals = pd.concat([load_deals_csv(f, "BLOCK" if "block" in f.lower() else "BULK") for f in a.deals_csv],
                          ignore_index=True)
        deals = deals[(deals["date"] >= start) & (deals["date"] <= end)]

    summary, reports = DealResearchAgent(cfg, nse).run(deals)
    md = render(summary, reports, start, end)
    with open(a.out, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"Wrote {a.out}: {len(summary)} symbol(s) with deals, {len(reports)} analysed.")
    for r in reports:
        pl = r.plan
        lvl = f" entry {pl.entry:,.2f} SL {pl.stop_loss:,.2f} T1 {pl.target1:,.2f}" if pl.entry else ""
        print(f"  {r.symbol:<12} {r.final_label:<20} {pl.direction}{lvl}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
