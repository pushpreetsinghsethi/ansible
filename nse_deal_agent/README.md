# NSE Bulk & Block Deal Research Agent

An agent that:

1. **Pulls bulk and block deals** from the NSE portal (`nseindia.com`) for a date range.
2. **Aggregates them per stock**: who bought, who sold, net ₹ crore, average deal price, and whether the
   activity is a real net buy/sell or just churn.
3. **Measures how the stock behaved after the deal**: deal-day move and volume spike, delivery %,
   1/3/5/10-day returns, change versus the deal price, and relative strength against NIFTY 50.
4. **Scores the trend** (−100 bearish … +100 bullish) from the 20/50/200-DMA, 9/21-EMA, MACD and RSI,
   the deal bias and the post-deal reaction. Every point is listed as a plain reason.
5. **Researches news and market sentiment with Claude** (live web search): why the deal happened (promoter sale,
   PE exit, FII buying, block placement…), key drivers, upcoming events, risks and headlines with links.
6. **Checks F&O eligibility** and reads the option chain: PCR, max pain, call/put OI walls (resistance/support)
   and ATM IV.
7. **Builds a trade plan**: direction, entry zone, ATR-based stop-loss adjusted to OI walls, two targets and
   R:R, a trailing rule, futures sizing (lot size, risk per lot, lots within your risk budget, approximate margin)
   and options ideas (ATM/OTM buys with premium SL/target from Black-Scholes, a debit spread and a writer view).

The final view blends 60% price/deal signals with 40% news sentiment.

## Install

```bash
cd nse_deal_agent
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...      # for the news/sentiment step
```

## Run

```bash
# Last 7 days of bulk + block deals, deals worth at least ₹10 Cr, top 10 by value
python -m nse_deal_agent

# Custom window, filters and position sizing (₹10 lakh capital, risk 1% per trade)
python -m nse_deal_agent --from 2026-09-28 --to 2026-10-03 --min-value 25 --top 5 \
    --capital 1000000 --risk-pct 1

# Only some stocks
python -m nse_deal_agent --symbols RELIANCE TATAMOTORS --from 2026-09-01

# NSE blocking you? Download the CSV from
# https://www.nseindia.com/report-detail/display-bulk-and-block-deals and pass it in
python -m nse_deal_agent --deals-csv bulk.csv block.csv --from 2026-09-28

# Skip the Claude step, or try the whole pipeline offline on synthetic data
python -m nse_deal_agent --no-ai
python -m nse_deal_agent --demo
```

The report goes to `nse_deal_report.md`, and a one-line verdict per stock is printed to the terminal.

## Notes and limitations

- NSE has no official API. The agent calls the same JSON endpoints the website uses, with browser headers and
  session cookies. NSE rate-limits and sometimes blocks cloud/datacenter IPs, so run it from a normal home or
  office connection. Install `yfinance` so price history still works if NSE refuses.
- Endpoint shapes change occasionally. Parsers accept the old and new key names, and every fetch fails soft.
- Option premiums use NSE's last traded price when available, otherwise a Black-Scholes estimate from ATM IV.
  Margin is a rough 20% estimate; your broker's SPAN + exposure figure is what counts.
- Stocks outside F&O can't be shorted overnight, so a bearish view there returns "AVOID / EXIT LONGS".

## Tests

```bash
python -m pytest -q tests
```

> **Not investment advice.** This tool is for research and education. The author is not a SEBI-registered
> adviser. Most retail F&O traders lose money. Verify every level with your broker before trading.
