import datetime as dt
from types import SimpleNamespace

import pandas as pd
import pytest

from nse_deal_agent import analysis, derivatives, sentiment
from nse_deal_agent.agent import Config, DealResearchAgent
from nse_deal_agent.demo import DemoNSE
from nse_deal_agent.nse_client import load_deals_csv, normalize_deal_rows
from nse_deal_agent.report import render


def test_normalize_handles_old_and_new_payload_keys():
    old = normalize_deal_rows([{"BD_DT_DATE": "03-Oct-2026", "BD_SYMBOL": "abc", "BD_CLIENT_NAME": "X",
                                "BD_BUY_SELL": "BUY", "BD_QTY_TRD": "1,00,000", "BD_TP_WATP": "250.5"}], "BULK")
    new = normalize_deal_rows([{"date": "03-Oct-2026", "symbol": "ABC", "clientName": "X", "buySell": "S",
                                "qty": 100000, "watp": 250.5}], "BLOCK")
    assert old.iloc[0]["symbol"] == "ABC" and old.iloc[0]["qty"] == 100000
    assert old.iloc[0]["date"] == dt.date(2026, 10, 3)
    assert new.iloc[0]["side"] == "SELL"


def test_load_deals_csv(tmp_path):
    f = tmp_path / "bulk.csv"
    f.write_text("Date,Symbol,Security Name,Client Name,Buy/Sell,Quantity Traded,"
                 "Trade Price / Wght. Avg. Price,Remarks\n"
                 "03-Oct-2026,XYZ,XYZ Ltd,FUND A,BUY,\"5,00,000\",400.00,-\n")
    df = load_deals_csv(str(f))
    assert df.iloc[0]["qty"] == 500000 and df.iloc[0]["price"] == 400.0


def test_summarize_deals_bias_and_filter():
    deals = DemoNSE(dt.date(2026, 10, 5)).deals(None, None)
    s = analysis.summarize_deals(deals, min_value_cr=1)
    bias = dict(zip(s["symbol"], s["deal_bias"]))
    assert bias == {"ALPHAINFRA": "NET BUY", "BETAPHARMA": "NET SELL", "GAMMASMALL": "MIXED / CHURN"}
    assert analysis.summarize_deals(deals, min_value_cr=10_000).empty


def test_option_chain_stats():
    stats = derivatives.parse_option_chain(DemoNSE(dt.date(2026, 10, 5)).option_chain("ALPHAINFRA"))
    assert stats.call_wall > stats.spot > stats.put_wall
    assert 0.2 < stats.atm_iv < 0.4
    assert stats.pcr_oi > 0


def test_black_scholes_put_call_parity():
    s, k, t, iv, r = 100, 105, 0.1, 0.3, derivatives.RISK_FREE
    import math
    c = derivatives.bs_price(s, k, t, iv, "CE")
    p = derivatives.bs_price(s, k, t, iv, "PE")
    assert c - p == pytest.approx(s - k * math.exp(-r * t), abs=1e-6)


@pytest.mark.parametrize("score,expected", [(60, "LONG"), (-60, "SHORT"), (0, "NO TRADE")])
def test_trade_plan_levels_are_consistent(score, expected):
    nse = DemoNSE(dt.date(2026, 10, 5))
    px = analysis.add_indicators(nse.price_history("ALPHAINFRA", None, None))
    chain = derivatives.parse_option_chain(nse.option_chain("ALPHAINFRA"))
    plan = derivatives.build_trade_plan(px, score, chain, 700, True, today=dt.date(2026, 10, 5))
    assert plan.direction == expected
    if expected == "LONG":
        assert plan.stop_loss < plan.entry < plan.target1 < plan.target2
        assert plan.reward_risk_t1 >= 1
    if expected == "SHORT":
        assert plan.stop_loss > plan.entry > plan.target1 > plan.target2
        assert plan.reward_risk_t1 >= 1
    if expected != "NO TRADE":
        assert plan.futures["lot_size"] == 700 and len(plan.options) == 4


def test_no_overnight_short_for_non_fno_stock():
    nse = DemoNSE(dt.date(2026, 10, 5))
    px = analysis.add_indicators(nse.price_history("GAMMASMALL", None, None))
    plan = derivatives.build_trade_plan(px, -50, None, None, False)
    assert plan.direction == "AVOID / EXIT LONGS"


class FakeClaude:
    """Mimics client.beta.messages.create: first pauses, then answers."""

    def __init__(self):
        self.calls = []
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self.create))

    def create(self, **kw):
        self.calls.append(kw)
        if len(self.calls) == 1:
            return SimpleNamespace(stop_reason="pause_turn", content=[SimpleNamespace(type="server_tool_use")])
        text = ('Found news.\n```json\n{"sentiment": "BULLISH", "sentiment_score": 50, '
                '"summary": "FII accumulation.", "key_drivers": ["Order wins"], "headlines": []}\n```')
        return SimpleNamespace(stop_reason="end_turn", content=[SimpleNamespace(type="text", text=text)])


def test_sentiment_resumes_pause_turn_and_parses_json():
    fake = FakeClaude()
    out = sentiment.research_sentiment("ABC", "ABC Ltd", "deal", "tech", client=fake)
    assert out["sentiment"] == "BULLISH" and out["sentiment_score"] == 50
    assert len(fake.calls) == 2
    assert fake.calls[0]["model"] == sentiment.MODEL
    assert fake.calls[1]["messages"][-1]["role"] == "assistant"


def test_full_pipeline_with_sentiment_blend():
    today = dt.date.today()
    cfg = Config(start=today - dt.timedelta(days=20), end=today, min_value_cr=1)
    summary, reports = DealResearchAgent(cfg, DemoNSE(today), claude_client=FakeClaude()).run()
    assert len(reports) == 3
    alpha = next(r for r in reports if r.symbol == "ALPHAINFRA")
    assert alpha.final_score == round(0.6 * alpha.trend.score + 0.4 * 50)
    md = render(summary, reports, cfg.start, cfg.end)
    assert "## ALPHAINFRA" in md and "Trade plan" in md and "Disclaimer" in md
