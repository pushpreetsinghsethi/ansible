"""Option-chain analytics and futures / options trade construction."""

from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass, field

import pandas as pd

RISK_FREE = 0.065  # ~ Indian 91-day T-bill; only used for option pricing estimates


# ------------------------------------------------------- Black-Scholes
def _ncdf(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def bs_price(spot: float, strike: float, years: float, iv: float, kind: str, r: float = RISK_FREE) -> float:
    if years <= 0 or iv <= 0:
        return max(0.0, spot - strike) if kind == "CE" else max(0.0, strike - spot)
    d1 = (math.log(spot / strike) + (r + iv * iv / 2) * years) / (iv * math.sqrt(years))
    d2 = d1 - iv * math.sqrt(years)
    if kind == "CE":
        return spot * _ncdf(d1) - strike * math.exp(-r * years) * _ncdf(d2)
    return strike * math.exp(-r * years) * _ncdf(-d2) - spot * _ncdf(-d1)


def bs_delta(spot: float, strike: float, years: float, iv: float, kind: str, r: float = RISK_FREE) -> float:
    if years <= 0 or iv <= 0:
        return (1.0 if spot > strike else 0.0) if kind == "CE" else (-1.0 if spot < strike else 0.0)
    d1 = (math.log(spot / strike) + (r + iv * iv / 2) * years) / (iv * math.sqrt(years))
    return _ncdf(d1) if kind == "CE" else _ncdf(d1) - 1


# --------------------------------------------------------- option chain
@dataclass
class ChainStats:
    expiry: str
    spot: float
    pcr_oi: float
    max_pain: float
    call_wall: float           # strike with highest call OI  -> resistance
    put_wall: float            # strike with highest put OI   -> support
    call_oi_build: float       # strike with largest fresh call OI addition
    put_oi_build: float
    atm_strike: float
    atm_iv: float              # decimal, e.g. 0.28
    strike_step: float
    chain: pd.DataFrame = field(repr=False, default_factory=pd.DataFrame)

    @property
    def bias(self) -> str:
        if self.pcr_oi >= 1.2:
            return "BULLISH (heavy put writing - writers defending downside)"
        if self.pcr_oi >= 0.9:
            return "MILDLY BULLISH / NEUTRAL"
        if self.pcr_oi >= 0.6:
            return "MILDLY BEARISH / NEUTRAL"
        return "BEARISH (heavy call writing - writers capping upside)"


def parse_option_chain(payload: dict, expiry: str | None = None) -> ChainStats | None:
    rec = (payload or {}).get("records") or {}
    data = rec.get("data") or []
    if not data:
        return None
    expiries = rec.get("expiryDates") or sorted({r.get("expiryDate") or r.get("expiryDates") for r in data})
    expiry = expiry or expiries[0]
    rows = []
    for r in data:
        e = r.get("expiryDate") or r.get("expiryDates")
        if e != expiry:
            continue
        ce, pe = r.get("CE") or {}, r.get("PE") or {}
        rows.append(
            {
                "strike": float(r["strikePrice"]),
                "ce_oi": float(ce.get("openInterest") or 0),
                "ce_chg_oi": float(ce.get("changeinOpenInterest") or 0),
                "ce_iv": float(ce.get("impliedVolatility") or 0),
                "ce_ltp": float(ce.get("lastPrice") or 0),
                "pe_oi": float(pe.get("openInterest") or 0),
                "pe_chg_oi": float(pe.get("changeinOpenInterest") or 0),
                "pe_iv": float(pe.get("impliedVolatility") or 0),
                "pe_ltp": float(pe.get("lastPrice") or 0),
            }
        )
    if not rows:
        return None
    ch = pd.DataFrame(rows).sort_values("strike").reset_index(drop=True)
    spot = float(rec.get("underlyingValue") or ch.loc[(ch["ce_oi"] + ch["pe_oi"]).idxmax(), "strike"])

    # Max pain: strike where option writers' total payout is smallest.
    strikes = ch["strike"].values
    pain = [
        ((s - ch["strike"]).clip(lower=0) * ch["ce_oi"]).sum() + ((ch["strike"] - s).clip(lower=0) * ch["pe_oi"]).sum()
        for s in strikes
    ]
    max_pain = float(strikes[int(pd.Series(pain).idxmin())])
    atm_i = int((ch["strike"] - spot).abs().idxmin())
    atm = ch.loc[atm_i]
    ivs = [v for v in (atm["ce_iv"], atm["pe_iv"]) if v > 0]
    step = float(pd.Series(strikes).diff().median()) if len(strikes) > 1 else spot * 0.01
    return ChainStats(
        expiry=expiry,
        spot=spot,
        pcr_oi=ch["pe_oi"].sum() / ch["ce_oi"].sum() if ch["ce_oi"].sum() else float("nan"),
        max_pain=max_pain,
        call_wall=float(ch.loc[ch["ce_oi"].idxmax(), "strike"]),
        put_wall=float(ch.loc[ch["pe_oi"].idxmax(), "strike"]),
        call_oi_build=float(ch.loc[ch["ce_chg_oi"].idxmax(), "strike"]),
        put_oi_build=float(ch.loc[ch["pe_chg_oi"].idxmax(), "strike"]),
        atm_strike=float(atm["strike"]),
        atm_iv=(sum(ivs) / len(ivs) / 100) if ivs else 0.0,
        strike_step=step,
        chain=ch,
    )


def years_to_expiry(expiry: str, today: dt.date | None = None) -> float:
    today = today or dt.date.today()
    try:
        e = dt.datetime.strptime(expiry, "%d-%b-%Y").date()
    except ValueError:
        return 7 / 365
    return max((e - today).days, 0.5) / 365


# ----------------------------------------------------------- trade plan
@dataclass
class TradePlan:
    direction: str                      # LONG / SHORT / NO TRADE
    rationale: str
    entry: float | None = None
    entry_zone: tuple[float, float] | None = None
    stop_loss: float | None = None
    target1: float | None = None
    target2: float | None = None
    trailing_rule: str = ""
    risk_per_share: float | None = None
    reward_risk_t1: float | None = None
    futures: dict = field(default_factory=dict)
    options: list[dict] = field(default_factory=list)
    invalidation: str = ""


def build_trade_plan(df: pd.DataFrame, trend_score: int, chain: ChainStats | None, lot_size: int | None,
                     fno_eligible: bool, capital: float = 500000.0, risk_pct: float = 1.0,
                     today: dt.date | None = None) -> TradePlan:
    """ATR-based levels, nudged by option-chain walls, sized for a fixed % risk."""
    last = df.iloc[-1]
    close, atr = float(last["close"]), float(last["atr14"])
    if trend_score >= 15:
        direction = "LONG"
    elif trend_score <= -15:
        direction = "SHORT"
        if not fno_eligible:
            return TradePlan(
                "AVOID / EXIT LONGS",
                f"Trend score {trend_score} is bearish, but the stock is not in F&O so it cannot be shorted "
                "overnight (cash-market shorts must be squared off intraday). Avoid fresh longs; holders can "
                f"exit or trail a stop above {last['swing_high10']:.2f}.",
                invalidation=f"Bearish view fails on a daily close above {last['swing_high10']:.2f}",
            )
    else:
        return TradePlan(
            "NO TRADE",
            f"Trend score {trend_score} is inside the -15..+15 neutral band - no directional edge. "
            "Wait for a breakout above the 10-day high or breakdown below the 10-day low.",
            invalidation=f"Watch {last['swing_high10']:.2f} (breakout) / {last['swing_low10']:.2f} (breakdown)",
        )

    if direction == "LONG":
        entry_lo, entry_hi = close - 0.3 * atr, close + 0.2 * atr
        entry = close
        sl = min(float(last["swing_low10"]), close - 1.5 * atr)
        if chain and sl - 0.75 * atr < chain.put_wall < close:
            sl = min(sl, chain.put_wall - 0.25 * atr)          # sit below the put-writers' support
        sl = max(sl, close - 3 * atr)                         # don't let a far level blow up risk
        risk = entry - sl
        t1, t2 = entry + 1.5 * risk, entry + 3 * risk
        if chain and entry + risk <= chain.call_wall < t1:
            t1 = chain.call_wall                               # first target = call-writer resistance
    else:
        entry_lo, entry_hi = close - 0.2 * atr, close + 0.3 * atr
        entry = close
        sl = max(float(last["swing_high10"]), close + 1.5 * atr)
        if chain and close < chain.call_wall < sl + 0.75 * atr:
            sl = max(sl, chain.call_wall + 0.25 * atr)         # sit above the call-writers' resistance
        sl = min(sl, close + 3 * atr)
        risk = sl - entry
        t1, t2 = entry - 1.5 * risk, entry - 3 * risk
        if chain and t1 < chain.put_wall <= entry - risk:
            t1 = chain.put_wall                                # first target = put-writer support

    plan = TradePlan(
        direction=direction,
        rationale=f"Trend score {trend_score:+d}. ATR(14) = {atr:.2f}.",
        entry=round(entry, 2),
        entry_zone=(round(entry_lo, 2), round(entry_hi, 2)),
        stop_loss=round(sl, 2),
        target1=round(t1, 2),
        target2=round(t2, 2),
        trailing_rule=(
            "After T1, book 50% and move the stop to entry (cost). Trail the rest with the 21-EMA "
            "on a closing basis."
        ),
        risk_per_share=round(risk, 2),
        reward_risk_t1=round(abs(t1 - entry) / risk, 2) if risk else None,
        invalidation=(
            f"Daily close {'below' if direction == 'LONG' else 'above'} {sl:.2f} invalidates the view."
        ),
    )

    risk_budget = capital * risk_pct / 100
    if fno_eligible and lot_size:
        lots_by_risk = int(risk_budget // (risk * lot_size)) if risk else 0
        plan.futures = {
            "instrument": "Near-month stock future",
            "side": "BUY" if direction == "LONG" else "SELL",
            "lot_size": lot_size,
            "risk_per_lot": round(risk * lot_size, 0),
            "reward_t1_per_lot": round(abs(t1 - entry) * lot_size, 0),
            "approx_contract_value": round(entry * lot_size, 0),
            "approx_margin_(~20%)": round(entry * lot_size * 0.20, 0),
            "lots_for_risk_budget": lots_by_risk,
            "risk_budget": round(risk_budget, 0),
            "note": (
                "Futures trade at a premium/discount to spot; apply the same point distances to the "
                "futures price at entry. Margin is an estimate - check your broker's SPAN+exposure."
                + (" One lot risks more than your budget - use options or skip." if lots_by_risk == 0 else "")
            ),
        }
    if fno_eligible and chain:
        plan.options = option_ideas(direction, entry, sl, t1, chain, lot_size, risk_budget, today)
    return plan


def option_ideas(direction: str, entry: float, sl: float, t1: float, chain: ChainStats,
                 lot_size: int | None, risk_budget: float, today: dt.date | None) -> list[dict]:
    T = years_to_expiry(chain.expiry, today)
    iv = chain.atm_iv or 0.30
    kind = "CE" if direction == "LONG" else "PE"
    step = chain.strike_step
    atm = chain.atm_strike
    otm = atm + step if kind == "CE" else atm - step
    ideas = []

    # T1 is usually hit a few sessions in; price the exit assuming ~3 sessions of decay.
    T_exit = max(T - 3 / 365, 0.5 / 365)
    for label, strike in (("ATM", atm), ("1 strike OTM", otm)):
        row = chain.chain[chain.chain["strike"] == strike]
        ltp = float(row[f"{kind.lower()}_ltp"].iloc[0]) if not row.empty else 0.0
        model_now = bs_price(entry, strike, T, iv, kind)
        prem = ltp if ltp > 0 else model_now
        prem_sl = bs_price(sl, strike, T_exit, iv, kind)
        prem_t1 = bs_price(t1, strike, T_exit, iv, kind)
        # Never risk more than ~40% of premium even if the underlying stop is far.
        prem_sl = max(prem_sl, prem * 0.6)
        risk_lot = (prem - prem_sl) * (lot_size or 0)
        ideas.append(
            {
                "strategy": f"Buy {label} {kind}",
                "strike": strike,
                "expiry": chain.expiry,
                "premium_entry": round(prem, 2),
                "premium_source": "NSE LTP" if ltp > 0 else "Black-Scholes estimate",
                "premium_stop_loss": round(prem_sl, 2),
                "premium_target": round(prem_t1, 2),
                "delta": round(bs_delta(entry, strike, T, iv, kind), 2),
                "risk_per_lot": round(risk_lot, 0) if lot_size else None,
                "lots_for_risk_budget": int(risk_budget // risk_lot) if lot_size and risk_lot > 0 else None,
                "exit_rule": "Exit the option if the UNDERLYING closes beyond the stop-loss, even if the "
                             "premium stop has not hit.",
            }
        )

    # Debit spread caps cost and IV crush when IV is rich.
    far = otm + 2 * step if kind == "CE" else otm - 2 * step
    buy_p = bs_price(entry, atm, T, iv, kind)
    sell_p = bs_price(entry, far, T, iv, kind)
    width = abs(far - atm)
    ideas.append(
        {
            "strategy": f"Bull call spread (buy {atm:g} CE / sell {far:g} CE)" if kind == "CE"
            else f"Bear put spread (buy {atm:g} PE / sell {far:g} PE)",
            "expiry": chain.expiry,
            "net_debit_est": round(buy_p - sell_p, 2),
            "max_profit_est": round(width - (buy_p - sell_p), 2),
            "max_loss_est": round(buy_p - sell_p, 2),
            "when": "Prefer this over naked buying when ATM IV is high "
                    f"(current ATM IV ≈ {iv * 100:.0f}%) or expiry is under 10 days away.",
        }
    )
    # Option-writer view for experienced traders.
    sell_kind = "PE" if kind == "CE" else "CE"
    wall = chain.put_wall if sell_kind == "PE" else chain.call_wall
    ideas.append(
        {
            "strategy": f"Sell {wall:g} {sell_kind} (writer view, needs full margin)",
            "expiry": chain.expiry,
            "why": f"Highest {sell_kind} open interest sits at {wall:g} - writers are defending it.",
            "stop": f"Exit if the underlying closes {'below' if sell_kind == 'PE' else 'above'} {wall:g}.",
            "warning": "Unlimited-risk position. Only with a hedge or strict stop.",
        }
    )
    return ideas
