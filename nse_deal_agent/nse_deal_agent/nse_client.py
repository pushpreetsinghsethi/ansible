"""Thin client for the public JSON endpoints behind nseindia.com.

NSE does not publish an official API. The site's own pages call these
endpoints, which require a browser-like User-Agent and the cookies set by a
first visit to the home page. Endpoint shapes change from time to time, so
every parser here accepts both the older and the newer key names and every
fetch fails soft (returns empty data) instead of crashing the whole run.
"""

from __future__ import annotations

import csv
import datetime as dt
import io
import logging
import time
from typing import Any

import pandas as pd
import requests

log = logging.getLogger(__name__)

BASE = "https://www.nseindia.com"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/128.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nseindia.com/",
}

DEAL_COLUMNS = ["date", "symbol", "name", "client", "side", "qty", "price", "deal_type"]


def _nse_date(d: dt.date) -> str:
    return d.strftime("%d-%m-%Y")


def _parse_date(value: Any) -> dt.date | None:
    if value is None:
        return None
    if isinstance(value, dt.date):
        return value
    s = str(value).strip()
    for fmt in ("%d-%b-%Y", "%d-%m-%Y", "%Y-%m-%d", "%d %b %Y", "%d-%B-%Y"):
        try:
            return dt.datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    try:
        return pd.to_datetime(s, dayfirst=True).date()
    except (ValueError, TypeError):
        return None


def _num(value: Any) -> float:
    if value is None or value == "" or value == "-":
        return float("nan")
    try:
        return float(str(value).replace(",", ""))
    except ValueError:
        return float("nan")


def _pick(row: dict, *keys: str, default: Any = None) -> Any:
    for k in keys:
        if k in row and row[k] not in (None, ""):
            return row[k]
    return default


def normalize_deal_rows(rows: list[dict], deal_type: str) -> pd.DataFrame:
    """Map any of NSE's deal payload shapes onto DEAL_COLUMNS."""
    out = []
    for r in rows:
        out.append(
            {
                "date": _parse_date(_pick(r, "BD_DT_DATE", "date", "Date", "mTIMESTAMP")),
                "symbol": str(_pick(r, "BD_SYMBOL", "symbol", "Symbol", default="")).strip().upper(),
                "name": _pick(r, "BD_SCRIP_NAME", "name", "Security Name", default=""),
                "client": _pick(r, "BD_CLIENT_NAME", "clientName", "Client Name", default=""),
                "side": str(_pick(r, "BD_BUY_SELL", "buySell", "Buy/Sell", default="")).strip().upper(),
                "qty": _num(_pick(r, "BD_QTY_TRD", "qty", "Quantity Traded")),
                "price": _num(
                    _pick(r, "BD_TP_WATP", "watp", "Trade Price / Wght. Avg. Price", "Price")
                ),
                "deal_type": deal_type,
            }
        )
    df = pd.DataFrame(out, columns=DEAL_COLUMNS)
    if not df.empty:
        df = df[df["symbol"] != ""]
        df["side"] = df["side"].replace({"B": "BUY", "S": "SELL"})
    return df


def load_deals_csv(path: str, deal_type: str = "BULK") -> pd.DataFrame:
    """Load the CSV you can download manually from NSE's bulk/block deal page."""
    with open(path, newline="", encoding="utf-8-sig") as f:
        text = f.read()
    reader = csv.DictReader(io.StringIO(text))
    rows = [{(k or "").strip(): (v or "").strip() for k, v in row.items()} for row in reader]
    return normalize_deal_rows(rows, deal_type)


class NSEClient:
    def __init__(self, timeout: float = 15.0, pause: float = 0.6):
        self.session = requests.Session()
        self.session.headers.update(HEADERS)
        self.timeout = timeout
        self.pause = pause
        self._warmed = False

    # -- transport -----------------------------------------------------
    def _warm(self) -> None:
        if self._warmed:
            return
        for path in ("/", "/report-detail/display-bulk-and-block-deals"):
            try:
                self.session.get(BASE + path, timeout=self.timeout)
            except requests.RequestException as e:
                log.warning("NSE warm-up %s failed: %s", path, e)
        self._warmed = True

    def get_json(self, path: str, params: dict | None = None, retries: int = 3) -> Any:
        self._warm()
        last: Exception | None = None
        for attempt in range(retries):
            try:
                resp = self.session.get(BASE + path, params=params, timeout=self.timeout)
                if resp.status_code == 401 or resp.status_code == 403:
                    # Cookies expired - refresh and retry.
                    self._warmed = False
                    self._warm()
                    raise requests.HTTPError(f"{resp.status_code} from NSE")
                resp.raise_for_status()
                time.sleep(self.pause)  # stay polite; NSE rate limits aggressively
                return resp.json()
            except (requests.RequestException, ValueError) as e:
                last = e
                time.sleep(1.5 * (attempt + 1))
        log.warning("NSE request %s failed after %d tries: %s", path, retries, last)
        return None

    # -- deals ---------------------------------------------------------
    def deals(self, start: dt.date, end: dt.date, kind: str = "bulk") -> pd.DataFrame:
        """Historical bulk ("bulk") or block ("block") deals between two dates."""
        payload = self.get_json(
            f"/api/historical/{kind}-deals",
            {"from": _nse_date(start), "to": _nse_date(end)},
        )
        rows = (payload or {}).get("data", []) if isinstance(payload, dict) else []
        return normalize_deal_rows(rows, kind.upper())

    def latest_large_deals(self) -> pd.DataFrame:
        """Today's bulk + block deals snapshot (what the NSE page shows by default)."""
        payload = self.get_json("/api/snapshot-capital-market-largedeal") or {}
        frames = [
            normalize_deal_rows(payload.get("BULK_DEALS_DATA") or [], "BULK"),
            normalize_deal_rows(payload.get("BLOCK_DEALS_DATA") or [], "BLOCK"),
        ]
        return pd.concat(frames, ignore_index=True)

    # -- prices --------------------------------------------------------
    def price_history(self, symbol: str, start: dt.date, end: dt.date) -> pd.DataFrame:
        """Daily OHLCV, oldest first. Tries NSE first, then yfinance if installed."""
        df = self._nse_history(symbol, start, end)
        if df.empty:
            df = yfinance_history(symbol + ".NS", start, end)
        return df

    def _nse_history(self, symbol: str, start: dt.date, end: dt.date) -> pd.DataFrame:
        frames = []
        # NSE caps a single request at roughly a year; chunk to be safe.
        cursor = start
        while cursor <= end:
            chunk_end = min(cursor + dt.timedelta(days=180), end)
            payload = self.get_json(
                "/api/historical/cm/equity",
                {
                    "symbol": symbol,
                    "series": '["EQ"]',
                    "from": _nse_date(cursor),
                    "to": _nse_date(chunk_end),
                },
            )
            rows = (payload or {}).get("data", []) if isinstance(payload, dict) else []
            if rows:
                frames.append(
                    pd.DataFrame(
                        {
                            "date": [_parse_date(_pick(r, "CH_TIMESTAMP", "mTIMESTAMP")) for r in rows],
                            "open": [_num(r.get("CH_OPENING_PRICE")) for r in rows],
                            "high": [_num(r.get("CH_TRADE_HIGH_PRICE")) for r in rows],
                            "low": [_num(r.get("CH_TRADE_LOW_PRICE")) for r in rows],
                            "close": [_num(r.get("CH_CLOSING_PRICE")) for r in rows],
                            "volume": [_num(r.get("CH_TOT_TRADED_QTY")) for r in rows],
                            "delivery_pct": [_num(r.get("COP_DELIV_PERC")) for r in rows],
                        }
                    )
                )
            cursor = chunk_end + dt.timedelta(days=1)
        if not frames:
            return pd.DataFrame()
        df = pd.concat(frames).dropna(subset=["date", "close"])
        return df.drop_duplicates("date").sort_values("date").set_index("date")

    def index_history(self, start: dt.date, end: dt.date, index: str = "NIFTY 50") -> pd.DataFrame:
        """Daily closes of an index (benchmark for relative strength)."""
        payload = self.get_json(
            "/api/historical/indicesHistory",
            {"indexType": index, "from": _nse_date(start), "to": _nse_date(end)},
        )
        rows = []
        if isinstance(payload, dict):
            data = payload.get("data", {})
            rows = data.get("indexCloseOnlineRecords", []) if isinstance(data, dict) else data
        if rows:
            df = pd.DataFrame(
                {
                    "date": [_parse_date(_pick(r, "EOD_TIMESTAMP", "TIMESTAMP")) for r in rows],
                    "close": [_num(_pick(r, "EOD_CLOSE_INDEX_VAL", "CLOSE")) for r in rows],
                }
            ).dropna()
            return df.drop_duplicates("date").sort_values("date").set_index("date")
        return yfinance_history("^NSEI", start, end)

    # -- derivatives ---------------------------------------------------
    def fno_symbols(self) -> set[str]:
        payload = self.get_json("/api/equity-stockIndices", {"index": "SECURITIES IN F&O"}) or {}
        return {
            str(r.get("symbol", "")).upper()
            for r in payload.get("data", [])
            if r.get("symbol") and r.get("priority", 0) == 0
        }

    def lot_size(self, symbol: str) -> int | None:
        payload = self.get_json("/api/quote-derivative", {"symbol": symbol}) or {}
        for s in payload.get("stocks", []):
            lot = (
                s.get("marketDeptOrderBook", {}).get("tradeInfo", {}).get("marketLot")
                or s.get("metadata", {}).get("marketLot")
            )
            if lot:
                return int(_num(lot))
        return None

    def option_chain(self, symbol: str) -> dict:
        """Raw option chain payload ({'records': {...}}), newest endpoint first."""
        meta = self.get_json("/api/option-chain-contract-info", {"symbol": symbol}) or {}
        expiries = meta.get("expiryDates") or []
        if expiries:
            payload = self.get_json(
                "/api/option-chain-v3",
                {"type": "Equity", "symbol": symbol, "expiry": expiries[0]},
            )
            if payload and payload.get("records", {}).get("data"):
                payload["records"].setdefault("expiryDates", expiries)
                return payload
        return self.get_json("/api/option-chain-equities", {"symbol": symbol}) or {}


def yfinance_history(ticker: str, start: dt.date, end: dt.date) -> pd.DataFrame:
    try:
        import yfinance as yf  # optional dependency
    except ImportError:
        return pd.DataFrame()
    try:
        raw = yf.download(
            ticker, start=start, end=end + dt.timedelta(days=1), progress=False, auto_adjust=False
        )
    except Exception as e:  # yfinance raises a zoo of exception types
        log.warning("yfinance %s failed: %s", ticker, e)
        return pd.DataFrame()
    if raw is None or raw.empty:
        return pd.DataFrame()
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    df = raw.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]]
    df.index = [d.date() for d in df.index]
    df.index.name = "date"
    return df
