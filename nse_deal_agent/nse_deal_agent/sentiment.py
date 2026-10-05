"""News and market-sentiment research with Claude + live web search."""

from __future__ import annotations

import json
import logging
import re

log = logging.getLogger(__name__)

MODEL = "claude-opus-5-5"

SYSTEM = """You are an equity research analyst covering Indian listed companies (NSE).
For the stock you are given, search the web for the most recent news and commentary and judge
the market sentiment around it, with special attention to WHY large bulk/block deals happened
(promoter stake sale, PE/VC exit, FII/DII accumulation, QIP/OFS, block placement, pledge
invocation, index rebalancing, etc.) and how the market interpreted them.

Prefer primary and reputable sources: NSE/BSE corporate announcements, Moneycontrol,
Economic Times, Business Standard, Mint, CNBC-TV18, Reuters, Bloomberg. Ignore social-media tips.
Do not invent facts; if you cannot find something, say so.

Finish with ONLY a JSON object (no prose after it) of this shape:
{
  "sentiment": "BULLISH" | "MILDLY BULLISH" | "NEUTRAL" | "MILDLY BEARISH" | "BEARISH",
  "sentiment_score": <integer -100..100>,
  "deal_interpretation": "<who was on the other side of the deal and what it signals>",
  "key_drivers": ["<driver>", ...],
  "upcoming_events": ["<results date, AGM, ex-date, lock-in expiry, etc.>", ...],
  "risks": ["<risk>", ...],
  "headlines": [{"date": "YYYY-MM-DD", "title": "...", "source": "...", "url": "..."}],
  "summary": "<3-5 sentence summary>"
}"""


def _extract_json(text: str) -> dict | None:
    fenced = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    candidates = fenced + [text[text.find("{"): text.rfind("}") + 1]] if "{" in text else fenced
    for c in reversed(candidates):
        try:
            return json.loads(c)
        except json.JSONDecodeError:
            continue
    return None


def research_sentiment(symbol: str, company: str, deal_context: str, technical_context: str,
                       client=None, max_searches: int = 6) -> dict:
    """Returns the parsed JSON dict, or {"error": ...} if Claude is unavailable."""
    try:
        import anthropic
    except ImportError:
        return {"error": "anthropic package not installed - run `pip install anthropic`"}

    client = client or anthropic.Anthropic()
    user = (
        f"Stock: {company} (NSE: {symbol})\n\n"
        f"Bulk/block deal activity:\n{deal_context}\n\n"
        f"Price/technical context computed from exchange data:\n{technical_context}\n\n"
        "Research the news flow and sentiment for this stock now, explain the deal, and return the JSON."
    )
    messages = [{"role": "user", "content": user}]
    tools = [{"type": "web_search_20260209", "name": "web_search", "max_uses": max_searches}]

    try:
        response = None
        for _ in range(5):  # resume server-side tool loop if it pauses
            response = client.beta.messages.create(
                model=MODEL,
                max_tokens=16000,
                system=SYSTEM,
                messages=messages,
                tools=tools,
                output_config={"effort": "medium"},
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
            )
            if response.stop_reason != "pause_turn":
                break
            messages = [messages[0], {"role": "assistant", "content": response.content}]
    except anthropic.AuthenticationError:
        return {"error": "Anthropic authentication failed - set ANTHROPIC_API_KEY"}
    except anthropic.RateLimitError:
        return {"error": "Anthropic rate limit hit - retry later"}
    except anthropic.APIStatusError as e:
        return {"error": f"Anthropic API error {e.status_code}: {e.message}"}
    except anthropic.APIConnectionError as e:
        return {"error": f"Could not reach the Anthropic API: {e}"}

    if response.stop_reason == "refusal":
        return {"error": "Model declined the request"}
    text = "".join(b.text for b in response.content if b.type == "text")
    parsed = _extract_json(text)
    if parsed is None:
        return {"error": "Could not parse sentiment JSON", "raw": text[-2000:]}
    return parsed
