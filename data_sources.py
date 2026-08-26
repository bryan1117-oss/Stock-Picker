import os, time, json, hashlib, re
from pathlib import Path
import requests
import pandas as pd
import yfinance as yf
from config import CONFIG

CACHE = Path(CONFIG.cache_dir)
CACHE.mkdir(exist_ok=True)
HEADERS = {
    "User-Agent": CONFIG.sec_user_agent or "7-factor-stock-agent/2.0 contact@example.com",
    "Accept-Encoding": "gzip, deflate",
}

def _cached_json(url, key, max_age=86400):
    p = CACHE / f"{key}.json"
    if p.exists() and time.time()-p.stat().st_mtime < max_age:
        return json.loads(p.read_text())
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    data = r.json()
    p.write_text(json.dumps(data))
    return data

def sec_company_tickers():
    data = _cached_json("https://www.sec.gov/files/company_tickers.json","company_tickers",86400)
    return pd.DataFrame([
        {"ticker":v["ticker"].upper(),"cik":str(v["cik_str"]).zfill(10),"title":v["title"]}
        for v in data.values()
    ])

def sec_company_facts(cik):
    return _cached_json(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json", f"facts_{cik}", 21600)

def sec_submissions(cik):
    return _cached_json(f"https://data.sec.gov/submissions/CIK{cik}.json", f"sub_{cik}", 21600)

def sp500_constituents():
    # NOTE: do NOT hand the URL directly to pd.read_html(). Pandas fetches via urllib
    # with a default User-Agent that Wikipedia rejects with HTTP 403 Forbidden --
    # especially from datacenter IPs (Streamlit Cloud, AWS, etc.), even though it often
    # works from a home connection. Fetch with requests + a real User-Agent, then parse
    # the HTML string we already have in hand.
    from io import StringIO
    r = requests.get(
        "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
        headers=HEADERS, timeout=30,
    )
    r.raise_for_status()
    tables = pd.read_html(StringIO(r.text))
    # The constituents table is normally first, but Wikipedia page edits do reorder
    # tables. Pick by column signature instead of trusting the index.
    df = next((t for t in tables if {"Symbol", "Security"}.issubset(t.columns)), None)
    if df is None:
        raise ValueError(
            "Could not find the S&P 500 constituents table on the Wikipedia page. "
            "The page layout may have changed."
        )
    return df.rename(columns={"Symbol":"ticker","Security":"company_name","GICS Sector":"sector"})[
        ["ticker","company_name","sector"]
    ].assign(ticker=lambda x:x.ticker.astype(str).str.replace(".","-",regex=False).str.upper())

def _stooq_history(ticker, period="15y"):
    """Fallback price source. Yahoo periodically breaks yfinance with 401 'Invalid Crumb'
    errors (anti-scraping tokens), which takes out prices for every ticker at once.
    Stooq is free, needs no auth, and serves plain CSV. US tickers get a '.us' suffix.
    Returns a DataFrame with a Close column and DatetimeIndex, matching yfinance's shape
    closely enough for this project's use (Close only).

    NOTE: Stooq prices are split-adjusted but NOT dividend-adjusted, which matches the
    auto_adjust=False convention used elsewhere here."""
    sym = ticker.lower().replace("-", "-") + ".us"
    url = f"https://stooq.com/q/d/l/?s={sym}&i=d"
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    text = r.text.strip()
    # Stooq returns the literal string "No data" (not a 404) for unknown symbols.
    if not text or text.lower().startswith("no data") or "Date" not in text.split("\n")[0]:
        return pd.DataFrame()
    from io import StringIO
    df = pd.read_csv(StringIO(text))
    if df.empty or "Date" not in df.columns or "Close" not in df.columns:
        return pd.DataFrame()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date").sort_index()
    # Trim to the requested lookback so downstream slicing behaves like yfinance's period.
    m = re.match(r"(\d+)(y|d|mo)", str(period))
    if m:
        n, unit = int(m.group(1)), m.group(2)
        days = {"y": 365, "mo": 31, "d": 1}[unit] * n
        cutoff = pd.Timestamp.today().normalize() - pd.Timedelta(days=days)
        df = df[df.index >= cutoff]
    return df


def price_history(ticker, period="15y"):
    # Try Yahoo first (richer data), fall back to Stooq when Yahoo is broken or blocked.
    try:
        h = yf.Ticker(ticker).history(period=period, auto_adjust=False)
        if not h.empty:
            return h
    except Exception:
        pass
    try:
        return _stooq_history(ticker, period)
    except Exception:
        return pd.DataFrame()

def latest_price(ticker):
    h = price_history(ticker, "10d")
    if h.empty:
        # A 10-day window can come back empty from Stooq around holidays/weekends or
        # when its feed lags; retry with a wider window before giving up.
        h = price_history(ticker, "3mo")
    return None if h.empty else float(h["Close"].dropna().iloc[-1])

def shares_info(ticker):
    # yf.Ticker().info is the piece most often broken by Yahoo's auth changes. There is
    # no Stooq equivalent for share counts, so FCF yield will be unavailable while Yahoo
    # is down -- that's surfaced as a warning rather than silently treated as zero.
    try:
        return yf.Ticker(ticker).info
    except Exception:
        return {}

def throttle():
    time.sleep(.10)
