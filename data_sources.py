import os, time, json, hashlib
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
    df = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")[0]
    return df.rename(columns={"Symbol":"ticker","Security":"company_name","GICS Sector":"sector"})[
        ["ticker","company_name","sector"]
    ].assign(ticker=lambda x:x.ticker.astype(str).str.replace(".","-",regex=False).str.upper())

def price_history(ticker, period="15y"):
    return yf.Ticker(ticker).history(period=period, auto_adjust=False)

def latest_price(ticker):
    h = price_history(ticker, "10d")
    return None if h.empty else float(h["Close"].dropna().iloc[-1])

def shares_info(ticker):
    try:
        return yf.Ticker(ticker).info
    except Exception:
        return {}

def throttle():
    time.sleep(.10)
