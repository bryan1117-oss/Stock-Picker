import numpy as np, pandas as pd
from analyzer import analyze_ticker
from data_sources import price_history, throttle

REPORTING_LAG_DAYS = 95  # approx time between fiscal year-end and 10-K availability

# In-memory, per-run cache of each ticker's full price history. FIX: the previous
# version called yf.download() fresh for every (ticker, date) pair -- with N tickers x
# M rebalance dates x 2 (entry+exit) that's a lot of redundant downloads and no
# throttling, risking Yahoo rate limits on any nontrivial backtest. We now fetch each
# ticker's history once and slice it.
_PRICE_CACHE = {}

def _price(ticker,date):
    d=pd.Timestamp(date)
    if ticker not in _PRICE_CACHE:
        try:
            h=price_history(ticker,"max")
            if not h.empty:
                h.index=pd.to_datetime(h.index)
                if h.index.tz is not None:
                    h.index=h.index.tz_localize(None)
            _PRICE_CACHE[ticker]=h
            throttle()
        except Exception:
            _PRICE_CACHE[ticker]=pd.DataFrame()
    h=_PRICE_CACHE[ticker]
    if h.empty:
        return None
    # NOTE: uses data_sources.price_history's auto_adjust=False convention (split-adjusted,
    # NOT dividend-adjusted), consistent with analyzer.py's historical P/E calc. This means
    # trade "return" below is a PRICE return, not a total return -- dividends are excluded.
    # That understates CAGR for dividend payers. Flagging rather than silently assuming
    # otherwise; switch to price_history(..., auto_adjust=True)-equivalent data if you want
    # total return instead, but then P/E and trade-return prices would use different bases.
    window=h[(h.index>=d)&(h.index<=d+pd.Timedelta(days=7))]
    if window.empty:
        return None
    return float(window["Close"].iloc[0])

def _passes_core(row,criteria,min_pass):
    """Re-check the core 8-factor screen (7 fundamental/valuation checks + the
    valuation-vs-history gate) against a single point-in-time fundamentals row. Mirrors
    score_core() in metrics.py. If metrics.py's logic changes, update this to match --
    it is intentionally self-contained so the backtest doesn't depend on
    calc_latest()/score_core(), which operate on *current* data only."""
    checks=[
        row.get("pe") is not None and row["pe"]<criteria["pe_max"],
        row.get("roic") is not None and row["roic"]>criteria["roic_min"],
        row.get("debt_to_equity") is not None and row["debt_to_equity"]<criteria["de_max"],
        row.get("eps_cagr") is not None and row["eps_cagr"]>criteria["eps_cagr_min"],
        row.get("roe") is not None and row["roe"]>criteria["roe_min"],
        row.get("ebit_margin") is not None and row["ebit_margin"]>criteria["ebit_margin_min"],
        row.get("gross_margin") is not None and row["gross_margin"]>criteria["gross_margin_min"],
        row.get("pe_discount_vs_median_10y") is not None and row["pe_discount_vs_median_10y"]>criteria["valuation_target_discount"],
    ]
    return sum(checks)>=min_pass

def _point_in_time_row(cache,ticker,d,criteria):
    """Return the most recent fiscal-year fundamentals for `ticker` that would actually
    have been PUBLICLY REPORTED as of date `d` (fiscal year-end + reporting lag), plus
    a trailing point-in-time P/E computed from the price at `d`. Returns None if no
    qualifying fiscal year or price data exists.

    NOTE: we still pass the real `criteria` dict into analyze_ticker() even though we
    ignore its current-day score_core()/enhanced_score() output (we only want `history`
    and `valuation_history`) -- metrics.py's score_core() does c["pe_max"] etc. with
    bracket access, so an empty/incomplete dict raises KeyError inside analyze_ticker's
    try/except and silently returns history=None, which would make every ticker
    invisible to the point-in-time screen without any visible error."""
    if ticker not in cache:
        out=analyze_ticker(ticker,criteria,include_history=True)
        cache[ticker]=out
    out=cache[ticker]
    op=out.get("history")
    if op is None or op.empty or "period_end" not in op.columns:
        return None
    known=op.dropna(subset=["period_end"]).copy()
    known=known[known["period_end"]+pd.Timedelta(days=REPORTING_LAG_DAYS)<=d]
    if known.empty:
        return None
    latest=known.sort_values("fiscal_year").iloc[-1]

    price=_price(ticker,d)

    # eps_diluted history lives in analyze_ticker's valuation_history (fiscal_year, eps, pe, price)
    vh=out.get("valuation_history")
    eps_val=None
    if vh is not None and not vh.empty:
        vrow=vh[vh.fiscal_year==latest.fiscal_year]
        if not vrow.empty:
            eps_val=float(vrow.iloc[0].eps)
    pe=(price/eps_val) if (price and eps_val and eps_val>0) else None

    debt_to_equity=None
    if pd.notna(latest.get("debt")) and pd.notna(latest.get("equity")) and latest.equity:
        debt_to_equity=latest.debt/latest.equity

    eps_cagr=None
    if vh is not None and not vh.empty:
        elig=vh[vh.fiscal_year<=latest.fiscal_year].sort_values("fiscal_year")
        if len(elig)>=2:
            e0,e1=elig.iloc[0].eps,elig.iloc[-1].eps
            yrs=elig.iloc[-1].fiscal_year-elig.iloc[0].fiscal_year
            if e0>0 and e1>0 and yrs>0:
                eps_cagr=(e1/e0)**(1/yrs)-1

    # Point-in-time valuation discount: median P/E of the fiscal years already known as
    # of date d (same reporting-lag filter as `known`), compared against the live P/E at
    # d (`pe`, computed above from price(d) and the most recent reported EPS). Using
    # only lag-filtered fiscal years for the median avoids leaking a not-yet-reported
    # year's P/E into the baseline.
    pe_discount=None
    if vh is not None and not vh.empty and pe is not None:
        known_fys=known.fiscal_year.tolist()
        vh_known=vh[vh.fiscal_year.isin(known_fys)].sort_values("fiscal_year").tail(10)
        if len(vh_known)>=1:
            median_pe=float(vh_known.pe.median())
            if median_pe>0:
                pe_discount=(median_pe-pe)/median_pe

    return {
        "price":price,"pe":pe,
        "roic":latest.get("roic"),"roe":latest.get("roe"),
        "ebit_margin":latest.get("ebit_margin"),"gross_margin":latest.get("gross_margin"),
        "debt_to_equity":debt_to_equity,"eps_cagr":eps_cagr,
        "pe_discount_vs_median_10y":pe_discount,
        "fiscal_year":latest.fiscal_year,
    }

def run_backtest(universe,criteria,start,end,holding_months=12,min_pass=5):
    if not {"date","ticker"}.issubset(universe.columns):
        raise ValueError("Universe CSV must contain date,ticker columns.")
    universe["date"]=pd.to_datetime(universe["date"])
    universe=universe.sort_values("date")
    dates=pd.date_range(start,end,freq=f"{holding_months}MS")
    trades=[]; curve=[{"date":pd.Timestamp(start),"equity":1.0}]
    data_warnings=[]
    cache={}

    for d in dates:
        eligible=universe[universe.date<=d].drop_duplicates("ticker")
        selected=[]
        for _,r in eligible.iterrows():
            pit=_point_in_time_row(cache,r.ticker,d,criteria)
            if pit is None:
                data_warnings.append(f"{r.ticker} @ {d.date()}: no reported fundamentals available as of this date (skipped).")
                continue
            if not _passes_core(pit,criteria,min_pass):
                continue
            selected.append((r.ticker,pit))

        for ticker,pit in selected:
            p0=pit["price"]
            p1=_price(ticker,d+pd.DateOffset(months=holding_months))
            if p0 and p1:
                trades.append({"entry_date":d,"exit_date":d+pd.DateOffset(months=holding_months),
                               "ticker":ticker,"return":p1/p0-1,"pe_at_entry":pit["pe"],
                               "core_criteria_met":True})
            else:
                data_warnings.append(f"{ticker} @ {d.date()}: missing entry/exit price data, excluded from portfolio return (not counted as a loss — verify separately if this was a delisting).")

        rets=[x["return"] for x in trades if x["entry_date"]==d]
        port=np.mean(rets) if rets else 0.0
        curve.append({"date":d+pd.DateOffset(months=holding_months),"equity":curve[-1]["equity"]*(1+port)})

    eq=pd.DataFrame(curve).drop_duplicates("date").sort_values("date")
    rets=eq.equity.pct_change().dropna()
    years=max((eq.date.iloc[-1]-eq.date.iloc[0]).days/365.25,0.01)
    cagr=eq.equity.iloc[-1]**(1/years)-1

    # FIX: rets is one observation per rebalance (every `holding_months` months), not
    # monthly. Annualizing with a fixed sqrt(12) overstated vol/Sharpe unless
    # holding_months==1. Use the actual number of rebalance periods per year.
    periods_per_year=12/holding_months
    vol=rets.std()*np.sqrt(periods_per_year) if len(rets)>1 else 0.0
    sharpe=(rets.mean()/rets.std()*np.sqrt(periods_per_year)) if rets.std()>0 else 0.0

    peak=eq.equity.cummax(); dd=eq.equity/peak-1
    return {"trades":pd.DataFrame(trades),"equity_curve":eq,"portfolio":eq,
            "cagr":cagr,"volatility":vol,"max_drawdown":dd.min(),"sharpe":sharpe,
            "warnings":data_warnings}
