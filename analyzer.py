import json, numpy as np, pandas as pd
from config import CONFIG
from data_sources import sec_company_tickers, sec_company_facts, latest_price, price_history, shares_info, sp500_constituents, throttle
from metrics import histories, operating_history, calc_latest, score_core, enhanced_score

def find_company(ticker):
    t=ticker.upper().replace("-",".")
    df=sec_company_tickers()
    x=df[df.ticker==t]
    if x.empty:x=df[df.ticker==ticker.upper()]
    if x.empty:return None
    return x.iloc[0]

def historical_pe(ticker, hist):
    prices=price_history(ticker,"15y")
    eps=hist["eps_diluted"][0]
    if prices.empty or eps.empty:return pd.DataFrame()
    prices=prices.copy()
    prices.index=pd.to_datetime(prices.index).tz_localize(None)
    rows=[]
    for _,r in eps.iterrows():
        fy=int(r.fy); end=pd.Timestamp(r.end)
        window=prices[(prices.index>=end-pd.Timedelta(days=10))&(prices.index<=end+pd.Timedelta(days=20))]
        if window.empty:continue
        # FIX: pick the trading day closest to fiscal year-end, not the last day in the
        # window. The old code biased toward prices ~2-4 weeks *after* FY end.
        idx=(window.index-end).map(lambda d: abs(d.days)).argmin()
        p=float(window.Close.iloc[idx]); e=float(r.val)
        if e>0: rows.append({"fiscal_year":fy,"price":p,"eps":e,"pe":p/e})
    return pd.DataFrame(rows).drop_duplicates("fiscal_year").sort_values("fiscal_year")

def analyze_ticker(ticker,criteria,include_history=True):
    errors=[]; warnings=[]
    try:
        co=find_company(ticker)
        if co is None:return {"metrics":None,"errors":[f"{ticker}: not found in SEC ticker list."],"warnings":[],"inputs":{}, "history":None,"valuation_history":None}
        facts=sec_company_facts(co.cik)
        hist=histories(facts)
        op=operating_history(hist)
        price=latest_price(ticker)
        info=shares_info(ticker)
        shares=info.get("sharesOutstanding")

        # FIX: the old code did `op["shares"]=shares`, which overwrote EVERY historical
        # fiscal year with today's current share count. That silently corrupts anything
        # downstream computing shares_cagr (buybacks/dilution would vanish -> ~0% CAGR
        # every time). We now only attach the current share count to the most recent row,
        # and leave earlier years alone (or NaN) unless operating_history() already
        # populated real historical share counts from SEC facts.
        if not op.empty:
            if "shares" not in op.columns:
                op["shares"]=np.nan
            if shares:
                op.loc[op.index[-1],"shares"]=shares
            if op["shares"].isna().sum()>1:
                warnings.append("Historical share counts unavailable for most fiscal years; "
                                 "shares_cagr is based on incomplete data.")
            # Attach each fiscal year's actual period-end date (from the EPS filings) so
            # downstream consumers (e.g. backtest.py) can apply a real reporting lag
            # instead of guessing at a calendar year-end. Best-effort: left as NaT if the
            # EPS history doesn't cover a given fiscal year.
            try:
                eps_dates=hist["eps_diluted"][0][["fy","end"]].drop_duplicates("fy")
                eps_dates["end"]=pd.to_datetime(eps_dates["end"])
                op=op.merge(eps_dates.rename(columns={"fy":"fiscal_year","end":"period_end"}),
                             on="fiscal_year",how="left")
            except Exception:
                op["period_end"]=pd.NaT

        m=calc_latest(hist,price)
        # FCF yield uses market cap rather than enterprise value.
        if m.get("fcf") is not None and price and shares:
            m["fcf_yield"]=m["fcf"]/(price*shares)
        else:m["fcf_yield"]=None
        m["ticker"]=ticker.upper(); m["company_name"]=co.title
        # Trend = latest minus 5Y-ago metric.
        for key in ["roic","roe","ebit_margin","gross_margin"]:
            x=op.dropna(subset=[key]).sort_values("fiscal_year")
            m[key+"_trend"]=float(x.iloc[-1][key]-x.iloc[0][key]) if len(x)>=2 else None
        vh=historical_pe(ticker,hist) if include_history else pd.DataFrame()
        if not vh.empty:
            # FIX: previously compared the current year's P/E against a window that
            # included itself (tail(10) with the last row being "today"), so it was
            # really only a 9-prior-year comparison mislabeled as 10Y. Now we explicitly
            # take up to 10 PRIOR years and compare the current P/E against them.
            hist_window=vh.tail(11)  # up to 10 prior years + current
            if len(hist_window)>=2:
                current_pe=hist_window.pe.iloc[-1]
                prior=hist_window.pe.iloc[:-1]
                m["pe_percentile_10y"]=float((prior<current_pe).mean())*100
                # Median P/E over the same up-to-10 prior years. Feeds enhanced_score's
                # valuation_target_discount ("Preferred P/E (%) below 10Y median" slider),
                # which previously had no metric wired to it and did nothing.
                median_pe=float(prior.median())
                m["pe_median_10y"]=median_pe
                m["pe_discount_vs_median_10y"]=(median_pe-current_pe)/median_pe if median_pe>0 else None
            else:
                m["pe_percentile_10y"]=None
                m["pe_median_10y"]=None
                m["pe_discount_vs_median_10y"]=None
        else:
            m["pe_percentile_10y"]=None
            m["pe_median_10y"]=None
            m["pe_discount_vs_median_10y"]=None
        score_core(m,criteria); enhanced_score(m,criteria)
        missing=[k for k in ["pe","roic","debt_to_equity","eps_cagr","roe","ebit_margin","gross_margin"] if m.get(k) is None]
        if missing:warnings.append("Missing metrics: "+", ".join(missing))
        if m.get("eps_cagr") is None:warnings.append("5Y EPS CAGR unavailable or invalid, often because an endpoint EPS was non-positive.")
        if m.get("pe_percentile_10y") is None:warnings.append("10Y valuation percentile unavailable.")
        inputs={"SEC CIK":co.cik,"latest_price":price,"shares_outstanding":shares,
                "latest_revenue":op.iloc[-1].revenue if not op.empty else None,
                "latest_EBIT":op.iloc[-1].ebit if not op.empty else None,
                "latest_net_income":op.iloc[-1].net_income if not op.empty else None,
                "latest_equity":op.iloc[-1].equity if not op.empty else None,
                "latest_debt":op.iloc[-1].debt if not op.empty else None,
                "latest_cash":op.iloc[-1].cash if not op.empty else None,
                "latest_FCF":op.iloc[-1].fcf if not op.empty else None}
        return {"metrics":m,"errors":errors,"warnings":warnings,"inputs":inputs,"history":op,"valuation_history":vh}
    except Exception as e:
        return {"metrics":None,"errors":[f"{ticker}: {type(e).__name__}: {e}"],"warnings":warnings,"inputs":{},"history":None,"valuation_history":None}

def sector_normalize(rows):
    if not rows:return rows
    df=pd.DataFrame(rows)
    # Peer percentile: higher is better except leverage and P/E.
    cols_hi=["ROIC %","EPS CAGR %","ROE %","EBIT Margin %","Gross Margin %","FCF Yield %"]
    cols_lo=["P/E","D/E"]
    for col in cols_hi:
        if col in df:
            df[col+"_pct"]=df.groupby("Sector")[col].rank(pct=True)
    for col in cols_lo:
        if col in df:
            df[col+"_pct"]=df.groupby("Sector")[col].rank(pct=True,ascending=False)
    pctcols=[c for c in df.columns if c.endswith("_pct")]
    if pctcols:
        df["sector_peer_score"]=df[pctcols].mean(axis=1)*100
        # FIX: scan_sp500 builds these dicts with the display key "Enhanced score"
        # (capitalized, spaced), not "enhanced_score". Referencing the snake_case name
        # raised KeyError as soon as enough rows survived for pctcols to be non-empty.
        if "Enhanced score" in df:
            df["Enhanced score"]=.80*df["Enhanced score"]+.20*df["sector_peer_score"]
    return df.to_dict("records")

def scan_sp500(criteria,min_pass=5,progress_callback=None,include_history=True):
    # NOTE: valuation_pass (in score_core) requires 10Y price/EPS history. With
    # include_history=False every company will fail that one check and cap at 7/8,
    # since a missing metric is scored as a fail like every other criterion here. Default
    # is True for correctness; set False only if you accept that valuation is excluded
    # and want the faster scan.
    constituents=sp500_constituents(); rows=[]; errors=[]
    for i,r in constituents.reset_index(drop=True).iterrows():
        if progress_callback:progress_callback((i+1)/len(constituents),f"{i+1}/{len(constituents)}: {r.ticker}")
        out=analyze_ticker(r.ticker,criteria,include_history=include_history)
        if out["metrics"]:
            m=out["metrics"]
            if m["score"]>=min_pass:
                rows.append({"Ticker":m["ticker"],"Company":m["company_name"],"Sector":r.sector,"Score":m["score"],
                    "Core score":round(m["weighted_score"],1),"Enhanced score":round(m["enhanced_score"],1),
                    "P/E":m["pe"],"ROIC %":None if m["roic"] is None else m["roic"]*100,
                    "D/E":m["debt_to_equity"],"EPS CAGR %":None if m["eps_cagr"] is None else m["eps_cagr"]*100,
                    "ROE %":None if m["roe"] is None else m["roe"]*100,
                    "EBIT Margin %":None if m["ebit_margin"] is None else m["ebit_margin"]*100,
                    "Gross Margin %":None if m["gross_margin"] is None else m["gross_margin"]*100,
                    "FCF Yield %":None if m.get("fcf_yield") is None else m["fcf_yield"]*100})
        errors.extend(out["errors"]); throttle()
    rows=sector_normalize(rows)
    return rows,errors

def make_ai_summary(result,model="gpt-5.6"):
    # FIX: don't let a missing/invalid OPENAI_API_KEY crash the whole Streamlit page.
    try:
        from openai import OpenAI
    except ImportError:
        return "_AI interpretation unavailable: the `openai` package is not installed._"
    m=result["metrics"]
    payload={"ticker":m["ticker"],"company":m["company_name"],"core_score":m["score"],
             "core_weighted":m["weighted_score"],"enhanced":m["enhanced_score"],
             "metrics":{k:m.get(k) for k in ["pe","roic","debt_to_equity","eps_cagr","roe","ebit_margin","gross_margin","fcf_yield","pe_percentile_10y"]},
             "trends":{k:m.get(k) for k in ["roic_trend","roe_trend","ebit_margin_trend","gross_margin_trend","revenue_cagr","fcf_cagr","shares_cagr"]},
             "warnings":result["warnings"]}
    prompt=f"""Act as a conservative quantitative equity analyst. Interpret only the deterministic data below.
Do not invent facts. Separate business quality, growth, valuation and balance-sheet risk.
Identify the strongest and weakest factors, explain important trends, and explain whether the valuation
looks favorable relative to the company's own history. Mention data limitations. Do not give personalized
financial advice. Data: {json.dumps(payload,default=str)}"""
    try:
        client=OpenAI(api_key=CONFIG.openai_api_key or None)
        return client.responses.create(model=model,input=prompt).output_text
    except Exception as e:
        return f"_AI interpretation failed ({type(e).__name__}: {e}). Check your OPENAI_API_KEY and model name._"
