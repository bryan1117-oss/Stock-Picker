import pandas as pd, numpy as np

CONCEPTS = {
"revenue":["Revenues","RevenueFromContractWithCustomerExcludingAssessedTax","SalesRevenueNet","SalesRevenueGoodsNet"],
"cost_of_revenue":["CostOfRevenue","CostOfGoodsAndServicesSold","CostOfGoodsSold"],
"operating_income":["OperatingIncomeLoss"],
"net_income":["NetIncomeLoss","ProfitLoss"],
"equity":["StockholdersEquity","StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest","PartnersCapital","MembersEquity"],
"debt":["LongTermDebtAndFinanceLeaseObligationsCurrent","LongTermDebtAndFinanceLeaseObligationsNoncurrent","LongTermDebtCurrent","LongTermDebtNoncurrent","ShortTermBorrowings","DebtCurrent","DebtLongtermAndShortterm"],
"cash":["CashAndCashEquivalentsAtCarryingValue","CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents"],
"tax_expense":["IncomeTaxExpenseBenefit"],
"eps_diluted":["EarningsPerShareDiluted"],
"shares_diluted":["WeightedAverageNumberOfDilutedSharesOutstanding"],
"cfo":["NetCashProvidedByUsedInOperatingActivities"],
"capex":["PaymentsToAcquirePropertyPlantAndEquipment","PaymentsToAcquireProductiveAssets"],
}

def concept_series(facts, candidates):
    us=facts.get("facts",{}).get("us-gaap",{})
    for name in candidates:
        concept=us.get(name)
        if not concept: continue
        units=concept.get("units",{})
        unit=next((u for u in ("USD","USD/shares","shares") if u in units), next(iter(units),None))
        if not unit: continue
        rows=[]
        for x in units[unit]:
            if x.get("form") not in ("10-K","10-K/A","20-F","40-F") or "fy" not in x or "val" not in x: continue
            if x.get("fp") not in (None,"FY"): continue
            rows.append({"fy":int(x["fy"]),"end":x.get("end"),"filed":x.get("filed"),"val":float(x["val"]),"start":x.get("start")})
        if rows:
            df=pd.DataFrame(rows).sort_values(["fy","filed"]).drop_duplicates("fy",keep="last")
            return df.reset_index(drop=True),name,unit
    return pd.DataFrame(),None,None

def histories(facts):
    return {k:concept_series(facts,v) for k,v in CONCEPTS.items()}

def val(hist,key,fy=None):
    df,_,_=hist[key]
    if df.empty:return None
    if fy is None:return float(df.iloc[-1].val)
    q=df[df.fy==fy]
    return None if q.empty else float(q.iloc[-1].val)

def cagr(hist,key,years=5):
    df,_,_=hist[key]
    if df.empty:return None
    df=df.sort_values("fy")
    cur=df.iloc[-1]; prior=df[df.fy<=cur.fy-years]
    if prior.empty or cur.val<=0 or prior.iloc[-1].val<=0:return None
    p=prior.iloc[-1]
    return (cur.val/p.val)**(1/(cur.fy-p.fy))-1

def operating_history(hist):
    rev=hist["revenue"][0]; ebit=hist["operating_income"][0]; ni=hist["net_income"][0]
    eq=hist["equity"][0]; debt=hist["debt"][0]; cash=hist["cash"][0]
    tax=hist["tax_expense"][0]; cfo=hist["cfo"][0]; capex=hist["capex"][0]
    years=sorted(set(rev.fy) & set(ebit.fy))
    rows=[]
    for fy in years:
        r=val(hist,"revenue",fy); e=val(hist,"operating_income",fy); n=val(hist,"net_income",fy)
        eqv=val(hist,"equity",fy); d=val(hist,"debt",fy) or 0; ca=val(hist,"cash",fy) or 0
        tx=val(hist,"tax_expense",fy)
        tr=max(0,min(.35,tx/e)) if tx is not None and e else .21
        nop=e*(1-tr) if e is not None else None
        ic=d+(eqv or 0)-ca if eqv is not None else None
        cfo_v=val(hist,"cfo",fy); cap_v=val(hist,"capex",fy)
        fcf=cfo_v+cap_v if cfo_v is not None and cap_v is not None else None
        rows.append({
            "fiscal_year":fy,"revenue":r,"ebit":e,"net_income":n,"equity":eqv,"debt":d,"cash":ca,
            "roic":nop/ic if nop is not None and ic and ic>0 else None,
            "roe":n/eqv if n is not None and eqv and eqv>0 else None,
            "ebit_margin":e/r if e is not None and r else None,
            "gross_margin":None, # populated below if COGS exists
            "fcf":fcf,
        })
    out=pd.DataFrame(rows)
    if not out.empty and not hist["cost_of_revenue"][0].empty:
        out["gross_margin"]=out.apply(lambda x:(x.revenue-val(hist,"cost_of_revenue",x.fiscal_year))/x.revenue if x.revenue and val(hist,"cost_of_revenue",x.fiscal_year) is not None else None,axis=1)
    return out

def calc_latest(hist,price):
    h=operating_history(hist)
    if h.empty:return {}
    latest=h.iloc[-1]
    eps=val(hist,"eps_diluted")
    pe=price/eps if price and eps and eps>0 else None
    de=latest.debt/latest.equity if latest.equity and latest.equity>0 else None
    eps_cagr=cagr(hist,"eps_diluted")
    rev_cagr=cagr(hist,"revenue")
    fcf_cagr=cagr_from_df(h,"fcf")
    shares_cagr=cagr(hist,"shares_diluted")
    fcf_yield=latest.fcf/(price*latest.shares) if "shares" in latest and latest.shares else None
    return {"price":price,"eps":eps,"pe":pe,"roic":latest.roic,"debt_to_equity":de,
            "eps_cagr":eps_cagr,"roe":latest.roe,"ebit_margin":latest.ebit_margin,
            "gross_margin":latest.gross_margin,"revenue_cagr":rev_cagr,"fcf":latest.fcf,
            "fcf_cagr":fcf_cagr,"shares_cagr":shares_cagr}

def cagr_from_df(df,key,years=5):
    x=df.dropna(subset=[key]).sort_values("fiscal_year")
    if len(x)<2:return None
    cur=x.iloc[-1]; prior=x[x.fiscal_year<=cur.fiscal_year-years]
    if prior.empty or cur[key]<=0 or prior.iloc[-1][key]<=0:return None
    p=prior.iloc[-1]
    return (cur[key]/p[key])**(1/(cur.fiscal_year-p.fiscal_year))-1

def score_core(m,c):
    checks={
    "pe_pass":m.get("pe") is not None and m["pe"]<c["pe_max"],
    "roic_pass":m.get("roic") is not None and m["roic"]>c["roic_min"],
    "de_pass":m.get("debt_to_equity") is not None and m["debt_to_equity"]<c["de_max"],
    "eps_cagr_pass":m.get("eps_cagr") is not None and m["eps_cagr"]>c["eps_cagr_min"],
    "roe_pass":m.get("roe") is not None and m["roe"]>c["roe_min"],
    "ebit_margin_pass":m.get("ebit_margin") is not None and m["ebit_margin"]>c["ebit_margin_min"],
    "gross_margin_pass":m.get("gross_margin") is not None and m["gross_margin"]>c["gross_margin_min"],
    # GATE: valuation now counts as a pass/fail criterion, not just an enhanced-score
    # nudge. Like every other check here, missing data (no 10Y valuation history) means
    # FAIL, not "skipped" -- so callers that don't fetch history (e.g. scan_sp500 with
    # include_history=False) will cap every stock at 7/8 unless history is fetched.
    "valuation_pass":m.get("pe_discount_vs_median_10y") is not None and m["pe_discount_vs_median_10y"]>c["valuation_target_discount"]}
    m.update(checks); m["score"]=sum(checks.values())

    def hi(v,t): return 0 if v is None else min(1.25,max(0,v/t))/1.25 if t else 0
    def lo(v,t): return 0 if v is None or v<=0 else min(1.25,max(0,t/v))/1.25
    # Weights re-balanced to fit an 8th factor: ebit_margin/gross_margin/debt-to-equity
    # each gave up .025 (from .10 to .075) to fund valuation at .075, so the total still
    # sums to 1.00. This is a judgment call -- retune freely.
    comps=[(hi(m.get("roic"),c["roic_min"]),.20),(hi(m.get("eps_cagr"),c["eps_cagr_min"]),.20),
           (lo(m.get("pe"),c["pe_max"]),.15),(hi(m.get("roe"),c["roe_min"]),.15),
           (hi(m.get("ebit_margin"),c["ebit_margin_min"]),.075),(hi(m.get("gross_margin"),c["gross_margin_min"]),.075),
           (lo(m.get("debt_to_equity"),c["de_max"]),.075),
           (hi(m.get("pe_discount_vs_median_10y"),c["valuation_target_discount"]),.075)]
    m["weighted_score"]=100*sum(v*w for v,w in comps)
    return m

def enhanced_score(m,c):
    # Additional factors: FCF yield and operating trend. Valuation is no longer scored
    # separately here -- it's now baked into weighted_score() itself via valuation_pass,
    # so a standalone valuation bucket here would double-count it. Rebalanced to 80/10/10.
    fcf = hi(m.get("fcf_yield"), c["fcf_yield_target"])
    trends=[m.get(x) for x in ["roic_trend","roe_trend","ebit_margin_trend","gross_margin_trend"]]
    trends=[x for x in trends if x is not None]
    trend=max(0,min(1,(sum(x>0 for x in trends)/len(trends))) if trends else 0)
    parts=[(m["weighted_score"],.80),(fcf*100,.10),(trend*100,.10)]
    available=sum(w for score,w in parts if score is not None)
    total=sum(score*w for score,w in parts if score is not None)
    m["enhanced_score"]=total/available if available else m["weighted_score"]
    return m

def hi(v,t):
    if v is None or not t:return 0
    return min(1.25,max(0,v/t))/1.25
