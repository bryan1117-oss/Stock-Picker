import streamlit as st
import pandas as pd
from analyzer import analyze_ticker, scan_sp500, make_ai_summary
from backtest import run_backtest

st.set_page_config(page_title="7-Factor Stock Agent V2", page_icon="📈", layout="wide")

st.title("📈 7-Factor Stock Analysis Agent — V2")
st.caption("Fundamental quality + valuation + trends + FCF + sector normalization")

with st.sidebar:
    st.header("Core 7-factor screen")
    pe_max = st.number_input("P/E maximum", value=20.0, min_value=0.0)
    roic_min = st.number_input("ROIC minimum (%)", value=15.0, min_value=0.0)
    de_max = st.number_input("Debt/Equity maximum", value=1.0, min_value=0.0)
    eps_cagr_min = st.number_input("5-year EPS CAGR minimum (%)", value=10.0)
    roe_min = st.number_input("ROE minimum (%)", value=15.0)
    ebit_margin_min = st.number_input("EBIT margin minimum (%)", value=10.0)
    gross_margin_min = st.number_input("Gross margin minimum (%)", value=40.0)

    st.divider()
    st.header("Enhanced ranking")
    fcf_yield_target = st.number_input("FCF yield target (%)", value=4.0, min_value=0.0)
    valuation_target = st.number_input("Preferred P/E (%) below 10Y median", value=15.0, min_value=0.0,
                                        help="Target discount to the stock's own 10-year median P/E. "
                                             "Feeds 10% of the Enhanced score via enhanced_score()'s valuation component.")
    use_ai = st.checkbox("Generate AI interpretation", value=True)
    model = st.text_input("OpenAI model", value="gpt-5.6",
                           help="Verify this against your OpenAI account's available models before running.")

criteria = {
    "pe_max": pe_max, "roic_min": roic_min/100, "de_max": de_max,
    "eps_cagr_min": eps_cagr_min/100, "roe_min": roe_min/100,
    "ebit_margin_min": ebit_margin_min/100, "gross_margin_min": gross_margin_min/100,
    "fcf_yield_target": fcf_yield_target/100,
    "valuation_target_discount": valuation_target/100,
}

tab1, tab2, tab3 = st.tabs(["Analyze ticker", "Rank S&P 500", "Backtest"])

with tab1:
    ticker = st.text_input("Ticker", value="AAPL").strip().upper()
    if st.button("Analyze", type="primary"):
        with st.spinner(f"Analyzing {ticker}..."):
            result = analyze_ticker(ticker, criteria, include_history=True)

        if result["errors"]:
            st.error("\n".join(result["errors"]))
        if result["metrics"]:
            m = result["metrics"]
            c1,c2,c3,c4,c5 = st.columns(5)
            c1.metric("Core score", f"{m['score']}/8")
            c2.metric("Core quality", f"{m['weighted_score']:.1f}/100")
            c3.metric("Enhanced score", f"{m['enhanced_score']:.1f}/100")
            c4.metric("Price", f"${m['price']:.2f}" if m["price"] else "N/A")
            c5.metric("P/E", f"{m['pe']:.1f}" if m["pe"] else "N/A")

            st.subheader("Core 8-factor screen")
            rows = []
            specs = [
                ("pe","P/E",f"< {pe_max:g}",m["pe_pass"],lambda x:f"{x:.2f}"),
                ("roic","ROIC",f"> {roic_min:g}%",m["roic_pass"],lambda x:f"{x*100:.2f}%"),
                ("debt_to_equity","Debt/Equity",f"< {de_max:g}",m["de_pass"],lambda x:f"{x:.2f}"),
                ("eps_cagr","5Y EPS CAGR",f"> {eps_cagr_min:g}%",m["eps_cagr_pass"],lambda x:f"{x*100:.2f}%"),
                ("roe","ROE",f"> {roe_min:g}%",m["roe_pass"],lambda x:f"{x*100:.2f}%"),
                ("ebit_margin","EBIT margin",f"> {ebit_margin_min:g}%",m["ebit_margin_pass"],lambda x:f"{x*100:.2f}%"),
                ("gross_margin","Gross margin",f"> {gross_margin_min:g}%",m["gross_margin_pass"],lambda x:f"{x*100:.2f}%"),
                ("pe_discount_vs_median_10y","Discount vs 10Y median P/E",f"> {valuation_target:g}%",m["valuation_pass"],lambda x:f"{x*100:.2f}%"),
            ]
            for key,label,threshold,passed,fmt in specs:
                rows.append({"Metric":label,"Actual":"N/A" if m[key] is None else fmt(m[key]),
                             "Threshold":threshold,"Result":"PASS" if passed else "FAIL"})
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

            st.subheader("V2 quantitative factors")
            # FIX: the old generic loop did `v*100` for every field including
            # "10Y P/E percentile", which is already stored on a 0-100 scale (e.g. 75.3
            # meaning the 75th percentile) — that multiplied it again into nonsense like
            # "7530.0%". Each field is now formatted explicitly by its actual unit instead
            # of guessing from the key name.
            v2_rows = [
                ("Free-cash-flow yield", m["fcf_yield"], "pct"),
                ("10Y P/E percentile", m["pe_percentile_10y"], "raw_pct"),
                ("10Y median P/E", m.get("pe_median_10y"), "ratio"),
                ("ROIC trend", m["roic_trend"], "pct"),
                ("ROE trend", m["roe_trend"], "pct"),
                ("EBIT margin trend", m["ebit_margin_trend"], "pct"),
                ("Gross margin trend", m["gross_margin_trend"], "pct"),
                ("Revenue CAGR (5Y)", m["revenue_cagr"], "pct"),
                ("FCF CAGR (5Y)", m["fcf_cagr"], "pct"),
                ("Share-count CAGR (5Y)", m["shares_cagr"], "pct"),
            ]
            def _fmt_v2(v, kind):
                if v is None: return "N/A"
                if kind == "pct": return f"{v*100:.2f}%"       # stored as a fraction (0.045 -> "4.50%")
                if kind == "raw_pct": return f"{v:.1f}%"        # already 0-100 (75.3 -> "75.3%")
                if kind == "ratio": return f"{v:.2f}"           # plain ratio, e.g. a P/E of 18.30
                return str(v)
            st.dataframe(pd.DataFrame([{"Factor":name,"Value":_fmt_v2(v,kind)} for name,v,kind in v2_rows]),
                         use_container_width=True, hide_index=True)

            if result["history"] is not None and not result["history"].empty:
                st.subheader("Historical operating trend")
                st.line_chart(result["history"].set_index("fiscal_year")[["roic","roe","ebit_margin","gross_margin"]])
                st.dataframe(result["history"], use_container_width=True, hide_index=True)

            if result["valuation_history"] is not None and not result["valuation_history"].empty:
                st.subheader("Historical valuation")
                st.line_chart(result["valuation_history"].set_index("fiscal_year")[["pe"]])
                st.dataframe(result["valuation_history"], use_container_width=True, hide_index=True)

            if result["warnings"]:
                st.warning("\n".join(result["warnings"]))

            st.subheader("Calculation provenance")
            st.json(result["inputs"])

            if use_ai:
                with st.spinner("Generating AI interpretation..."):
                    st.markdown(make_ai_summary(result, model=model))

with tab2:
    st.write("Ranks the current S&P 500 using the core screen. Sector-normalized peer scores are calculated from the companies successfully analyzed.")
    st.write("The valuation check requires 10Y price/EPS history per company, which is slower to fetch for 500 tickers. "
             "Disable it below for a faster scan, but note every company will then fail that one check "
             "and cap at 7/8 (missing data counts as a fail, same as every other criterion here).")
    fetch_valuation_history = st.checkbox("Include valuation history (required for the 8th criterion; slower)", value=True)
    min_pass_rank = st.slider("Minimum core criteria passed", 1, 8, 6, key="min_pass_rank")
    top_n = st.number_input("Return top N", min_value=5, max_value=100, value=25)
    if st.button("Run S&P 500 ranking", type="primary"):
        progress = st.progress(0)
        status = st.empty()
        rows, errors = scan_sp500(
            criteria, min_pass=min_pass_rank, include_history=fetch_valuation_history,
            progress_callback=lambda p,s:(progress.progress(p), status.write(s))
        )
        if rows:
            df = pd.DataFrame(rows).sort_values(["enhanced_score","weighted_score"], ascending=False).head(int(top_n))
            st.success(f"Ranked {len(df)} qualifying companies.")
            st.dataframe(df, use_container_width=True, hide_index=True)
            st.download_button("Download ranked CSV", df.to_csv(index=False), "sp500_v2_ranking.csv", "text/csv")
        if errors:
            with st.expander(f"{len(errors)} data issues"):
                st.write("\n".join(errors))

with tab3:
    st.write("Backtesting requires a point-in-time universe to avoid survivorship bias. V2 supports a CSV of historical constituents.")
    st.write("At each rebalance date, the backtest now re-applies the core 8-factor screen using fundamentals that "
             "would actually have been reported by that date (fiscal year-end + a ~95-day filing lag) — it no longer "
             "just buys every ticker in the uploaded CSV regardless of whether it passed the screen.")
    st.code("date,ticker\n2018-01-01,AAPL\n2018-01-01,MSFT\n2019-01-01,AAPL")
    universe_file = st.file_uploader("Historical universe CSV", type=["csv"])
    start = st.date_input("Backtest start", value=pd.Timestamp("2016-01-01"))
    end = st.date_input("Backtest end", value=pd.Timestamp.today())
    holding = st.selectbox("Holding period (months)", [1,3,6,12], index=3)
    min_pass_bt = st.slider("Minimum core criteria passed to enter a position", 1, 8, 6, key="min_pass_bt")
    if st.button("Run backtest", type="primary"):
        if universe_file is None:
            st.error("Upload a point-in-time universe CSV first. A current S&P 500 list would create survivorship bias.")
        else:
            with st.spinner("Running backtest... this re-fetches SEC fundamentals per ticker per rebalance date and can take a while."):
                try:
                    uni = pd.read_csv(universe_file)
                    out = run_backtest(uni, criteria, str(start), str(end), holding, min_pass=min_pass_bt)
                    st.dataframe(out["portfolio"], use_container_width=True, hide_index=True)
                    c1,c2,c3,c4 = st.columns(4)
                    c1.metric("CAGR", f"{out['cagr']*100:.2f}%")
                    c2.metric("Annualized volatility", f"{out['volatility']*100:.2f}%")
                    c3.metric("Max drawdown", f"{out['max_drawdown']*100:.2f}%")
                    c4.metric("Sharpe", f"{out['sharpe']:.2f}")
                    st.line_chart(out["equity_curve"].set_index("date"))
                    st.download_button("Download backtest trades", out["trades"].to_csv(index=False), "backtest_trades.csv", "text/csv")
                    if out.get("warnings"):
                        with st.expander(f"{len(out['warnings'])} data-quality notes (tickers skipped or excluded from a rebalance)"):
                            st.write("\n".join(out["warnings"]))
                except Exception as e:
                    st.error(str(e))
