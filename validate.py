"""
validate.py -- cross-check the 7/8-factor pipeline against reality.

Three independent layers, weakest to strongest:

  1. INVARIANTS   Arithmetic that must hold for any company. Catches unit errors,
                  sign flips, and bad XBRL concept mapping. No external truth needed.
  2. CROSS-SOURCE Compare SEC-XBRL-derived metrics against Yahoo Finance's own
                  independently-computed figures. Disagreement doesn't prove which
                  side is wrong, but it flags what to inspect by hand.
  3. GROUND TRUTH Compare against figures YOU transcribed from actual 10-K filings
                  (ground_truth.json). This is the only layer that can truly confirm
                  correctness -- the other two can only find contradictions.

Usage:
    python validate.py                      # default ticker set
    python validate.py AAPL MSFT JNJ
    python validate.py --sectors            # sector-stress set (financials, REITs, etc.)
    python validate.py --write-template     # emit a ground_truth.json skeleton to fill in

Exit code is nonzero if any FAIL-level check trips, so this can gate CI.
"""
import argparse, json, sys, math
from pathlib import Path

import pandas as pd

from analyzer import analyze_ticker
from data_sources import shares_info, throttle
from metrics import histories, operating_history

GROUND_TRUTH_PATH = Path("ground_truth.json")

# Criteria values are irrelevant to whether the *metrics* are correct, but score_core()
# requires every key to be present (it uses bracket access), so pass a complete dict.
DUMMY_CRITERIA = {
    "pe_max": 20.0, "roic_min": .15, "de_max": 1.0, "eps_cagr_min": .10,
    "roe_min": .15, "ebit_margin_min": .10, "gross_margin_min": .40,
    "fcf_yield_target": .04, "valuation_target_discount": .15,
}

DEFAULT_TICKERS = ["AAPL", "MSFT", "JNJ", "KO", "HD"]

# Sectors where this model's assumptions are known to break down. ROIC/debt-to-equity
# are close to meaningless for banks (deposits aren't "debt" in the operating sense)
# and gross margin is undefined for most financials and REITs.
SECTOR_STRESS_TICKERS = ["JPM", "BAC", "PLD", "AMT", "XOM", "BRK-B", "NEE"]

PASS, WARN, FAIL = "PASS", "WARN", "FAIL"


class Results:
    def __init__(self):
        self.rows = []

    def add(self, ticker, layer, check, status, detail=""):
        self.rows.append({"ticker": ticker, "layer": layer, "check": check,
                          "status": status, "detail": detail})

    def failed(self):
        return any(r["status"] == FAIL for r in self.rows)

    def to_frame(self):
        return pd.DataFrame(self.rows)


def _close(a, b, tol):
    """Relative comparison that degrades gracefully near zero."""
    if a is None or b is None:
        return None
    if b == 0:
        return abs(a) < tol
    return abs(a - b) / abs(b) <= tol


# ----------------------------------------------------------------------------
# Layer 1: invariants
# ----------------------------------------------------------------------------
def check_invariants(ticker, result, res):
    m = result.get("metrics") or {}
    op = result.get("history")
    L = "invariant"

    if not m:
        res.add(ticker, L, "metrics produced", FAIL,
                "; ".join(result.get("errors", [])) or "no metrics returned")
        return

    # P/E must equal price / EPS. Catches a mismatched price or EPS unit.
    if m.get("pe") is not None and m.get("price") and m.get("eps"):
        expect = m["price"] / m["eps"]
        res.add(ticker, L, "pe == price/eps",
                PASS if _close(m["pe"], expect, .01) else FAIL,
                f"reported {m['pe']:.3f} vs recomputed {expect:.3f}")

    # Margins are ratios, not percentages. A value >1.5 almost always means a x100
    # scaling bug crept in somewhere.
    for key in ("ebit_margin", "gross_margin", "roe", "roic"):
        v = m.get(key)
        if v is None:
            continue
        if abs(v) > 1.5:
            res.add(ticker, L, f"{key} on ratio scale", FAIL,
                    f"{v:.3f} -- looks like a percentage, expected a fraction")
        else:
            res.add(ticker, L, f"{key} on ratio scale", PASS, f"{v:.4f}")

    # Gross margin should exceed EBIT margin: opex is subtracted after COGS.
    gm, em = m.get("gross_margin"), m.get("ebit_margin")
    if gm is not None and em is not None:
        res.add(ticker, L, "gross_margin >= ebit_margin",
                PASS if gm >= em - 1e-9 else FAIL,
                f"gross {gm:.4f} vs ebit {em:.4f}")

    # Recompute ROE from the raw operating history rather than trusting calc_latest().
    if op is not None and not op.empty:
        last = op.iloc[-1]
        if pd.notna(last.get("net_income")) and pd.notna(last.get("equity")) and last.get("equity"):
            expect = last["net_income"] / last["equity"]
            if m.get("roe") is not None:
                res.add(ticker, L, "roe reproducible from history",
                        PASS if _close(m["roe"], expect, .02) else FAIL,
                        f"metric {m['roe']:.4f} vs history {expect:.4f}")

        # Fiscal years should be unique, ordered, and gap-free-ish. Duplicates mean the
        # XBRL dedupe in concept_series() let a restatement through.
        fys = op["fiscal_year"].tolist()
        res.add(ticker, L, "fiscal years unique",
                PASS if len(fys) == len(set(fys)) else FAIL, f"{fys}")
        res.add(ticker, L, "fiscal years ascending",
                PASS if fys == sorted(fys) else FAIL, f"{fys}")

        # period_end powers the backtest's reporting lag. Missing -> backtest silently
        # skips the ticker entirely.
        if "period_end" not in op.columns:
            res.add(ticker, L, "period_end present", FAIL, "column missing")
        else:
            n_missing = int(op["period_end"].isna().sum())
            res.add(ticker, L, "period_end populated",
                    PASS if n_missing == 0 else WARN,
                    f"{n_missing}/{len(op)} fiscal years missing a period end date")

    # Valuation discount and percentile should agree in direction: a below-median P/E
    # (positive discount) implies a percentile under 50.
    disc, pct = m.get("pe_discount_vs_median_10y"), m.get("pe_percentile_10y")
    if disc is not None and pct is not None:
        consistent = (disc > 0 and pct <= 50) or (disc < 0 and pct >= 50) or abs(disc) < .01
        res.add(ticker, L, "valuation discount/percentile agree",
                PASS if consistent else WARN,
                f"discount {disc:+.3f} vs percentile {pct:.1f}")

    # Score must equal the number of individual pass flags.
    flags = ["pe_pass", "roic_pass", "de_pass", "eps_cagr_pass", "roe_pass",
             "ebit_margin_pass", "gross_margin_pass", "valuation_pass"]
    present = [f for f in flags if f in m]
    if present:
        expect = sum(bool(m[f]) for f in present)
        res.add(ticker, L, "score == sum(pass flags)",
                PASS if m.get("score") == expect else FAIL,
                f"score {m.get('score')} vs {expect} flags true (of {len(present)} checks)")

    # Scores must be in range.
    for key in ("weighted_score", "enhanced_score"):
        v = m.get(key)
        if v is not None:
            res.add(ticker, L, f"{key} in [0,100]",
                    PASS if -1e-6 <= v <= 100 + 1e-6 else FAIL, f"{v:.2f}")


# ----------------------------------------------------------------------------
# Layer 2: cross-source
# ----------------------------------------------------------------------------
def check_cross_source(ticker, result, res):
    """Compare against Yahoo's independently-derived figures.

    A mismatch does NOT establish which source is right. Yahoo uses trailing-twelve-month
    data while this pipeline uses last completed fiscal year, so differences are expected
    and mid-year gaps can be large. These are WARN, never FAIL -- they mark what to
    verify by hand against the filing.
    """
    L = "cross-source"
    m = result.get("metrics") or {}
    if not m:
        return
    try:
        info = shares_info(ticker)
        throttle()
    except Exception as e:
        res.add(ticker, L, "yahoo reachable", WARN, f"{type(e).__name__}: {e}")
        return
    if not info:
        res.add(ticker, L, "yahoo reachable", WARN, "empty .info (unofficial API, breaks often)")
        return

    pairs = [
        ("pe", info.get("trailingPE"), .25, "P/E (ours: last FY EPS, Yahoo: TTM)"),
        ("roe", info.get("returnOnEquity"), .25, "ROE"),
        ("gross_margin", info.get("grossMargins"), .15, "gross margin"),
        ("ebit_margin", info.get("operatingMargins"), .20, "EBIT/operating margin"),
    ]
    for key, theirs, tol, label in pairs:
        ours = m.get(key)
        if ours is None or theirs is None:
            res.add(ticker, L, label, WARN,
                    f"unavailable (ours={ours}, yahoo={theirs})")
            continue
        ok = _close(ours, theirs, tol)
        res.add(ticker, L, label, PASS if ok else WARN,
                f"ours {ours:.4f} vs yahoo {theirs:.4f}"
                + ("" if ok else f"  <-- differs >{tol:.0%}, inspect the 10-K"))


# ----------------------------------------------------------------------------
# Layer 3: ground truth
# ----------------------------------------------------------------------------
def check_ground_truth(ticker, result, res, truth):
    L = "ground-truth"
    m = result.get("metrics") or {}
    entry = truth.get(ticker)
    if not entry:
        res.add(ticker, L, "ground truth available", WARN,
                "no entry in ground_truth.json -- correctness UNVERIFIED for this ticker")
        return
    if not m:
        return

    fy = entry.get("fiscal_year")
    op = result.get("history")
    tol = entry.get("tolerance", .02)

    # Raw financial statement line items, compared for the specific fiscal year the
    # user transcribed. This is the check that actually validates XBRL concept mapping.
    if op is not None and not op.empty and fy is not None:
        row = op[op["fiscal_year"] == fy]
        if row.empty:
            res.add(ticker, L, f"FY{fy} present in history", FAIL,
                    f"available: {op['fiscal_year'].tolist()}")
        else:
            row = row.iloc[0]
            for field in ("revenue", "ebit", "net_income", "equity", "debt", "cash"):
                if field not in entry:
                    continue
                ours, theirs = row.get(field), entry[field]
                if ours is None or (isinstance(ours, float) and math.isnan(ours)):
                    res.add(ticker, L, f"FY{fy} {field}", FAIL, f"pipeline produced no value (10-K: {theirs:,.0f})")
                    continue
                ok = _close(float(ours), float(theirs), tol)
                res.add(ticker, L, f"FY{fy} {field}", PASS if ok else FAIL,
                        f"ours {float(ours):,.0f} vs 10-K {float(theirs):,.0f}"
                        + ("" if ok else "  <-- CONCEPT MAPPING LIKELY WRONG"))

    # Derived metrics, if the user chose to record them.
    for field in ("roe", "roic", "gross_margin", "ebit_margin"):
        if field not in entry:
            continue
        ours, theirs = m.get(field), entry[field]
        if ours is None:
            res.add(ticker, L, f"{field}", FAIL, f"pipeline produced no value (expected {theirs})")
            continue
        ok = _close(ours, float(theirs), entry.get("metric_tolerance", .05))
        res.add(ticker, L, f"{field}", PASS if ok else FAIL,
                f"ours {ours:.4f} vs expected {float(theirs):.4f}")


# ----------------------------------------------------------------------------
def write_template(tickers):
    template = {
        "_README": [
            "Fill these in from the company's actual 10-K filing, then rerun validate.py.",
            "Until a ticker has an entry here, its correctness is UNVERIFIED -- the",
            "invariant and cross-source layers can only detect contradictions, not confirm",
            "that the numbers match the filing.",
            "Units must match SEC XBRL: raw dollars, not millions. 'debt' should be the sum",
            "of the debt concepts listed in metrics.py CONCEPTS['debt'], which is the most",
            "common source of mismatch.",
            "Ratios (roe, gross_margin, ...) are fractions: 0.42 not 42.",
        ],
        "_example": {
            "fiscal_year": 2023,
            "revenue": 0, "ebit": 0, "net_income": 0,
            "equity": 0, "debt": 0, "cash": 0,
            "roe": 0.0, "gross_margin": 0.0,
            "tolerance": 0.02, "metric_tolerance": 0.05,
        },
    }
    for t in tickers:
        template[t] = {"fiscal_year": None, "revenue": None, "ebit": None,
                       "net_income": None, "equity": None, "debt": None, "cash": None}
    GROUND_TRUTH_PATH.write_text(json.dumps(template, indent=2))
    print(f"Wrote {GROUND_TRUTH_PATH}. Fill it from the 10-Ks, then rerun.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tickers", nargs="*", default=None)
    ap.add_argument("--sectors", action="store_true",
                    help="use the sector-stress set (financials, REITs, energy)")
    ap.add_argument("--write-template", action="store_true")
    ap.add_argument("--csv", help="write full results to this path")
    args = ap.parse_args()

    tickers = args.tickers or (SECTOR_STRESS_TICKERS if args.sectors else DEFAULT_TICKERS)

    if args.write_template:
        write_template(tickers)
        return 0

    truth = {}
    if GROUND_TRUTH_PATH.exists():
        truth = {k: v for k, v in json.loads(GROUND_TRUTH_PATH.read_text()).items()
                 if not k.startswith("_")}
    else:
        print(f"NOTE: {GROUND_TRUTH_PATH} not found. Running layers 1-2 only.\n"
              f"      Run with --write-template to create one.\n"
              f"      Without it, nothing here CONFIRMS correctness -- it can only find contradictions.\n")

    res = Results()
    for t in tickers:
        print(f"Validating {t} ...", flush=True)
        result = analyze_ticker(t, DUMMY_CRITERIA, include_history=True)
        for w in result.get("warnings", []):
            res.add(t, "pipeline", "warning", WARN, w)
        for e in result.get("errors", []):
            res.add(t, "pipeline", "error", FAIL, e)
        check_invariants(t, result, res)
        check_cross_source(t, result, res)
        check_ground_truth(t, result, res, truth)
        throttle()

    df = res.to_frame()
    if df.empty:
        print("No checks ran.")
        return 1

    print("\n" + "=" * 78)
    for status in (FAIL, WARN, PASS):
        sub = df[df.status == status]
        if sub.empty:
            continue
        print(f"\n{status} ({len(sub)})")
        print("-" * 78)
        show = sub if status != PASS else sub[["ticker", "layer", "check"]]
        for _, r in show.iterrows():
            detail = f"  |  {r['detail']}" if status != PASS and r.get("detail") else ""
            print(f"  {r['ticker']:<7} {r['layer']:<14} {r['check']}{detail}")

    n_fail = int((df.status == FAIL).sum())
    n_warn = int((df.status == WARN).sum())
    unverified = [t for t in tickers if t not in truth]

    print("\n" + "=" * 78)
    print(f"{len(df)} checks: {int((df.status==PASS).sum())} pass, {n_warn} warn, {n_fail} fail")
    if unverified:
        print(f"\nUNVERIFIED (no ground truth entry): {', '.join(unverified)}")
        print("These passed only contradiction-checks. Passing is not the same as being correct.")

    if args.csv:
        df.to_csv(args.csv, index=False)
        print(f"\nFull results -> {args.csv}")

    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
