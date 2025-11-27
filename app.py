import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import math
import random
import warnings
warnings.filterwarnings("ignore")

# -------------------------
# App config
# -------------------------
st.set_page_config(layout="wide", page_title="Insight Alpha 2.0 — Full")

# -------------------------
# CSS (no f-strings to avoid brace issues)
# -------------------------
st.markdown("""
<style>
  body { background-color: #f7f9fc; color: #222; }
  .rec-box { padding: 18px; border-radius: 12px; color: white; text-align:center; }
  .rec-title { font-size: 24px; font-weight:800; }
  .metric-card { padding: 12px; border-radius: 10px; background: #fff; border:1px solid #eee; text-align:center; }
  .grade-badge { font-weight:800; padding:6px 10px; border-radius:8px; color:white; display:inline-block; }
  .small { font-size:12px; color:#666; }
</style>
""", unsafe_allow_html=True)

# -------------------------
# Utilities
# -------------------------
def safe_float(x):
    try:
        return float(x)
    except Exception:
        return None

def format_large(v):
    try:
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return "N/A"
        v = float(v)
        if v >= 1e12:
            return f"${v/1e12:.2f}T"
        if v >= 1e9:
            return f"${v/1e9:.2f}B"
        if v >= 1e6:
            return f"${v/1e6:.2f}M"
        return f"${v:,.0f}"
    except:
        return "N/A"

def grade_from_score(s):
    if s >= 90: return "A+"
    if s >= 80: return "A"
    if s >= 70: return "B"
    if s >= 60: return "C"
    if s >= 40: return "D"
    return "F"

def color_from_score(s):
    if s >= 80: return "#00C853"
    if s >= 60: return "#FFD600"
    return "#FF3D00"

# -------------------------
# Industry rules
# -------------------------
def industry_rule_map(sector, industry):
    s = (sector or "").lower()
    i = (industry or "").lower()
    rule = {
        "gm_scale": 1.0,
        "growth_scale": 1.0,
        "de_norm": lambda x: x
    }
    if "financial" in s or "bank" in i or "insurance" in i:
        rule["gm_scale"] = 0.6
        rule["growth_scale"] = 0.5
        rule["de_norm"] = lambda x: x / 5.0
    if "technology" in s or "software" in i or "internet" in i or "semiconductor" in i:
        rule["gm_scale"] = 1.0
        rule["growth_scale"] = 1.0
        rule["de_norm"] = lambda x: x
    if "energy" in s or "utility" in s or "mining" in i or "crypto" in i or "mining" in s:
        rule["gm_scale"] = 0.6
        rule["growth_scale"] = 0.8
        rule["de_norm"] = lambda x: x * 1.5
    if "real estate" in s or "reit" in i:
        rule["gm_scale"] = 0.7
        rule["growth_scale"] = 0.8
        rule["de_norm"] = lambda x: x / 4.0
    if "biotech" in i or "pharmaceutical" in s:
        rule["gm_scale"] = 0.3
        rule["growth_scale"] = 1.3
        rule["de_norm"] = lambda x: x * 2.0
    return rule

# -------------------------
# Scoring functions
# -------------------------
def score_valuation(info):
    peg = safe_float(info.get("pegRatio"))
    per = safe_float(info.get("forwardPE"))
    ps = safe_float(info.get("priceToSalesTrailing12Months"))
    if peg and peg > 0:
        v = peg
        if v <= 0.5: s = 100
        elif v <= 0.8: s = 90
        elif v <= 1.0: s = 80
        elif v <= 1.5: s = 60
        elif v <= 2.0: s = 40
        else: s = 20
        return float(s), f"PEG: {v:.2f}"
    if per and per > 0:
        v = per
        if v <= 10: s = 100
        elif v <= 15: s = 90
        elif v <= 20: s = 80
        elif v <= 30: s = 60
        elif v <= 40: s = 40
        else: s = 20
        return float(s), f"Forward P/E: {v:.1f}"
    if ps and ps > 0:
        v = ps
        if v <= 1: s = 100
        elif v <= 2: s = 90
        elif v <= 5: s = 80
        elif v <= 10: s = 60
        elif v <= 20: s = 40
        else: s = 20
        return float(s), f"P/S: {v:.2f}"
    return 50.0, "Valuation: insufficient data"

def score_profitability(info, rule):
    gm_raw = safe_float(info.get("grossMargins"))
    gm_pct = (gm_raw or 0) * 100
    gm_pct = gm_pct * rule.get("gm_scale",1.0)
    if gm_pct <= 0:
        s = 20.0
    elif gm_pct < 5:
        s = 30.0
    elif gm_pct < 10:
        s = 40.0
    elif gm_pct < 20:
        s = 60.0
    elif gm_pct < 40:
        s = 80.0
    else:
        s = 100.0
    return float(s), f"Gross Margin: {gm_pct:.1f}%"

def score_growth(info, rule):
    rev_g_raw = safe_float(info.get("revenueGrowth"))
    rev_g = (rev_g_raw or 0) * 100
    rev_g = rev_g * rule.get("growth_scale",1.0)
    if rev_g <= -20:
        s = 20.0
    elif rev_g < 0:
        s = 30.0
    elif rev_g < 5:
        s = 40.0
    elif rev_g < 10:
        s = 60.0
    elif rev_g < 20:
        s = 80.0
    else:
        s = 100.0
    return float(s), f"Revenue Growth: {rev_g:.1f}%"

def score_safety(info, rule):
    de_raw = safe_float(info.get("debtToEquity"))
    if de_raw is None:
        return 50.0, "D/E: N/A"
    try:
        de_adj = rule.get("de_norm", lambda x:x)(de_raw)
    except Exception:
        de_adj = de_raw
    if de_adj <= 0:
        s = 100.0
    elif de_adj <= 50:
        s = 90.0
    elif de_adj <= 150:
        s = 60.0
    elif de_adj <= 300:
        s = 40.0
    else:
        s = 20.0
    return float(s), f"D/E: {de_raw:.1f}% (adj {de_adj:.1f})"

# -------------------------
# Momentum + Risk scoring
# -------------------------
def score_momentum_and_risk(hist, weights=(0.5,0.3,0.2)):
    # weights correspond to 1M,3M,12M
    try:
        close = hist['Close'].dropna()
        ln = len(close)
        if ln < 21:
            return 50.0, "Insufficient history"
        # indices
        idx_1m = max(0, ln - 21)
        idx_3m = max(0, ln - 63)
        idx_12m = 0
        r1 = (close.iloc[-1] / close.iloc[idx_1m] - 1) * 100
        r3 = (close.iloc[-1] / close.iloc[idx_3m] - 1) * 100 if ln >= 63 else r1
        r12 = (close.iloc[-1] / close.iloc[idx_12m] - 1) * 100 if ln >= 252 else r3
        # cap values
        r1c = np.clip(r1, -90, 1000)
        r3c = np.clip(r3, -200, 2000)
        r12c = np.clip(r12, -500, 5000)
        mom_raw = weights[0]*r1c + weights[1]*r3c + weights[2]*r12c
        # volatility (63 trading days ~ 3 months)
        ret = close.pct_change().dropna()
        vol = float(ret.rolling(63).std().iloc[-1]) if len(ret) >= 63 else float(ret.std())
        vol = 0.0 if np.isnan(vol) else vol
        # map mom_raw to baseline
        if mom_raw <= -50:
            base = 20.0
        elif mom_raw < 0:
            base = 40.0
        elif mom_raw < 20:
            base = 60.0
        elif mom_raw < 60:
            base = 80.0
        else:
            base = 100.0
        vol_pen = min(60, vol * 100)
        score = max(20.0, base - vol_pen * 0.4)
        detail = f"1M:{r1:.1f}% 3M:{r3:.1f}% 1Y:{r12:.1f}% Vol:{vol:.3f}"
        return float(min(100.0, score)), detail
    except Exception as e:
        return 50.0, f"Mom err: {e}"

# -------------------------
# Adaptive weight optimizer (random search)
# -------------------------
def optimize_weights_random(factors_df, forward_returns, n_iter=2000, seed=123):
    # simple random search over simplex
    np.random.seed(seed)
    best = {"w": None, "score": -1e9}
    if len(forward_returns) == 0 or np.nanstd(forward_returns) == 0:
        # no info to optimize
        w0 = np.array([0.25,0.25,0.2,0.2,0.1])  # default
        return w0, -9999.0
    for _ in range(n_iter):
        w = np.random.random(5)
        w = w / w.sum()
        signal = factors_df.values.dot(w)
        # use simple correlation as proxy: corr(signal, forward_returns)
        try:
            if np.nanstd(signal) == 0 or np.nanstd(forward_returns) == 0:
                score = -1e9
            else:
                score = np.nanmean((signal - np.nanmean(signal)) * (forward_returns - np.nanmean(forward_returns))) / (np.nanstd(signal) * np.nanstd(forward_returns))
        except Exception:
            score = -1e9
        if score > best["score"]:
            best = {"w": w.copy(), "score": score}
    if best["w"] is None:
        return np.array([0.25,0.25,0.2,0.2,0.1]), -9999.0
    return best["w"], best["score"]

# -------------------------
# Full pipeline for universe
# -------------------------
def compute_universe_scores(tickers, hist_period="1y"):
    results = {}
    factor_rows = []
    tick_list = []
    forward_returns = []
    for t in tickers:
        try:
            tk = yf.Ticker(t)
            info = tk.info
        except Exception:
            info = {}
        # history
        try:
            hist = tk.history(period=hist_period, auto_adjust=True)
        except Exception:
            hist = pd.DataFrame()
        sector = info.get("sector","")
        industry = info.get("industry","")
        rule = industry_rule_map(sector, industry)
        v_s, v_det = score_valuation(info)
        p_s, p_det = score_profitability(info, rule)
        g_s, g_det = score_growth(info, rule)
        mom_s, mom_det = score_momentum_and_risk(hist)
        s_s, s_det = score_safety(info, rule)
        # Market Quality (institutional ownership)
        inst = safe_float(info.get("heldPercentInstitutions")) or 0.0
        mq = float(100 if inst>=0.6 else 80 if inst>=0.4 else 60 if inst>=0.2 else 40)
        mq_det = f"Institutional: {inst*100:.1f}%"
        final_rule_based = int(round(v_s*0.25 + p_s*0.2 + g_s*0.2 + mom_s*0.2 + s_s*0.1 + mq*0.05))
        results[t] = {
            "info": info,
            "hist": hist,
            "scores": {"valuation":v_s, "profit":p_s, "growth":g_s, "momentum":mom_s, "safety":s_s, "mq":mq},
            "details": {"valuation":v_det, "profit":p_det, "growth":g_det, "momentum":mom_det, "safety":s_det, "mq":mq_det},
            "final_rule": final_rule_based
        }
        # prepare for optimizer
        factor_rows.append([v_s, p_s, g_s, mom_s, s_s])
        tick_list.append(t)
        # compute forward return proxy: 3m forward (if possible)
        try:
            close = hist['Close'].dropna()
            if len(close) >= 63:
                forward = (close.iloc[-1] / close.iloc[int(-63)] - 1)
            else:
                forward = 0.0
        except Exception:
            forward = 0.0
        forward_returns.append(forward)
    # optimizer
    factors_df = pd.DataFrame(factor_rows, index=tick_list, columns=['val','prof','grow','mom','safe']) if len(factor_rows)>0 else pd.DataFrame(columns=['val','prof','grow','mom','safe'])
    fr = np.array(forward_returns)
    weights, opt_score = optimize_weights_random(factors_df, fr, n_iter=3000)
    # composite scores
    composite = factors_df.values.dot(weights) if not factors_df.empty else np.array([0]*len(tick_list))
    # scale composite to 0-100
    if composite.size > 1:
        cs = (composite - np.nanmean(composite)) / (np.nanstd(composite) if np.nanstd(composite)>0 else 1.0)
        comp_scaled = 50 + 10*cs
    else:
        comp_scaled = composite
    for i,t in enumerate(tick_list):
        results[t]["adaptive_weights"] = weights.tolist()
        results[t]["adaptive_score_raw"] = float(composite[i]) if composite.size>0 else 0.0
        results[t]["adaptive_score"] = float(comp_scaled[i]) if composite.size>0 else float(composite[i])
    return results, weights, opt_score

# -------------------------
# Simple backtest: monthly rebalance long top-N
# -------------------------
def backtest_topn(tickers, start, end, weights, topn=3, rebalance_months=1):
    # returns dictionary with equity curve etc.
    # weights is array of 5 factor weights used to compute composite
    # We'll fetch monthly close price series and simulate equally weighted long top-N
    price_data = {}
    for t in tickers:
        try:
            df = yf.Ticker(t).history(start=start, end=end, auto_adjust=True)
            price_data[t] = df['Close'].dropna()
        except Exception:
            price_data[t] = pd.Series(dtype=float)
    # build monthly dates
    all_dates = pd.date_range(start=start, end=end, freq='M')
    portfolio_value = []
    dates = []
    cash = 1.0  # start with 1.0 (normalized)
    holdings = {}
    for dt in all_dates:
        # compute scores at this date: approximate by using trailing 1y data up to dt
        current_scores = {}
        for t in tickers:
            try:
                df = price_data[t]
                # build pseudo-hist for scoring using last 1y up to dt
                hist = df[df.index <= dt].tail(252)
                # if no data, skip
                if hist.empty:
                    v = p = g = m = s = 50.0
                else:
                    info = yf.Ticker(t).info
                    rule = industry_rule_map(info.get("sector",""), info.get("industry",""))
                    v, _ = score_valuation(info)
                    p, _ = score_profitability(info, rule)
                    g, _ = score_growth(info, rule)
                    m, _ = score_momentum_and_risk(hist)
                    s, _ = score_safety(info, rule)
                current_scores[t] = np.dot([v,p,g,m,s], weights)
            except Exception:
                current_scores[t] = 0.0
        # rank
        ranked = sorted(current_scores.items(), key=lambda x: x[1], reverse=True)
        top = [x[0] for x in ranked[:topn]]
        # allocate equally
        allocation = {t: 1.0/len(top) for t in top} if len(top)>0 else {}
        # compute portfolio value at next month end (approx)
        next_dt = dt + pd.DateOffset(months=rebalance_months)
        # get prices at dt and next_dt
        val = 0.0
        for t, alloc in allocation.items():
            p_now = price_data[t][price_data[t].index <= dt]
            p_next = price_data[t][price_data[t].index <= next_dt]
            if p_now.empty or p_next.empty:
                continue
            p_now = p_now.iloc[-1]
            p_next = p_next.iloc[-1]
            ret = p_next / p_now
            val += alloc * ret
        if val == 0.0:
            # no movement data -> keep cash
            portfolio_value.append(cash)
        else:
            cash = cash * val
            portfolio_value.append(cash)
        dates.append(next_dt)
    if len(dates) == 0:
        return pd.DataFrame()
    eq = pd.Series(portfolio_value, index=dates)
    return eq

# -------------------------
# Streamlit UI
# -------------------------
st.title("Insight Alpha — Quant Score 2.0 (Full Edition)")

with st.expander("Overview — What this app does"):
    st.markdown("""
    - Industry-aware factor scoring (valuation, profitability, growth, momentum, safety)
    - Multi-horizon momentum + volatility risk adjustment
    - Adaptive weight optimizer (random search) to align factor signals with forward returns proxy
    - Simple monthly backtest (top-N long strategy)
    """)

col1, col2 = st.columns([1,3])
with col1:
    tick_input = st.text_input("Tickers (comma separated)", value="BITF,TSLA,AAPL,IREN")
    hist_period = st.selectbox("History period for signals", ["1y","2y","3y"], index=0)
    run_button = st.button("Run Quant 2.0")
with col2:
    st.write("Instructions:")
    st.write("- 입력 예: BITF,TSLA,AAPL,IREN")
    st.write("- Run을 누르면 adaptive weights 계산 후 개별 티커 결과와 간단 백테스트를 보여줍니다.")

if run_button:
    with st.spinner("Calculating universe scores..."):
        tickers = [t.strip().upper() for t in tick_input.split(",") if t.strip()]
        if len(tickers) == 0:
            st.error("티커를 입력하세요.")
        else:
            results, weights, opt_score = compute_universe_scores(tickers, hist_period)
            st.subheader("Adaptive Weights (optimized)")
            st.write("Order: [Valuation, Profitability, Growth, Momentum, Safety]")
            st.write(np.round(weights, 4).tolist())
            st.write(f"Optimizer proxy score: {opt_score:.4f}")
            # summary table
            rows = []
            for t in tickers:
                r = results.get(t, {})
                rows.append({
                    "Ticker": t,
                    "Name": r.get("info", {}).get("shortName", ""),
                    "Adaptive Score": r.get("adaptive_score", 0.0),
                    "Rule Final": r.get("final_rule", 0),
                    "Valuation": r.get("scores", {}).get("valuation", 0),
                    "Profit": r.get("scores", {}).get("profit", 0),
                    "Growth": r.get("scores", {}).get("growth", 0),
                    "Momentum": r.get("scores", {}).get("momentum", 0),
                    "Safety": r.get("scores", {}).get("safety", 0)
                })
            df = pd.DataFrame(rows).set_index("Ticker")
            st.dataframe(df.style.format("{:.2f}"))
            st.subheader("Rankings (Adaptive Score)")
            st.write(df.sort_values("Adaptive Score", ascending=False)[["Name","Adaptive Score","Rule Final"]])
            # detailed cards and mini charts
            for t in tickers:
                r = results.get(t)
                if r is None:
                    st.warning(f"No data for {t}")
                    continue
                st.markdown(f"### {t} — {r.get('info',{}).get('shortName','')}")
                cols = st.columns(6)
                labels = ["Valuation","Profit","Growth","Momentum","Safety","MarketQ"]
                keys = ["valuation","profit","growth","momentum","safety","mq"]
                for i,k in enumerate(keys):
                    val = r.get("scores",{}).get(k, 0)
                    det = r.get("details",{}).get(k, "")
                    grade = grade_from_score(val if val is not None else 0)
                    color = color_from_score(val if val is not None else 0)
                    cols[i].markdown(f"<div class='metric-card'><div style='font-weight:700'>{labels[i]}</div><div style='font-size:20px;color:{color};font-weight:800'>{val:.1f}</div><div class='small'>{det}</div></div>", unsafe_allow_html=True)
                # price chart
                hist = r.get("hist")
                if hist is not None and not hist.empty:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=hist.index, y=hist['Close'], name='Close'))
                    fig.update_layout(height=250, margin=dict(t=20,b=10))
                    st.plotly_chart(fig, use_container_width=True)
                st.markdown("---")
            # backtest UI
            st.subheader("Quick Backtest (Top-N monthly)")
            bt_col1, bt_col2 = st.columns([1,1])
            with bt_col1:
                topn = st.number_input("Top N", value=3, min_value=1, max_value=len(tickers))
                rebalance_months = st.number_input("Rebalance months", value=1, min_value=1)
            with bt_col2:
                start_date = st.date_input("Start date", value=datetime.now().date() - timedelta(days=365))
                end_date = st.date_input("End date", value=datetime.now().date())
            if st.button("Run Backtest"):
                with st.spinner("Running backtest..."):
                    eq = backtest_topn(tickers, start_date, end_date, weights, topn=int(topn), rebalance_months=int(rebalance_months))
                    if eq.empty:
                        st.warning("백테스트 결과가 없습니다. 데이터가 충분한지 확인하세요.")
                    else:
                        st.line_chart(eq)
                        total_return = (eq.iloc[-1] / eq.iloc[0] - 1) * 100 if len(eq)>1 else 0.0
                        st.write(f"Total return: {total_return:.2f}%")
                        # basic stats
                        returns = eq.pct_change().dropna()
                        if returns.empty:
                            st.write("Insufficient returns for stats.")
                        else:
                            ann_ret = (1+returns.mean())**12 - 1
                            vol = returns.std() * math.sqrt(12)
                            sharpe = ann_ret / vol if vol>0 else float('nan')
                            st.write(f"Ann return (approx): {ann_ret*100:.2f}%  Vol(ann): {vol*100:.2f}%  Sharpe(ann): {sharpe:.2f}")
            st.success("Done.")
