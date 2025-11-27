import streamlit as st
import pandas as pd
import numpy as np
from functools import lru_cache

# =====================
#  UI CONFIG
# =====================
st.set_page_config(page_title="Quant Score 2.0", layout="wide")

st.markdown("""
<style>
    .main-title {
        font-size: 42px;
        font-weight: 800;
        text-align: center;
        margin-bottom: 20px;
    }
    .section-title {
        font-size: 26px;
        font-weight: 600;
        margin-top: 40px;
    }
    .footer {
        margin-top: 80px;
        text-align: center;
        font-weight: 600;
        opacity: 0.4;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">📊 Quant Score 2.0 — Premium Edition</div>', unsafe_allow_html=True)

# =====================
#  MOCK DATA FETCHER
# =====================
@lru_cache(maxsize=None)
def fetch_stock(ticker: str):
    # Placeholder (user should add API later)
    return {
        "PER": np.random.uniform(5, 40),
        "PSR": np.random.uniform(0.5, 10),
        "PBR": np.random.uniform(0.5, 8),
        "PEG": np.random.uniform(0.2, 4),
        "Revenue_Growth": np.random.uniform(-10, 40),
        "EPS_Growth": np.random.uniform(-20, 50),
        "Debt_Ratio": np.random.uniform(20, 200),
        "ROE": np.random.uniform(-5, 30),
        "Momentum_3M": np.random.uniform(-10, 25),
    }

# =====================
#  SCORING FUNCTIONS
# =====================
def inverse_score(value, low, high):
    return max(0, min(100, 100 * (high - value) / (high - low)))

def direct_score(value, low, high):
    return max(0, min(100, 100 * (value - low) / (high - low)))

# =====================
#  QUANT SCORE ENGINE
# =====================
def compute_quant_score(data):
    valuation_scores = [
        inverse_score(data["PER"], 5, 40),
        inverse_score(data["PSR"], 0.5, 10),
        inverse_score(data["PBR"], 0.5, 8),
        inverse_score(data["PEG"], 0.2, 4),
    ]
    valuation = max(valuation_scores)

    growth = (
        direct_score(data["Revenue_Growth"], -10, 40) * 0.5 +
        direct_score(data["EPS_Growth"], -20, 50) * 0.5
    )

    stability = inverse_score(data["Debt_Ratio"], 20, 200)

    profitability = direct_score(data["ROE"], -5, 30)

    momentum = direct_score(data["Momentum_3M"], -10, 25)

    final_score = (
        valuation * 0.35 +
        growth * 0.25 +
        stability * 0.15 +
        profitability * 0.15 +
        momentum * 0.10
    )

    return {
        "Valuation": round(valuation, 2),
        "Growth": round(growth, 2),
        "Stability": round(stability, 2),
        "Profitability": round(profitability, 2),
        "Momentum": round(momentum, 2),
        "Final": round(final_score, 2)
    }

# =====================
#  UI INPUT AREA
# =====================
st.markdown('<div class="section-title">1️⃣ 티커 입력</div>', unsafe_allow_html=True)
ticker = st.text_input("티커(symbol) 입력", "AAPL")

if st.button("🔍 분석 실행"):
    raw = fetch_stock(ticker.upper())
    score = compute_quant_score(raw)

    st.markdown('<div class="section-title">2️⃣ 결과 분석</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("Valuation", score["Valuation"])
    c2.metric("Growth", score["Growth"])
    c3.metric("Stability", score["Stability"])

    c4, c5, c6 = st.columns(3)
    c4.metric("Profitability", score["Profitability"])
    c5.metric("Momentum", score["Momentum"])
    c6.metric("Final Score", score["Final"])

    st.divider()
    st.subheader("📌 Raw Data")
    st.json(raw)

st.markdown('<div class="footer">MADE BY YOUNG YOON</div>', unsafe_allow_html=True)
