from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils.prediction_service import available_datasets, build_prediction, load_price_history


st.set_page_config(page_title="CryptoPredictions Pro", layout="wide")

st.markdown(
    """
    <style>
        :root {
            color-scheme: dark;
        }
        .main {
            background-color: #0b0f19;
        }
        header, .css-18ni7ap, .css-1dp5vir {
            background: #0b0f19;
        }
        .trading-card {
            background: linear-gradient(145deg, #151a2d 0%, #0e1222 100%);
            border: 1px solid #20263a;
            padding: 16px 20px;
            border-radius: 16px;
            box-shadow: 0 12px 30px rgba(0, 0, 0, 0.4);
        }
        .muted {
            color: #94a3b8;
        }
        .tag {
            background: #1f2937;
            color: #e2e8f0;
            padding: 4px 10px;
            border-radius: 999px;
            font-size: 12px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


datasets = available_datasets()
if not datasets:
    st.warning("No datasets found in the ./data directory.")
    st.stop()

symbols = sorted({item.symbol for item in datasets})
intervals = sorted({item.interval for item in datasets})

with st.sidebar:
    st.markdown("### Market selector")
    symbol = st.selectbox("Symbol", symbols, index=0)
    interval = st.selectbox("Interval", intervals, index=0)
    model_type = st.selectbox(
        "Model",
        ["random_forest", "xgboost", "lstm", "gru", "prophet", "orbit"],
        index=0,
    )
    st.caption("Pick a dataset from the local ./data folder.")

selected_path = next(
    (item.path for item in datasets if item.symbol == symbol and item.interval == interval),
    None,
)

if selected_path is None:
    st.error("Selected dataset is missing. Please pick another combination.")
    st.stop()

price_history = load_price_history(Path(selected_path))
latest_row = price_history.iloc[-1]
close_col = "close" if "close" in price_history.columns else "Close"
open_col = "open" if "open" in price_history.columns else "Open"

price_change = latest_row[close_col] - price_history[close_col].iloc[-2]
price_change_pct = (price_change / price_history[close_col].iloc[-2]) * 100

st.markdown("## CryptoPredictions Pro")
st.markdown("<span class='muted'>Predictive insights with a trading-terminal vibe.</span>", unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown("<div class='trading-card'>", unsafe_allow_html=True)
    st.metric("Last Close", f"${latest_row[close_col]:,.2f}")
    st.markdown("</div>", unsafe_allow_html=True)
with col2:
    st.markdown("<div class='trading-card'>", unsafe_allow_html=True)
    st.metric("24H Change", f"{price_change:+.2f}", f"{price_change_pct:+.2f}%")
    st.markdown("</div>", unsafe_allow_html=True)
with col3:
    st.markdown("<div class='trading-card'>", unsafe_allow_html=True)
    st.metric("Last Open", f"${latest_row[open_col]:,.2f}")
    st.markdown("</div>", unsafe_allow_html=True)
with col4:
    st.markdown("<div class='trading-card'>", unsafe_allow_html=True)
    st.metric("Volume", f"{latest_row['volume']:,.0f}" if "volume" in latest_row else "—")
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("### Price action")

fig = go.Figure(
    data=[
        go.Candlestick(
            x=price_history["Date"],
            open=price_history[open_col],
            high=price_history["High" if "High" in price_history.columns else "high"],
            low=price_history["Low" if "Low" in price_history.columns else "low"],
            close=price_history[close_col],
            increasing_line_color="#22c55e",
            decreasing_line_color="#ef4444",
        )
    ]
)
fig.update_layout(
    height=520,
    margin=dict(l=10, r=10, t=30, b=10),
    template="plotly_dark",
    xaxis_rangeslider_visible=False,
)

st.plotly_chart(fig, use_container_width=True)

st.markdown("### Live prediction")

prediction_col, info_col = st.columns([2, 3])
with prediction_col:
    st.markdown("<div class='trading-card'>", unsafe_allow_html=True)
    if st.button("Generate forecast", type="primary"):
        with st.spinner("Training model and generating prediction..."):
            result = build_prediction(symbol, interval, model_type=model_type)
        st.metric("Predicted Mean", f"${result.predicted_mean:,.2f}")
        st.caption(
            f"Model: {result.model} · Last close: ${result.last_close:,.2f} on {result.last_timestamp}"
        )
    else:
        st.caption("Click the button to compute a model-based forecast.")
    st.markdown("</div>", unsafe_allow_html=True)

with info_col:
    st.markdown("<div class='trading-card'>", unsafe_allow_html=True)
    st.markdown("#### Market snapshots")
    st.markdown(
        "<span class='tag'>Trend</span> <span class='tag'>Momentum</span> <span class='tag'>Volatility</span>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p class='muted'>Use the sidebar to switch symbols and intervals. Forecasts are created from the same"
        " local datasets used in training.</p>",
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("### Recent data")
st.dataframe(price_history.tail(12), use_container_width=True)
