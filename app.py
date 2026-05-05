import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import os

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Stock Prediction — 446 S&P 500 Stocks",
    page_icon="📈",
    layout="wide"
)

# ── PATHS ─────────────────────────────────────────────────────────────────────
CNN_RESULTS_PATH  = 'cnn_results.csv'    # CNN-Deep return prediction results
LSTM_RESULTS_PATH = 'cnnlstm_results.csv'  # CNN-LSTM return prediction results
ALL_RESULTS_PATH  = 'all_results.csv'    # CNN-Deep price prediction results

# ── DATA LOADING ──────────────────────────────────────────────────────────────
@st.cache_data
def load_price_data():
    """CNN-Deep price prediction results (all_results.csv)."""
    if os.path.exists(ALL_RESULTS_PATH):
        df = pd.read_csv(ALL_RESULTS_PATH)
        return df
    return pd.DataFrame()

@st.cache_data
def load_cnnlstm_data():
    """CNN-LSTM return prediction results (cnnlstm_results.csv)."""
    if os.path.exists(LSTM_RESULTS_PATH):
        return pd.read_csv(LSTM_RESULTS_PATH)
    return pd.DataFrame()

@st.cache_data
def load_cnn_return_data():
    """CNN-Deep return prediction results (cnn_results.csv)."""
    if os.path.exists(CNN_RESULTS_PATH):
        return pd.read_csv(CNN_RESULTS_PATH)
    return pd.DataFrame()

price_df  = load_price_data()
lstm_df   = load_cnnlstm_data()
cnn_ret_df = load_cnn_return_data()

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
st.sidebar.title("📈 Stock Prediction")
st.sidebar.markdown("AI-Based Stock Market Prediction")
st.sidebar.markdown("University of Hertfordshire | 2026")
st.sidebar.markdown("---")

active_model = st.sidebar.radio(
    "Active model",
    ["CNN-Deep (price)", "CNN-LSTM (returns)"],
    index=0
)

st.sidebar.markdown("---")
menu = st.sidebar.radio(
    "Navigation",
    ["🏠 Dashboard", "📊 Results Explorer", "📈 Charts",
     "🎯 Stock Analysis", "📋 Benchmark", "ℹ️ About"]
)

st.sidebar.markdown("---")
st.sidebar.caption("Youssef Derraz\nSupervisor: Dr. Ahmed Salaheldin")

# ── HELPER ────────────────────────────────────────────────────────────────────
def colour_edge(val):
    """Colour edge score cells green if positive, red if negative."""
    try:
        v = float(val)
        color = '#c6efce' if v > 0 else '#ffc7ce' if v < 0 else ''
        return f'background-color: {color}'
    except:
        return ''

# ══════════════════════════════════════════════════════════════════════════════
# DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
if menu == "🏠 Dashboard":

    if active_model == "CNN-Deep (price)":
        st.title("Dashboard — CNN-Deep Price Prediction")
        st.caption("CNN-Deep — price prediction")

        if st.button("🔄 Refresh Data"):
            st.cache_data.clear()
            st.rerun()

        # Exact poster metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Stocks trained", "332")
        col2.metric("Positive R²", "164", "49.4% of trained")
        col3.metric("Best R²", "0.961")
        col4.metric("Avg R² (positive)", "0.558")

        st.markdown("---")
        st.subheader("Performance snapshot")

        if not price_df.empty:
            s = price_df[price_df['status'] == 'success']
            col1, col2 = st.columns(2)
            with col1:
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.hist(s['r2'], bins=30, color='#7B4FBE', edgecolor='white', alpha=0.9)
                ax.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero')
                ax.set_xlabel('R² Score')
                ax.set_ylabel('Count')
                ax.set_title('R² Score Distribution — CNN-Deep')
                ax.legend()
                st.pyplot(fig)
                plt.close()

            with col2:
                top10 = s.nlargest(10, 'r2')
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.barh(top10['ticker'][::-1], top10['r2'][::-1], color='#7B4FBE')
                ax.set_xlabel('R²')
                ax.set_title('Top 10 Stocks by R² — CNN-Deep')
                ax.set_xlim(0, 1)
                st.pyplot(fig)
                plt.close()
        else:
            st.warning("Price prediction results CSV not found.")

    else:  # CNN-LSTM
        st.title("Dashboard — CNN-LSTM Return Prediction")
        st.caption("CNN-LSTM — return prediction")

        if st.button("🔄 Refresh Data"):
            st.cache_data.clear()
            st.rerun()

        # Exact poster metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Stocks trained", "421")
        col2.metric("With real edge", "102", "Long>0 & Short<0")
        col3.metric("Avg DPA", "50.47%", "50% = random")
        col4.metric("Avg edge score", "0.0263", "+0.026 actual")

        st.markdown("---")
        st.subheader("Performance snapshot")

        if not lstm_df.empty:
            s = lstm_df[lstm_df['status'] == 'success']
            col1, col2 = st.columns(2)
            with col1:
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.hist(s['dpa'], bins=25, color='#3AAFA9', edgecolor='white', alpha=0.9)
                ax.axvline(50, color='red', linestyle='--', linewidth=2, label='Random (50%)')
                ax.set_xlabel('DPA (%)')
                ax.set_ylabel('Count')
                ax.set_title('Direction Prediction Accuracy — CNN-LSTM')
                ax.legend()
                st.pyplot(fig)
                plt.close()

            with col2:
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.hist(s['edge_score'], bins=40, color='#3AAFA9', edgecolor='white', alpha=0.9)
                ax.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero edge')
                ax.set_xlabel('Edge Score')
                ax.set_ylabel('Count')
                ax.set_title('Edge Score Distribution — CNN-LSTM')
                ax.legend()
                st.pyplot(fig)
                plt.close()
        else:
            st.warning("CNN-LSTM results CSV not found.")

# ══════════════════════════════════════════════════════════════════════════════
# RESULTS EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
elif menu == "📊 Results Explorer":

    if active_model == "CNN-Deep (price)":
        st.title("Results Explorer — CNN-Deep Price Prediction")
        if price_df.empty:
            st.error("Price results CSV not found.")
        else:
            s = price_df[price_df['status'] == 'success'].copy()
            filt = st.radio("Filter:", ["All", "Positive R² only", "R² > 0.5", "R² > 0.7"], horizontal=True)
            if filt == "Positive R² only": s = s[s['r2'] > 0]
            elif filt == "R² > 0.5": s = s[s['r2'] > 0.5]
            elif filt == "R² > 0.7": s = s[s['r2'] > 0.7]

            search = st.text_input("🔍 Search ticker:", "")
            if search:
                s = s[s['ticker'].str.contains(search.upper(), na=False)]

            s = s.sort_values('r2', ascending=False).reset_index(drop=True)
            s.insert(0, 'Rank', range(1, len(s) + 1))
            st.write(f"**Showing {len(s)} stocks**")
            st.dataframe(
                s[['Rank','ticker','r2','dpa','mae','rmse']].style
                .format({'r2':'{:.4f}','dpa':'{:.2f}%','mae':'{:.4f}','rmse':'{:.4f}'})
                .background_gradient(subset=['r2'], cmap='RdYlGn', vmin=-1, vmax=1),
                use_container_width=True, height=600
            )
            csv = s[['Rank','ticker','r2','dpa','mae','rmse']].to_csv(index=False)
            st.download_button("💾 Download CSV", csv, "cnn_price_results.csv", "text/csv")

    else:  # CNN-LSTM
        st.title("Results Explorer — CNN-LSTM Return Prediction")
        if lstm_df.empty:
            st.error("CNN-LSTM results CSV not found.")
        else:
            s = lstm_df[lstm_df['status'] == 'success'].copy()
            filt = st.radio("Filter:", ["All", "Positive edge", "Genuine edge (Long>0 & Short<0)", "DPA > 52%"], horizontal=True)
            if filt == "Positive edge":
                s = s[s['edge_score'] > 0]
            elif filt == "Genuine edge (Long>0 & Short<0)":
                s = s[(s['long_ret_mean'] > 0) & (s['short_ret_mean'] < 0)]
            elif filt == "DPA > 52%":
                s = s[s['dpa'] > 52]

            search = st.text_input("🔍 Search ticker:", "")
            if search:
                s = s[s['ticker'].str.contains(search.upper(), na=False)]

            s = s.sort_values('edge_score', ascending=False).reset_index(drop=True)
            s.insert(0, 'Rank', range(1, len(s) + 1))
            s_display = s[['Rank','ticker','dpa','long_ret_mean','short_ret_mean','edge_score','mae']].copy()
            s_display.columns = ['Rank','Ticker','DPA (%)','Long Ret (%)','Short Ret (%)','Edge Score','MAE (%)']

            st.write(f"**Showing {len(s_display)} stocks**")
            st.dataframe(
                s_display.style
                .format({'DPA (%)':'{:.2f}','Long Ret (%)':'{:.3f}','Short Ret (%)':'{:.3f}',
                         'Edge Score':'{:.3f}','MAE (%)':'{:.3f}'})
                .map(colour_edge, subset=['Edge Score']),
                use_container_width=True, height=600
            )
            csv = s_display.to_csv(index=False)
            st.download_button("💾 Download CSV", csv, "cnnlstm_results.csv", "text/csv")

# ══════════════════════════════════════════════════════════════════════════════
# CHARTS
# ══════════════════════════════════════════════════════════════════════════════
elif menu == "📈 Charts":

    if active_model == "CNN-Deep (price)":
        st.title("Charts — CNN-Deep Price Prediction")
        if price_df.empty:
            st.error("Price results CSV not found.")
        else:
            s = price_df[price_df['status'] == 'success']
            pos = s[s['r2'] > 0]
            fig, axes = plt.subplots(2, 2, figsize=(14, 9))

            axes[0,0].hist(s['r2'], bins=30, color='#7B4FBE', edgecolor='white')
            axes[0,0].axvline(0, color='red', linestyle='--')
            axes[0,0].set_title('R² Distribution')
            axes[0,0].set_xlabel('R²')

            axes[0,1].hist(pos['mae'], bins=30, color='#27ae60', edgecolor='white')
            axes[0,1].set_title('MAE Distribution (Positive R² stocks)')
            axes[0,1].set_xlabel('MAE')

            sc = axes[1,0].scatter(s['r2'], s['mae'], c=s['r2'], cmap='RdYlGn', alpha=0.6)
            axes[1,0].set_title('R² vs MAE')
            axes[1,0].set_xlabel('R²')
            axes[1,0].set_ylabel('MAE')
            fig.colorbar(sc, ax=axes[1,0])

            axes[1,1].hist(pos['dpa'], bins=20, color='#f39c12', edgecolor='white')
            axes[1,1].axvline(50, color='red', linestyle='--', label='Random (50%)')
            axes[1,1].set_title('DPA Distribution (Positive R² stocks)')
            axes[1,1].set_xlabel('DPA (%)')
            axes[1,1].legend()

            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

    else:  # CNN-LSTM
        st.title("Charts — CNN-LSTM Return Prediction")
        if lstm_df.empty:
            st.error("CNN-LSTM results CSV not found.")
        else:
            s = lstm_df[lstm_df['status'] == 'success']
            genuine = s[(s['long_ret_mean'] > 0) & (s['short_ret_mean'] < 0)]
            fig, axes = plt.subplots(2, 2, figsize=(14, 9))

            axes[0,0].hist(s['dpa'], bins=25, color='#3AAFA9', edgecolor='white')
            axes[0,0].axvline(50, color='red', linestyle='--', label='Random')
            axes[0,0].set_title('DPA Distribution (421 stocks)')
            axes[0,0].set_xlabel('DPA (%)')
            axes[0,0].legend()

            axes[0,1].hist(s['edge_score'], bins=40, color='#3AAFA9', edgecolor='white')
            axes[0,1].axvline(0, color='red', linestyle='--', label='Zero edge')
            axes[0,1].set_title('Edge Score Distribution')
            axes[0,1].set_xlabel('Edge Score (%)')
            axes[0,1].legend()

            top15 = genuine.nlargest(15, 'edge_score')
            axes[1,0].barh(top15['ticker'][::-1], top15['edge_score'][::-1], color='#27ae60')
            axes[1,0].set_title('Top 15 by Edge Score (genuine edge only)')
            axes[1,0].set_xlabel('Edge Score (%)')

            axes[1,1].scatter(s['dpa'], s['edge_score'], c=s['edge_score'],
                              cmap='RdYlGn', alpha=0.5, vmin=-0.5, vmax=0.5)
            axes[1,1].axvline(50, color='gray', linestyle='--', linewidth=0.8)
            axes[1,1].axhline(0, color='gray', linestyle='--', linewidth=0.8)
            axes[1,1].set_title('DPA vs Edge Score')
            axes[1,1].set_xlabel('DPA (%)')
            axes[1,1].set_ylabel('Edge Score (%)')

            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# STOCK ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
elif menu == "🎯 Stock Analysis":
    st.title("Individual Stock Analysis")

    if active_model == "CNN-Deep (price)":
        df_use = price_df[price_df['status'] == 'success'] if not price_df.empty else pd.DataFrame()
        mode = 'price'
    else:
        df_use = lstm_df[lstm_df['status'] == 'success'] if not lstm_df.empty else pd.DataFrame()
        mode = 'return'

    if df_use.empty:
        st.error("Results CSV not found.")
    else:
        ticker = st.selectbox("Select a stock:", sorted(df_use['ticker'].unique()))
        row = df_use[df_use['ticker'] == ticker].iloc[0]

        st.markdown(f"## {ticker}")
        st.markdown("---")

        if mode == 'price':
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("R²", f"{row['r2']:.4f}")
            col2.metric("MAE", f"{row['mae']:.4f}")
            col3.metric("RMSE", f"{row['rmse']:.4f}")
            col4.metric("DPA", f"{row['dpa']:.2f}%")

            st.subheader("Interpretation")
            r2 = row['r2']
            if r2 > 0.7:
                st.success(f"**Excellent:** Model explains {r2*100:.1f}% of price variance.")
            elif r2 > 0.3:
                st.info(f"**Moderate:** Model explains {r2*100:.1f}% of price variance.")
            elif r2 > 0:
                st.warning(f"**Weak:** Model explains only {r2*100:.1f}% of price variance.")
            else:
                st.error(f"**Poor:** Model fails to find learnable price pattern (R²={r2:.4f}).")

        else:
            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("DPA", f"{row['dpa']:.2f}%")
            col2.metric("Long Ret", f"{row['long_ret_mean']:.3f}%")
            col3.metric("Short Ret", f"{row['short_ret_mean']:.3f}%")
            col4.metric("Edge Score", f"{row['edge_score']:.3f}%")
            col5.metric("MAE", f"{row['mae']:.3f}%")

            st.subheader("Interpretation")
            edge = row['edge_score']
            long_r = row['long_ret_mean']
            short_r = row['short_ret_mean']
            if long_r > 0 and short_r < 0:
                st.success(f"**Genuine bidirectional edge.** Long days average +{long_r:.3f}%, Short days average {short_r:.3f}%. Edge: {edge:.3f}%.")
            elif edge > 0:
                st.info(f"**Positive edge but one-sided.** Edge score: {edge:.3f}%.")
            else:
                st.error(f"**No directional edge detected.** Edge score: {edge:.3f}%.")

        st.markdown("---")
        st.subheader("📊 1-Year Price History")
        with st.spinner("Fetching market data..."):
            try:
                hist = yf.Ticker(ticker).history(period="1y")
                if not hist.empty:
                    st.line_chart(hist['Close'])
                else:
                    st.error("Could not fetch price history.")
            except:
                st.error("Error fetching data.")

# ══════════════════════════════════════════════════════════════════════════════
# BENCHMARK
# ══════════════════════════════════════════════════════════════════════════════
elif menu == "📋 Benchmark":
    st.title("Model Benchmark Comparison")
    st.markdown("All 4 models tested on the same 38 randomly sampled stocks, return prediction task.")

    # Hardcoded from poster — these are the verified final numbers
    benchmark = pd.DataFrame({
        'Model':       ['Linear Regression', 'LSTM (2-layer)', 'CNN-Deep', 'CNN-LSTM Hybrid'],
        'Avg DPA':     ['49.98%', '50.88%', '50.78%', '50.47%'],
        'Long Ret':    ['+0.053%', '+0.080%', '+0.094%', '+0.077%'],
        'Short Ret':   ['+0.037%', '+0.036%', '+0.104%', '+0.051%'],
        'Edge Score':  [0.016, 0.044, -0.010, 0.026],
        'Avg MAE':     ['4.953%', '1.636%', '1.579%', '1.607%'],
        'Verdict':     ['Weak', 'Good', 'Broken (MaxPool)', 'Best ✓'],
    })

    st.dataframe(
        benchmark.style
        .map(colour_edge, subset=['Edge Score'])
        .format({'Edge Score': '{:+.3f}%'}),
        use_container_width=True,
        hide_index=True
    )

    st.markdown("---")
    st.subheader("Why CNN-Deep is Broken for Return Prediction")
    st.info(
        "**MaxPool Temporal Destruction:** CNN-Deep uses MaxPool1d which discards the sequential "
        "ordering of technical events. On days the model predicts DOWN, stocks actually rose +0.104% "
        "on average — an inverted signal. Removing MaxPool and adding LSTM restores the temporal "
        "ordering and recovers a positive edge (+0.026%)."
    )

    st.markdown("---")
    st.subheader("Full-Scale CNN-LSTM Results (421 stocks)")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Stocks trained", "421")
    col2.metric("Genuine edge stocks", "102", "Long>0 & Short<0")
    col3.metric("Best DPA", "58.1%", "TRGP")
    col4.metric("Avg edge score", "+0.026%")

    st.markdown("---")
    st.subheader("Top Performers — CNN-LSTM")
    top = pd.DataFrame({
        'Ticker':      ['MDB', 'CRWD', 'GS', 'CSCO', 'BOLD', 'AMAT', 'TVTX', 'VLO', 'APTV', 'TRGP'],
        'DPA':         ['57.00%','52.52%','53.76%','53.35%','50.88%','51.60%','50.21%','52.42%','51.34%','58.14%'],
        'Long Ret':    ['+0.899%','+1.031%','+0.148%','+0.092%','+0.536%','+0.167%','+0.700%','+0.118%','+0.588%','+0.206%'],
        'Short Ret':   ['-0.367%','-0.085%','-0.820%','-0.804%','-0.331%','-0.590%','-0.050%','-0.537%','-0.062%','0.000%'],
        'Edge Score':  [1.265, 1.115, 0.968, 0.897, 0.866, 0.757, 0.750, 0.655, 0.650, 0.206],
        'MAE':         ['2.500%','2.023%','1.265%','1.006%','1.764%','2.005%','2.869%','1.706%','1.676%','1.372%'],
    })
    st.dataframe(
        top.style
        .map(colour_edge, subset=['Edge Score'])
        .format({'Edge Score': '{:+.3f}%'}),
        use_container_width=True,
        hide_index=True
    )

# ══════════════════════════════════════════════════════════════════════════════
# ABOUT
# ══════════════════════════════════════════════════════════════════════════════
elif menu == "ℹ️ About":
    st.title("About the Project")
    st.markdown("""
    ### AI-Based Stock Market Prediction Using Deep Convolutional Neural Networks
    **Author:** Youssef Derraz  
    **Supervisor:** Dr. Ahmed Salaheldin  
    **Institution:** University of Hertfordshire, School of Engineering & Computer Science  
    **Year:** 2026  

    ---

    ### Project Summary
    This system trains deep learning models per-stock across **446 S&P 500 stocks** using **25 years** 
    of daily OHLCV data (2000–2025) and **24 technical indicators**.

    The target variable is the **next-day percentage return** — a directly tradeable signal.

    ---

    ### Key Results
    | Phase | Finding |
    |---|---|
    | CNN-Deep (price) | R² up to 0.961, 164/332 stocks predictable, avg R² 0.558 |
    | Return benchmark | CNN-Deep broken (edge −0.010%) due to MaxPool |
    | CNN-LSTM (returns) | 102/421 stocks with genuine edge, avg +0.026% |
    | CNN + ML Hybrid | XGBoost beats CNN baseline on 69.5% of stocks |

    ---

    ### Architecture
    - **CNN-Deep:** 4 × Conv1D (32→64→128→128), BatchNorm, Dropout(0.2), MaxPool1d, FC head
    - **CNN-LSTM Hybrid:** 2 × Conv1D (64→128), no MaxPool, 2-layer LSTM (hidden=128), FC head

    ---

    ### Data Pipeline
    - **446 stocks** collected via yfinance · **332 retained** after volatility filter (std > 5% excluded)
    - **24 features:** SMA(20/50), EMA(12), RSI(14), MACD, Bollinger Bands, ATR, OBV, 4 return lags, 4 price lags
    - **60-day** sliding windows · **70/15/15** chronological train/val/test split

    ---

    ### Tools
    Python 3 · PyTorch · scikit-learn · XGBoost · yfinance · pandas · NumPy · CUDA (RTX 5070)  
    **GitHub:** github.com/youssefdraz/AI-stock-prediction-app
    """)
