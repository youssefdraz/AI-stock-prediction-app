import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import yfinance as yf
import os
import subprocess

# ── CONFIGURATION ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Stock Prediction — Youssef Derraz",
    page_icon="📈",
    layout="wide"
)

# Paths — CSV files must be in the root of your GitHub repo
CNN_RESULTS_PATH      = 'cnn_results.csv'
CNNLSTM_RESULTS_PATH  = 'cnnlstm_results.csv'
CNN_TRAINING_SCRIPT   = 'Training_All_445.py'
CNNLSTM_TRAINING_SCRIPT = 'Training_CNNLSTM_445.py'

# Benchmark comparison data (from your actual results)
BENCHMARK_DATA = {
    'Model':       ['Linear Regression', 'LSTM (2-layer)', 'CNN-Deep', 'CNN-LSTM Hybrid'],
    'Avg DPA (%)': [49.98, 50.88, 50.78, 50.47],
    'Long Ret (%)': [0.053, 0.080, 0.094, 0.077],
    'Short Ret (%)': [0.037, 0.036, 0.104, 0.051],
    'Edge Score':  [0.016, 0.044, -0.010, 0.026],
    'Avg MAE (%)': [4.953, 1.636, 1.579, 1.607],
}


# ── DATA LOADING ────────────────────────────────────────────────────────────────
@st.cache_data
def load_csv(path, label):
    if os.path.exists(path):
        df = pd.read_csv(path)
        return df
    return None


def make_dummy_cnn():
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'JPM', 'V', 'WMT']
    np.random.seed(42)
    return pd.DataFrame({
        'ticker': tickers,
        'mae': np.random.uniform(1.5, 15.0, len(tickers)),
        'rmse': np.random.uniform(2.0, 20.0, len(tickers)),
        'r2': np.random.uniform(0.3, 0.98, len(tickers)),
        'dpa': np.random.uniform(45, 65, len(tickers)),
        'status': ['success'] * len(tickers)
    })


def make_dummy_cnnlstm():
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'JPM', 'V', 'WMT']
    np.random.seed(99)
    return pd.DataFrame({
        'ticker': tickers,
        'mae': np.random.uniform(1.0, 3.0, len(tickers)),
        'rmse': np.random.uniform(1.5, 4.0, len(tickers)),
        'r2': np.random.uniform(-0.05, 0.02, len(tickers)),
        'dpa': np.random.uniform(49, 58, len(tickers)),
        'long_ret_mean': np.random.uniform(0.05, 0.3, len(tickers)),
        'short_ret_mean': np.random.uniform(-0.2, 0.05, len(tickers)),
        'edge_score': np.random.uniform(-0.1, 0.5, len(tickers)),
        'status': ['success'] * len(tickers)
    })


# Load both datasets
raw_cnn     = load_csv(CNN_RESULTS_PATH, 'CNN')
raw_cnnlstm = load_csv(CNNLSTM_RESULTS_PATH, 'CNN-LSTM')

cnn_df     = raw_cnn     if raw_cnn     is not None else make_dummy_cnn()
cnnlstm_df = raw_cnnlstm if raw_cnnlstm is not None else make_dummy_cnnlstm()

# Filter and prep CNN data (price prediction — use R²)
cnn_df = cnn_df[cnn_df['r2'] > -5]
cnn_success = cnn_df[cnn_df['status'] == 'success']
cnn_positive = cnn_success[cnn_success['r2'] > 0]

# Filter and prep CNN-LSTM data (return prediction — use edge_score)
cnnlstm_df = cnnlstm_df[cnnlstm_df['status'] == 'success']
# Edge-positive stocks: long_ret > 0 AND short_ret < 0
has_edge_cols = ('long_ret_mean' in cnnlstm_df.columns and 'short_ret_mean' in cnnlstm_df.columns)
if has_edge_cols:
    cnnlstm_edge = cnnlstm_df[
        (cnnlstm_df['long_ret_mean'] > 0) & (cnnlstm_df['short_ret_mean'] < 0)
    ]
else:
    cnnlstm_edge = cnnlstm_df


# ── SIDEBAR ──────────────────────────────────────────────────────────────────
st.sidebar.title("📈 Stock Prediction")
st.sidebar.markdown("**AI-Based Stock Market Prediction**")
st.sidebar.markdown("University of Hertfordshire | 2026")
st.sidebar.markdown("---")

# Model selector
model_choice = st.sidebar.radio(
    "Active model",
    ["CNN-Deep (price)", "CNN-LSTM (returns)"],
    help="CNN-Deep predicts price level. CNN-LSTM predicts next-day % return."
)

st.sidebar.markdown("---")

menu = st.sidebar.radio(
    "Navigation",
    ["🏠 Dashboard", "📊 Results Explorer", "📈 Charts", "🎯 Stock Analysis",
     "🔬 Benchmark", "⚙️ Training", "ℹ️ About"]
)

st.sidebar.markdown("---")
st.sidebar.caption("Youssef Derraz\nSupervisor: Dr. Ahmed Salaheldin")

# Pick active dataframe based on model choice
is_cnnlstm = (model_choice == "CNN-LSTM (returns)")
active_df  = cnnlstm_df  if is_cnnlstm else cnn_success
active_label = "CNN-LSTM Hybrid" if is_cnnlstm else "CNN-Deep"


# ── HELPER ───────────────────────────────────────────────────────────────────
def model_badge(is_lstm):
    if is_lstm:
        return "🟢 CNN-LSTM — return prediction"
    return "🟣 CNN-Deep — price prediction"


# ══════════════════════════════════════════════════════════════════════════════
# DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
if menu == "🏠 Dashboard":
    st.title("Dashboard — AI Stock Prediction")
    st.caption(model_badge(is_cnnlstm))

    if st.button("🔄 Refresh"):
        st.cache_data.clear()
        st.rerun()

    st.markdown("---")

    # ── Metric cards ──
    if is_cnnlstm:
        avg_dpa   = cnnlstm_df['dpa'].mean() if 'dpa' in cnnlstm_df.columns else 0
        avg_edge  = cnnlstm_df['edge_score'].mean() if 'edge_score' in cnnlstm_df.columns else 0
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Stocks trained",    len(cnnlstm_df))
        col2.metric("With real edge",    len(cnnlstm_edge), "Long>0 & Short<0")
        col3.metric("Avg DPA",           f"{avg_dpa:.2f}%", "50% = random")
        col4.metric("Avg edge score",    f"{avg_edge:.4f}", "+0.026 actual")
    else:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Stocks trained",    len(cnn_success))
        col2.metric("Positive R²",       len(cnn_positive),
                    f"{len(cnn_positive)/len(cnn_success)*100:.1f}% of trained")
        col3.metric("Best R²",           f"{cnn_positive['r2'].max():.3f}" if not cnn_positive.empty else "N/A")
        col4.metric("Avg R² (positive)", f"{cnn_positive['r2'].mean():.3f}" if not cnn_positive.empty else "N/A")

    st.markdown("---")

    # ── Charts ──
    st.subheader("Performance snapshot")
    col1, col2 = st.columns(2)

    if is_cnnlstm:
        with col1:
            fig, ax = plt.subplots(figsize=(8, 4))
            if 'dpa' in cnnlstm_df.columns:
                ax.hist(cnnlstm_df['dpa'], bins=30, color='#0D7377', edgecolor='black', alpha=0.8)
                ax.axvline(50, color='red', linestyle='--', linewidth=2, label='Random (50%)')
                ax.set_xlabel('DPA (%)')
                ax.set_title('Direction Prediction Accuracy — CNN-LSTM')
                ax.legend()
            st.pyplot(fig)
            plt.close()

        with col2:
            fig, ax = plt.subplots(figsize=(8, 4))
            if 'edge_score' in cnnlstm_df.columns:
                ax.hist(cnnlstm_df['edge_score'], bins=30, color='#0D7377', edgecolor='black', alpha=0.8)
                ax.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero edge')
                ax.set_xlabel('Edge Score')
                ax.set_title('Edge Score Distribution — CNN-LSTM')
                ax.legend()
            st.pyplot(fig)
            plt.close()
    else:
        with col1:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.hist(cnn_success['r2'], bins=30, color='#5B2D8E', edgecolor='black', alpha=0.8)
            ax.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero')
            ax.set_xlabel('R² Score')
            ax.set_title('R² Score Distribution — CNN-Deep')
            ax.legend()
            st.pyplot(fig)
            plt.close()

        with col2:
            top10 = cnn_positive.nlargest(10, 'r2') if not cnn_positive.empty else cnn_success.head(10)
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.barh(top10['ticker'], top10['r2'], color='#5B2D8E')
            ax.set_xlabel('R²')
            ax.set_title('Top 10 Stocks by R² — CNN-Deep')
            st.pyplot(fig)
            plt.close()

    st.markdown("---")

    # ── Full results table ──
    st.subheader(f"All {len(active_df)} stocks — {active_label} results")
    search = st.text_input("🔍 Search ticker:", "")
    display_df = active_df.copy()

    if search:
        display_df = display_df[display_df['ticker'].str.contains(search.upper(), na=False)]

    if is_cnnlstm:
        sort_col = 'edge_score' if 'edge_score' in display_df.columns else 'dpa'
        display_df = display_df.sort_values(sort_col, ascending=False).reset_index(drop=True)
        display_df.insert(0, 'Rank', range(1, len(display_df) + 1))
        cols = ['Rank', 'ticker', 'dpa', 'long_ret_mean', 'short_ret_mean', 'edge_score', 'mae']
        cols = [c for c in cols if c in display_df.columns]
        fmt = {c: '{:.4f}' for c in cols if c not in ('Rank', 'ticker', 'status')}
        if 'dpa' in fmt: fmt['dpa'] = '{:.2f}%'
        st.dataframe(
            display_df[cols].style.format(fmt)
            .background_gradient(subset=['edge_score'] if 'edge_score' in cols else ['dpa'],
                                 cmap='RdYlGn', vmin=-0.5, vmax=0.5),
            use_container_width=True, height=500
        )
    else:
        display_df = display_df.sort_values('r2', ascending=False).reset_index(drop=True)
        display_df.insert(0, 'Rank', range(1, len(display_df) + 1))
        cols = ['Rank', 'ticker', 'mae', 'rmse', 'r2', 'dpa']
        cols = [c for c in cols if c in display_df.columns]
        st.dataframe(
            display_df[cols].style.format({
                'mae': '{:.4f}', 'rmse': '{:.4f}',
                'r2': '{:.4f}', 'dpa': '{:.2f}%'
            })
            .background_gradient(subset=['r2'], cmap='RdYlGn', vmin=-1, vmax=1),
            use_container_width=True, height=500
        )

    csv = display_df.to_csv(index=False)
    st.download_button(
        label="💾 Download results CSV",
        data=csv,
        file_name=f"{active_label.replace(' ', '_')}_results.csv",
        mime="text/csv"
    )


# ══════════════════════════════════════════════════════════════════════════════
# RESULTS EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
elif menu == "📊 Results Explorer":
    st.title("Results Explorer")
    st.caption(model_badge(is_cnnlstm))

    if is_cnnlstm:
        filter_val = st.radio(
            "Filter by:",
            ["All stocks", "Real edge only (Long>0 & Short<0)", "Edge score > 0.1", "DPA > 52%"],
            horizontal=True
        )
        display_df = cnnlstm_df.copy()
        if filter_val == "Real edge only (Long>0 & Short<0)" and has_edge_cols:
            display_df = display_df[(display_df['long_ret_mean'] > 0) & (display_df['short_ret_mean'] < 0)]
        elif filter_val == "Edge score > 0.1" and 'edge_score' in display_df.columns:
            display_df = display_df[display_df['edge_score'] > 0.1]
        elif filter_val == "DPA > 52%" and 'dpa' in display_df.columns:
            display_df = display_df[display_df['dpa'] > 52]

        sort_col = 'edge_score' if 'edge_score' in display_df.columns else 'dpa'
        display_df = display_df.sort_values(sort_col, ascending=False).reset_index(drop=True)
        st.write(f"**{len(display_df)} stocks shown**")
        cols = [c for c in ['ticker', 'dpa', 'long_ret_mean', 'short_ret_mean', 'edge_score', 'mae', 'status']
                if c in display_df.columns]
        fmt = {c: '{:.4f}' for c in cols if c not in ('ticker', 'status')}
        if 'dpa' in fmt: fmt['dpa'] = '{:.2f}%'
        st.dataframe(
            display_df[cols].style.format(fmt)
            .background_gradient(subset=['edge_score'] if 'edge_score' in cols else [],
                                 cmap='RdYlGn', vmin=-0.5, vmax=0.5),
            use_container_width=True, height=600
        )
    else:
        filter_val = st.radio(
            "Filter by R²:",
            ["All stocks", "Positive R² only", "R² > 0.5", "R² > 0.7"],
            horizontal=True
        )
        display_df = cnn_success.copy()
        if filter_val == "Positive R² only":    display_df = display_df[display_df['r2'] > 0]
        elif filter_val == "R² > 0.5":          display_df = display_df[display_df['r2'] > 0.5]
        elif filter_val == "R² > 0.7":          display_df = display_df[display_df['r2'] > 0.7]
        display_df = display_df.sort_values('r2', ascending=False).reset_index(drop=True)
        st.write(f"**{len(display_df)} stocks shown**")
        cols = [c for c in ['ticker', 'mae', 'rmse', 'r2', 'dpa', 'status'] if c in display_df.columns]
        st.dataframe(
            display_df[cols].style.format({
                'mae': '{:.4f}', 'rmse': '{:.4f}',
                'r2': '{:.4f}', 'dpa': '{:.2f}%'
            })
            .background_gradient(subset=['r2'], cmap='RdYlGn', vmin=-1, vmax=1),
            use_container_width=True, height=600
        )


# ══════════════════════════════════════════════════════════════════════════════
# CHARTS
# ══════════════════════════════════════════════════════════════════════════════
elif menu == "📈 Charts":
    st.title("Performance Visualisations")
    st.caption(model_badge(is_cnnlstm))

    if is_cnnlstm:
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('CNN-LSTM Hybrid — Return Prediction Results', fontsize=14)

        if 'dpa' in cnnlstm_df.columns:
            axes[0, 0].hist(cnnlstm_df['dpa'], bins=30, color='#0D7377', edgecolor='black', alpha=0.8)
            axes[0, 0].axvline(50, color='red', linestyle='--', label='Random (50%)')
            axes[0, 0].set_title('DPA Distribution')
            axes[0, 0].set_xlabel('DPA (%)')
            axes[0, 0].legend()

        if 'edge_score' in cnnlstm_df.columns:
            axes[0, 1].hist(cnnlstm_df['edge_score'], bins=30, color='#0D7377', edgecolor='black', alpha=0.8)
            axes[0, 1].axvline(0, color='red', linestyle='--', label='Zero edge')
            axes[0, 1].set_title('Edge Score Distribution')
            axes[0, 1].set_xlabel('Edge Score')
            axes[0, 1].legend()

        if has_edge_cols:
            axes[1, 0].scatter(
                cnnlstm_df['long_ret_mean'], cnnlstm_df['short_ret_mean'],
                c=cnnlstm_df['edge_score'] if 'edge_score' in cnnlstm_df.columns else '#0D7377',
                cmap='RdYlGn', alpha=0.6
            )
            axes[1, 0].axhline(0, color='black', linestyle='--', linewidth=0.8)
            axes[1, 0].axvline(0, color='black', linestyle='--', linewidth=0.8)
            axes[1, 0].set_xlabel('Long Return Mean (%)')
            axes[1, 0].set_ylabel('Short Return Mean (%)')
            axes[1, 0].set_title('Long vs Short Returns\n(ideal: top-left quadrant)')

        if 'edge_score' in cnnlstm_df.columns:
            top10 = cnnlstm_df.nlargest(10, 'edge_score')
            axes[1, 1].barh(top10['ticker'], top10['edge_score'], color='#0D7377')
            axes[1, 1].axvline(0, color='red', linestyle='--')
            axes[1, 1].set_title('Top 10 by Edge Score')
            axes[1, 1].set_xlabel('Edge Score')

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    else:
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('CNN-Deep — Price Prediction Results', fontsize=14)

        axes[0, 0].hist(cnn_success['r2'], bins=30, color='#5B2D8E', edgecolor='black', alpha=0.8)
        axes[0, 0].axvline(0, color='red', linestyle='--', label='Zero')
        axes[0, 0].set_title('R² Score Distribution')
        axes[0, 0].legend()

        if not cnn_positive.empty:
            axes[0, 1].hist(cnn_positive['mae'], bins=30, color='#27ae60', edgecolor='black', alpha=0.8)
            axes[0, 1].set_title('MAE Distribution (Positive R² stocks)')
            axes[0, 1].set_xlabel('MAE')

        scatter = axes[1, 0].scatter(
            cnn_success['r2'], cnn_success['mae'],
            c=cnn_success['r2'], cmap='RdYlGn', alpha=0.6
        )
        axes[1, 0].set_title('R² vs MAE')
        axes[1, 0].set_xlabel('R²')
        axes[1, 0].set_ylabel('MAE')
        fig.colorbar(scatter, ax=axes[1, 0])

        axes[1, 1].hist(cnn_success['dpa'], bins=20, color='#f39c12', edgecolor='black', alpha=0.8)
        axes[1, 1].axvline(50, color='red', linestyle='--', label='Random (50%)')
        axes[1, 1].set_title('DPA Distribution')
        axes[1, 1].legend()

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# STOCK ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
elif menu == "🎯 Stock Analysis":
    st.title("Individual Stock Analysis")
    st.caption(model_badge(is_cnnlstm))

    ticker_list = sorted(active_df['ticker'].unique())
    ticker = st.selectbox("Select a stock:", ticker_list)
    row = active_df[active_df['ticker'] == ticker].iloc[0]

    st.markdown(f"## {ticker}")

    if is_cnnlstm:
        cols = st.columns(4)
        cols[0].metric("DPA", f"{row['dpa']:.2f}%" if 'dpa' in row else "N/A")
        cols[1].metric("Edge score", f"{row['edge_score']:.4f}" if 'edge_score' in row else "N/A")
        cols[2].metric("Long ret mean", f"{row['long_ret_mean']:.4f}%" if 'long_ret_mean' in row else "N/A")
        cols[3].metric("Short ret mean", f"{row['short_ret_mean']:.4f}%" if 'short_ret_mean' in row else "N/A")
    else:
        cols = st.columns(4)
        cols[0].metric("R²",   f"{row['r2']:.4f}")
        cols[1].metric("MAE",  f"{row['mae']:.4f}")
        cols[2].metric("RMSE", f"{row['rmse']:.4f}" if 'rmse' in row else "N/A")
        cols[3].metric("DPA",  f"{row['dpa']:.2f}%" if 'dpa' in row else "N/A")

    st.markdown("---")
    col_left, col_right = st.columns([1, 1.5])

    with col_left:
        st.subheader("Interpretation")
        if is_cnnlstm:
            edge = row['edge_score'] if 'edge_score' in row else 0
            dpa  = row['dpa'] if 'dpa' in row else 50
            long_r = row['long_ret_mean'] if 'long_ret_mean' in row else 0
            short_r = row['short_ret_mean'] if 'short_ret_mean' in row else 0

            if edge > 0.2:
                st.success(f"**Strong edge:** Score {edge:.4f} — clear directional signal.")
            elif edge > 0:
                st.info(f"**Modest edge:** Score {edge:.4f} — weak but positive signal.")
            else:
                st.error(f"**No edge:** Score {edge:.4f} — model lacks directional signal for this stock.")

            if long_r > 0 and short_r < 0:
                st.success("**Real directional edge confirmed** — Long>0, Short<0.")
            else:
                st.warning("**Partial signal** — short side not fully corrected.")

            if dpa > 54:
                st.success(f"**DPA {dpa:.2f}%** — meaningfully above random baseline.")
            elif dpa > 51:
                st.info(f"**DPA {dpa:.2f}%** — slightly above random.")
            else:
                st.warning(f"**DPA {dpa:.2f}%** — close to random (50%).")
        else:
            r2  = row['r2']
            dpa = row['dpa'] if 'dpa' in row else 50
            if r2 > 0.7:
                st.success(f"**Excellent:** Model explains {r2*100:.1f}% of price variance.")
            elif r2 > 0.3:
                st.info(f"**Moderate:** Model explains {r2*100:.1f}% of price variance.")
            elif r2 > 0:
                st.warning(f"**Weak:** Model explains {r2*100:.1f}% of price variance.")
            else:
                st.error(f"**Poor:** R² = {r2:.4f} — model cannot predict this stock reliably.")
            if dpa > 52:
                st.success(f"**DPA {dpa:.2f}%** — better than random direction.")
            else:
                st.warning(f"**DPA {dpa:.2f}%** — close to random direction.")

    with col_right:
        st.subheader("1-year price history")
        with st.spinner("Fetching from Yahoo Finance..."):
            try:
                hist = yf.Ticker(ticker).history(period="1y")
                if not hist.empty:
                    fig, ax = plt.subplots(figsize=(9, 4))
                    ax.plot(hist.index, hist['Close'], color='#5B2D8E', linewidth=1.5)
                    ax.set_title(f"{ticker} — Closing Price (1 Year)")
                    ax.set_xlabel("Date")
                    ax.set_ylabel("Price (USD)")
                    ax.grid(alpha=0.3)
                    st.pyplot(fig)
                    plt.close()
                else:
                    st.error("Could not fetch data from Yahoo Finance.")
            except Exception as e:
                st.error(f"Error: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# BENCHMARK COMPARISON
# ══════════════════════════════════════════════════════════════════════════════
elif menu == "🔬 Benchmark":
    st.title("Model Benchmark Comparison")
    st.markdown("All 4 models tested on the same 38 randomly sampled stocks, return prediction task.")

    bench_df = pd.DataFrame(BENCHMARK_DATA)

    # Colour the edge score column
    def colour_edge(val):
        if val > 0.02:  return 'background-color: #d4edda; color: #155724'
        elif val < 0:   return 'background-color: #f8d7da; color: #721c24'
        else:           return 'background-color: #fff3cd; color: #856404'

    st.dataframe(
        bench_df.style
        .format({
            'Avg DPA (%)': '{:.2f}%',
            'Long Ret (%)': '{:.3f}%',
            'Short Ret (%)': '{:.3f}%',
            'Edge Score': '{:.3f}',
            'Avg MAE (%)': '{:.3f}%',
        })
        .applymap(colour_edge, subset=['Edge Score']),
        use_container_width=True
    )

    st.markdown("---")
    st.subheader("Edge Score comparison")
    st.caption("Edge Score = avg return on predicted-UP days − avg return on predicted-DOWN days. Positive = genuine directional signal.")

    fig, ax = plt.subplots(figsize=(10, 4))
    colors = ['#f39c12', '#27ae60', '#e74c3c', '#0D7377']
    bars = ax.bar(bench_df['Model'], bench_df['Edge Score'], color=colors, edgecolor='black', alpha=0.85)
    ax.axhline(0, color='black', linewidth=1)
    ax.set_ylabel('Edge Score')
    ax.set_title('Edge Score by Model')
    for bar, val in zip(bars, bench_df['Edge Score']):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f'{val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    st.pyplot(fig)
    plt.close()

    st.markdown("---")
    st.subheader("Key findings")

    col1, col2 = st.columns(2)
    with col1:
        st.error("**CNN-Deep problem**\nMaxPool destroys temporal order. On predicted-DOWN days, stocks actually rose +0.104% on average — inverted signal.")
    with col2:
        st.success("**CNN-LSTM fix**\nPreserving sequence order flips the edge from −0.010 to +0.026. 102/421 stocks show genuine two-sided directional edge.")


# ══════════════════════════════════════════════════════════════════════════════
# TRAINING
# ══════════════════════════════════════════════════════════════════════════════
elif menu == "⚙️ Training":
    st.title("Model Training")

    st.info("""
    **CNN-Deep** — 4× Conv1D, BatchNorm, Dropout(0.2), MaxPool, FC head. Target: closing price. Adam LR 0.001.

    **CNN-LSTM Hybrid** — 2× Conv1D (no MaxPool), 2-layer LSTM (hidden=128), FC head. Target: next-day % return. Adam LR 0.0005.
    """)

    st.warning("⚠️ Training is GPU-intensive. Best run from terminal, not via this button.")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("CNN-Deep (price prediction)")
        if st.button("▶ Train CNN-Deep"):
            with st.spinner("Running — check terminal for progress..."):
                try:
                    if os.path.exists(CNN_TRAINING_SCRIPT):
                        subprocess.run(["python", CNN_TRAINING_SCRIPT], check=True)
                        st.success("CNN-Deep training complete!")
                        st.cache_data.clear()
                    else:
                        st.error(f"Script not found: {CNN_TRAINING_SCRIPT}")
                except Exception as e:
                    st.error(f"Failed: {e}")

    with col2:
        st.subheader("CNN-LSTM Hybrid (return prediction)")
        if st.button("▶ Train CNN-LSTM"):
            with st.spinner("Running — check terminal for progress..."):
                try:
                    if os.path.exists(CNNLSTM_TRAINING_SCRIPT):
                        subprocess.run(["python", CNNLSTM_TRAINING_SCRIPT], check=True)
                        st.success("CNN-LSTM training complete!")
                        st.cache_data.clear()
                    else:
                        st.error(f"Script not found: {CNNLSTM_TRAINING_SCRIPT}")
                except Exception as e:
                    st.error(f"Failed: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# ABOUT
# ══════════════════════════════════════════════════════════════════════════════
elif menu == "ℹ️ About":
    st.title("About the Project")
    st.markdown("""
    ### AI-Based Stock Market Prediction Using Deep Convolutional Neural Networks

    **Author:** Youssef Derraz
    **Supervisor:** Dr. Ahmed Salaheldin
    **Institution:** University of Hertfordshire | Computer Science | 2026

    ---

    ### Project Summary
    This system investigates whether deep learning can identify predictable patterns across the S&P 500.
    Two architectures are trained and compared:

    | Model | Target | Key metric | Result |
    |---|---|---|---|
    | CNN-Deep | Closing price | R² | Up to 0.979 (IONS) |
    | CNN-LSTM Hybrid | Next-day % return | Edge Score | +0.026 avg, 102 stocks with real edge |

    ---

    ### Architecture Details

    **CNN-Deep**
    - 4× Conv1D layers (32→64→128→128 channels)
    - BatchNorm + Dropout(0.2) + MaxPool1d
    - Flatten → FC(64) → FC(1)
    - 1.2M parameters | Adam LR 0.001 | 50 epochs

    **CNN-LSTM Hybrid**
    - 2× Conv1D without MaxPool (preserves all 60 timesteps)
    - 2-layer LSTM (hidden=128)
    - FC(64) → FC(1)
    - 0.9M parameters | Adam LR 0.0005 | 60 epochs

    ---

    ### Dataset
    - 446 S&P 500 stocks via Yahoo Finance (2000–2025)
    - 24 technical features per timestep
    - 60-day sequence windows
    - 70% train / 15% val / 15% test (chronological)
    - Volatility filter: exclude stocks with daily return std > 5%

    ---

    ### References
    - Fama, E. F. (1970). Efficient capital markets. *The Journal of Finance, 25*(2), 383–417.
    - LeCun, Y., Bengio, Y., & Hinton, G. (2015). Deep learning. *Nature, 521*(7553), 436–444.
    - Jiang, W. (2021). Applications of deep learning in stock market prediction. *Expert Systems with Applications, 184*, 115537.
    - Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. *Neural Computation, 9*(8), 1735–1780.
    """)
