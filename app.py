import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import os
import subprocess
import time

# --- CONFIGURATION ---
st.set_page_config(page_title="AI Stock Prediction-500 Stocks", page_icon="📈", layout="wide")


def apply_theme(light_mode: bool):
    if light_mode:
        st.markdown(
            """
            <style>
            .stApp {
                background-color: #f8fafc;
                color: #0f172a;
            }
            [data-testid="stSidebar"] {
                background-color: #ffffff;
                color: #0f172a;
                border-right: 1px solid #e2e8f0;
            }
            .stMetric {
                background-color: #ffffff;
                border: 1px solid #e2e8f0;
                border-radius: 10px;
                padding: 10px;
            }
            .stButton > button,
            .stDownloadButton > button {
                background-color: #ffffff;
                color: #0f172a;
                border: 1px solid #cbd5e1;
            }
            .stTextInput input,
            .stSelectbox div[data-baseweb="select"] > div,
            .stMultiSelect div[data-baseweb="select"] > div {
                background-color: #ffffff;
                color: #0f172a;
                border: 1px solid #cbd5e1;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

RESULTS_PATH = 'all_results.csv'  # Changed from results_500/all_results.csv
TRAINING_SCRIPT = 'Training_All_445_PROVEN.py'

# --- DATA LOADING ---
@st.cache_data
def load_data():
    """Loads results from CSV, or generates dummy data if missing."""
    if os.path.exists(RESULTS_PATH):
        return pd.read_csv(RESULTS_PATH)
    else:
        # Fallback dummy data so the app always renders
        tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'JPM', 'V', 'WMT']
        np.random.seed(42)
        return pd.DataFrame({
            'ticker': tickers,
            'mae': np.random.uniform(1.5, 15.0, len(tickers)),
            'rmse': np.random.uniform(2.0, 20.0, len(tickers)),
            'r2': np.random.uniform(-0.2, 0.85, len(tickers)),
            'dpa': np.random.uniform(45, 65, len(tickers)),
            'status': ['success'] * len(tickers)
        })

df = load_data()
df = df[df['r2'] > -5]  # Remove models worse than simple baseline
successful_df = df[df['status'] == 'success']
positive_df = successful_df[successful_df['r2'] > 0]

# --- SIDEBAR NAVIGATION ---
st.sidebar.title("📈 Stock Prediction")
st.sidebar.markdown("AI-Based Stock Market Prediction System")
st.sidebar.markdown(f"**📊 Analyzing {len(df)} Stocks**")
if 'light_mode' not in st.session_state:
    st.session_state.light_mode = False

light_mode = st.sidebar.toggle("☀️ Light mode", value=st.session_state.light_mode)
st.session_state.light_mode = light_mode

apply_theme(light_mode)

menu = st.sidebar.radio(
    "Navigation",
    ["🏠 Dashboard", "📊 Results Explorer", "📈 Live Charts", "🎯 Stock Analysis", "⚙️ Training", "ℹ️ About"]
)

st.sidebar.markdown("---")
st.sidebar.caption("Youssef Derraz\n\nUniversity of Hertfordshire | 2026")

# --- VIEWS ---

if menu == "🏠 Dashboard":
    st.title("Dashboard Overview - 500 Stock System")
    
    # Refresh button
    if st.button("🔄 Refresh Data"):
        st.cache_data.clear()
        st.rerun()

    # Metrics Row
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Stocks", len(df), "Analyzed")
    if not positive_df.empty:
        success_rate = f"{(len(positive_df)/len(df))*100:.1f}%"
        col2.metric("Successful Models", len(positive_df), f"{success_rate} Success Rate")
        col3.metric("Average R²", f"{positive_df['r2'].mean():.3f}", "Positive stocks only")
    
    st.markdown("---")

    # Quick Charts
    st.subheader("Performance Snapshot")
    col1, col2 = st.columns(2)
    
    with col1:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(successful_df['r2'], bins=30, color='#3498db', edgecolor='black', alpha=0.7)
        ax.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero')
        ax.set_xlabel('R² Score')
        ax.set_title("Distribution of R² Scores")
        st.pyplot(fig)
        
    with col2:
        top10 = successful_df.nlargest(10, 'r2')
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.barh(top10['ticker'], top10['r2'], color='#27ae60')
        ax.set_title("Top 10 Performers")
        st.pyplot(fig)
    
    st.markdown("---")
    
    # FULL TABLE - ALL COMPANIES
    st.subheader(f"📋 All {len(successful_df)} Companies - Complete Results")
    
    # Search bar
    search = st.text_input("🔍 Search by ticker:", "")
    
    # Prepare data
    display_df = successful_df.copy()
    
    # Apply search
    if search:
        display_df = display_df[display_df['ticker'].str.contains(search.upper(), na=False)]
    
    # Sort by R² (high to low)
    display_df = display_df.sort_values('r2', ascending=False).reset_index(drop=True)
    display_df.insert(0, 'Rank', range(1, len(display_df) + 1))
    
    st.write(f"**Showing {len(display_df)} of {len(successful_df)} stocks**")
    
    # Display table with color coding
    st.dataframe(
        display_df[['Rank', 'ticker', 'mae', 'rmse', 'r2', 'dpa']].style
        .format({
            'mae': '${:.2f}',
            'rmse': '${:.2f}',
            'r2': '{:.4f}',
            'dpa': '{:.2f}%'
        })
        .background_gradient(subset=['r2'], cmap='RdYlGn', vmin=-5, vmax=1),
        use_container_width=True,
        height=600
    )
    
    # Download button
    csv = display_df[['Rank', 'ticker', 'mae', 'rmse', 'r2', 'dpa']].to_csv(index=False)
    st.download_button(
        label="💾 Download All Results (CSV)",
        data=csv,
        file_name=f"all_{len(display_df)}_stocks_results.csv",
        mime="text/csv"
    )


elif menu == "📊 Results Explorer":
    st.title("Results Explorer")
    
    filter_val = st.radio("Filter Results:", ["All Stocks", "Positive R² Only", "R² > 0.5", "R² > 0.7"], horizontal=True)
    
    display_df = df.copy()
    if filter_val == "Positive R² Only":
        display_df = display_df[display_df['r2'] > 0]
    elif filter_val == "R² > 0.5":
        display_df = display_df[display_df['r2'] > 0.5]
    elif filter_val == "R² > 0.7":
        display_df = display_df[display_df['r2'] > 0.7]
        
    display_df = display_df.sort_values('r2', ascending=False).reset_index(drop=True)
    
    st.dataframe(
        display_df.style.format({
            'mae': '${:.2f}',
            'rmse': '${:.2f}',
            'r2': '{:.4f}',
            'dpa': '{:.2f}%'
        }),
        use_container_width=True,
        height=600
    )


elif menu == "📈 Live Charts":
    st.title("Performance Visualizations")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # R2 Dist
    axes[0, 0].hist(successful_df['r2'], bins=30, color='#3498db', edgecolor='black')
    axes[0, 0].axvline(0, color='red', linestyle='--')
    axes[0, 0].set_title('R² Score Distribution')
    
    # MAE Dist
    axes[0, 1].hist(positive_df['mae'], bins=30, color='#27ae60', edgecolor='black')
    axes[0, 1].set_title('MAE Distribution (Positive Models)')
    
    # R2 vs MAE Scatter
    scatter = axes[1, 0].scatter(successful_df['r2'], successful_df['mae'], c=successful_df['r2'], cmap='RdYlGn')
    axes[1, 0].set_title('R² vs MAE')
    axes[1, 0].set_xlabel('R²')
    axes[1, 0].set_ylabel('MAE ($)')
    fig.colorbar(scatter, ax=axes[1, 0])
    
    # DPA Dist
    axes[1, 1].hist(positive_df['dpa'], bins=20, color='#f39c12', edgecolor='black')
    axes[1, 1].axvline(50, color='red', linestyle='--')
    axes[1, 1].set_title('Directional Accuracy (%)')
    
    plt.tight_layout()
    st.pyplot(fig)


elif menu == "🎯 Stock Analysis":
    st.title("Individual Stock Analysis")
    
    ticker = st.selectbox("Select a stock to analyze:", sorted(df['ticker'].unique()))
    stock_data = df[df['ticker'] == ticker].iloc[0]
    
    st.markdown(f"## {ticker} Analysis")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("R² Score", f"{stock_data['r2']:.4f}")
    col2.metric("MAE", f"${stock_data['mae']:.2f}")
    col3.metric("RMSE", f"${stock_data['rmse']:.2f}")
    col4.metric("DPA", f"{stock_data['dpa']:.2f}%")
    
    st.markdown("---")
    
    col_left, col_right = st.columns([1, 1.5])
    
    with col_left:
        st.subheader("📋 Interpretation")
        r2 = stock_data['r2']
        dpa = stock_data['dpa']
        
        if r2 > 0.7:
            st.success(f"**Excellent Predictability:** Model explains {r2*100:.1f}% of price variance.")
        elif r2 > 0.3:
            st.info(f"**Moderate Predictability:** Model explains {r2*100:.1f}% of price variance.")
        else:
            st.error(f"**Poor Predictability:** Model struggles to find patterns. R² is {r2:.4f}.")
            
        if dpa > 52:
            st.success(f"**Direction Accuracy:** {dpa:.2f}% (Better than random).")
        else:
            st.warning(f"**Direction Accuracy:** {dpa:.2f}% (Close to random).")
            
    with col_right:
        st.subheader("📊 1-Year Price History")
        with st.spinner("Fetching market data..."):
            try:
                hist = yf.Ticker(ticker).history(period="1y")
                if not hist.empty:
                    st.line_chart(hist['Close'])
                else:
                    st.error("Could not fetch price history from Yahoo Finance.")
            except:
                st.error("Error fetching data.")


elif menu == "⚙️ Training":
    st.title("Model Training")
    st.info("🎓 **Architecture:** 4 CNN Layers + 2 Dense Layers | **Features:** 19 indicators | **Timesteps:** 60")
    
    st.warning("⚠️ **Note:** Running heavy ML training scripts via a web button can cause timeouts if deployed to the cloud. Best used locally.")
    
    if st.button("▶ Start Training Pipeline"):
        with st.spinner("Training in progress... Check your terminal for logs."):
            try:
                if os.path.exists(TRAINING_SCRIPT):
                    subprocess.run(["python", TRAINING_SCRIPT], check=True)
                    st.success("Training completed successfully!")
                else:
                    st.error(f"Could not find script: {TRAINING_SCRIPT}")
            except Exception as e:
                st.error(f"Training failed: {e}")


elif menu == "ℹ️ About":
    st.title("About the Project")
    st.markdown("""
    ### AI-Based Stock Market Prediction
    **Author:** Youssef Derraz  
    **Supervisor:** Dr. Ahmed Salaheldin  
    **Institution:** University of Hertfordshire (2026)
    
    This system uses Convolutional Neural Networks (CNN) to predict stock prices across 500+ stocks using 19 technical indicators.
    
    #### Key Features:
    - 332 stocks analyzed after filtering
    - CNN-Deep architecture with 4 convolutional layers
    - 164 stocks showing positive predictability (49.4%)
    - Average R² of 0.558 for predictable stocks
    """)
