"""
OPTIMIZED PREPROCESSING - 500 STOCKS
With parallel processing and progress tracking
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import warnings
import os
import pickle
import json
from datetime import datetime
from multiprocessing import Pool, cpu_count
import time

warnings.filterwarnings('ignore')


def calculate_technical_indicators(df):
    """Calculate all technical indicators for a stock"""
    
    # Make a copy
    data = df.copy()
    
    # Moving Averages
    data['SMA_20'] = data['Close'].rolling(window=20).mean()
    data['SMA_50'] = data['Close'].rolling(window=50).mean()
    data['EMA_12'] = data['Close'].ewm(span=12, adjust=False).mean()
    
    # RSI (Relative Strength Index)
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
    ema_12 = data['Close'].ewm(span=12, adjust=False).mean()
    ema_26 = data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = ema_12 - ema_26
    data['MACD_signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
    
    # Bollinger Bands
    data['BB_middle'] = data['Close'].rolling(window=20).mean()
    bb_std = data['Close'].rolling(window=20).std()
    data['BB_upper'] = data['BB_middle'] + (2 * bb_std)
    data['BB_lower'] = data['BB_middle'] - (2 * bb_std)
    
    # ATR (Average True Range)
    high_low = data['High'] - data['Low']
    high_close = np.abs(data['High'] - data['Close'].shift())
    low_close = np.abs(data['Low'] - data['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    data['ATR'] = true_range.rolling(14).mean()
    
    # OBV (On-Balance Volume)
    obv = [0]
    for i in range(1, len(data)):
        if data['Close'].iloc[i] > data['Close'].iloc[i-1]:
            obv.append(obv[-1] + data['Volume'].iloc[i])
        elif data['Close'].iloc[i] < data['Close'].iloc[i-1]:
            obv.append(obv[-1] - data['Volume'].iloc[i])
        else:
            obv.append(obv[-1])
    data['OBV'] = obv
    
    # Lagged features
    data['Close_Lag_1'] = data['Close'].shift(1)
    data['Close_Lag_2'] = data['Close'].shift(2)
    data['Close_Lag_3'] = data['Close'].shift(3)
    data['Close_Lag_5'] = data['Close'].shift(5)
    
    return data


def preprocess_single_stock(args):
    """Process single stock (for parallel processing)"""
    
    ticker, raw_path, output_dir, seq_length = args
    
    try:
        # Load data
        df = pd.read_csv(f'{raw_path}/{ticker}_prices.csv', index_col=0, parse_dates=True)
        
        # Calculate indicators
        df = calculate_technical_indicators(df)
        
        # Remove NaN
        df = df.dropna()
        
        if len(df) < seq_length + 100:  # Need minimum data
            return ticker, False, f"Insufficient data after cleanup: {len(df)} rows"
        
        # Features and target
        feature_cols = [
            'Open', 'High', 'Low', 'Volume',
            'SMA_20', 'SMA_50', 'EMA_12',
            'RSI', 'MACD', 'MACD_signal',
            'BB_middle', 'BB_upper', 'BB_lower',
            'ATR', 'OBV',
            'Close_Lag_1', 'Close_Lag_2', 'Close_Lag_3', 'Close_Lag_5'
        ]
        
        X = df[feature_cols].values
        y = df['Close'].values
        
        # Scale features
        scaler_X = StandardScaler()
        X_scaled = scaler_X.fit_transform(X)
        
        # Scale target
        scaler_y = StandardScaler()
        y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
        
        # Create sequences
        X_seq, y_seq = [], []
        for i in range(len(X_scaled) - seq_length):
            X_seq.append(X_scaled[i:i+seq_length])
            y_seq.append(y_scaled[i+seq_length])
        
        X_seq = np.array(X_seq)
        y_seq = np.array(y_seq)
        
        # Split: 70% train, 15% val, 15% test
        train_size = int(0.7 * len(X_seq))
        val_size = int(0.15 * len(X_seq))
        
        X_train = X_seq[:train_size]
        y_train = y_seq[:train_size]
        
        X_val = X_seq[train_size:train_size+val_size]
        y_val = y_seq[train_size:train_size+val_size]
        
        X_test = X_seq[train_size+val_size:]
        y_test = y_seq[train_size+val_size:]
        
        # Save arrays
        np.save(f'{output_dir}/{ticker}_X_train.npy', X_train)
        np.save(f'{output_dir}/{ticker}_y_train.npy', y_train)
        np.save(f'{output_dir}/{ticker}_X_val.npy', X_val)
        np.save(f'{output_dir}/{ticker}_y_val.npy', y_val)
        np.save(f'{output_dir}/{ticker}_X_test.npy', X_test)
        np.save(f'{output_dir}/{ticker}_y_test.npy', y_test)
        
        # Save scalers
        with open(f'{output_dir}/{ticker}_scaler_X.pkl', 'wb') as f:
            pickle.dump(scaler_X, f)
        
        with open(f'{output_dir}/{ticker}_scaler_y.pkl', 'wb') as f:
            pickle.dump(scaler_y, f)
        
        return ticker, True, f"Train:{len(X_train)} Val:{len(X_val)} Test:{len(X_test)}"
        
    except Exception as e:
        return ticker, False, str(e)


def main():
    """Main preprocessing with parallel processing"""
    
    # Configuration
    RAW_DATA_PATH = r'D:\Grad Project\stock_data_500'
    OUTPUT_DIR = r'D:\Grad Project\processed_data_500'
    SEQ_LENGTH = 60
    
    print("\n" + "="*70)
    print("PREPROCESSING 500 STOCKS - PARALLEL MODE")
    print("="*70 + "\n")
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Get tickers from metadata
    metadata_file = f'{RAW_DATA_PATH}/metadata.json'
    if os.path.exists(metadata_file):
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
            # Get list of files
            files = [f for f in os.listdir(RAW_DATA_PATH) if f.endswith('_prices.csv')]
            tickers = [f.replace('_prices.csv', '') for f in files]
    else:
        # Scan directory
        files = [f for f in os.listdir(RAW_DATA_PATH) if f.endswith('_prices.csv')]
        tickers = [f.replace('_prices.csv', '') for f in files]
    
    print(f"Found {len(tickers)} stocks to preprocess")
    print(f"Sequence length: {SEQ_LENGTH}")
    print(f"Features: 19 technical indicators")
    
    # Check for already processed
    processed = []
    for ticker in tickers:
        if os.path.exists(f'{OUTPUT_DIR}/{ticker}_X_train.npy'):
            processed.append(ticker)
    
    if processed:
        print(f"\n✓ {len(processed)} stocks already processed")
        response = input("Skip already processed stocks? (y/n): ")
        if response.lower() == 'y':
            tickers = [t for t in tickers if t not in processed]
            print(f"  Will process {len(tickers)} remaining stocks")
    
    if not tickers:
        print("\n✓ All stocks already processed!")
        return
    
    # Prepare arguments for parallel processing
    args_list = [(ticker, RAW_DATA_PATH, OUTPUT_DIR, SEQ_LENGTH) for ticker in tickers]
    
    # Determine number of processes
    num_processes = max(1, cpu_count() - 1)  # Leave 1 core free
    print(f"\nUsing {num_processes} parallel processes")
    print(f"Estimated time: {len(tickers) * 2 / num_processes / 60:.1f} minutes\n")
    
    # Process in parallel
    start_time = time.time()
    successful = []
    failed = []
    
    print(f"{'='*70}")
    print("PROCESSING...")
    print(f"{'='*70}\n")
    
    with Pool(processes=num_processes) as pool:
        results = pool.map(preprocess_single_stock, args_list)
    
    # Collect results
    for ticker, success, message in results:
        if success:
            successful.append(ticker)
            print(f"✓ {ticker:6s} - {message}")
        else:
            failed.append(ticker)
            print(f"✗ {ticker:6s} - FAILED: {message}")
    
    # Summary
    elapsed = time.time() - start_time
    
    print(f"\n{'='*70}")
    print("PREPROCESSING COMPLETE")
    print(f"{'='*70}")
    print(f"✓ Successful: {len(successful)}/{len(tickers)}")
    print(f"✗ Failed:     {len(failed)}/{len(tickers)}")
    print(f"⏱️ Time:       {elapsed/60:.1f} minutes")
    print(f"📁 Location:   {OUTPUT_DIR}/")
    
    if failed:
        print(f"\n⚠️ Failed stocks ({len(failed)}):")
        print(f"   {', '.join(failed)}")
    
    # Save metadata
    preprocessing_metadata = {
        'total_stocks': len(tickers),
        'successful': len(successful),
        'failed': len(failed),
        'failed_tickers': failed,
        'seq_length': SEQ_LENGTH,
        'num_features': 19,
        'train_split': 0.70,
        'val_split': 0.15,
        'test_split': 0.15,
        'preprocessing_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'elapsed_minutes': elapsed/60,
        'num_processes': num_processes
    }
    
    with open(f'{OUTPUT_DIR}/preprocessing_metadata.json', 'w') as f:
        json.dump(preprocessing_metadata, f, indent=2)
    
    print(f"\n✓ Metadata saved")
    print(f"\n✅ Ready for training! Run FINAL_Training_500.py")


if __name__ == "__main__":
    main()
