import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import ta
import warnings
import os
import pickle

warnings.filterwarnings('ignore')

class SimpleStockPreprocessor:
    """Simplified preprocessing using sklearn's StandardScaler - proven to work"""
    def __init__(self, data_path):
        self.data_path = data_path
        self.feature_scaler = StandardScaler()
        self.target_scaler = StandardScaler()
        self.feature_columns = []
        
    def load_data(self, ticker):
        path = f'{self.data_path}\\{ticker}_prices.csv'
        if not os.path.exists(path):
            raise FileNotFoundError(f"Could not find file for {ticker}")
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        return df
    
    def add_technical_indicators(self, df):
        """Add technical indicators"""
        df['SMA_20'] = ta.trend.sma_indicator(df['Close'], window=20)
        df['SMA_50'] = ta.trend.sma_indicator(df['Close'], window=50)
        df['EMA_12'] = ta.trend.ema_indicator(df['Close'], window=12)
        df['RSI'] = ta.momentum.rsi(df['Close'], window=14)
        
        macd = ta.trend.MACD(df['Close'])
        df['MACD'] = macd.macd()
        df['MACD_Signal'] = macd.macd_signal()
        
        bollinger = ta.volatility.BollingerBands(df['Close'])
        df['BB_Width'] = bollinger.bollinger_wband()
        df['ATR'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'])
        
        df['OBV'] = ta.volume.on_balance_volume(df['Close'], df['Volume'])
        df['Returns'] = df['Close'].pct_change()
        df['Volatility'] = df['Returns'].rolling(window=20).std()
        
        # Lagged features (NOT including current Close!)
        for lag in [1, 2, 3, 5]:
            df[f'Close_Lag_{lag}'] = df['Close'].shift(lag)
        
        return df.ffill().bfill().dropna()
    
    def create_sequences(self, X, y, seq_length=60):
        """Create sequences"""
        X_seq, y_seq = [], []
        for i in range(seq_length, len(X)):
            X_seq.append(X[i-seq_length:i])
            y_seq.append(y[i])
        return np.array(X_seq), np.array(y_seq)

    def preprocess_pipeline(self, ticker, seq_length=60):
        print(f'\n{"="*60}')
        print(f'Processing {ticker} (Simple StandardScaler Approach)')
        print(f'{"="*60}')
        
        # Load and add indicators
        df = self.load_data(ticker)
        df = self.add_technical_indicators(df)
        
        # Define features (NO Close!)
        all_features = [c for c in df.columns if c not in ['Close', 'Dividends', 'Stock Splits']]
        target_col = 'Close'
        
        # Split data
        train_size = int(len(df) * 0.7)
        val_size = int(len(df) * 0.15)
        
        train_df = df.iloc[:train_size].copy()
        val_df = df.iloc[train_size:train_size+val_size].copy()
        test_df = df.iloc[train_size+val_size:].copy()
        
        print(f'  Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}')
        print(f'  Features: {len(all_features)}')
        print(f'  Price range: ${train_df[target_col].min():.2f} to ${train_df[target_col].max():.2f}')
        
        # Store feature names
        self.feature_columns = all_features
        
        # Fit scalers on training data ONLY
        self.feature_scaler.fit(train_df[all_features].values)
        self.target_scaler.fit(train_df[[target_col]].values)
        
        # Transform all sets
        X_train = self.feature_scaler.transform(train_df[all_features].values)
        X_val = self.feature_scaler.transform(val_df[all_features].values)
        X_test = self.feature_scaler.transform(test_df[all_features].values)
        
        y_train = self.target_scaler.transform(train_df[[target_col]].values)
        y_val = self.target_scaler.transform(val_df[[target_col]].values)
        y_test = self.target_scaler.transform(test_df[[target_col]].values)
        
        # Create sequences
        X_train_seq, y_train_seq = self.create_sequences(X_train, y_train, seq_length)
        X_val_seq, y_val_seq = self.create_sequences(X_val, y_val, seq_length)
        X_test_seq, y_test_seq = self.create_sequences(X_test, y_test, seq_length)
        
        print(f'  Sequences: Train={len(X_train_seq)}, Val={len(X_val_seq)}, Test={len(X_test_seq)}')
        print(f'  Shape: X={X_train_seq.shape}, y={y_train_seq.shape}')
        
        return {
            'X_train': X_train_seq,
            'y_train': y_train_seq,
            'X_val': X_val_seq,
            'y_val': y_val_seq,
            'X_test': X_test_seq,
            'y_test': y_test_seq,
            'feature_columns': self.feature_columns
        }
    
    def save_processed_data(self, data_dict, ticker, output_path):
        """Save with scalers for proper inverse transform"""
        os.makedirs(output_path, exist_ok=True)
        
        # Save arrays
        np.save(f'{output_path}\\{ticker}_X_train.npy', data_dict['X_train'])
        np.save(f'{output_path}\\{ticker}_y_train.npy', data_dict['y_train'])
        np.save(f'{output_path}\\{ticker}_X_val.npy', data_dict['X_val'])
        np.save(f'{output_path}\\{ticker}_y_val.npy', data_dict['y_val'])
        np.save(f'{output_path}\\{ticker}_X_test.npy', data_dict['X_test'])
        np.save(f'{output_path}\\{ticker}_y_test.npy', data_dict['y_test'])
        
        # Save scalers (critical for inverse transform!)
        with open(f'{output_path}\\{ticker}_target_scaler.pkl', 'wb') as f:
            pickle.dump(self.target_scaler, f)
        
        # Save metadata
        metadata = {
            'features': data_dict['feature_columns'],
            'n_features': len(data_dict['feature_columns'])
        }
        pd.DataFrame([metadata]).to_json(f'{output_path}\\{ticker}_metadata.json', orient='records', indent=2)
        
        print(f'  [OK] Saved NPY files and scaler for {ticker}')

if __name__ == "__main__":
    DATA_PATH = r'D:\Grad Project\Improved_stock_data'
    OUTPUT_PATH = r'D:\Grad Project\processed_data_simple'
    
    # All 100 tickers
    TICKERS = [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK-B', 'LLY', 'AVGO',
        'JPM', 'V', 'XOM', 'UNH', 'MA', 'HD', 'PG', 'JNJ', 'COST', 'ABBV',
        'NFLX', 'BAC', 'CRM', 'CVX', 'MRK', 'KO', 'ADBE', 'WMT', 'PEP', 'TMO',
        'CSCO', 'ACN', 'LIN', 'MCD', 'AMD', 'ABT', 'DIS', 'IBM', 'GE', 'CMCSA',
        'INTC', 'DHR', 'INTU', 'VZ', 'TXN', 'QCOM', 'AMGN', 'PFE', 'CAT', 'NEE',
        'AMAT', 'HON', 'ORCL', 'PM', 'UNP', 'COP', 'SPGI', 'RTX', 'GS', 'LOW',
        'BA', 'T', 'NKE', 'UPS', 'DE', 'SBUX', 'AXP', 'BLK', 'MDT', 'LMT',
        'BKNG', 'ELV', 'PLD', 'GILD', 'ADI', 'TJX', 'SYK', 'MMC', 'C', 'REGN',
        'VRTX', 'ISRG', 'MDLZ', 'ADP', 'CVS', 'SCHW', 'MO', 'PGR', 'CB', 'CI',
        'ZTS', 'BMY', 'SO', 'ETN', 'DUK', 'EOG', 'WM', 'BSX', 'ITW', 'APD'
    ]
    
    preprocessor = SimpleStockPreprocessor(DATA_PATH)
    
    successful = []
    failed = []
    
    print(f'\n{"="*60}')
    print(f'PREPROCESSING {len(TICKERS)} TICKERS')
    print(f'{"="*60}\n')
    
    for idx, ticker in enumerate(TICKERS, 1):
        try:
            print(f'[{idx}/{len(TICKERS)}] Processing {ticker}...')
            processed_data = preprocessor.preprocess_pipeline(ticker, seq_length=60)
            preprocessor.save_processed_data(processed_data, ticker, OUTPUT_PATH)
            successful.append(ticker)
        except Exception as e:
            print(f'  [ERROR] {ticker}: {e}')
            failed.append(ticker)
            continue
    
    print(f'\n{"="*60}')
    print('PREPROCESSING SUMMARY')
    print(f'{"="*60}')
    print(f'[OK] Successful: {len(successful)}/{len(TICKERS)}')
    print(f'[FAILED] {len(failed)}/{len(TICKERS)}')
    if failed:
        print(f'Failed tickers: {", ".join(failed)}')
    print(f'{"="*60}\n')
