import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import pickle
import pandas as pd
import warnings
import time
warnings.filterwarnings('ignore')

# CNN Model (the winner!)
class CNN_Deep(nn.Module):
    """Deep CNN with 4 conv layers - our best performer"""
    def __init__(self, input_size, seq_len):
        super(CNN_Deep, self).__init__()
        self.conv1 = nn.Conv1d(input_size, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv1d(128, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2)
        self.flatten = nn.Flatten()
        flattened_size = 128 * (seq_len // 2)
        self.fc1 = nn.Linear(flattened_size, 64)
        self.fc2 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.batch_norm1 = nn.BatchNorm1d(32)
        self.batch_norm2 = nn.BatchNorm1d(64)
        self.batch_norm3 = nn.BatchNorm1d(128)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.dropout(self.relu(self.batch_norm1(self.conv1(x))))
        x = self.dropout(self.relu(self.batch_norm2(self.conv2(x))))
        x = self.dropout(self.relu(self.batch_norm3(self.conv3(x))))
        x = self.dropout(self.relu(self.conv4(x)))
        x = self.pool(x)
        x = self.flatten(x)
        x = self.dropout(self.relu(self.fc1(x)))
        return self.fc2(x)


class MultiTickerTrainer:
    def __init__(self, data_path, results_path='results'):
        self.data_path = data_path
        self.results_path = results_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.results = []
        print(f'Device: {self.device}')
        
        import os
        os.makedirs(results_path, exist_ok=True)
    
    def load_data(self, ticker):
        """Load preprocessed data for a ticker"""
        X_train = np.load(f'{self.data_path}\\{ticker}_X_train.npy')
        y_train = np.load(f'{self.data_path}\\{ticker}_y_train.npy')
        X_val = np.load(f'{self.data_path}\\{ticker}_X_val.npy')
        y_val = np.load(f'{self.data_path}\\{ticker}_y_val.npy')
        X_test = np.load(f'{self.data_path}\\{ticker}_X_test.npy')
        y_test = np.load(f'{self.data_path}\\{ticker}_y_test.npy')
        
        with open(f'{self.data_path}\\{ticker}_target_scaler.pkl', 'rb') as f:
            target_scaler = pickle.load(f)
        
        return X_train, X_val, X_test, y_train, y_val, y_test, target_scaler
    
    def train_model(self, X_train, y_train, X_val, y_val, input_size, seq_len, epochs=50, batch_size=32):
        """Train CNN model"""
        model = CNN_Deep(input_size, seq_len).to(self.device)
        
        X_train_t = torch.FloatTensor(X_train).to(self.device)
        y_train_t = torch.FloatTensor(y_train).to(self.device)
        X_val_t = torch.FloatTensor(X_val).to(self.device)
        y_val_t = torch.FloatTensor(y_val).to(self.device)
        
        dataset = TensorDataset(X_train_t, y_train_t)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
        
        best_val_loss = float('inf')
        patience_counter = 0
        patience = 10
        
        for epoch in range(epochs):
            model.train()
            train_loss = 0
            
            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                pred = model(batch_X)
                loss = criterion(pred, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                train_loss += loss.item()
            
            train_loss /= len(loader)
            
            model.eval()
            with torch.no_grad():
                val_pred = model(X_val_t)
                val_loss = criterion(val_pred, y_val_t).item()
            
            scheduler.step(val_loss)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                break
        
        return model
    
    def evaluate_model(self, model, X_test, y_test, target_scaler):
        """Evaluate model on test set"""
        model.eval()
        X_test_t = torch.FloatTensor(X_test).to(self.device)
        
        with torch.no_grad():
            y_pred_scaled = model(X_test_t).cpu().numpy()
        
        y_pred = target_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
        y_actual = target_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
        
        mae = mean_absolute_error(y_actual, y_pred)
        rmse = np.sqrt(mean_squared_error(y_actual, y_pred))
        r2 = r2_score(y_actual, y_pred)
        
        direction_actual = np.diff(y_actual) > 0
        direction_pred = np.diff(y_pred) > 0
        dpa = np.mean(direction_actual == direction_pred) * 100
        
        return {'mae': mae, 'rmse': rmse, 'r2': r2, 'dpa': dpa}
    
    def train_ticker(self, ticker):
        """Train and evaluate on a single ticker"""
        print(f'\n{"="*60}')
        print(f'Training {ticker}')
        print(f'{"="*60}')
        
        try:
            # Load data
            X_train, X_val, X_test, y_train, y_val, y_test, scaler = self.load_data(ticker)
            
            input_size = X_train.shape[2]
            seq_len = X_train.shape[1]
            
            print(f'  Data: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}')
            
            # Train model
            model = self.train_model(X_train, y_train, X_val, y_val, input_size, seq_len, epochs=50)
            
            # Evaluate
            metrics = self.evaluate_model(model, X_test, y_test, scaler)
            
            print(f'  Results: MAE=${metrics["mae"]:.2f}, R²={metrics["r2"]:.4f}, DPA={metrics["dpa"]:.2f}%')
            
            # Store results
            result = {
                'ticker': ticker,
                **metrics,
                'status': 'success'
            }
            self.results.append(result)
            
            return result
            
        except Exception as e:
            print(f'  [ERROR] {e}')
            self.results.append({
                'ticker': ticker,
                'mae': np.nan,
                'rmse': np.nan,
                'r2': np.nan,
                'dpa': np.nan,
                'status': 'failed'
            })
            return None
    
    def train_all_tickers(self, tickers):
        """Train on all tickers"""
        print(f'\n{"="*60}')
        print(f'TRAINING {len(tickers)} TICKERS')
        print(f'{"="*60}\n')

        start_time = time.time()
        
        for idx, ticker in enumerate(tickers, 1):
            elapsed = time.time() - start_time
            if idx > 1:
                avg_per_ticker = elapsed / (idx - 1)
                remaining = avg_per_ticker * (len(tickers) - (idx - 1))
                elapsed_min = elapsed / 60
                eta_min = remaining / 60
                print(f'[Time] Elapsed: {elapsed_min:.1f} min | ETA: {eta_min:.1f} min')
            else:
                print('[Time] Elapsed: 0.0 min | ETA: calculating...')

            print(f'[{idx}/{len(tickers)}] {ticker}')
            self.train_ticker(ticker)

        total_minutes = (time.time() - start_time) / 60
        print(f'\n[Time] Total training time: {total_minutes:.1f} min')
        
        # Generate report
        self.generate_report()
    
    def generate_report(self):
        """Generate comprehensive results report"""
        print(f'\n{"="*60}')
        print('GENERATING RESULTS REPORT')
        print(f'{"="*60}')
        
        df = pd.DataFrame(self.results)
        
        # Save full results
        df.to_csv(f'{self.results_path}/all_results.csv', index=False)
        print(f'[OK] Saved: {self.results_path}/all_results.csv')
        
        # Calculate statistics
        successful = df[df['status'] == 'success']
        
        summary = {
            'Total Tickers': len(df),
            'Successful': len(successful),
            'Failed': len(df) - len(successful),
            'Avg MAE': successful['mae'].mean(),
            'Avg RMSE': successful['rmse'].mean(),
            'Avg R²': successful['r2'].mean(),
            'Avg DPA': successful['dpa'].mean(),
            'Median R²': successful['r2'].median(),
            'Best R² Ticker': successful.loc[successful['r2'].idxmax(), 'ticker'],
            'Best R² Value': successful['r2'].max(),
            'Worst R² Ticker': successful.loc[successful['r2'].idxmin(), 'ticker'],
            'Worst R² Value': successful['r2'].min()
        }
        
        # Print summary
        print(f'\n{"="*60}')
        print('SUMMARY STATISTICS')
        print(f'{"="*60}')
        for key, value in summary.items():
            if isinstance(value, (int, np.integer)):
                print(f'{key:25s}: {value}')
            elif isinstance(value, str):
                print(f'{key:25s}: {value}')
            else:
                print(f'{key:25s}: {value:.4f}')
        
        # Save summary
        pd.DataFrame([summary]).to_csv(f'{self.results_path}/summary.csv', index=False)
        print(f'\n[OK] Saved: {self.results_path}/summary.csv')
        
        # Plot distribution
        self.plot_results_distribution(successful)
        
        # Top and bottom performers
        print(f'\n{"="*60}')
        print('TOP 10 PERFORMERS (by R²)')
        print(f'{"="*60}')
        top10 = successful.nlargest(10, 'r2')[['ticker', 'mae', 'r2', 'dpa']]
        print(top10.to_string(index=False))
        
        print(f'\n{"="*60}')
        print('BOTTOM 10 PERFORMERS (by R²)')
        print(f'{"="*60}')
        bottom10 = successful.nsmallest(10, 'r2')[['ticker', 'mae', 'r2', 'dpa']]
        print(bottom10.to_string(index=False))
    
    def plot_results_distribution(self, df):
        """Plot distribution of results"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # R² distribution
        axes[0, 0].hist(df['r2'], bins=30, edgecolor='black', alpha=0.7)
        axes[0, 0].axvline(df['r2'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {df["r2"].mean():.3f}')
        axes[0, 0].set_xlabel('R² Score')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Distribution of R² Scores')
        axes[0, 0].legend()
        axes[0, 0].grid(alpha=0.3)
        
        # MAE distribution
        axes[0, 1].hist(df['mae'], bins=30, edgecolor='black', alpha=0.7, color='orange')
        axes[0, 1].axvline(df['mae'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: ${df["mae"].mean():.2f}')
        axes[0, 1].set_xlabel('MAE ($)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Distribution of MAE')
        axes[0, 1].legend()
        axes[0, 1].grid(alpha=0.3)
        
        # DPA distribution
        axes[1, 0].hist(df['dpa'], bins=30, edgecolor='black', alpha=0.7, color='green')
        axes[1, 0].axvline(df['dpa'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {df["dpa"].mean():.2f}%')
        axes[1, 0].set_xlabel('Direction Prediction Accuracy (%)')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Distribution of DPA')
        axes[1, 0].legend()
        axes[1, 0].grid(alpha=0.3)
        
        # R² vs MAE scatter
        axes[1, 1].scatter(df['r2'], df['mae'], alpha=0.6)
        axes[1, 1].set_xlabel('R² Score')
        axes[1, 1].set_ylabel('MAE ($)')
        axes[1, 1].set_title('R² vs MAE')
        axes[1, 1].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.results_path}/results_distribution.png', dpi=300)
        print(f'[OK] Saved: {self.results_path}/results_distribution.png')
        plt.close()


if __name__ == "__main__":
    DATA_PATH = r'D:\Grad Project\processed_data_simple'
    RESULTS_PATH = r'D:\Grad Project\results_final'
    
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
    
    trainer = MultiTickerTrainer(DATA_PATH, RESULTS_PATH)
    trainer.train_all_tickers(TICKERS)
    
    print(f'\n{"="*60}')
    print('[OK] ALL 100 TICKERS COMPLETE!')
    print(f'{"="*60}\n')
