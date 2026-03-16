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
import os
from pathlib import Path

warnings.filterwarnings('ignore')


# ══════════════════════════════════════════════════════════
# OLD MODEL — kept for meeting discussion, not used in training
# ══════════════════════════════════════════════════════════

class CNN_Deep(nn.Module):
    """
    Original CNN — good for price prediction (shape patterns),
    weaker for return prediction (sequence memory needed).
    Kept here for reference and meeting comparison.
    """
    def __init__(self, input_size, seq_len):
        super(CNN_Deep, self).__init__()
        self.conv1 = nn.Conv1d(input_size, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv1d(128, 128, kernel_size=3, padding=1)
        self.pool  = nn.MaxPool1d(2)
        self.flatten = nn.Flatten()
        flattened_size = 128 * (seq_len // 2)
        self.fc1 = nn.Linear(flattened_size, 64)
        self.fc2 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.dropout    = nn.Dropout(0.2)
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


# ══════════════════════════════════════════════════════════
# NEW MODEL — CNN-LSTM Hybrid
# ══════════════════════════════════════════════════════════

class CNN_LSTM(nn.Module):
    """
    CNN-LSTM Hybrid for return prediction.

    Architecture:
      1. CNN block: extracts local technical features from each
         short window (momentum bursts, volatility spikes, RSI crossings)
      2. LSTM block: reads the sequence of CNN feature vectors across
         the full 60-day window, preserving temporal order and memory
      3. FC head: maps final LSTM state to predicted return

    Why it works better than CNN alone:
      - CNN never sees the full 60-day context — MaxPool destroys order
      - LSTM reads ALL 60 CNN feature vectors in order
      - Long-range patterns (3-week momentum, earnings drift) are captured
    """
    def __init__(self, input_size, seq_len,
                 cnn_channels=64,
                 lstm_hidden=128,
                 lstm_layers=2,
                 dropout=0.3):
        super(CNN_LSTM, self).__init__()

        # ── CNN feature extractor (no pooling — preserve all timesteps) ──
        # kernel_size=3 means each position sees itself + 1 day each side
        self.conv1 = nn.Conv1d(input_size,   cnn_channels,     kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(cnn_channels, cnn_channels * 2, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm1d(cnn_channels)
        self.bn2   = nn.BatchNorm1d(cnn_channels * 2)

        cnn_out_channels = cnn_channels * 2  # 128

        # ── LSTM sequence reader ──
        # Input: (batch, seq_len, cnn_out_channels)
        # Reads all 60 CNN feature vectors in order
        self.lstm = nn.LSTM(
            input_size  = cnn_out_channels,
            hidden_size = lstm_hidden,
            num_layers  = lstm_layers,
            batch_first = True,
            dropout     = dropout if lstm_layers > 1 else 0.0,
        )

        # ── FC prediction head ──
        self.fc1     = nn.Linear(lstm_hidden, 64)
        self.fc2     = nn.Linear(64, 1)
        self.relu    = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x shape: (batch, seq_len, input_size)

        # CNN expects (batch, channels, seq_len)
        x = x.permute(0, 2, 1)
        x = self.dropout(self.relu(self.bn1(self.conv1(x))))
        x = self.dropout(self.relu(self.bn2(self.conv2(x))))

        # Back to (batch, seq_len, cnn_channels) for LSTM
        x = x.permute(0, 2, 1)

        # LSTM reads all 60 feature vectors in order
        lstm_out, _ = self.lstm(x)

        # Take only the last timestep's hidden state
        x = lstm_out[:, -1, :]

        # FC head
        x = self.dropout(self.relu(self.fc1(x)))
        return self.fc2(x)


# ══════════════════════════════════════════════════════════
# TRAINER
# ══════════════════════════════════════════════════════════

class MultiTickerTrainer:
    def __init__(self, data_path, results_path='results'):
        self.data_path   = Path(data_path)
        self.results_path = Path(results_path)
        self.device      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.results     = []
        print(f'Device: {self.device}')
        print(f'Model:  CNN-LSTM Hybrid')

        if not self.data_path.exists():
            raise FileNotFoundError(
                f"Data path does not exist: {self.data_path}\n"
                f"Please check the DATA_PATH variable at the bottom of this script."
            )

        self.results_path.mkdir(parents=True, exist_ok=True)

    def load_data(self, ticker):
        """Load preprocessed return-based data for a ticker"""
        try:
            X_train = np.load(self.data_path / f'{ticker}_X_train.npy')
            y_train = np.load(self.data_path / f'{ticker}_y_train.npy')
            X_val   = np.load(self.data_path / f'{ticker}_X_val.npy')
            y_val   = np.load(self.data_path / f'{ticker}_y_val.npy')
            X_test  = np.load(self.data_path / f'{ticker}_X_test.npy')
            y_test  = np.load(self.data_path / f'{ticker}_y_test.npy')
            with open(self.data_path / f'{ticker}_scaler_y.pkl', 'rb') as f:
                target_scaler = pickle.load(f)
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Missing file for ticker '{ticker}': {e}")
        except Exception as e:
            raise RuntimeError(f"Failed to load data for ticker '{ticker}': {e}")

        return X_train, X_val, X_test, y_train, y_val, y_test, target_scaler

    def train_model(self, X_train, y_train, X_val, y_val,
                    input_size, seq_len, epochs=60, batch_size=32):
        """Train CNN-LSTM model"""
        model = CNN_LSTM(input_size, seq_len).to(self.device)

        y_train = np.asarray(y_train).reshape(-1, 1)
        y_val   = np.asarray(y_val).reshape(-1, 1)

        X_train_t = torch.FloatTensor(X_train).to(self.device)
        y_train_t = torch.FloatTensor(y_train).to(self.device)
        X_val_t   = torch.FloatTensor(X_val).to(self.device)
        y_val_t   = torch.FloatTensor(y_val).to(self.device)

        dataset = TensorDataset(X_train_t, y_train_t)
        loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        criterion = nn.MSELoss()
        # Slightly lower LR than CNN — LSTM benefits from slower convergence
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=7, min_lr=1e-6
        )

        best_val_loss    = float('inf')
        best_model_state = None
        patience_counter = 0
        patience         = 15  # slightly more patience than CNN (LSTM learns slower)

        for epoch in range(epochs):
            model.train()
            train_loss = 0

            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                pred = model(batch_X)
                loss = criterion(pred.squeeze(), batch_y.squeeze())
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                train_loss += loss.item()

            train_loss /= len(loader)

            model.eval()
            with torch.no_grad():
                val_loss = criterion(
                    model(X_val_t).squeeze(), y_val_t.squeeze()
                ).item()

            scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss    = val_loss
                best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                break

        if best_model_state:
            model.load_state_dict(best_model_state)

        return model

    def evaluate_model(self, model, X_test, y_test, target_scaler):
        """Evaluate model — target is % return, no inverse_transform needed"""
        model.eval()
        y_test   = np.asarray(y_test).flatten()
        X_test_t = torch.FloatTensor(X_test).to(self.device)

        with torch.no_grad():
            y_pred = model(X_test_t).cpu().numpy().flatten()

        mae  = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2   = r2_score(y_test, y_pred)

        # DPA: did we call the direction right?
        direction_actual = y_test > 0
        direction_pred   = y_pred > 0
        dpa = np.mean(direction_actual == direction_pred) * 100

        # Edge metrics: avg actual return on days we predicted UP vs DOWN
        long_ret  = y_test[direction_pred].mean()  * 100 if direction_pred.any()   else 0.0
        short_ret = y_test[~direction_pred].mean() * 100 if (~direction_pred).any() else 0.0

        return {
            'mae': mae * 100,
            'rmse': rmse * 100,
            'r2': r2,
            'dpa': dpa,
            'long_ret_mean':  long_ret,
            'short_ret_mean': short_ret,
            'edge_score':     long_ret - short_ret,
        }

    def train_ticker(self, ticker):
        """Train and evaluate CNN-LSTM on a single ticker"""
        print(f'\n{"="*60}')
        print(f'Training {ticker}')
        print(f'{"="*60}')

        try:
            X_train, X_val, X_test, y_train, y_val, y_test, scaler = self.load_data(ticker)

            if len(X_train) == 0:
                raise ValueError(f"Training set is empty for ticker '{ticker}'.")

            # Volatility filter — skip biotech/event stocks (daily std > 5%)
            y_std = np.std(y_train)
            if y_std > 0.05:
                print(f'  [SKIP] {ticker} — too volatile (daily std={y_std*100:.2f}%), skipping.')
                self.results.append({
                    'ticker': ticker,
                    'mae': np.nan, 'rmse': np.nan, 'r2': np.nan,
                    'dpa': np.nan, 'long_ret_mean': np.nan,
                    'short_ret_mean': np.nan, 'edge_score': np.nan,
                    'status': 'skipped_volatile'
                })
                self._save_checkpoint()
                return None

            print(f'  Volatility check: daily std={y_std*100:.2f}% — OK')

            input_size = X_train.shape[2]
            seq_len    = X_train.shape[1]

            print(f'  Data: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}')

            model   = self.train_model(X_train, y_train, X_val, y_val, input_size, seq_len, epochs=60)
            metrics = self.evaluate_model(model, X_test, y_test, scaler)

            print(
                f'  Results: MAE={metrics["mae"]:.4f}% | '
                f'R²={metrics["r2"]:.4f} | '
                f'DPA={metrics["dpa"]:.2f}% | '
                f'Edge={metrics["edge_score"]:.3f}% '
                f'(Long: {metrics["long_ret_mean"]:.3f}% / Short: {metrics["short_ret_mean"]:.3f}%)'
            )

            result = {'ticker': ticker, **metrics, 'status': 'success'}
            self.results.append(result)
            self._save_checkpoint()
            return result

        except Exception as e:
            print(f'  [ERROR] {ticker} failed: {e}')
            self.results.append({
                'ticker': ticker,
                'mae': np.nan, 'rmse': np.nan, 'r2': np.nan,
                'dpa': np.nan, 'long_ret_mean': np.nan,
                'short_ret_mean': np.nan, 'edge_score': np.nan,
                'status': 'failed'
            })
            self._save_checkpoint()
            return None

    def _save_checkpoint(self):
        """Save results after every ticker — crash-safe"""
        pd.DataFrame(self.results).to_csv(
            self.results_path / 'checkpoint_results.csv', index=False
        )

    def _load_checkpoint(self):
        """Resume from checkpoint if one exists"""
        checkpoint_path = self.results_path / 'checkpoint_results.csv'
        if checkpoint_path.exists():
            df = pd.read_csv(checkpoint_path)
            self.results = df.to_dict('records')
            done = set(df['ticker'].tolist())
            print(f'[Checkpoint] Resuming — {len(done)} tickers already done.')
            return done
        return set()

    def train_all_tickers(self, tickers):
        """Train on all tickers with checkpoint resume"""
        if not tickers:
            raise ValueError(
                "No tickers found! Check DATA_PATH contains '*_X_train.npy' files."
            )

        already_done = self._load_checkpoint()
        tickers = [t for t in tickers if t not in already_done]
        if already_done:
            print(f'[Checkpoint] {len(already_done)} done, {len(tickers)} remaining.')

        print(f'\n{"="*60}')
        print(f'TRAINING {len(tickers)} TICKERS — CNN-LSTM HYBRID')
        print(f'{"="*60}\n')

        start_time = time.time()

        for idx, ticker in enumerate(tickers, 1):
            elapsed = time.time() - start_time
            if idx > 1:
                avg   = elapsed / (idx - 1)
                eta   = avg * (len(tickers) - (idx - 1))
                print(f'[Time] Elapsed: {elapsed/60:.1f} min | ETA: {eta/60:.1f} min')
            else:
                print('[Time] Elapsed: 0.0 min | ETA: calculating...')

            print(f'[{idx}/{len(tickers)}] {ticker}')
            self.train_ticker(ticker)

        total = (time.time() - start_time) / 60
        print(f'\n[Time] Total training time: {total:.1f} min')
        self.generate_report()

    def generate_report(self):
        """Generate results report with edge-score focused summary"""
        print(f'\n{"="*60}')
        print('GENERATING RESULTS REPORT')
        print(f'{"="*60}')

        df         = pd.DataFrame(self.results)
        successful = df[df['status'] == 'success']
        skipped    = df[df['status'] == 'skipped_volatile']

        df.to_csv(self.results_path / 'all_results.csv', index=False)
        print(f'[OK] Saved: {self.results_path / "all_results.csv"}')

        if len(skipped) > 0:
            print(f'\n[INFO] Skipped {len(skipped)} volatile tickers: {list(skipped["ticker"])}')

        if successful.empty:
            print('[WARNING] No tickers trained successfully.')
            return

        # Stocks with genuine directional edge (long > 0 AND short < 0)
        with_edge = successful[
            (successful['long_ret_mean'] > 0) & (successful['short_ret_mean'] < 0)
        ]

        summary = {
            'Total Tickers':       len(df),
            'Successful':          len(successful),
            'Failed':              len(df[df['status'] == 'failed']),
            'Skipped (volatile)':  len(skipped),
            'With real edge':      len(with_edge),
            'Avg DPA (%)':         successful['dpa'].mean(),
            'Avg Edge Score':      successful['edge_score'].mean(),
            'Avg Long Ret (%)':    successful['long_ret_mean'].mean(),
            'Avg Short Ret (%)':   successful['short_ret_mean'].mean(),
            'Avg MAE (%)':         successful['mae'].mean(),
            'Avg R²':              successful['r2'].mean(),
            'Best Edge Ticker':    successful.loc[successful['edge_score'].idxmax(), 'ticker'],
            'Best Edge Score':     successful['edge_score'].max(),
            'Best DPA Ticker':     successful.loc[successful['dpa'].idxmax(), 'ticker'],
            'Best DPA Value':      successful['dpa'].max(),
        }

        print(f'\n{"="*60}')
        print('SUMMARY STATISTICS')
        print(f'{"="*60}')
        for key, value in summary.items():
            if isinstance(value, (int, np.integer)):
                print(f'{key:28s}: {value}')
            elif isinstance(value, str):
                print(f'{key:28s}: {value}')
            else:
                print(f'{key:28s}: {value:.4f}')

        pd.DataFrame([summary]).to_csv(self.results_path / 'summary.csv', index=False)
        print(f'\n[OK] Saved: {self.results_path / "summary.csv"}')

        self.plot_results(successful)

        # Top performers by edge score (most useful for trading)
        print(f'\n{"="*60}')
        print('TOP 10 BY EDGE SCORE (best for trading)')
        print(f'{"="*60}')
        cols = ['ticker', 'dpa', 'long_ret_mean', 'short_ret_mean', 'edge_score', 'mae']
        print(successful.nlargest(10, 'edge_score')[cols].to_string(index=False))

        print(f'\n{"="*60}')
        print('TOP 10 BY DPA')
        print(f'{"="*60}')
        print(successful.nlargest(10, 'dpa')[cols].to_string(index=False))

        print(f'\n{"="*60}')
        print(f'STOCKS WITH REAL DIRECTIONAL EDGE: {len(with_edge)}')
        print('(long_ret > 0 AND short_ret < 0 — these are tradeable)')
        print(f'{"="*60}')
        if not with_edge.empty:
            print(with_edge.nlargest(20, 'edge_score')[cols].to_string(index=False))

    def plot_results(self, df):
        """Plot distribution of results focused on return metrics"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('CNN-LSTM Hybrid — Results Distribution', fontsize=14)

        # DPA distribution
        axes[0, 0].hist(df['dpa'], bins=30, edgecolor='black', alpha=0.7, color='steelblue')
        axes[0, 0].axvline(50, color='black', linestyle=':', linewidth=1, label='Random baseline (50%)')
        axes[0, 0].axvline(df['dpa'].mean(), color='red', linestyle='--', linewidth=2,
                           label=f'Mean: {df["dpa"].mean():.2f}%')
        axes[0, 0].set_xlabel('Direction Prediction Accuracy (%)')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('DPA Distribution')
        axes[0, 0].legend()
        axes[0, 0].grid(alpha=0.3)

        # Edge score distribution
        axes[0, 1].hist(df['edge_score'], bins=30, edgecolor='black', alpha=0.7, color='green')
        axes[0, 1].axvline(0, color='black', linestyle=':', linewidth=1, label='Zero edge')
        axes[0, 1].axvline(df['edge_score'].mean(), color='red', linestyle='--', linewidth=2,
                           label=f'Mean: {df["edge_score"].mean():.3f}%')
        axes[0, 1].set_xlabel('Edge Score (Long% - Short%)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Edge Score Distribution')
        axes[0, 1].legend()
        axes[0, 1].grid(alpha=0.3)

        # Long ret vs Short ret scatter
        axes[1, 0].scatter(df['long_ret_mean'], df['short_ret_mean'], alpha=0.5, color='purple')
        axes[1, 0].axhline(0, color='black', linestyle=':', linewidth=1)
        axes[1, 0].axvline(0, color='black', linestyle=':', linewidth=1)
        axes[1, 0].set_xlabel('Long Return Mean (%)')
        axes[1, 0].set_ylabel('Short Return Mean (%)')
        axes[1, 0].set_title('Long vs Short Returns\n(ideal: top-left quadrant)')
        axes[1, 0].grid(alpha=0.3)
        # Shade the ideal quadrant
        axes[1, 0].axhspan(
            df['short_ret_mean'].min(), 0,
            xmin=0.5, xmax=1.0,
            alpha=0.08, color='green', label='Ideal zone'
        )
        axes[1, 0].legend()

        # MAE distribution
        axes[1, 1].hist(df['mae'], bins=30, edgecolor='black', alpha=0.7, color='orange')
        axes[1, 1].axvline(df['mae'].mean(), color='red', linestyle='--', linewidth=2,
                           label=f'Mean: {df["mae"].mean():.4f}%')
        axes[1, 1].set_xlabel('MAE (% return)')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('MAE Distribution')
        axes[1, 1].legend()
        axes[1, 1].grid(alpha=0.3)

        plt.tight_layout()
        out_path = self.results_path / 'results_distribution.png'
        plt.savefig(out_path, dpi=300)
        print(f'[OK] Saved: {out_path}')
        plt.close()


# ══════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════

if __name__ == "__main__":
    DATA_PATH    = r'E:\Grad Project\processed_data_500_returns'
    RESULTS_PATH = r'E:\Grad Project\results_500_cnnlstm'   # separate folder from CNN results

    files   = [f for f in os.listdir(DATA_PATH) if f.endswith('_X_train.npy')]
    TICKERS = sorted([f.replace('_X_train.npy', '') for f in files])

    print(f"Found {len(TICKERS)} preprocessed stocks")
    print(f"First 10: {TICKERS[:10]}")
    print(f"Last 10:  {TICKERS[-10:]}")

    trainer = MultiTickerTrainer(DATA_PATH, RESULTS_PATH)
    trainer.train_all_tickers(TICKERS)

    print(f'\n{"="*60}')
    print(f'[OK] ALL {len(TICKERS)} STOCKS COMPLETE — CNN-LSTM HYBRID')
    print(f'{"="*60}\n')
