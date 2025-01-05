import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import logging
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from pathlib import Path
from typing import List, Dict, Tuple
from dataclasses import dataclass
import math
from torch.nn import functional as F
from datetime import datetime

# Fixed model configuration for TimesNet
MODEL_CONFIG = {
    "hidden_dim": 256,
    "num_kernels": 6,
    "num_layers": 3,
    "learning_rate": 1e-3,
    "weight_decay": 0.01,
    "dropout": 0.3
}

@dataclass
class Config:
    data_path: Path
    performance_output_path: Path
    start_date: pd.Timestamp
    device: str
    batch_size: int = 2048
    epochs: int = 100
    early_stopping_patience: int = 15
    min_train_samples: int = 252
    warmup_epochs: int = 10
    max_seq_length: int = 40
    feature_dropout: float = 0.2

class Inception_Block_V1(nn.Module):
    def __init__(self, in_channels, out_channels, num_kernels=6, init_weight=True):
        super(Inception_Block_V1, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_kernels = num_kernels
        kernels = []
        for i in range(self.num_kernels):
            kernels.append(nn.Conv2d(in_channels, out_channels, 
                                   kernel_size=2*i+1, padding=i))
        self.kernels = nn.ModuleList(kernels)
        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', 
                                      nonlinearity='relu')

    def forward(self, x):
        return torch.stack([kernel(x) for kernel in self.kernels], 
                         dim=-1).sum(-1)

class TimesBlock(nn.Module):
    def __init__(self, hidden_dim, num_kernels=6):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.conv_layer = Inception_Block_V1(1, hidden_dim, num_kernels)
        self.layernorm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        x = self.layernorm(x)
        x = x.unsqueeze(1)
        x = self.conv_layer(x)
        x = x.permute(0, 2, 3, 1)
        x = self.ffn(x)
        x = x.mean(-2)
        return x

class TimesNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=3, num_kernels=6, dropout=0.1):
        super().__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.times_blocks = nn.ModuleList([
            TimesBlock(hidden_dim, num_kernels) for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x):
        # Embedding and reshape
        x = self.embedding(x)
        x = self.dropout(x)
        
        # TimesNet blocks with residual connections
        for block in self.times_blocks:
            x = x + block(x)
        
        x = self.norm(x)
        x = x.mean(dim=1)  # Global average pooling
        return self.head(x)

class StockMetrics:
    def __init__(self, stock_id: int):
        self.stock_id = stock_id
        self.quarterly_metrics = []
        self.training_start_time = None
        self.training_end_time = None
    
    def log_quarterly_performance(self, quarter, predictions, targets):
        returns = pd.Series(predictions)
        metrics = {
            'quarter': quarter,
            'r2_oos': r2_score(targets, predictions),
            'mse': mean_squared_error(targets, predictions),
            'mae': mean_absolute_error(targets, predictions),
            'pct_correct_direction': np.mean(np.sign(predictions) == np.sign(targets)),
            'sharpe_ratio': returns.mean() / returns.std() if returns.std() != 0 else 0,
            'max_drawdown': (returns.cummax() - returns).max()
        }
        self.quarterly_metrics.append(metrics)
    
    def get_summary_metrics(self):
        df = pd.DataFrame(self.quarterly_metrics)
        training_time = (self.training_end_time - self.training_start_time).total_seconds()
        
        return {
            'stock_id': self.stock_id,
            'r2_oos': df['r2_oos'].mean(),
            'r2_std': df['r2_oos'].std(),
            'pct_positive_quarters': (df['r2_oos'] > 0).mean(),
            'avg_correct_direction': df['pct_correct_direction'].mean(),
            'sharpe_ratio': df['sharpe_ratio'].mean(),
            'max_drawdown': df['max_drawdown'].max(),
            'training_time': training_time,
            'num_quarters': len(df)
        }

class FinancialPredictor:
    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device(config.device)
        self.logger = self._setup_logger()
        
    @staticmethod
    def _setup_logger() -> logging.Logger:
        logger = logging.getLogger('FinancialPredictor')
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(handler)
        return logger

    def create_dataloaders(self, X: np.ndarray, y: np.ndarray, 
                          sample_weights: np.ndarray = None) -> DataLoader:
        dataset = TensorDataset(
            torch.FloatTensor(X).to(self.device),
            torch.FloatTensor(y).to(self.device)
        )
        
        if sample_weights is not None:
            sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(sample_weights),
                replacement=True
            )
            return DataLoader(dataset, batch_size=self.config.batch_size, sampler=sampler)
        
        return DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True)

    def train_epoch(self, model: TimesNet, train_loader: DataLoader, 
                   optimizer: optim.Optimizer, criterion: nn.Module) -> float:
        model.train()
        total_loss = 0
        
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            
            output = model(X_batch)
            loss = criterion(output, y_batch.view(-1, 1))
            
            # Add L1 regularization
            l1_reg = 0.
            for param in model.parameters():
                l1_reg += torch.norm(param, 1)
            loss += 1e-5 * l1_reg
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(train_loader)

    def validate(self, model: TimesNet, val_loader: DataLoader, 
                criterion: nn.Module) -> Tuple[float, float, np.ndarray]:
        model.eval()
        total_loss = 0
        predictions = []
        targets = []
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                output = model(X_batch)
                loss = criterion(output, y_batch.view(-1, 1))
                total_loss += loss.item()
                predictions.extend(output.cpu().numpy())
                targets.extend(y_batch.cpu().numpy())
        
        val_loss = total_loss / len(val_loader)
        r2 = r2_score(targets, predictions)
        return val_loss, r2, np.array(predictions) 

    def get_training_windows(self, data: pd.DataFrame) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        quarters = sorted(data["quarter"].unique())
        windows = []
        
        for i in range(4, len(quarters) - 1):
            train_quarters = quarters[i-4:i]
            val_quarter = quarters[i]
            
            train_data = data[data["quarter"].isin(train_quarters)]
            val_data = data[data["quarter"] == val_quarter]
            
            if len(train_data) >= self.config.min_train_samples and len(val_data) >= 2:
                windows.append((train_data, val_data))
        
        return windows

    def run_prediction(self) -> pd.DataFrame:
        self.logger.info("Starting prediction process...")
        
        data = pd.read_csv(str(self.config.data_path), parse_dates=["date"])
        data = data[data["date"] >= self.config.start_date].copy()
        
        data["quarter"] = data["date"].dt.to_period("Q")
        data["same_day_return"] = data["ret"]
        data = data.sort_values(by=["permno", "date"])
        
        feature_cols = [col for col in data.columns 
                       if "Lag" in col or "MA" in col or "StdDev" in col or "EWMA" in col]
        
        all_metrics = []
        
        for stock_id in data["permno"].unique():
            self.logger.info(f"Processing stock {stock_id}")
            stock_data = data[data["permno"] == stock_id].copy()
            
            if len(stock_data) < self.config.min_train_samples:
                continue
            
            try:
                stock_metrics = StockMetrics(stock_id)
                stock_metrics.training_start_time = datetime.now()
                
                windows = self.get_training_windows(stock_data)
                
                for train_data, val_data in windows:
                    scaler = RobustScaler()
                    X_train = scaler.fit_transform(train_data[feature_cols])
                    X_val = scaler.transform(val_data[feature_cols])
                    
                    y_train = train_data["same_day_return"].values
                    y_val = val_data["same_day_return"].values
                    
                    train_weights = np.exp(np.linspace(-1, 0, len(y_train)))
                    
                    train_loader = self.create_dataloaders(X_train, y_train, train_weights)
                    val_loader = self.create_dataloaders(X_val, y_val)
                    
                    model = TimesNet(
                        input_dim=X_train.shape[1],
                        hidden_dim=MODEL_CONFIG["hidden_dim"],
                        num_layers=MODEL_CONFIG["num_layers"],
                        num_kernels=MODEL_CONFIG["num_kernels"],
                        dropout=MODEL_CONFIG["dropout"]
                    ).to(self.device)
                    
                    optimizer = optim.AdamW(
                        model.parameters(),
                        lr=MODEL_CONFIG["learning_rate"],
                        weight_decay=MODEL_CONFIG["weight_decay"]
                    )
                    
                    scheduler = optim.lr_scheduler.OneCycleLR(
                        optimizer,
                        max_lr=MODEL_CONFIG["learning_rate"],
                        epochs=self.config.epochs,
                        steps_per_epoch=len(train_loader),
                        pct_start=0.1
                    )
                    
                    criterion = nn.HuberLoss()
                    best_val_r2 = float('-inf')
                    best_predictions = None
                    patience_counter = 0
                    
                    for epoch in range(self.config.epochs):
                        train_loss = self.train_epoch(model, train_loader, optimizer, criterion)
                        val_loss, val_r2, predictions = self.validate(model, val_loader, criterion)
                        
                        scheduler.step()
                        
                        if val_r2 > best_val_r2:
                            best_val_r2 = val_r2
                            best_predictions = predictions
                            patience_counter = 0
                        else:
                            patience_counter += 1
                        
                        if patience_counter >= self.config.early_stopping_patience:
                            break
                    
                    quarter = val_data['quarter'].iloc[0]
                    stock_metrics.log_quarterly_performance(quarter, best_predictions, y_val)
                
                stock_metrics.training_end_time = datetime.now()
                summary_metrics = stock_metrics.get_summary_metrics()
                all_metrics.append(summary_metrics)
                
                self.logger.info(f"Stock {stock_id} R² OOS: {summary_metrics['r2_oos']:.4f}")
                
            except Exception as e:
                self.logger.error(f"Error processing stock {stock_id}: {str(e)}")
                continue
            
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        metrics_df = pd.DataFrame(all_metrics)
        metrics_df.to_csv(self.config.performance_output_path, index=False)
        
        return metrics_df

def main():
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Device name: {torch.cuda.get_device_name()}")
        
    config = Config(
        data_path=Path(r"C:\Users\srb019\foreign_signals_prediction\data\raw\combined_data.csv"),
        performance_output_path=Path("results/metrics/timesnet_metrics.csv"),
        start_date=pd.Timestamp("2010-03-01"),
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    predictor = FinancialPredictor(config)
    metrics_df = predictor.run_prediction()
    
    print("\nPrediction Results:")
    print(f"Total stocks processed: {len(metrics_df)}")
    print(f"Average R² OOS: {metrics_df['r2_oos'].mean():.4f}")
    print(f"Stocks with positive R²: {(metrics_df['r2_oos'] > 0).mean()*100:.1f}%")

if __name__ == "__main__":
    main()