import os
import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import logging
import gc
import yaml
from pathlib import Path
from typing import List, Dict, Tuple
from dataclasses import dataclass
import math
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from datetime import datetime
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))


from utils.model_tracker import ModelTracker


# 1. CONFIG CLASS & LOADING FROM YAML

@dataclass
class Config:
    data_path: Path
    performance_output_path: Path
    start_date: pd.Timestamp
    device: str
    model_params: Dict
    training_params: Dict
    data_params: Dict

    def __post_init__(self):
        # Validate paths
        if not isinstance(self.data_path, Path):
            self.data_path = Path(self.data_path)
        if not isinstance(self.performance_output_path, Path):
            self.performance_output_path = Path(self.performance_output_path)
        
        # Create output directory if it doesn't exist
        self.performance_output_path.parent.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_yaml(cls, model_config_path: str, data_config_path: str):
        # Load model config
        with open(model_config_path, 'r') as f:
            model_dict = yaml.safe_load(f)
        # Load data config
        with open(data_config_path, 'r') as f:
            data_dict = yaml.safe_load(f)

        
        project_root = Path(__file__).resolve().parent.parent.parent

        
        # Construct absolute paths
        data_path = project_root / data_dict['paths']['data_path']
        perf_path = project_root / data_dict['paths']['performance_output_path']
        
        return cls(
            data_path=data_path,
            performance_output_path=perf_path,
            start_date=pd.Timestamp(data_dict['dates']['start_date']),
            device=data_dict['hardware']['device'],
            model_params=model_dict['model'],        # e.g. hidden_dim, dropout, etc.
            training_params=model_dict['training'],  # e.g. epochs, early_stopping_patience, etc.
            data_params=model_dict['data']           # e.g. min_train_samples, max_seq_length, etc.
        )


# 2. MODEL COMPONENTS

class Inception_Block_V1(nn.Module):
    """
    Example Inception-like block from your code.
    """
    def __init__(self, in_channels, out_channels, num_kernels=6, init_weight=True):
        super(Inception_Block_V1, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_kernels = num_kernels
        kernels = []
        for i in range(self.num_kernels):
            kernels.append(nn.Conv2d(in_channels, out_channels, kernel_size=2 * i + 1, padding=i))
        self.kernels = nn.ModuleList(kernels)
        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        # x shape: (B, C=1, seq_len, hidden_dim) for instance
        return torch.stack([kernel(x) for kernel in self.kernels], dim=-1).sum(-1)


class TimesBlock(nn.Module):
    """
    One 'TimesBlock' that uses Inception, LayerNorm, and an FFN.
    """
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
        """
        x: shape (batch_size, seq_len, hidden_dim)
        """
        batch_size, seq_len, hidden_dim = x.shape
        # Normalization
        x = self.layernorm(x)
        # Expand channel dimension for Conv2d
        x = x.unsqueeze(1)  # (B, 1, seq_len, hidden_dim)
        x = self.conv_layer(x)  # -> shape (B, hidden_dim, seq_len, ???) but we sum along last dim
        # Permute to (B, seq_len, hidden_dim)
        x = x.permute(0, 2, 3, 1)  # shape: (B, seq_len, ???, hidden_dim)
        # Possibly the shape might differ if we need to reshape. We'll assume ??? = 1
        # We'll reduce along dimension -2
        x = self.ffn(x)
        # mean across seq_len dimension (some aggregator)
        x = x.mean(dim=-2)  # shape: (B, seq_len, hidden_dim) -> (B, ???, hidden_dim)
        return x


class TimesNet(nn.Module):
    """
    A TimesNet-inspired model that takes in time series features and produces a single next-day return.
    """
    def __init__(self, 
                 input_dim: int, 
                 hidden_dim: int = 256, 
                 num_layers: int = 3, 
                 num_kernels: int = 6, 
                 dropout: float = 0.3, 
                 feature_dropout: float = 0.2):
        super().__init__()
        # Feature dropout before embedding
        self.feature_dropout = nn.Dropout(feature_dropout)
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
        
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        # x shape: (B, seq_len, input_dim)
        batch_size, seq_len, input_dim = x.shape
        
        # Feature dropout: drop some of the input features
        x = self.feature_dropout(x)
        
        # Project input_dim -> hidden_dim
        x = self.embedding(x)  # (B, seq_len, hidden_dim)
        x = self.dropout(x)    # additional dropout

        for block in self.times_blocks:
            x = x + block(x)  # Residual

        # Final layer norm
        x = self.norm(x)  
        
        # Pool across sequence dimension
        x = x.mean(dim=1)  # shape: (B, hidden_dim)
        
        # Output single next-day return
        return self.head(x)


# 3. METRICS & LOGGING

class StockMetrics:
    """
    Collects and stores metrics for each stock across quarters.
    """
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
        if df.empty:
            return {}
        training_time = 0
        if self.training_start_time and self.training_end_time:
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


# 4. MAIN FINANCIAL PREDICTOR CLASS

class FinancialPredictor:
    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device(config.device)
        self.logger = self._setup_logger()
        self.tracker = ModelTracker(Path(__file__).parent.parent.parent)

    @staticmethod
    def _setup_logger() -> logging.Logger:
        logger = logging.getLogger('FinancialPredictor')
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
            logger.addHandler(handler)
        return logger

    def create_dataloaders(self, X: np.ndarray, y: np.ndarray, sample_weights: np.ndarray = None) -> DataLoader:
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
            return DataLoader(dataset, batch_size=self.config.training_params['batch_size'], sampler=sampler)
        
        return DataLoader(dataset, batch_size=self.config.training_params['batch_size'], shuffle=True)

    def train_epoch(self, 
                    model: TimesNet, 
                    train_loader: DataLoader, 
                    optimizer: optim.Optimizer, 
                    scheduler: optim.lr_scheduler.OneCycleLR,
                    criterion: nn.Module,
                    l1_alpha: float = 1e-5) -> float:
        model.train()
        total_loss = 0.0
        
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            
            output = model(X_batch)
            loss = criterion(output, y_batch.view(-1, 1))
            
            # L1 regularization
            l1_reg = sum(torch.norm(p, 1) for p in model.parameters())
            loss += l1_alpha * l1_reg
            
            loss.backward()
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
        
        return total_loss / len(train_loader)

    def validate(self, 
                 model: TimesNet, 
                 val_loader: DataLoader, 
                 criterion: nn.Module) -> Tuple[float, float, np.ndarray]:
        model.eval()
        total_loss = 0.0
        predictions = []
        targets = []
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                output = model(X_batch)
                loss = criterion(output, y_batch.view(-1, 1))
                total_loss += loss.item()
                predictions.extend(output.cpu().numpy().flatten())
                targets.extend(y_batch.cpu().numpy().flatten())
        
        predictions = np.array(predictions)
        targets = np.array(targets)
        val_loss = total_loss / len(val_loader)
        r2 = r2_score(targets, predictions) if len(np.unique(targets)) > 1 else 0
        
        return val_loss, r2, predictions

    def get_training_windows(self, data: pd.DataFrame) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Create rolling training windows by quarter.
        e.g. 4 quarters for training, next quarter for validation, sliding forward.
        """
        quarters = sorted(data['quarter'].unique())
        windows = []
        stock_id = data['permno'].iloc[0]
        
        self.logger.info(f"Creating windows for stock {stock_id}")
        self.logger.info(f"Total quarters available: {len(quarters)} from {quarters[0]} to {quarters[-1]}")
        
        for i in range(4, len(quarters) - 1):
            train_quarters = quarters[i-4:i]
            val_quarter = quarters[i]
            
            train_data = data[data['quarter'].isin(train_quarters)]
            val_data = data[data['quarter'] == val_quarter]
            
            if len(train_data) >= self.config.data_params['min_train_samples'] and len(val_data) >= 2:
                windows.append((train_data, val_data))
                
                # Only log the first and last window
                if len(windows) == 1 or i == len(quarters) - 2:
                    self.logger.info(
                        f"Window {len(windows)}: Train {train_quarters[0]}-{train_quarters[-1]}, Val {val_quarter}"
                    )
        
        self.logger.info(f"Created {len(windows)} windows for stock {stock_id}\n")
        return windows

    def run_prediction(self) -> pd.DataFrame:
        self.logger.info("Starting prediction process...")
        all_metrics = []
        
        try:
            # Get unique stock IDs
            chunk_size = 90000
            unique_permnos = []
            for chunk in pd.read_csv(str(self.config.data_path), usecols=['permno'], chunksize=chunk_size):
                unique_permnos.extend(chunk['permno'].unique())
            unique_permnos = np.unique(unique_permnos)
            
            self.logger.info(f"Total stocks to process: {len(unique_permnos)}")
            total_batches = (len(unique_permnos) + 9) // 10
            
            # Process in batches of 10 stocks
            batch_size = 10
            all_val_true, all_val_pred = [], []
            
            for batch_idx, batch_start in enumerate(range(0, len(unique_permnos), batch_size)):
                batch_permnos = unique_permnos[batch_start:batch_start + batch_size]
                self.logger.info(f"Processing batch {batch_idx + 1}/{total_batches}")
                
                # Load batch data
                chunks = []
                for chunk in pd.read_csv(str(self.config.data_path), chunksize=chunk_size):
                    chunk_filtered = chunk[chunk['permno'].isin(batch_permnos)].copy()
                    if len(chunk_filtered) > 0:
                        chunk_filtered['date'] = pd.to_datetime(chunk_filtered['date'])
                        chunk_filtered['quarter'] = chunk_filtered['quarter'].astype(str)
                        chunk_filtered = chunk_filtered[chunk_filtered['date'] >= self.config.start_date]
                        if len(chunk_filtered) > 0:
                            chunks.append(chunk_filtered)
                    del chunk_filtered
                    gc.collect()
                
                if not chunks:
                    continue
                
                data = pd.concat(chunks, ignore_index=True)
                feature_cols = [col for col in data.columns 
                            if any(x in col for x in ['Lag', 'MA', 'StdDev', 'EWMA'])]
                data["same_day_return"] = data["ret"]
                data = data.sort_values(by=["permno", "date"])
                
                for stock_id in batch_permnos:
                    try:
                        stock_data = data[data["permno"] == stock_id].copy()
                        if len(stock_data) < self.config.data_params['min_train_samples']:
                            continue
                        
                        stock_metrics = StockMetrics(stock_id)
                        stock_metrics.training_start_time = datetime.now()
                        windows = self.get_training_windows(stock_data)
                        
                        for train_data, val_data in windows:
                            # Prepare data
                            scaler = RobustScaler()
                            X_train = scaler.fit_transform(train_data[feature_cols])
                            X_val = scaler.transform(val_data[feature_cols])
                            y_train = train_data["same_day_return"].values
                            y_val = val_data["same_day_return"].values
                            
                            # Create dataloaders
                            train_weights = np.exp(np.linspace(-1, 0, len(y_train)))
                            train_loader = self.create_dataloaders(X_train, y_train, train_weights)
                            val_loader = self.create_dataloaders(X_val, y_val)
                            
                            # Initialize model
                            model = TimesNet(
                                input_dim=X_train.shape[1],
                                hidden_dim=self.config.model_params['hidden_dim'],
                                num_layers=self.config.model_params['num_layers'],
                                num_kernels=self.config.model_params['num_kernels'],
                                dropout=self.config.model_params['dropout'],
                                feature_dropout=self.config.data_params['feature_dropout']
                            ).to(self.device)
                            
                            # Training setup
                            optimizer = optim.AdamW(
                                model.parameters(),
                                lr=self.config.model_params['learning_rate'],
                                weight_decay=self.config.model_params['weight_decay']
                            )
                            
                            scheduler = optim.lr_scheduler.OneCycleLR(
                                optimizer,
                                max_lr=self.config.model_params['learning_rate'],
                                epochs=self.config.training_params['epochs'],
                                steps_per_epoch=len(train_loader),
                                pct_start=0.1
                            )
                            
                            criterion = nn.HuberLoss()
                            best_val_r2 = float('-inf')
                            best_predictions = None
                            patience_counter = 0
                            
                            # Training loop
                            for epoch in range(self.config.training_params['epochs']):
                                train_loss = self.train_epoch(
                                    model, train_loader, optimizer, scheduler, criterion
                                )
                                val_loss, val_r2, predictions = self.validate(
                                    model, val_loader, criterion
                                )
                                
                                if val_r2 > best_val_r2:
                                    best_val_r2 = val_r2
                                    best_predictions = predictions
                                    patience_counter = 0
                                else:
                                    patience_counter += 1
                                
                                if patience_counter >= self.config.training_params['early_stopping_patience']:
                                    break
                            
                            # Log performance
                            quarter = val_data['quarter'].iloc[0]
                            stock_metrics.log_quarterly_performance(quarter, best_predictions, y_val)
                            
                            # Track metrics
                            self.tracker.save_model_metrics(
                                model_name="TimesNet",
                                y_true=y_val,
                                y_pred=best_predictions,
                                hyperparameters=self.config.model_params,
                                notes=f"Stock {stock_id}, Quarter {quarter}",
                                fold="validation"
                            )
                            
                            # Collect all predictions
                            all_val_true.extend(y_val)
                            all_val_pred.extend(best_predictions)
                        
                        # Summarize stock performance
                        stock_metrics.training_end_time = datetime.now()
                        summary_metrics = stock_metrics.get_summary_metrics()
                        all_metrics.append(summary_metrics)
                        
                        self.logger.info(f"Stock {stock_id} R² OOS: {summary_metrics['r2_oos']:.4f}")
                        
                    except Exception as e:
                        self.logger.error(f"Error processing stock {stock_id}: {str(e)}")
                        continue
                    
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                
                # Cleanup batch data
                del data, chunks
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # Save overall metrics
            if all_metrics:
                metrics_df = pd.DataFrame(all_metrics)
                metrics_df.to_csv(self.config.performance_output_path, index=False)
                
                # Track overall model performance
                if all_val_true and all_val_pred:
                    self.tracker.save_model_metrics(
                        model_name="TimesNet",
                        y_true=np.array(all_val_true),
                        y_pred=np.array(all_val_pred),
                        hyperparameters=self.config.model_params,
                        notes="Full model validation performance",
                        fold="validation"
                    )
                
                return metrics_df
            else:
                self.logger.warning("No metrics collected")
                return pd.DataFrame()
                
        except Exception as e:
            self.logger.error(f"Fatal error in run_prediction: {str(e)}")
            raise


# 5. MAIN FUNCTION

def main():
    try:
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"Device name: {torch.cuda.get_device_name(0)}")

        # Adjust these paths to where your configs live
        project_root = Path(__file__).parent.parent.parent
        model_config_path = project_root / 'configs' / 'model_config.yaml'
        data_config_path = project_root / 'configs' / 'data_config.yaml'

        # Load Config
        config = Config.from_yaml(
            model_config_path=str(model_config_path),
            data_config_path=str(data_config_path)
        )

        predictor = FinancialPredictor(config)
        metrics_df = predictor.run_prediction()
        
        if not metrics_df.empty:
            print("\n=== Final Prediction Results ===")
            print(f"Total stocks processed: {len(metrics_df)}")
            print(f"Average R² OOS: {metrics_df['r2_oos'].mean():.4f}")
            print(f"Fraction of Stocks with Positive R²: {(metrics_df['r2_oos'] > 0).mean() * 100:.1f}%")
        else:
            print("\nNo results to display. Check the logs for more details.")

    except Exception as e:
        print(f"Fatal error in main: {str(e)}")
        raise


if __name__ == "__main__":
    main()
