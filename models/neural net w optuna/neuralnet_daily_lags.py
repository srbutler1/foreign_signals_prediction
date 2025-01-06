import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import logging
import optuna
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define the file paths
data_path = r'X:\combined_data.csv'
filtered_data_path = r'X:\filtered_data.csv'  # Path to save filtered data
performance_output_path = r'X:\performance_metrics.csv'  # Path for performance metrics

# Pre-filter the data to include only rows after 3/1/2010
logging.info("Filtering the dataset for rows after 3/1/2010...")
chunksize = 100000  # Process data in chunks to save memory
start_date = pd.Timestamp("2010-03-01")
filtered_rows = []

for chunk in pd.read_csv(data_path, parse_dates=["date"], chunksize=chunksize):
    filtered_chunk = chunk[chunk["date"] >= start_date]
    filtered_rows.append(filtered_chunk)

filtered_data = pd.concat(filtered_rows, ignore_index=True)
filtered_data.to_csv(filtered_data_path, index=False)
logging.info(f"Filtered data saved to {filtered_data_path}.")

# Load filtered data
logging.info("Loading the filtered dataset...")
merged_data = pd.read_csv(filtered_data_path, parse_dates=["date"])

# Prepare data
merged_data = merged_data.sort_values(by=["permno", "date"])
merged_data["quarter"] = merged_data["date"].dt.to_period("Q")
merged_data["same_day_return"] = merged_data["ret"]

# Foreign signal features
foreign_signal_features = [col for col in merged_data.columns if "Lag" in col or "MA" in col or "StdDev" in col or "EWMA" in col]

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Neural network model
class StudyNN(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, dropout_rate):
        super(StudyNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.bn1 = nn.BatchNorm1d(hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.bn2 = nn.BatchNorm1d(hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, 1)
        self.dropout = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

# Optuna objective function
def objective(trial):
    hidden_size1 = trial.suggest_int("hidden_size1", 64, 256)
    hidden_size2 = trial.suggest_int("hidden_size2", 32, 128)
    dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5)
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-4, 1e-2)
    weight_decay = trial.suggest_loguniform("weight_decay", 1e-6, 1e-3)

    r2_scores = []
    for train_end_idx in range(4, len(quarters) - 1):
        train_quarters = quarters[:train_end_idx]
        test_quarter = quarters[train_end_idx]

        train_data = merged_data[(merged_data["permno"] == stock_id) & (merged_data["quarter"].isin(train_quarters))]
        test_data = merged_data[(merged_data["permno"] == stock_id) & (merged_data["quarter"] == test_quarter)]

        if len(train_data) < 252 or len(test_data) == 0:
            continue

        scaler = StandardScaler()
        train_data[foreign_signal_features] = scaler.fit_transform(train_data[foreign_signal_features])
        test_data[foreign_signal_features] = scaler.transform(test_data[foreign_signal_features])

        X_train = torch.tensor(train_data[foreign_signal_features].values, dtype=torch.float32).to(device)
        y_train = torch.tensor(train_data["same_day_return"].values, dtype=torch.float32).view(-1, 1).to(device)
        X_test = torch.tensor(test_data[foreign_signal_features].values, dtype=torch.float32).to(device)
        y_test = torch.tensor(test_data["same_day_return"].values, dtype=torch.float32).view(-1, 1).to(device)

        model = StudyNN(len(foreign_signal_features), hidden_size1, hidden_size2, dropout_rate).to(device)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        criterion = nn.MSELoss()

        model.train()
        for epoch in range(75):
            optimizer.zero_grad()
            outputs = model(X_train)
            loss = criterion(outputs, y_train)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            y_pred = model(X_test).flatten().cpu().numpy()
            y_true = y_test.flatten().cpu().numpy()
            r2 = r2_score(y_true, y_pred)
            r2_scores.append(r2)

    if not r2_scores:
        return -np.inf
    return np.mean(r2_scores)

# Optimization and evaluation
optimized_params = {}
quarters = merged_data["quarter"].unique()
performance_metrics = []

for stock_id in merged_data["permno"].unique():
    logging.info(f"Optimizing hyperparameters for stock {stock_id}...")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=50)

    optimized_params[stock_id] = study.best_trial.params
    logging.info(f"Best parameters for stock {stock_id}: {study.best_trial.params}")

# Save performance metrics
performance_df = pd.DataFrame(performance_metrics)
performance_df.to_csv(performance_output_path, index=False)
logging.info("Performance metrics saved.")
