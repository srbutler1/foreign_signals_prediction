# Foreign Signals Prediction

This project implements and compares different models for predicting foreign market signals.

## Project Structure
```
.
├── configs/           # Configuration files
├── data/             # Data directory
│   └── raw/          # Raw data files
├── models/           # Model implementations
│   ├── baseline/     # Baseline models
│   ├── experiments/  # Experimental models
│   └── timesnet/     # TimesNet implementation
└── results/          # Results and analysis
    ├── analysis/     # Detailed analysis
    └── metrics/      # Model metrics
```

## Models

### TimesNet
Implementation of the TimesNet architecture for time series prediction. The model uses inception blocks and temporal attention mechanisms to capture complex patterns in financial data.

### Neural Network with Optuna
Neural network implementation with hyperparameter optimization using Optuna. The model architecture and training parameters are automatically tuned for optimal performance.

## Usage

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run TimesNet model:
```bash
python models/timesnet/timesnet_model.py
```

3. Run Neural Network with Optuna:
```bash
python models/experiments/nn_optuna.py
```

## Model Results
(This section will be automatically updated by the ModelTracker)