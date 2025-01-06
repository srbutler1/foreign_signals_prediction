# Foreign Signals Prediction

This project implements and compares different models for predicting foreign market signals.

## Project Structure
```
.
├───configs
├───data
│   └───raw
├───models
│   ├───baseline
│   ├───experiments
│   └───timesnet
├───results
│   ├───analysis
│   └───metrics
└───utils
    └───__pycache_
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

## Known Issues and Debug Status

### Current Errors (As of <!-- LAST_UPDATED -->)

1. **Data Type Comparison Error**
   - **Error**: '<=` not supported between instances of 'float' and 'str'
   - **Status**: Working on resolution
   - **Resolution**: Believe it will be quarter handling to use string comparisons

### Resolved Issues

1. **Memory Management**
   - **Issue**: Memory errors with large datasets
   - **Resolution**: Implemented chunk-based processing
   - **Implementation**: Using chunks of 90,000 rows and batches of 10 stocks

2. **Quarter Processing Issue**
   - **Error**: Getting repeated quarters ('2010Q1') in debug output
   - **Status**: Investigating data loading and filtering
   - **Debug Output**: Sample chunks showing only '2010Q1' quarters
   - **Next Steps**: 
     - Add additional debugging for date ranges
     - Verify quarter column formatting
     - Check data filtering logic

### Debugging Status

Current debug implementations:
```python
self.logger.info(f"DEBUG - All unique quarters in chunk: {chunk_filtered['quarter'].unique()}")
self.logger.info(f"DEBUG - Date range: {chunk_filtered['date'].min()} to {chunk_filtered['date'].max()}")
self.logger.info(f"DEBUG - Number of rows in chunk: {len(chunk_filtered)}")
```

### Future Improvements

1. Add data validation for quarter formats
2. Implement more robust error handling
3. Add performance monitoring



## Last Updated
2025-01-06 15:53:12 UTC