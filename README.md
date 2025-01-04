# foreign_signals_prediction

This project implements a machine learning approach to study the impact of foreign information on asset prices, focusing on detecting foreign signals that predict daily U.S. stock returns.

---

## Project Overview

The study analyzes how foreign market information affects U.S. stock returns using neural networks to capture stock-specific time-varying relationships. The implementation includes:

- Processing of lagged returns from multiple foreign markets
- Neural network models with dynamic hyperparameter optimization
- Out-of-sample return predictability analysis
- Rolling window validation approach

---

## Features

- **Data Preprocessing**: Includes scripts for preparing lagged returns from foreign markets and handling missing data.
- **Dynamic Neural Network Modeling**: Implements stock-specific neural network models with hyperparameter tuning.
- **Rolling Window Validation**: Uses rolling window validation for robust out-of-sample testing.
- **Scalability**: The project is designed to be modular and extensible for additional markets or assets.

---

## Prerequisites

Before starting, ensure you have the following installed:
- Python 3.8+ 
- pip (Python package manager)
- Git

---

## Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/foreign_signals_prediction.git
   cd foreign_signals_prediction
