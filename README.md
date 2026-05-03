# Time Series Forecasting using Hybrid Models

## Overview
This project demonstrates a structured approach to time series forecasting using statistical and machine learning methods. The objective is to model temporal dynamics such as trend and seasonality and improve prediction accuracy through hybrid modeling.

## Note on Data
Due to confidentiality and NDA constraints, the original dataset is not included. Synthetic data is used to demonstrate the methodology while preserving the core modeling principles.

## Methodology

### 1. Data Generation
Synthetic time series data is generated to simulate:
- Trend
- Seasonality
- Noise

### 2. Statistical Modeling (SARIMA)
A SARIMA model is used to capture:
- Temporal dependencies
- Seasonal structure
- Underlying system dynamics

### 3. Machine Learning Model
A Gradient Boosting model is applied with:
- Lag-based features (lag1, lag12)
- Nonlinear pattern learning

### 4. Hybrid Modeling
A hybrid approach is implemented by combining:
- SARIMA predictions
- Machine learning predictions

This improves robustness and forecasting accuracy.

## Results
The hybrid model demonstrates improved performance by leveraging:
- Statistical structure (SARIMA)
- Nonlinear learning (ML)

## Key Insight
Combining model-based and data-driven approaches leads to better representation of complex temporal systems.

## Future Work
- Extend to deep learning models (LSTM, Transformers)
- Incorporate external variables
- Apply to real-world datasets

## How to Run
```bash
pip install numpy pandas matplotlib scikit-learn statsmodels
python main.py
