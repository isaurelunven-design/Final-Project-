# Final-Project-
Portfolio analysis 

# Volatility Forecasting: GARCH vs Machine Learning

## Description
This project compares the performance of econometric and machine learning models in forecasting 1-day-ahead realized volatility of the S&P 500 index. The models used are: GARCH(1,1) (econometric benchmark), Random Forest, and XGBoost. The goal is to determine which model predicts volatility most accurately and to evaluate the trade-off between predictive accuracy and interpretability.

## Project Structure
```

.
├── data/
│   ├── raw/             # Raw SP500 and VIX data
│   └── processed/       # Features for ML models
├── results/             # Forecasts and performance metrics
├── src/                 # Python modules
│   ├── data_loader.py
│   ├── models.py
│   └── evaluation.py
└── main.py              # Main script to run the pipeline

````

## Installation / Dependencies
Make sure you have Python 3.8+ and install required packages:
```bash
pip install pandas numpy yfinance arch scikit-learn xgboost tqdm
````

## Usage

1. Run the full pipeline:

```bash
python main.py
```

This executes all steps: downloading SP500 and VIX data, computing realized volatility, feature engineering for ML, GARCH, Random Forest, and XGBoost forecasts, and model evaluation (RMSE, MAE, correlation).

2. Outputs:

* `results/performance_metrics.csv` → performance table
* `results/GARCH_forecast.csv`, `results/RF_forecast.csv`, `results/XGBoost_forecast.csv` → model forecasts

## Main Modules

* **data_loader.py**: data download and preparation, realized volatility computation, ML feature creation.
* **models.py**: GARCH estimation and ML model training / forecasting (RF and XGBoost).
* **evaluation.py**: functions to compare models and compute RMSE, MAE, and correlation.

## Notes

* **Expanding-window forecast**: forecasts are generated using a rolling window to simulate out-of-sample predictions.
* **Caching**: forecasts are saved to avoid recalculating heavy models on every run.
* **ML features**: include lagged variables, rolling statistics, and GARCH volatility as a benchmark.

## Author

Isaure Lunven

```

This is a direct copy-paste ready version for your `README.md`.  

If you want, I can also make **a more GitHub-friendly version** with badges and headings for a cleaner look. Do you want me to do that?
```
