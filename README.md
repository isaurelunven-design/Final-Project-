# Volatility Forecasting: GARCH vs. Machine Learning

## Purpose
The goal of this project is to evaluate whether modern Machine Learning models (Random Forest, XGBoost) can outperform traditional econometric models (GARCH) in predicting stock market volatility. We use the S&P 500 index for forecasting and the VIX Index as a real-world benchmark to compare our predictions.

## Project Structure
## Project Structure

```text
Final-Project/
├── PROPOSAL.md            # Initial project proposal
├── README.md              # Project documentation
├── environment.yml        # Conda dependencies
├── requirements.txt       # Pip dependencies
├── data/
│   ├── raw/               # Original data files
│   └── processed/         # Generated features for ML
├── main.py                # Main entry point
├── src/                   # Python source code
│   ├── data_loader.py     # Data & Volatility computation
│   ├── models.py          # GARCH, RF, and XGBoost forecasting
│   └── evaluation.py      # Metrics (MAE, RMSE) calculations
├── results/               # Output metrics and forecasts
└── tests/                 # Unit tests
    ├── test_data_loader.py
    ├── test_models.py
    └── test_evaluation.py
```


## Environment & Dependencies
This project was developed using Python 3.13.5.

Prerequisites

Install the required libraries using one of the following:

Conda: conda env create -f environment.yml

Pip: pip install -r requirements.txt

Main libraries: pandas, numpy, arch, scikit-learn, xgboost, yfinance, and pytest

## How to Use It
To run the complete analysis from scratch, execute the following command in your terminal: "python main.py"

Detailed Component Breakdown

1. src/ (Source Code)

- data_loader.py: Handles data acquisition via yfinance, merging datasets, and calculating the 30-day rolling Realized Volatility.
- models.py: Implements the Expanding Window forecasting logic for GARCH(1,1), Random Forest, and XGBoost models
- evaluation.py: Computes statistical error metrics (MAE, RMSE) to compare forecasts against realized data.

2. results/ (Outputs)

Once the execution is finished, the project generates:
- performance_metrics.csv: A summary table comparing all models.
- GARCH_forecast.csv, RF_forecast.csv, XGBoost_forecast.csv: Daily volatility predictions.

## Testing
To verify the code integrity before running a full simulation: "pytest tests/"
