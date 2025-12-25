# Volatility Forecasting: GARCH vs. Machine Learning

## Purpose
The goal of this project is to evaluate whether modern Machine Learning models (Random Forest, XGBoost) can outperform traditional econometric models (GARCH) in predicting stock market volatility. We use the S&P 500 index for forecasting and the VIX Index as a real-world benchmark to compare our predictions.

## Data
This project utilizes publicly available financial data imported via the `yfinance` API. We specifically collect:
* **S&P 500 (GSPC):** Used to calculate historical returns and realized volatility.
* **VIX Index (VIX):** Used as an external market sentiment feature and performance benchmark. 

The data is open-access, ensuring that the results of this study are fully reproducible by any user without requiring proprietary API keys.

## Project Structure
```text
Final-Project/
├── PROPOSAL.md            # Initial project proposal
├── README.md              # Project documentation
├── environment.yml        # Conda dependencies
├── requirements.txt       # Pip dependencies
├── data/
│   ├── raw/               # Original data files (SPX & VIX)
│   └── processed/         # Generated features for ML
├── main.py                # Main entry point
├── src/                   # Python source code (Modular logic)
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
Python ≥ 3.10 recommended
Developed and tested using Python 3.13.5

Prerequisites

Install the required libraries using one of the following:

```conda env create -f environment.yml```

or 

```Pip: pip install -r requirements.txt```

Main libraries: pandas, numpy, arch, scikit-learn, xgboost, yfinance, and pytest

## How to Use It
To execute the full volatility forecasting pipeline from scratch, navigate to the project folder and run in the terminal:  

```python main.py```

This command performs data loading, feature engineering, expanding-window model training, and out-of-sample evaluation. 

## Code Overview

* **<u>src/</u>**

<u>data_loader.py:</u> Retrieves S&P 500 and VIX data from Yahoo Finance, aligns time series, and computes 30-day rolling realized volatility.

<u>models.py:</u> Implements expanding-window forecasts for GARCH(1,1), Random Forest, and XGBoost models.

<u>evaluation.py:</u>Computes statistical performance metrics such as RMSE and MAE.

* **<u>results/</u>** After execution, the following outputs are generated:

<u>performance_metrics.csv:</u>Summary table comparing all models.

<u>GARCH_forecast.csv:</u> Daily volatility forecasts for GARCH.

<u>RF_forecast.csv:</u> Daily volatility forecasts for Random Forest.

<u>XGBoost_forecast.csv:</u> Daily volatility forecasts for XGBoost.


## Testing
To verify code integrity before running a full simulation, run in the terminal: 

```pytest tests/```
 
 Done by Isaure Lunven 
