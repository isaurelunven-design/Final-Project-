# Project Proposal: Volatility Forecasting

Isaure Lunven  
Project Proposition: Volatility Forecasting: Econometric vs Machine Learning Models

## Problem Statement and Motivation

The project addresses the following research question: Which model most accurately forecasts one day ahead realized volatility of the S&P 500, GARCH(1,1), Random Forest, or XGBoost?

Accurate volatility forecasting is a fundamental requirement in quantitative finance, as it is critical for effective risk management, option pricing, and dynamic portfolio allocation. Volatility is a complex and unobservable variable that exhibits clustering, persistence, and non linear dynamics.

This project is driven by my professional interest in asset management, where volatility modeling is a foundational skill for constructing risk adjusted investment portfolios. The motivation is rooted in connecting theoretical knowledge from concurrent courses, including Investments, Corporate Finance, and Advanced Programming, to a rigorous real world financial problem.

The empirical objective of the project is to perform a direct comparison between a traditional econometric benchmark, GARCH, and modern non linear machine learning ensemble models. The analysis will quantify performance differences while addressing the trade off between predictive accuracy and model interpretability.

## Planned Approach and Technologies

The project will be implemented in Python using industry standard libraries for financial data science.

Pandas and NumPy will be used for data collection, cleaning, and numerical computations.  
The arch library will be used to estimate the GARCH(1,1) model via maximum likelihood estimation.  
Scikit learn and XGBoost will be used to implement and train the machine learning models.  
Yfinance and fredapi will be used to collect daily financial data from Yahoo Finance and FRED.

## Planned Steps

Data collection and target computation will involve retrieving daily adjusted close prices for the S&P 500 index and the VIX index over the period 2010 to 2024. The target variable, one day ahead realized volatility, will be computed using a 30 day rolling window of squared daily returns, strictly lagged by one day to prevent data leakage.

Feature engineering will consist of constructing lagged explanatory variables for the machine learning models, including lagged realized volatility, lagged absolute returns, rolling standard deviation, and the lagged VIX value.

Model estimation and validation will include the implementation of a GARCH(1,1) model alongside Random Forest and XGBoost regressors. All models will be evaluated using an expanding window backtesting framework to replicate true out of sample forecasting conditions.

Forecasting and evaluation will generate one day ahead volatility forecasts for all models as well as a VIX based benchmark. Model performance will be assessed using standard evaluation metrics such as root mean squared error, mean absolute error, and correlation.

Reporting will consist of a structured academic report presenting the methodology, results, and conclusions, supported by a modular and well documented codebase managed using Git.

## Expected Challenges and How to Address Them

A primary challenge is the high computational cost associated with expanding window backtesting. This issue will be addressed by implementing a caching strategy that stores pre computed forecasts to disk, significantly reducing execution time and ensuring feasibility.

Another challenge concerns data quality and integrity. Financial time series data retrieved from APIs may contain missing values or misaligned indexes between the S&P 500 and VIX series. This will be mitigated through careful data cleaning procedures, including handling missing values via forward filling, re indexing to ensure date alignment, and robustness checks to confirm that the required time range is fully usable.

## Success Criteria

The project will deliver a modular and well documented codebase structured within a src Python package.  
A fully reproducible forecasting pipeline will be executable from a single entry point.  
A clear empirical conclusion will quantify the difference in predictive performance between econometric and machine learning models using RMSE and MAE, directly answering the research question.

## Stretch Goals

If time permits, the project will be extended to include a suite of unit tests to ensure robustness, code quality, and long term reliability.
