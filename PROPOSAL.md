# PROJECT PROPOSAL: VOLATILITY FORECASTING

**Auteur:** Isaure Lunven
**Proposition de Projet:** Volatility Forecasting: Econometric vs Machine Learning Models

---

## Problem Statement and Motivation

The project addresses the critical research question: **“Which model most accurately forecasts 1-day-ahead realized volatility of the S&P 500: GARCH(1,1), Random Forest, or XGBoost?”**

Accurate volatility forecasting is a fundamental requirement in quantitative finance, critical for effective risk management, option pricing, and dynamic portfolio allocation. Volatility is a complex, unobservable variable exhibiting characteristics.

This project is directly driven by my strong professional interest in **Asset Management**, where volatility modeling is a foundational skill for constructing risk-adjusted investment portfolios. The motivation is deeply rooted in connecting theoretical knowledge from various concurrent courses—including **Investments**, **Corporate Finance**, and **Advanced Programming**—to a rigorous, real-world financial problem.

The project’s empirical objective is to perform a direct comparison between the traditional econometric benchmark (GARCH) and modern, non-linear machine learning ensemble models, quantifying the performance gain and addressing the strategic trade-off between model accuracy and regulatory interpretability.

---

## Planned Approach and Technologies

The project will be implemented in Python, using industry-standard libraries essential for financial data science:

* **pandas / NumPy:** For efficient data collection, cleaning, and numerical computations.
* **arch:** To estimate the GARCH(1,1) model via Maximum Likelihood Estimation (MLE).
* **scikit-learn / xgboost:** For implementing and training the Machine Learning models.
* **yfinance / fredapi:** To collect daily stock, index, and ETF data from Yahoo Finance and FRED.

---

## Planned Steps

* **Data Collection and Target Computation:** Daily adjusted close prices for the S&P 500 Index (SPX) and the VIX Index (2010–2024) will be retrieved. The target variable, 1-day-ahead Realized Volatility (RV), will be computed using a 30-day rolling window of squared daily returns, strictly lagged by one day to prevent data leakage.
* **Feature Engineering:** For machine learning models, a set of lagged features will be constructed, including lagged RV, lagged absolute returns, rolling standard deviation, and the lagged VIX value.
* **Model Estimation and Validation:** The GARCH(1,1) model will be implemented alongside Random Forest and XGBoost regressors. All models will be validated using a rigorous **expanding-window backtesting** scheme to simulate true out-of-sample forecasting conditions.
* **Forecasting and Evaluation:** One-day-ahead volatility forecasts will be generated for all models and the VIX benchmark. Model performance will be evaluated using standard metrics: Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), and Correlation.
* **Reporting:** A structured academic report will be produced, detailing the methodology, results, and conclusions, along with a modular codebase managed via Git.

---

## Expected Challenges and How to Address Them

The primary challenge anticipated is the **high computational cost** associated with expanding-window backtesting. **Mitigation:** A custom caching strategy will be implemented to store pre-computed forecasts to disk, significantly reducing execution time and ensuring project feasibility.

Another crucial challenge relates to **data quality and integrity**. Financial time-series data retrieved from APIs often contain missing values or misaligned indexes between the S\&P 500 and VIX series. **Mitigation:** The initial data loading phase will involve extensive cleaning, including **imputing missing values** using standard techniques like forward filling, **re-indexing** the series to ensure perfect date synchronization, and performing **robustness checks** to confirm the required time range is fully covered and usable.

---

## Success Criteria

* A modular and well-documented codebase will be implemented within a structured `src/` Python package.
* A robust, reproducible forecasting pipeline executable from a single entry point.
* A clear empirical conclusion will be drawn, quantifying the difference in predictive power (RMSE, MAE) between econometric and machine learning models, providing a definitive answer to the research question.

---

## Stretch Goals (If Time Allows)

If the core objectives are achieved ahead of schedule, the project will be extended to include the implementation of a comprehensive suite of **unit tests** to ensure the robustness, quality, and reliability of the entire codebase.