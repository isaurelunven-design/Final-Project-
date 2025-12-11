import pytest
import pandas as pd
from src.evaluation import evaluate_models

def simple_model_data():
    dates = pd.date_range("2023-01-01", periods=5)
    df = pd.DataFrame({
        "RealizedVol": [0.1, 0.2, 0.15, 0.18, 0.12],
        "RF_Forecast": [0.11, 0.19, 0.14, 0.17, 0.13]
    }, index=dates)
    return [df]

def test_evaluate_single_model(simple_model_data):
    metrics_df = evaluate_models(simple_model_data)
    assert "RF" in metrics_df.index                   
    assert all(col in metrics_df.columns for col in ["RMSE", "MAE", "Correlation"])
    assert (metrics_df >= 0).all().all()            # RMSE et MAE must be positif
    corr = metrics_df.loc["RF", "Correlation"]
    assert -1 <= corr <= 1                          # corrélation must be valid

@pytest.fixture
def multi_model_data():
    dates = pd.date_range("2023-01-01", periods=5)
    df = pd.DataFrame({
        "RealizedVol": [0.1, 0.2, 0.15, 0.18, 0.12],
        "RF_Forecast": [0.11, 0.19, 0.14, 0.17, 0.13],
        "GARCH_Forecast": [0.09, 0.21, 0.16, 0.19, 0.11]
    }, index=dates)
    return [df]

def test_evaluate_multiple_models(multi_model_data):
    metrics_df = evaluate_models(multi_model_data)
    assert set(metrics_df.index) == {"RF", "GARCH"}        
    rmse_values = metrics_df["RMSE"].values
    assert all(rmse_values[i] <= rmse_values[i+1] for i in range(len(rmse_values)-1))
def test_evaluate_empty_list():
    assert evaluate_models([]).empty

def test_evaluate_missing_forecast():
    df = pd.DataFrame({"RealizedVol": [0.1, 0.2]})
    assert evaluate_models([df]).empty
