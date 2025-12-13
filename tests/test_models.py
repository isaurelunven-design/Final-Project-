import pandas as pd
import numpy as np
import pytest
from unittest.mock import patch, MagicMock
from arch import arch_model

from src.models import (
    compute_log_returns, 
    garch_expanding_window_forecast, 
    ml_expanding_window_forecast
)

@pytest.fixture
def sample_garch_data():
    dates = pd.date_range(start="2023-01-01", periods=10, freq='D')
    sp_prices = np.array([100, 101, 102, 103, 104, 105, 106, 107, 108, 109])
    realized_vol = np.linspace(0.1, 0.2, 10)                                    # Realized Volatility data aligned with returns
    
    data = pd.DataFrame({
        "SP500": sp_prices,
        "RealizedVol": realized_vol
    }, index=dates)
    
    return data

@pytest.fixture
def sample_ml_data():
    dates = pd.date_range(start="2023-01-01", periods=10, freq='D')
    df = pd.DataFrame({
        "Feature1": np.random.rand(10),
        "Feature2": np.random.rand(10),
        "Feature3": np.random.rand(10),
        "Target_Vol": np.linspace(0.01, 0.1, 10) 
    }, index=dates)
    
    return df

def test_compute_log_returns(sample_garch_data):
    log_returns = compute_log_returns(sample_garch_data)
    assert len(log_returns) == len(sample_garch_data) - 1                       # Lengt = len - 1 because the first value should be NaN so removed
    assert log_returns.index[0] == sample_garch_data.index[1]                   # we check that the index start one day after 
    expected_return_1 = np.log(101/100)
    expected_return_2 = np.log(102/101)
    assert log_returns.iloc[0] == pytest.approx(expected_return_1)
    assert log_returns.iloc[1] == pytest.approx(expected_return_2)

@patch('src.models.arch_model')                                                 # avoid running the model that can be very slow 
def test_garch_expanding_window_forecast(mock_arch_model, sample_garch_data):
    mock_fitted = MagicMock()
    mock_fitted.forecast.return_value.variance.values = np.array([[1.0]])       # MOCK: Simulate variance=1.0. Rescaled volatility should be sqrt(1.0 / 100**2) = 0.01. 
    mock_arch_model.return_value.fit.return_value = mock_fitted
    start_window = 5                                                            # little window because 1000 too much 
    result = garch_expanding_window_forecast(sample_garch_data, start_window=start_window)

    # 1. check the output form 
    assert isinstance(result, pd.DataFrame)
    assert 'GARCH_Forecast' in result.columns
    assert 'RealizedVol' in result.columns
    
    # 2. check the lenght of the prevision 
    assert len(result) == (len(sample_garch_data) - 1) - start_window + 1  
    assert len(result) == 4                                                     # Total iterations = (Total_Data - Start_Window). 9-5=4 iterations.

    # 3. check of the scaling and the forecast value
    assert (result['GARCH_Forecast'].iloc[0]) == pytest.approx(np.sqrt(1.0 / 100**2)) 
    assert (result['GARCH_Forecast'] == 0.01).all()

    # 4. check the RealizedVol allignment
    expected_realized_vol = sample_garch_data.loc[result.index, 'RealizedVol'].values
    assert (result['RealizedVol'].values == expected_realized_vol).all()

@patch('src.models.XGBRegressor')               # same strcuture as before 
def test_ml_expanding_window_forecast_xgboost(mock_xgb_model, sample_ml_data):
    mock_model_instance = MagicMock()                                           # same here we use the mock fct 
    mock_model_instance.predict.return_value = np.array([0.5]) 
    mock_xgb_model.return_value = mock_model_instance
    start_window = 5
    model_type = 'XGBoost'
    
    result = ml_expanding_window_forecast(sample_ml_data, model_type=model_type, start_window=start_window)
    assert isinstance(result, pd.DataFrame)
    assert f'{model_type}_Forecast' in result.columns
    assert 'RealizedVol' in result.columns
    assert len(result) == len(sample_ml_data) - start_window # 10 - 5 = 5 forecast 
    assert mock_model_instance.fit.call_count == 5
    assert (result[f'{model_type}_Forecast'] == 0.5).all()
    expected_realized_vol = sample_ml_data['Target_Vol'].iloc[start_window:].values
    assert (result['RealizedVol'].values == expected_realized_vol).all()


def test_ml_expanding_window_forecast_rf(sample_ml_data):
    start_window = 5
    model_type = 'RF'
    result = ml_expanding_window_forecast(sample_ml_data, model_type=model_type, start_window=start_window)
    assert not result.empty
    assert f'{model_type}_Forecast' in result.columns
    assert len(result) == 5

def test_ml_expanding_window_forecast_unsupported_model(sample_ml_data):           # we Test that an exception is raised for an unsupported model type 
    with pytest.raises(ValueError, match="Model type UNSUPPORTED not supported."):
        ml_expanding_window_forecast(sample_ml_data, model_type='UNSUPPORTED')