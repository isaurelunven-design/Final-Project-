import os
import pandas as pd
import pytest
import src.data_loader as dl
from src.data_loader import download_sp500, download_vix, merge_data, compute_realized_volatility, save_data, save_features, load_or_run_forecast

#play with simulated data 
@pytest.fixture
def sample_data():
    dates = pd.date_range(start="2023-01-01", periods=5, freq='D')
    sp = pd.DataFrame({"SP500": [100, 101, 102, 103, 104]}, index=dates)
    vix = pd.DataFrame({"VIX": [20, 21, 19, 18, 22]}, index=dates)
    return sp, vix

# test for the function merge data 
def test_merge_data(sample_data):
    sp, vix = sample_data
    merged = merge_data(sp, vix)
    assert "SP500" in merged.columns
    assert "VIX" in merged.columns
    assert len(merged) == 5

# we test the function comput realized volatility 
def test_compute_realized_volatility(sample_data):
    sp, vix = sample_data
    data = merge_data(sp, vix)
    df = compute_realized_volatility(data, window=2)
    assert "RealizedVol" in df.columns
    assert (df['RealizedVol'].dropna() >= 0).all()            # value must be positive 

# we test that the caching function is well utilised and executed 
def dummy_forecast(data):
    return pd.DataFrame({"a": [1, 2], "b": [3, 4]})

def test_load_or_run_forecast(tmp_path, monkeypatch):
    # tmp_path est un dossier temporaire fourni par pytest
    monkeypatch.chdir(tmp_path)  # <-- change le répertoire courant vers tmp_path

    result = load_or_run_forecast("DummyModel", dummy_forecast, pd.DataFrame({"x": [1]}))
    assert not result.empty


