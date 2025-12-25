import pandas as pd
import pytest
import src.data_loader as dl
from src.data_loader import download_sp500, download_vix, compute_realized_volatility, save_features, load_or_run_forecast

@pytest.fixture                # this function simulate data 
def sample_data():
    dates = pd.date_range(start="2023-01-01", periods=5, freq='D')
    sp = pd.DataFrame({"SP500": [100, 101, 102, 103, 104]}, index=dates)
    vix = pd.DataFrame({"VIX": [20, 21, 19, 18, 22]}, index=dates)
    return sp, vix

def test_compute_realized_volatility(sample_data):
    sp, vix = sample_data
    data = sp.join(vix, how="inner")
    df = compute_realized_volatility(data, window=2)
    assert "RealizedVol" in df.columns
    assert (df['RealizedVol'].dropna() >= 0).all()            # value must be positive 

def dummy_forecast(data):                                     # we test that the caching function is well utilised and executed 
    return pd.DataFrame({"a": [1, 2], "b": [3, 4]})

def test_load_or_run_forecast(tmp_path, monkeypatch):         # tmp_path eis a temporary file 
    monkeypatch.chdir(tmp_path) 
    (tmp_path / "results").mkdir()
    result = load_or_run_forecast("DummyModel", dummy_forecast, pd.DataFrame({"x": [1]}))
    assert not result.empty


