import pandas as pd
import pytest
from numpy import sqrt
from sklearn.metrics import mean_squared_error, mean_absolute_error

def evaluate_models(results_list: list) -> pd.DataFrame:            # Simulating the function 
    metrics = {}
    if not results_list:
        return pd.DataFrame()
    
    realized_vol = results_list[0]['RealizedVol']                   # Use the RealizedVol from the first DataFrame as the common benchmark
    for result_df in results_list:
        forecast_cols = [col for col in result_df.columns if 'Forecast' in col]
        if not forecast_cols: continue
        model_name = forecast_cols[0].replace('_Forecast', '')
        forecast = result_df[f'{model_name}_Forecast']

        aligned_data = pd.concat([realized_vol, forecast], axis=1).dropna()
        RV_actual = aligned_data['RealizedVol']
        RV_predicted = aligned_data[f'{model_name}_Forecast']
        
        rmse = sqrt(mean_squared_error(RV_actual, RV_predicted))
        mae = mean_absolute_error(RV_actual, RV_predicted)
        correlation = RV_actual.corr(RV_predicted)
        
        metrics[model_name] = {
            'RMSE': rmse,
            'MAE': mae,
            'Correlation': correlation
        }
        
    metrics_df = pd.DataFrame.from_dict(metrics, orient='index')
    return metrics_df.sort_values(by='RMSE')

@pytest.fixture
def sample_results():
    """Creates simulated forecast data for testing."""
    dates = pd.date_range(start="2023-01-01", periods=10, freq='D')
    realized_vol = pd.Series([0.1, 0.15, 0.12, 0.2, 0.18, 0.1, 0.15, 0.12, 0.2, 0.18], index=dates, name='RealizedVol')

    # Model A: Good performance (close forecasts)
    forecast_a = pd.Series([0.11, 0.16, 0.13, 0.21, 0.19, 0.11, 0.16, 0.13, 0.21, 0.19], index=dates, name='ModelA_Forecast')
    df_a = pd.concat([realized_vol, forecast_a], axis=1)
    
    # Model B: Poor performance (far forecasts)
    forecast_b = pd.Series([0.05, 0.25, 0.08, 0.25, 0.10, 0.05, 0.25, 0.08, 0.25, 0.10], index=dates, name='ModelB_Forecast')
    df_b = pd.concat([realized_vol, forecast_b], axis=1)

    # Model C: With misaligned indices (to test alignment and NaN handling)
    dates_c = pd.date_range(start="2023-01-03", periods=8, freq='D')
    forecast_c = pd.Series([0.13, 0.21, 0.19, 0.11, 0.16, 0.13, 0.21, 0.19], index=dates_c, name='ModelC_Forecast')
    df_c = pd.DataFrame({'RealizedVol': realized_vol, 'ModelC_Forecast': forecast_c}) 

    return [df_a, df_b, df_c]

def test_evaluate_models_empty_list():                              # We test the case when the results list is empty
    results = evaluate_models([])
    assert results.empty

def test_evaluate_models_output_format(sample_results):             # We test that the output has the correct format
    results_df = evaluate_models(sample_results)
    
    assert isinstance(results_df, pd.DataFrame)
    assert not results_df.empty
    assert set(results_df.columns) == {'RMSE', 'MAE', 'Correlation'}
    assert set(results_df.index) == {'ModelA', 'ModelB', 'ModelC'}

def test_evaluate_models_metrics_calculation(sample_results):       # We test that metrics are calculated correctly and sorting is respected 
    results_df = evaluate_models(sample_results)
    
    # Model A is better than B in the fixture, so RMSE(A) < RMSE(B)
    rmse_a = results_df.loc['ModelA', 'RMSE']
    rmse_b = results_df.loc['ModelB', 'RMSE']
    
    assert rmse_a < rmse_b
    assert list(results_df.index) == ['ModelA', 'ModelC', 'ModelB']

    RV_actual_A = sample_results[0]['RealizedVol'].values
    RV_predicted_A = sample_results[0]['ModelA_Forecast'].values
    expected_rmse_a = sqrt(mean_squared_error(RV_actual_A, RV_predicted_A))
    
    assert results_df.loc['ModelA', 'RMSE'] == pytest.approx(expected_rmse_a)
    
def test_evaluate_models_alignment(sample_results):
    results_df = evaluate_models(sample_results)
    realized_vol = sample_results[0]['RealizedVol']                 # Recalculate RMSE for ModelC manually on the aligned sub-section
    forecast_c = sample_results[2]['ModelC_Forecast']
    aligned_data = pd.concat([realized_vol, forecast_c], axis=1).dropna()
    RV_actual = aligned_data['RealizedVol'].values
    RV_predicted = aligned_data['ModelC_Forecast'].values
    expected_rmse_c = sqrt(mean_squared_error(RV_actual, RV_predicted))
    
    assert results_df.loc['ModelC', 'RMSE'] == pytest.approx(expected_rmse_c)
    assert not pd.isna(results_df.loc['ModelC', 'Correlation'])   # we check that the calculation was successful (correlation not NaN)