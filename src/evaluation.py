import pandas as pd
from numpy import sqrt
from typing import List, Dict, Any
from sklearn.metrics import mean_squared_error, mean_absolute_error

def evaluate_models(results_list: List[pd.DataFrame]) -> pd.DataFrame:
    metrics: Dict[str, Dict[str, Any]] = {}
    if not results_list:                                                            # Check for empty input list
        return pd.DataFrame()
    realized_vol: pd.Series = results_list[0]['RealizedVol']
    for result_df in results_list:
        forecast_cols: List[str] = [col for col in result_df.columns if 'Forecast' in col]
        if not forecast_cols:
            continue
            
        model_name: str = forecast_cols[0].replace('_Forecast', '')                 # Extract the model name (e.g., GARCH, RF, XGBoost, VIX)
        forecast: pd.Series = result_df[f'{model_name}_Forecast']
        aligned_data: pd.DataFrame = pd.concat([realized_vol, forecast], axis=1).dropna()
        if aligned_data.empty:
            continue

        RV_actual: pd.Series = aligned_data['RealizedVol']
        RV_predicted: pd.Series = aligned_data[f'{model_name}_Forecast']
        
        rmse: float = sqrt(mean_squared_error(RV_actual, RV_predicted))
        mae: float = mean_absolute_error(RV_actual, RV_predicted)                     # Mean Absolute Error (MAE)
        correlation: float = RV_actual.corr(RV_predicted)                             # Correlation between actual and predicted values

        metrics[model_name] = {
            'RMSE': rmse,
            'MAE': mae,
            'Correlation': correlation
        }

    metrics_df: pd.DataFrame = pd.DataFrame.from_dict(metrics, orient='index')        # Convert the dictionary of metrics into a DataFrame
    
    return metrics_df.sort_values(by='RMSE')