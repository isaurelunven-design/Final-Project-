import pandas as pd
from numpy import sqrt
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Step 5: We compare and evaluate all models (ML and Econometrics) 
def evaluate_models(results_list: list) -> pd.DataFrame:
    metrics = {}
    if not results_list:                                                                        # Use the RealizedVol from the first DataFrame as the common benchmark
        return pd.DataFrame()  
    realized_vol = results_list[0]['RealizedVol']
    for result_df in results_list:
        forecast_cols = [col for col in result_df.columns if 'Forecast' in col]                 # Extract the model name (e.g., GARCH, RF, XGBoost, VIX)
        if not forecast_cols: continue 
        model_name = forecast_cols[0].replace('_Forecast', '')
        forecast = result_df[f'{model_name}_Forecast']
        aligned_data = pd.concat([realized_vol, forecast], axis=1).dropna()                     # Align predictions with the true realized volatility
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