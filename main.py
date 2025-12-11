import os
import pandas as pd
import numpy as np
from numpy import sqrt
import yfinance as yf
from src.data_loader import download_sp500, download_vix, merge_data, compute_realized_volatility, save_data, save_features, create_ml_features, load_or_run_forecast
from src.models import garch_expanding_window_forecast, ml_expanding_window_forecast
from src.evaluation import evaluate_models

if __name__ == "__main__":
    # step 1/2 Download data and compute realized volatility
    sp500 = download_sp500()
    vix = download_vix()
    data = merge_data(sp500, vix)
    data.columns = [col[0] if isinstance(col, tuple) else col for col in data.columns]                      # Ensure column names are clean (solves potential KeyError)    
    print("Colonnes après merge:", data.columns.tolist()) 
    data['SP500'] = pd.to_numeric(data['SP500'].squeeze(), errors='coerce')          
    data['VIX']   = pd.to_numeric(data['VIX'].squeeze(), errors='coerce')
    data = data.dropna(subset=['SP500', 'VIX'])                                     
    data = compute_realized_volatility(data, window=30) 
    save_data(data)
    save_features(data)

    # step 3 Feature engineering for ML
    print("\n--- 2. FEATURE ENGINEERING & ML PREPARATION ---")
    processed_data = create_ml_features(data) 
    save_features(processed_data, path="data/processed/features.csv")
    start_window_size = 1000

    # step 4 GARCH forecast (CACHED)
    print("\n--- 3. GARCH Forecast (Econometric Benchmark) ---")
    garch_result = load_or_run_forecast(
        'GARCH', 
        garch_expanding_window_forecast, 
        data=data, 
        start_window=start_window_size
    )
    print("GARCH Forecast Head (from cache or run):")
    print(garch_result.head()) 

    # 4. ML Forecasts (Random Forest and XGBoost) (CACHED)
    print("\n--- 4. ML Forecasts (RF & XGBoost) ---")
    rf_result = load_or_run_forecast(
        'RF', 
        ml_expanding_window_forecast, 
        processed_data=processed_data, 
        model_type='RF', 
        start_window=start_window_size
    )

    xgb_result = load_or_run_forecast(
        'XGBoost', 
        ml_expanding_window_forecast, 
        processed_data=processed_data, 
        model_type='XGBoost', 
        start_window=start_window_size
    )

    # Setp 5/6 
    vix_benchmark = data.loc[garch_result.index].copy()
    vix_benchmark = vix_benchmark[['VIX']].rename(columns={'VIX': 'VIX_Forecast'})
    vix_benchmark['VIX_Forecast'] = vix_benchmark['VIX_Forecast'] / np.sqrt(252)                       #le désanualiser 
    vix_benchmark['RealizedVol'] = garch_result['RealizedVol']
    print("\n--- 5. EVALUATION OF PERFORMANCE ---")
    all_results = [garch_result, rf_result, xgb_result, vix_benchmark]       
    performance_table = evaluate_models(all_results)                         
    
    print("\nMODEL PERFORMANCE TABLE:")
    print(performance_table)

    os.makedirs("results", exist_ok=True)
    performance_table.to_csv("results/performance_metrics.csv")
    print("\n✔ Performance metrics saved to results/performance_metrics.csv")

