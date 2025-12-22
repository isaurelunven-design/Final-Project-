import pandas as pd
import numpy as np

from src.data_loader import download_sp500, download_vix, compute_realized_volatility, save_features, create_ml_features, load_or_run_forecast
from src.models import garch_expanding_window_forecast, ml_expanding_window_forecast, run_garch_estimation_for_report # <--- CORRECTION 1: Ajout de la fonction et fin de la ligne
from src.evaluation import evaluate_models 

RANDOM_SEED = 42 
np.random.seed(RANDOM_SEED)                                                     # to be reproductible 

if __name__ == "__main__":

    ## Step 0 and 1: Download data and compute realized volatility

    sp500 = download_sp500()
    vix = download_vix()
    data = sp500.merge(vix, left_index=True, right_index=True, how="inner")
    data.columns = [col[0] if isinstance(col, tuple) else col for col in data.columns]
    print("Column after merge:", data.columns.tolist()) 
    data['SP500'] = pd.to_numeric(data['SP500'], errors='coerce')          
    data['VIX']   = pd.to_numeric(data['VIX'], errors='coerce')
    data = data.dropna(subset=['SP500', 'VIX'])                                     
    data = compute_realized_volatility(data, window=30) 
    from src.models import compute_log_returns # Assurez-vous d'importer ceci
    log_returns = compute_log_returns(data)
    print("\n 1.b. GARCH Parameter Estimation for LaTeX Report")
    run_garch_estimation_for_report(log_returns)
    data.to_csv("data/raw/merged_data.csv")
    print("✔ Data saved to data/raw/merged_data.csv")

    ## Step 2: Feature engineering for ML

    print("\n 2. Feature Engineering & ML Preparation")
    processed_data = create_ml_features(data) 
    save_features(processed_data, path="data/processed/features.csv")
    start_window_size = 1000

    ## Step 3: We forecast through GARCH (CACHED)

    print("\n 3. GARCH Forecast (Econometric Benchmark)")
    garch_result = load_or_run_forecast(
        'GARCH', 
        garch_expanding_window_forecast, 
        data=data, 
        start_window=start_window_size
    )
    print("GARCH Forecast Head:")
    print(garch_result.head()) 

    ## Step 4: We forecast through ML models (Random Forest and XGBoost) (CACHED)

    print("\n 4. ML Forecasts (RF & XGBoost)")
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

    ## Step 5: We evaluate and compare models 

    vix_benchmark = data.loc[garch_result.index]
    vix_benchmark = vix_benchmark[['VIX']].rename(columns={'VIX': 'VIX_Forecast'})
    vix_benchmark['VIX_Forecast'] = (vix_benchmark['VIX_Forecast'] / 100) / np.sqrt(252)                       #le désanualiser 
    vix_benchmark['RealizedVol'] = garch_result['RealizedVol']
    print("\n 5. Evaluation of Model's performance")

    all_results = [garch_result, rf_result, xgb_result, vix_benchmark]       
    performance_table = evaluate_models(all_results)                         
    print("\n Model Performance table:")
    print(performance_table)

    performance_table.to_csv("results/performance_metrics.csv")
    print("\n✔ Performance metrics saved to results/performance_metrics.csv")

