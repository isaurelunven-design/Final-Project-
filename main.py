import os
import pandas as pd
import yfinance as yf
import numpy as np
from arch import arch_model
from tqdm import tqdm 
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from numpy import sqrt


# DATA LOADING FUNCTIONS 
def download_sp500(start_date="2010-01-01", end_date="2024-12-31"):        
    sp500 = yf.download("^GSPC", start=start_date, end=end_date, auto_adjust=True)  # ^GSPC is the ticker for S&P 500 in Yahoo 
    sp500 = sp500[["Close"]].rename(columns={"Close": "SP500"})                     # we only keep the column "Close"
    sp500.index = pd.to_datetime(sp500.index)                                       # ensure the index is datetime
    return sp500 

def download_vix(start_date="2010-01-01", end_date="2024-12-31"):          
    vix = yf.download("^VIX", start=start_date, end=end_date)                      # ^VIX is the ticker for VIX
    vix = vix[["Close"]].rename(columns={"Close": "VIX"})                           # keep only "Close" column
    vix.index = pd.to_datetime(vix.index)                                           # ensure the index is datetime
    return vix

def merge_data(sp500, vix):                                                     
    data = sp500.merge(vix, left_index=True, right_index=True, how="inner")       # inner merge keeps only dates present in both
    return data

def save_data(data, path="data/raw/merged_data.csv"):
    os.makedirs(os.path.dirname(path), exist_ok=True)                             # create folder if it doesn't exist
    data.to_csv(path)                                                              # save dataframe to CSV
    print(f"Data saved to {path}")  

def load_data(path="data/raw/merged_data.csv"):
    data = pd.read_csv(path, index_col=0, parse_dates=True)                        # load merged SP500 & VIX data
    return data

# REALIZED VOLATILITY FUNCTIONS (MODIFIED TO USE PANDAS ROLLING) 
def compute_realized_volatility(data, window=30):
    df = data.copy()  # Copy data to avoid modifying the original DataFrame
    df['LogReturn'] = np.log(df['SP500'] / df['SP500'].shift(1))  # Calculate daily log returns
    df['SquaredReturn'] = df['LogReturn']**2                       # Square log returns
    df['RealizedVol'] = np.sqrt(df['SquaredReturn'].rolling(window=window).sum() / window)  # Rolling RV
    df['RealizedVol'] = df['RealizedVol'].shift(1)  # Shift by 1 to use information up to t-1
    df = df.drop(columns=['LogReturn', 'SquaredReturn'])  # Clean up intermediate columns
    return df  # Return DataFrame with new 'RealizedVol' column

def save_features(data, path="data/processed/features.csv"):
    os.makedirs(os.path.dirname(path), exist_ok=True)                             # create folder if it doesn't exist
    data.to_csv(path)                                                              # save features dataframe to CSV
    print(f"✔ Features saved to {path}")  


# FEATURE ENGINEERING FOR MACHINE LEARNING
def create_ml_features(data: pd.DataFrame, lags: int = 5, rolling_window: int = 5) -> pd.DataFrame:
    df = data.copy()
    y = df['RealizedVol'].squeeze().dropna()
    scaling_factor = 100       #SCALING to avoid GARCH's warning 
    y_scaled = y * scaling_factor
    model = arch_model(y_scaled, vol='Garch', p=1, q=1)
    res = model.fit(disp="off")
    garch_vol_scaled = pd.Series(res.conditional_volatility.values, index=y.index)
    df['garch_vol'] = garch_vol_scaled / scaling_factor

    if 'LogReturn' not in df.columns:
        df['LogReturn'] = np.log(df['SP500'] / df['SP500'].shift(1))
    for lag in range(1, lags + 1):
        df[f'vol_lag_{lag}'] = df['garch_vol'].shift(lag) 
    df['AbsReturn'] = np.abs(df['LogReturn']) 
    for lag in range(1, lags + 1): 
        df[f'AbsReturn_Lag_{lag}'] = df['AbsReturn'].shift(lag) 
    df['AbsReturn_RollStd'] = df['AbsReturn'].rolling(window=rolling_window).std().shift(1) 
    df['AbsReturn_RollMean'] = df['AbsReturn'].rolling(window=rolling_window).mean().shift(1) 
    df['VIX_Lag_1'] = df['VIX'].shift(1) 
    df['Target_Vol'] = df['RealizedVol'].shift(-1) 
    features_to_keep = [col for col in df.columns if 'Lag' in col or 'Roll' in col or col == 'Target_Vol'] 
    df = df[features_to_keep].dropna() 
    print(f"Features generated. Final shape: {df.shape}") 
    return df

def process_and_save_features():
    print("\n--- STARTING FEATURE ENGINEERING ---")
    raw_data = load_data("data/raw/merged_data.csv")
    if raw_data is None:
        print("Error: Raw data not found. Run main.py first.")
        return None
    data_with_rv = raw_data.copy() 
    if 'RealizedVol' not in data_with_rv.columns:
         data_with_rv = compute_realized_volatility(data_with_rv)      
    processed_data = create_ml_features(data_with_rv)                       
    save_features(processed_data, path="data/processed/features.csv")       
    return processed_data


if __name__ == "__main__":
    process_and_save_features()

#GARCH (1,1) FORECAST
def compute_log_returns(data:pd.DataFrame) -> pd.Series:        #compute daily log returns from the SP500 column 
    return np.log(data['SP500']/data['SP500'].shift(1)).dropna()

def garch_expanding_window_forecast(
    data: pd.DataFrame,                                        #DataFrame contains SP500 and realizedVol 
    start_window: int = 1000                                   #start_window: number of initial observations before forecasting starts 
) -> pd.DataFrame: 
    log_returns = compute_log_returns(data)
    scaling_factor = 100
    log_returns_scaled = log_returns * scaling_factor
    realized_vol = data.loc[log_returns.index, 'RealizedVol']      #align relaized vol with reutrns 
    garch_forecasts = []
    dates = []

    for t in tqdm(range(start_window, len(log_returns_scaled))): 
        train_returns_scaled = log_returns_scaled.iloc[:t]            # Utilise les données scalées pour l'entraînement GARCH
        model = arch_model(
            train_returns_scaled, # DONNÉES SCALÉES
            p=1,
            q=1,
            vol="GARCH", 
            dist="normal",
            rescale=False # Nous gérons le scaling manuellement
        )
        try:
            fitted = model.fit(disp="off")
        except:                                     #fallback in case convergence fails 
            continue 
        forecast = fitted.forecast(horizon=1)                           #1 step ahead forecast 
        vol_forecast_variance = forecast.variance.values[-1, 0] / (scaling_factor**2)    # Le résultat est en variance. Pour revenir à l'échelle d'origine, on divise par scaling_factor^2
        vol_forecast = np.sqrt(vol_forecast_variance)
        garch_forecasts.append(vol_forecast)
        dates.append(log_returns.index[t])
    result = pd.DataFrame({
        "Date": dates, 
        "GARCH_Forecast": garch_forecasts,
    })
    result["RealizedVol"] = data.loc[result["Date"], "RealizedVol"].values   #add realized vol for evaluation 
    return result.set_index("Date")

# MACHINE LEARNING EXPANDING WINDOW FORECAST 
def ml_expanding_window_forecast(
    processed_data: pd.DataFrame, 
    model_type: str, 
    start_window: int = 1000
) -> pd.DataFrame:
    df = processed_data.copy()                  #data preparation and variable identification  
    X = df.drop(columns=['Target_Vol'])         # X contains all features except the target (Target_Vol)
    y = df['Target_Vol']                        # Y is the target (next-day realized volatility)
    if model_type == 'RF':                      #Model initialization
        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    elif model_type == 'XGBoost':
        model = XGBRegressor(n_estimators=100, objective='reg:squarederror', random_state=42, n_jobs=-1)
    else:
        raise ValueError(f"Model type {model_type} not supported.")
    ml_forecasts = []

    for t in tqdm(range(start_window, len(df)), desc=f"Forecasting with {model_type}"):          #Training and forecasting loop (starts after the initial window)
        X_train = X.iloc[:t]                     # Split: Train on everything up to 't-1'
        y_train = y.iloc[:t]
        X_test = X.iloc[t].to_frame().T            #Test: The next line 't' (1-day-ahead forecast)
        model.fit(X_train, y_train)
        forecast = model.predict(X_test)[0]
        ml_forecasts.append(forecast)

    dates = df.index[start_window:]         #Building the results DataFrame
    result = pd.DataFrame({
        "Date": dates,
        f"{model_type}_Forecast": ml_forecasts,
    }).set_index("Date")
    result["RealizedVol"] = df.loc[dates, 'Target_Vol']
    return result

#EVALUATION/COMPARISON OF ALL MODELS 
def evaluate_models(results_list: list) -> pd.DataFrame:
    metrics = {}
    if not results_list:                    # Use the RealizedVol from the first DataFrame as the common benchmark
        return pd.DataFrame()  
    realized_vol = results_list[0]['RealizedVol']
    for result_df in results_list:
        forecast_cols = [col for col in result_df.columns if 'Forecast' in col]          # Extract the model name (e.g., GARCH, RF, XGBoost, VIX)
        if not forecast_cols: continue 
        model_name = forecast_cols[0].replace('_Forecast', '')
        forecast = result_df[f'{model_name}_Forecast']
        aligned_data = pd.concat([realized_vol, forecast], axis=1).dropna()        # Align predictions with the true realized volatility
        RV_actual = aligned_data['RealizedVol']
        RV_predicted = aligned_data[f'{model_name}_Forecast']
        
        # Calculate metrics
        rmse = sqrt(mean_squared_error(RV_actual, RV_predicted))
        mae = mean_absolute_error(RV_actual, RV_predicted)
        correlation = RV_actual.corr(RV_predicted)
        metrics[model_name] = {
            'RMSE': rmse,
            'MAE': mae,
            'Correlation': correlation
        }
    metrics_df = pd.DataFrame.from_dict(metrics, orient='index')        # Create a summary DataFrame, sorted by RMSE
    return metrics_df.sort_values(by='RMSE') 

#MAIN PIPELINE
if __name__ == "__main__":
    # 1. Download data and compute realized volatility
    sp500 = download_sp500()
    vix = download_vix()
    data = merge_data(sp500, vix)
    data.columns = [col[0] if isinstance(col, tuple) else col for col in data.columns]      #Ensure column names are clean (solves potential KeyError)    
    print("Colonnes après merge:", data.columns.tolist()) 
    data['SP500'] = pd.to_numeric(data['SP500'].squeeze(), errors='coerce')          
    data['VIX']   = pd.to_numeric(data['VIX'].squeeze(), errors='coerce')
    data = data.dropna(subset=['SP500', 'VIX'])                                     
    data = compute_realized_volatility(data, window=30) # Ajoute la colonne 'RealizedVol'
    save_data(data)
    save_features(data)

    # 2. Feature engineering for ML
    print("\n--- 2. FEATURE ENGINEERING & ML PREPARATION ---")
    processed_data = create_ml_features(data) 
    save_features(processed_data, path="data/processed/features.csv")

     # FORECASTING AND EVALUATION 
    start_window_size = 1000

    # 3. GARCH forecast
    print("\n--- 3. GARCH Forecast (Econometric Benchmark) ---")
    garch_result = garch_expanding_window_forecast(data, start_window=start_window_size)
    print("GARCH Forecast Head:")
    print(garch_result.head())

    # 4. ML Forecasts (Random Forest and XGBoost)
    print("\n--- 4. ML Forecasts (RF & XGBoost) ---")
    rf_result = ml_expanding_window_forecast(processed_data, model_type='RF', start_window=start_window_size)
    xgb_result = ml_expanding_window_forecast(processed_data, model_type='XGBoost', start_window=start_window_size)
    
    # 5. VIX Benchmark (Implied Volatility)
    # Align VIX data with the forecast dates for comparison
    vix_benchmark = data.loc[garch_result.index].copy()
    vix_benchmark = vix_benchmark[['VIX']].rename(columns={'VIX': 'VIX_Forecast'})
    vix_benchmark['RealizedVol'] = garch_result['RealizedVol']

    # 6. Evaluation
    print("\n--- 5. EVALUATION OF PERFORMANCE ---")
    all_results = [garch_result, rf_result, xgb_result, vix_benchmark]       # Combine results from all models and benchmarks
    performance_table = evaluate_models(all_results)                         # Calculate performance metrics
    
    print("\nMODEL PERFORMANCE TABLE:")
    print(performance_table)

    os.makedirs("results", exist_ok=True)
    performance_table.to_csv("results/performance_metrics.csv")
    print("\n✔ Performance metrics saved to results/performance_metrics.csv")