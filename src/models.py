import pandas as pd
import numpy as np
from arch import arch_model
from tqdm import tqdm 
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

## Step 3: We forecast through GARCH  
def compute_log_returns(data:pd.DataFrame) -> pd.Series:                                    # we compute daily log returns from the SP500 column 
    return np.log(data['SP500']/data['SP500'].shift(1)).dropna()

def garch_expanding_window_forecast(
    data: pd.DataFrame,                                        
    start_window: int = 1000                                                                # start_window: number of initial observations before forecasting starts 
) -> pd.DataFrame: 
    log_returns = compute_log_returns(data)
    scaling_factor = 100
    log_returns_scaled = log_returns * scaling_factor
    realized_vol = data.loc[log_returns.index, 'RealizedVol']                               # we align realized vol with reutrns 
    garch_forecasts = []
    dates = []

    for t in tqdm(range(start_window, len(log_returns_scaled))): 
        train_returns_scaled = log_returns_scaled.iloc[:t]            
        model = arch_model(
            train_returns_scaled, 
            p=1,
            q=1,
            vol="GARCH", 
            dist="normal",
            rescale=False 
        )
        try:
            fitted = model.fit(disp="off")
        except:                                                                             # fallback in case convergence fails 
            continue 
        forecast = fitted.forecast(horizon=1)                           
        vol_forecast_variance = forecast.variance.values[-1, 0] / (scaling_factor**2)       # To revert to the original scale, we divide by the scaling factor squared.
        vol_forecast = np.sqrt(vol_forecast_variance)
        garch_forecasts.append(vol_forecast)
        dates.append(log_returns.index[t])
    result = pd.DataFrame({
        "Date": dates, 
        "GARCH_Forecast": garch_forecasts,
    })
    result["RealizedVol"] = data.loc[result["Date"], "RealizedVol"].values                 
    return result.set_index("Date")

## Step 4: We forecast by using ML model 
def ml_expanding_window_forecast(
    processed_data: pd.DataFrame, 
    model_type: str, 
    start_window: int = 1000
) -> pd.DataFrame:
    df = processed_data.copy()                                                                  # data preparation and variable identification  
    X = df.drop(columns=['Target_Vol'])                                                         # X contains all features except the target (Target_Vol)
    y = df['Target_Vol']                                                                        # Y is the target (next-day realized volatility)
    if model_type == 'RF':                      
        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    elif model_type == 'XGBoost':
        model = XGBRegressor(n_estimators=100, objective='reg:squarederror', random_state=42, n_jobs=-1)
    else:
        raise ValueError(f"Model type {model_type} not supported.")
    ml_forecasts = []

    for t in tqdm(range(start_window, len(df)), desc=f"Forecasting with {model_type}"):          #Training and forecasting loop (starts after the initial window)
        X_train = X.iloc[:t]                     
        y_train = y.iloc[:t]
        X_test = X.iloc[t].to_frame().T            
        model.fit(X_train, y_train)
        forecast = model.predict(X_test)[0]
        ml_forecasts.append(forecast)

    dates = df.index[start_window:]         
    result = pd.DataFrame({
        "Date": dates,
        f"{model_type}_Forecast": ml_forecasts,
    }).set_index("Date")
    result["RealizedVol"] = df.loc[dates, 'Target_Vol']
    return result