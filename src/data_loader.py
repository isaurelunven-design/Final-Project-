import pandas as pd
import yfinance as yf
import numpy as np
from arch import arch_model
from tqdm import tqdm 

# Step 0: dowload SP500 and VIX  
def download_sp500(start_date="2010-01-01", end_date="2024-12-31"):        
    sp500 = yf.download("^GSPC", start=start_date, end=end_date, auto_adjust=True)          # ^GSPC is the ticker for S&P 500 in Yahoo 
    sp500 = sp500[["Close"]].rename(columns={"Close": "SP500"})                             # we only keep the column "Close"
    sp500.index = pd.to_datetime(sp500.index)                                       
    return sp500 

def download_vix(start_date="2010-01-01", end_date="2024-12-31"):          
    vix = yf.download("^VIX", start=start_date, end=end_date)                               # ^VIX is the ticker for VIX
    vix = vix[["Close"]].rename(columns={"Close": "VIX"})                           
    vix.index = pd.to_datetime(vix.index)                                           
    return vix

def load_data(path="data/raw/merged_data.csv"):
    data = pd.read_csv(path, index_col=0, parse_dates=True)                       
    return data

# Step 1: We compute the REALIZED VOLATILITY  
def compute_realized_volatility(data, window=30):
    df = data.copy()                                                                        # Copy data to avoid modifying the original DataFrame
    df['LogReturn'] = np.log(df['SP500'] / df['SP500'].shift(1))                            # = calculation of the daily log return 
    df['SquaredReturn'] = df['LogReturn']**2                       
    df['RealizedVol'] = np.sqrt(df['SquaredReturn'].rolling(window=window).sum() / window)  # Rolling RV
    df['RealizedVol'] = df['RealizedVol'].shift(1)                                          # to use information up to t-1
    df = df.drop(columns=['LogReturn', 'SquaredReturn'])                                    # Clean up intermediate columns
    return df 

def save_features(data, path="data/processed/features.csv"):                            
    data.to_csv(path)                                                                       # save features dataframe to CSV
    print(f"✔ Features saved to {path}")  

# Step 2: We create FEATURE ENGINEERING for ML 
def create_ml_features(data: pd.DataFrame, lags: int = 5, rolling_window: int = 5) -> pd.DataFrame:
    df = data.copy()
    y = df['RealizedVol'].squeeze().dropna()
    scaling_factor = 100                                                                    # SCALING to avoid GARCH's warning 
    y_scaled = y * scaling_factor
    model = arch_model(y_scaled, vol='Garch', p=1, q=1)                                     # first model introduced: GARCH (benchmark feature)
    res = model.fit(disp="off")
    garch_vol_scaled = pd.Series(res.conditional_volatility.values, index=y.index)
    df['garch_vol'] = garch_vol_scaled / scaling_factor

    if 'LogReturn' not in df.columns:
        df['LogReturn'] = np.log(df['SP500'] / df['SP500'].shift(1))

    for lag in range(1, lags + 1):                                                           # we add past volatility (help predict the futur)
        df[f'vol_lag_{lag}'] = df['garch_vol'].shift(lag)

    df['AbsReturn'] = np.abs(df['LogReturn']) 
    
    for lag in range(1, lags + 1):                                                           # add information about past absolute returns (important for volatility clustering)
        df[f'AbsReturn_Lag_{lag}'] = df['AbsReturn'].shift(lag) 

    df['AbsReturn_RollStd'] = df['AbsReturn'].rolling(window=rolling_window).std().shift(1)  # we add rolling statistics (mean/std) to capture recent trends.
    df['AbsReturn_RollMean'] = df['AbsReturn'].rolling(window=rolling_window).mean().shift(1) 
    df['VIX_Lag_1'] = df['VIX'].shift(1) 
    df['Target_Vol'] = df['RealizedVol'].shift(-1) 
    features_to_keep = [col for col in df.columns if 'Lag' in col or 'Roll' in col or col == 'Target_Vol'] 
    df = df[features_to_keep].dropna() 
    print(f"Features generated. Final shape: {df.shape}") 
    return df

def process_and_save_features():
    print("\n Starting Feature Engineering ")
    raw_data = load_data("data/raw/merged_data.csv")
    if raw_data is None:
        print("Error: Raw data not found. Run main.py first.")
        return None
    data_with_rv = raw_data
    if 'RealizedVol' not in data_with_rv.columns:
         data_with_rv = compute_realized_volatility(data_with_rv)      
    processed_data = create_ml_features(data_with_rv)                       
    save_features(processed_data, path="data/processed/features.csv")       
    return processed_data

    # We use a caching function to help the execution 
def load_or_run_forecast(model_name, forecast_func, *args, **kwargs):
    path = f"results/{model_name}_forecast.csv"
    
    try:
        # Essaie de charger le fichier
        result = pd.read_csv(path, index_col=0, parse_dates=True)
        print(f"Loading {model_name} results from cache: {path}")
        return result
    except FileNotFoundError:
        # Sinon, on calcule
        print(f"Running full {model_name} forecast (may take a while)...")
        result = forecast_func(*args, **kwargs)
        result.to_csv(path)
        print(f"Saved {model_name} results to cache.")
        return result

