import os
import pandas as pd
import yfinance as yf
import numpy as np

# --- DATA LOADING FUNCTIONS ---
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

# --- REALIZED VOLATILITY FUNCTIONS (MODIFIED TO USE PANDAS ROLLING) ---
def compute_realized_volatility(data, window=30):
    """
    Compute realized volatility of SP500 using a rolling window, 
    efficiently implemented with Pandas rolling function.
    """
    df = data.copy()  # Copy data to avoid modifying the original DataFrame
    
    df['LogReturn'] = np.log(df['SP500'] / df['SP500'].shift(1))  # Calculate daily log returns
    df['SquaredReturn'] = df['LogReturn']**2                       # Square log returns
    
    # Compute realized volatility over rolling window and annualize if needed
    df['RealizedVol'] = np.sqrt(df['SquaredReturn'].rolling(window=window).sum() / window)  # Rolling RV
    
    df['RealizedVol'] = df['RealizedVol'].shift(1)  # Shift by 1 to use information up to t-1
    
    df = df.drop(columns=['LogReturn', 'SquaredReturn'])  # Clean up intermediate columns
    
    return df  # Return DataFrame with new 'RealizedVol' column

def save_features(data, path="data/processed/features.csv"):
    os.makedirs(os.path.dirname(path), exist_ok=True)                             # create folder if it doesn't exist
    data.to_csv(path)                                                              # save features dataframe to CSV
    print(f"✔ Features saved to {path}")  

# --- MAIN FUNCTION ---
def main():
   
    sp500 = download_sp500()                                                       # download S&P500 adjusted close prices
    vix = download_vix()                                                           # download VIX adjusted close prices
    data = merge_data(sp500, vix)                                                 # merge on date
    
    # flatten MultiIndex columns if exists
    data.columns = [col[0] if isinstance(col, tuple) else col for col in data.columns]
    
    # check columns and first rows
    print(data.head())                                                             # show first 5 rows
    print(data.columns.tolist())                                                   # list all columns
    
    data['SP500'] = pd.to_numeric(data['SP500'], errors='coerce')                  # convert SP500 to numeric
    data['VIX']   = pd.to_numeric(data['VIX'], errors='coerce')                    # convert VIX to numeric
    data = data.dropna(subset=['SP500', 'VIX'])                                    # remove rows with NA in SP500 or VIX
    save_data(data)                                                                # save merged CSV
    data = compute_realized_volatility(data, window=30)                            # compute 30-day realized volatility
    save_features(data)                                                            # save features CSV

if __name__ == "__main__":
    main()     


# --- FEATURE ENGINEERING FOR MACHINE LEARNING ---
def create_ml_features(data: pd.DataFrame, lags: int = 5, rolling_window: int = 5) -> pd.DataFrame:
    """
    Create lagged and rolling features for ML models to forecast 1-day-ahead realized volatility.

    Args:
        data (pd.DataFrame): Must contain 'RealizedVol' and 'SP500'.
        lags (int): Number of lagged days for RV and absolute returns.
        rolling_window (int): Window size for rolling statistics (Std Dev and Mean).

    Returns:
        pd.DataFrame: Features for ML models including target variable.
    """
    df = data.copy()  # Work on a copy to avoid modifying original data

    # --- Ensure Log Returns exist ---
    if 'LogReturn' not in df.columns:
        df['LogReturn'] = np.log(df['SP500'] / df['SP500'].shift(1))  # ln(today price / yesterday price)

    # --- 1. Lagged realized volatility features ---
    for lag in range(1, lags + 1):
        df[f'RV_Lag_{lag}'] = df['RealizedVol'].shift(lag)  # RV from previous days

    # --- 2. Lagged absolute returns ---
    df['AbsReturn'] = np.abs(df['LogReturn'])  # daily absolute log returns
    for lag in range(1, lags + 1):
        df[f'AbsReturn_Lag_{lag}'] = df['AbsReturn'].shift(lag)  # previous day absolute returns

    # --- 3. Rolling standard deviation of absolute returns ---
    df['AbsReturn_RollStd'] = df['AbsReturn'].rolling(window=rolling_window).std().shift(1)  # shift(1) to avoid lookahead

    # --- 4. Rolling mean of absolute returns ---
    df['AbsReturn_RollMean'] = df['AbsReturn'].rolling(window=rolling_window).mean().shift(1)

    # --- 5. Include VIX as feature ---
    df['VIX_Lag_1'] = df['VIX'].shift(1)  # previous day's VIX

    # --- 6. Target variable ---
    df['Target_Vol'] = df['RealizedVol'].shift(-1)  # next-day realized volatility

    # --- 7. Final cleanup ---
    features_to_keep = [col for col in df.columns if 'Lag' in col or 'Roll' in col or col == 'Target_Vol']      #Keep only relevant features and target, drop rows with NaNs caused by shifts/rolling
    df = df[features_to_keep].dropna()

    print(f"Features generated. Final shape: {df.shape}")  # confirm size of processed dataset
    return df


# --- MAIN FUNCTION TO PROCESS AND SAVE FEATURES ---
def process_and_save_features():
    print("\n--- STARTING FEATURE ENGINEERING ---")

    # 1. Load raw merged data
    raw_data = load_data("data/raw/merged_data.csv")
    if raw_data is None:
        print("Error: Raw data not found. Run main.py first.")
        return None

    data_with_rv = compute_realized_volatility(raw_data)                    #Compute Realized Volatility if not already present

    processed_data = create_ml_features(data_with_rv)                       #Generate ML features and target

    save_features(processed_data, path="data/processed/features.csv")       #Save processed features

    return processed_data


if __name__ == "__main__":
    process_and_save_features()

