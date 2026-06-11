"""
Download intraday (hourly) HSI stock data - EXTENDED VERSION
Gets multiple years of data by downloading in chunks
"""

import yfinance as yf
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

def download_hourly_hsi_extended(years=2):
    """
    Download hourly HSI data for multiple years
    Yahoo Finance allows 60 days at a time, so we download in chunks
    """
    
    print("="*80)
    print(f"DOWNLOADING {years} YEARS OF HOURLY HSI DATA")
    print("="*80)
    
    ticker = "^HSI"
    all_data = []
    
    # Calculate date ranges (60 days at a time)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365*years)
    
    print(f"\nTarget period: {start_date.date()} to {end_date.date()}")
    print(f"Total days: {(end_date - start_date).days}")
    
    current_end = end_date
    chunk_days = 60
    
    chunk_num = 1
    while current_end > start_date:
        current_start = current_end - timedelta(days=chunk_days)
        
        if current_start < start_date:
            current_start = start_date
        
        print(f"\n[Chunk {chunk_num}] Downloading: {current_start.date()} to {current_end.date()}...")
        
        try:
            df_chunk = yf.download(
                ticker,
                start=current_start.date(),
                end=current_end.date(),
                interval="1h",
                progress=False
            )
            
            if len(df_chunk) > 0:
                print(f"  ✅ Got {len(df_chunk)} hourly candles")
                all_data.append(df_chunk)
            else:
                print(f"  ⚠️  No data for this period")
                
        except Exception as e:
            print(f"  ❌ Error: {e}")
        
        current_end = current_start - timedelta(days=1)
        chunk_num += 1
    
    if len(all_data) == 0:
        print("\n❌ No data downloaded!")
        return False
    
    # Combine all chunks
    print(f"\n\nCombining {len(all_data)} chunks...")
    df = pd.concat(all_data, ignore_index=False)
    df = df.sort_index()
    
    print(f"✅ Combined total: {len(df)} hourly candles")
    
    # Clean up
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    df = df.reset_index()
    
    # Rename date column
    if 'Datetime' in df.columns:
        df.rename(columns={'Datetime': 'date'}, inplace=True)
    elif 'Date' in df.columns:
        df.rename(columns={'Date': 'date'}, inplace=True)
    
    print(f"Columns: {list(df.columns)}")
    
    # Calculate technical indicators
    print("\nCalculating technical indicators...")
    
    # MACD (12, 26, 9)
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    # RSI (14)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Drop NaN values
    df.dropna(inplace=True)
    
    print(f"After dropping NaN: {len(df)} rows")
    
    # Format date
    df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d %H:%M:%S')
    
    # Reorder columns
    feature_order = ['date', 'Open', 'High', 'Low', 'Volume', 'RSI', 'MACD', 'MACD_Signal', 'Close']
    df = df[feature_order]
    
    # Save
    os.makedirs('./datasets', exist_ok=True)
    csv_path = f'./datasets/hsi_hourly_{years}y.csv'
    df.to_csv(csv_path, index=False)
    
    print(f"\n{'='*80}")
    print(f"✅ HOURLY DATA SAVED")
    print(f"{'='*80}")
    print(f"File: {csv_path}")
    print(f"Total hourly candles: {len(df)}")
    print(f"Total features: {len(feature_order) - 1}")
    print(f"Date range: {df['date'].iloc[0]} to {df['date'].iloc[-1]}")
    print(f"Data shape: {df.shape}")
    
    # Calculate splits
    total = len(df)
    train_size = int(total * 0.7)
    val_size = int(total * 0.1)
    test_size = total - train_size - val_size
    
    print(f"\nData split (70-10-20):")
    print(f"  Train: {train_size} samples")
    print(f"  Val:   {val_size} samples")
    print(f"  Test:  {test_size} samples")
    
    # Show sample
    print(f"\nFirst 5 rows:")
    print(df.head())
    
    print(f"\nLast 5 rows:")
    print(df.tail())
    
    return True

def compare_all_data():
    """
    Compare daily vs hourly data
    """
    
    print(f"\n{'='*80}")
    print("DATA COMPARISON")
    print(f"{'='*80}")
    
    daily_path = './datasets/hsi_custom.csv'
    hourly_1y = './datasets/hsi_hourly_1y.csv'
    hourly_2y = './datasets/hsi_hourly_2y.csv'
    
    data_info = []
    
    if os.path.exists(daily_path):
        df = pd.read_csv(daily_path)
        data_info.append(('Daily (16 years)', len(df)))
        print(f"\n📊 Daily (16 years):")
        print(f"   Rows: {len(df)}")
        print(f"   Train/Val/Test: {int(len(df)*0.7)} / {int(len(df)*0.1)} / {int(len(df)*0.2)}")
    
    if os.path.exists(hourly_1y):
        df = pd.read_csv(hourly_1y)
        data_info.append(('Hourly (1 year)', len(df)))
        print(f"\n📊 Hourly (1 year):")
        print(f"   Rows: {len(df)}")
        print(f"   Train/Val/Test: {int(len(df)*0.7)} / {int(len(df)*0.1)} / {int(len(df)*0.2)}")
    
    if os.path.exists(hourly_2y):
        df = pd.read_csv(hourly_2y)
        data_info.append(('Hourly (2 years)', len(df)))
        print(f"\n📊 Hourly (2 years):")
        print(f"   Rows: {len(df)}")
        print(f"   Train/Val/Test: {int(len(df)*0.7)} / {int(len(df)*0.1)} / {int(len(df)*0.2)}")
        print(f"   ✅ Similar to ETT datasets (~17,520)")

if __name__ == "__main__":
    # Try to download 2 years first, then 1 year if that fails
    print("Attempting to download 2 years of data...\n")
    
    success = download_hourly_hsi_extended(years=2)
    
    if success:
        compare_all_data()
        print(f"\n{'='*80}")
        print("✅ Ready to train with hourly data!")
        print(f"{'='*80}")
    else:
        print("\n⚠️  Trying 1 year instead...\n")
        success = download_hourly_hsi_extended(years=1)
        
        if success:
            compare_all_data()
        else:
            print("\n❌ Failed to download data")