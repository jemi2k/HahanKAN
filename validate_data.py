"""
Data Validation Script for HahanKAN Stock Market Dataset
This script validates the custom stock data before training
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

def validate_stock_data(csv_path):
    """
    Comprehensive validation of stock market dataset
    """
    print("="*80)
    print("HAHANK STOCK DATA VALIDATION REPORT")
    print("="*80)
    print(f"\nValidating: {csv_path}\n")
    
    # Check if file exists
    if not os.path.exists(csv_path):
        print("❌ ERROR: File not found!")
        return False
    
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"❌ ERROR: Could not read CSV file: {e}")
        return False
    
    all_valid = True
    
    # 1. Check shape and basic info
    print("1. DATASET SHAPE AND BASIC INFO")
    print("-" * 80)
    print(f"   Total rows: {df.shape[0]}")
    print(f"   Total columns: {df.shape[1]}")
    print(f"   Columns: {list(df.columns)}")
    print()
    
    # 2. Check required columns
    print("2. REQUIRED COLUMNS CHECK")
    print("-" * 80)
    required_columns = ['date', 'Open', 'High', 'Low', 'Volume', 'RSI', 'MACD', 'MACD_Signal', 'Close']
    missing_cols = set(required_columns) - set(df.columns)
    
    if missing_cols:
        print(f"   ❌ MISSING COLUMNS: {missing_cols}")
        all_valid = False
    else:
        print("   ✅ All required columns present")
    print()
    
    # 3. Check data types
    print("3. DATA TYPES CHECK")
    print("-" * 80)
    print(f"   Date column type: {df['date'].dtype}")
    
    # Try to parse date
    try:
        df['date'] = pd.to_datetime(df['date'])
        print("   ✅ Date column is valid datetime")
    except Exception as e:
        print(f"   ❌ Date column cannot be parsed: {e}")
        all_valid = False
    
    numeric_cols = ['Open', 'High', 'Low', 'Volume', 'RSI', 'MACD', 'MACD_Signal', 'Close']
    for col in numeric_cols:
        if col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                print(f"   ✅ {col}: numeric")
            else:
                print(f"   ❌ {col}: NOT numeric (type: {df[col].dtype})")
                all_valid = False
    print()
    
    # 4. Check for missing values
    print("4. MISSING VALUES CHECK")
    print("-" * 80)
    missing_data = df.isnull().sum()
    if missing_data.sum() > 0:
        print("   ⚠️  MISSING VALUES DETECTED:")
        print(missing_data[missing_data > 0])
        all_valid = False
    else:
        print("   ✅ No missing values")
    print()
    
    # 5. Check for duplicates
    print("5. DUPLICATE DATES CHECK")
    print("-" * 80)
    duplicate_dates = df['date'].duplicated().sum()
    if duplicate_dates > 0:
        print(f"   ❌ Found {duplicate_dates} duplicate dates")
        all_valid = False
    else:
        print("   ✅ No duplicate dates")
    print()
    
    # 6. Check date continuity (for daily data)
    print("6. DATE CONTINUITY CHECK")
    print("-" * 80)
    df_sorted = df.sort_values('date').reset_index(drop=True)
    date_diffs = df_sorted['date'].diff()
    
    # For trading days, most gaps should be 1 day (weekends/holidays allowed)
    one_day = timedelta(days=1)
    gap_counts = date_diffs.value_counts().head(10)
    print("   Date differences distribution (top 10):")
    for idx, (gap, count) in enumerate(gap_counts.items(), 1):
        print(f"   {idx}. {gap}: {count} occurrences")
    
    # Check for extreme gaps
    extreme_gaps = date_diffs[date_diffs > timedelta(days=30)].count()
    if extreme_gaps > 0:
        print(f"   ⚠️  Found {extreme_gaps} gaps > 30 days (may indicate data issues)")
    else:
        print("   ✅ No extreme gaps detected")
    print()
    
    # 7. Check data value ranges
    print("7. DATA VALUE RANGES CHECK")
    print("-" * 80)
    print("   Statistical Summary:")
    print(df[numeric_cols].describe().to_string())
    print()
    
    # Specific checks
    print("   Range Validations:")
    
    # Price logic check: High >= Low
    invalid_prices = (df['High'] < df['Low']).sum()
    if invalid_prices > 0:
        print(f"   ❌ Found {invalid_prices} rows where High < Low")
        all_valid = False
    else:
        print("   ✅ High >= Low for all rows")
    
    # Price logic check: High >= Open, Close; Low <= Open, Close
    invalid_ohlc = ((df['Open'] > df['High']) | (df['Open'] < df['Low']) |
                    (df['Close'] > df['High']) | (df['Close'] < df['Low'])).sum()
    if invalid_ohlc > 0:
        print(f"   ❌ Found {invalid_ohlc} rows with invalid OHLC relationships")
        all_valid = False
    else:
        print("   ✅ Valid OHLC relationships")
    
    # Volume check
    zero_volume = (df['Volume'] <= 0).sum()
    if zero_volume > 0:
        print(f"   ⚠️  Found {zero_volume} rows with zero/negative volume")
    else:
        print("   ✅ All volumes are positive")
    
    # RSI check (should be 0-100)
    invalid_rsi = ((df['RSI'] < 0) | (df['RSI'] > 100)).sum()
    if invalid_rsi > 0:
        print(f"   ❌ Found {invalid_rsi} RSI values outside 0-100 range")
        all_valid = False
    else:
        print("   ✅ RSI values within 0-100 range")
    
    print()
    
    # 8. Data Split Simulation
    print("8. TRAIN/VAL/TEST SPLIT SIMULATION")
    print("-" * 80)
    total_rows = len(df)
    num_train = int(total_rows * 0.7)
    num_test = int(total_rows * 0.2)
    num_vali = total_rows - num_train - num_test
    
    print(f"   Total samples: {total_rows}")
    print(f"   Train set: {num_train} samples (70%)")
    print(f"   Val set: {num_vali} samples")
    print(f"   Test set: {num_test} samples (20%)")
    print()
    
    # Check if we have enough data for sequence modeling
    seq_len = 60
    label_len = 30
    pred_len = 30
    min_required = seq_len + label_len + pred_len
    
    print(f"   Minimum required for seq_len={seq_len}, label_len={label_len}, pred_len={pred_len}:")
    print(f"   {min_required} samples total")
    
    if total_rows < min_required:
        print(f"   ❌ NOT ENOUGH DATA! Need at least {min_required}, have {total_rows}")
        all_valid = False
    else:
        print(f"   ✅ Sufficient data for training")
    
    if num_train < min_required:
        print(f"   ⚠️  Training set smaller than sequence requirements")
    else:
        print(f"   ✅ Training set is adequate")
    
    print()
    
    # 9. Model Configuration Requirements
    print("9. MODEL CONFIGURATION RECOMMENDATIONS")
    print("-" * 80)
    print(f"   Number of features (enc_in): {len(numeric_cols)}")
    print(f"   Target column: Close")
    print(f"   Frequency: d (daily)")
    print(f"   Suggested seq_len: 60 (2-3 months)")
    print(f"   Suggested label_len: 30 (1 month)")
    print(f"   Suggested pred_len: 30 (1 month)")
    print()
    
    # Final report
    print("="*80)
    if all_valid:
        print("✅ DATA VALIDATION: PASSED - Ready for training!")
    else:
        print("❌ DATA VALIDATION: FAILED - Please fix issues above before training")
    print("="*80)
    
    return all_valid

if __name__ == "__main__":
    # Validate the stock data
    csv_file = "./datasets/hsi_custom.csv"
    
    # Check if datasets directory exists
    if not os.path.exists("./datasets"):
        print("Creating ./datasets directory...")
        os.makedirs("./datasets")
    
    if os.path.exists(csv_file):
        is_valid = validate_stock_data(csv_file)
        exit(0 if is_valid else 1)
    else:
        print(f"❌ File not found: {csv_file}")
        print(f"Please make sure your hsi_custom.csv is in ./datasets/")
        exit(1)