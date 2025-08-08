# src/data/clean_data.py
import pandas as pd
import numpy as np

def remove_duplicates(df):
    """Remove duplicate rows"""
    initial_rows = len(df)
    df_clean = df.drop_duplicates()
    removed_rows = initial_rows - len(df_clean)
    print(f"Removed {removed_rows} duplicate rows")
    return df_clean

def handle_missing_values(df):
    """Handle missing values in the dataset"""
    df_clean = df.copy()
    
    # Convert total_charges to numeric (sometimes it's stored as string)
    if 'total_charges' in df_clean.columns:
        df_clean['total_charges'] = pd.to_numeric(df_clean['total_charges'], errors='coerce')
    
    # Fill missing values
    for column in df_clean.columns:
        if df_clean[column].isnull().sum() > 0:
            if df_clean[column].dtype in ['int64', 'float64']:
                # For numerical columns, use median
                df_clean[column].fillna(df_clean[column].median(), inplace=True)
            else:
                # For categorical columns, use mode
                df_clean[column].fillna(df_clean[column].mode()[0], inplace=True)
            
            print(f"Filled missing values in {column}")
    
    return df_clean

def remove_outliers(df):
    """Remove outliers using IQR method"""
    df_clean = df.copy()
    numerical_cols = ['age', 'tenure', 'monthly_charges', 'total_charges']
    
    for column in numerical_cols:
        if column in df_clean.columns:
            Q1 = df_clean[column].quantile(0.25)
            Q3 = df_clean[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            initial_rows = len(df_clean)
            df_clean = df_clean[(df_clean[column] >= lower_bound) & 
                              (df_clean[column] <= upper_bound)]
            removed_rows = initial_rows - len(df_clean)
            
            if removed_rows > 0:
                print(f"Removed {removed_rows} outliers from {column}")
    
    return df_clean

def clean_data(df):
    """Complete data cleaning pipeline"""
    print("Starting data cleaning...")
    
    df_clean = remove_duplicates(df)
    df_clean = handle_missing_values(df_clean)
    df_clean = remove_outliers(df_clean)
    
    print(f"Data cleaning completed. Final shape: {df_clean.shape}")
    return df_clean


