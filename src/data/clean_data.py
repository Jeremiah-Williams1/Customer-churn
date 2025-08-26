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
    """
    Clean and handle missing data with domain-specific logic
    """
    df_clean = df.copy()
    
    # Convert TotalCharges to numeric (often stored as string with spaces)
    if 'total_charges' in df_clean.columns:
        df_clean['total_charges'] = pd.to_numeric(df_clean['total_charges'], errors='coerce')
        
        # For customers with 0 tenure, total_charges should be 0
        df_clean.loc[df_clean['tenure'] == 0, 'total_charges'] = 0
    
        # Fill remaining missing values with median based on tenure groups
        median_by_tenure = df_clean.groupby(pd.cut(df_clean['tenure'], bins=[0, 12, 24, 48, 72]))['total_charges'].median()
        for idx in df_clean[df_clean['total_charges'].isna()].index:
            tenure_group = pd.cut([df_clean.loc[idx, 'tenure']], bins=[0, 12, 24, 48, 72])[0]
            df_clean.loc[idx, 'total_charges'] = median_by_tenure.get(tenure_group, df_clean['total_charges'].median())
    
    # Handle any other missing values
    for col in df_clean.select_dtypes(include=[np.number]).columns:
        if df_clean[col].isna().sum() > 0:
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())
    
    for col in df_clean.select_dtypes(include=['object']).columns:
        if df_clean[col].isna().sum() > 0:
            df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])
    
    print("Data cleaning completed")
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


