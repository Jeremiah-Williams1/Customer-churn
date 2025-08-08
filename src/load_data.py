import pandas as pd
from pathlib import Path

def load_data(file_path):
    """Load data from CSV file"""
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")
    
    df = pd.read_csv(file_path)
    print(f"Loaded {len(df)} rows and {len(df.columns)} columns")
    return df

def get_data_info(df):
    """Get basic information about the dataset"""
    print(f"Dataset shape: {df.shape}")
    print(f"Missing values per column:")
    print(df.isnull().sum())
    print(f"\nData types:")
    print(df.dtypes)
    return df.info()