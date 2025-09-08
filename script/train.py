import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.data.load_data import load_data
from src.data.clean_data import clean_data
from src.data.transform_data import transform_data

from model_configs.config import train_all_models


def main():
    """Main training pipeline"""
    print("=== CUSTOMER CHURN PREDICTION TRAINING ===")
    
    # Load data
    print("\n1. Loading data...")
    df = load_data("dave/raw/customer_churn.csv")
    
    # Clean data
    print("\n2. Cleaning data...")
    df_clean = clean_data(df)
    
    # Transform data
    print("\n3. Transforming data...")
    X_train, X_test, y_train, y_test = transform_data(df_clean)
    
    # Train models
    print("\n4. Training models...")
    models = train_all_models(X_train, y_train)
    
    print(f"\nTraining completed! {len(models)} models trained and saved.")

if __name__ == "__main__":
    main()