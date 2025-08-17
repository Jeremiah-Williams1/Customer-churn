import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
from pathlib import Path

def encode_categorical_features(df):
    """Encode categorical features using one-hot encoding"""
    df_encoded = df.copy()
    
    categorical_cols = ['gender', 'contract', 'internet_service', 'online_security',
                       'tech_support', 'streaming_tv', 'payment_method', 'paperless_billing']
    
    for col in categorical_cols:
        if col in df_encoded.columns:
            dummies = pd.get_dummies(df_encoded[col], prefix=col, drop_first=True)
            df_encoded = pd.concat([df_encoded.drop(col, axis=1), dummies], axis=1)
            print(f"Encoded {col}")
    
    return df_encoded

def scale_numerical_features(df, numerical_cols, fit_scaler=True, scaler=None):
    """Scale numerical features using StandardScaler"""
    df_scaled = df.copy()
    # Ensure all expected numerical columns exist, even if missing
    for col in numerical_cols:
        if col not in df_scaled.columns:
            df_scaled[col] = 0  # or np.nan and impute later
    if fit_scaler:
        scaler = StandardScaler()
        df_scaled[numerical_cols] = scaler.fit_transform(df_scaled[numerical_cols])
        return df_scaled, scaler
    else:
        df_scaled[numerical_cols] = scaler.transform(df_scaled[numerical_cols])
        return df_scaled


def create_churn_features(df):
    """Create new features for churn prediction"""
    df_features = df.copy()
    
    # Tenure groups
    if 'tenure' in df_features.columns:
        df_features['tenure_group'] = pd.cut(df_features['tenure'], 
                                           bins=[0, 12, 24, 48, 72], 
                                           labels=[0, 1, 2, 3])
        df_features['tenure_group'] = df_features['tenure_group'].astype(int)
    
    # Average monthly charges per tenure
    if 'total_charges' in df_features.columns and 'tenure' in df_features.columns:
        df_features['avg_monthly_charges'] = np.where(
            df_features['tenure'] > 0,
            df_features['total_charges'] / df_features['tenure'],
            df_features['total_charges']
        )
    
    # High value customer flag
    if 'monthly_charges' in df_features.columns:
        threshold = df_features['monthly_charges'].quantile(0.75)
        df_features['high_value_customer'] = (df_features['monthly_charges'] > threshold).astype(int)
    
    # Service count
    service_cols = ['online_security', 'tech_support', 'streaming_tv']
    df_features['service_count'] = 0
    for col in service_cols:
        if col in df_features.columns:
            df_features['service_count'] += (df_features[col] == 'Yes').astype(int)
    
    print("Created churn-specific features")
    return df_features

def split_data(df, target_column='churn', test_size=0.2):
    """Split data into train and test sets"""
    if 'customer_id' in df.columns:
        df = df.drop('customer_id', axis=1)
    
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Encode target variable
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=test_size, random_state=42, stratify=y_encoded
    )
    
    print(f"Train set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    return X_train, X_test, y_train, y_test, le, X, y

def save_transformers(scaler, label_encoder, filepath="Models/transformers.pkl"):
    """Save transformers for later use"""
    
   
    script_dir = Path(__file__).parent
    main_folder = script_dir.parent.parent
    
    # Create the full path to the Models folder in the main directory
    full_filepath = main_folder / filepath
    
    # Create the Models directory if it doesn't exist
    full_filepath.parent.mkdir(parents=True, exist_ok=True)
    
    transformers = {
        'scaler': scaler,
        'label_encoder': label_encoder
    }
    joblib.dump(transformers, full_filepath)
    print(f"Transformers saved to {full_filepath}")

def load_transformers(filepath="Models/transformers.pkl"):
    """Load saved transformers"""
    script_dir = Path(__file__).parent    
    main_folder = script_dir.parent.parent
    
    # full path
    full_filepath = main_folder / filepath
    
    transformers = joblib.load(full_filepath)
    print(f"Transformers loaded from {full_filepath}")
    return transformers['scaler'], transformers['label_encoder']

def transform_data(df):
    print("Starting data transformation...")
    
    df_features = create_churn_features(df)
    df_encoded = encode_categorical_features(df_features)
    
    X_train, X_test, y_train, y_test, label_encoder, X, y = split_data(df_encoded)

    # Decide numerical columns from training set
    numerical_cols = ['age', 'tenure', 'monthly_charges', 'total_charges']
    
    X_train_scaled, scaler = scale_numerical_features(X_train, numerical_cols, fit_scaler=True)
    X_test_scaled = scale_numerical_features(X_test, numerical_cols, fit_scaler=False, scaler=scaler)

    # Get the directory where this script is located
    script_dir = Path(__file__).parent
    
    # Go up to the main folder (churn) - since this script is in src/data/
    main_folder = script_dir.parent.parent
    
    # Create processed data directory in main folder
    output_dir = main_folder / "data" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save processed train and test data
    X_train_scaled.to_csv(output_dir / "X_train.csv", index=False)
    X_test_scaled.to_csv(output_dir / "X_test.csv", index=False)
    y_train_saved = pd.DataFrame(y_train)
    y_train_saved.to_csv(output_dir / "y_train.csv", index=False)
    y_test_saved = pd.DataFrame(y_test)
    y_test_saved.to_csv(output_dir / "y_test.csv", index=False)

    # Save features and target name in main directory
    col_features = pd.DataFrame(list(X.columns))
    col_features.to_csv(main_folder / 'features.csv', header=False, index=False)
    target_col = pd.DataFrame([y.name])
    target_col.to_csv(main_folder / 'target.csv', header=False, index=False)
    
    save_transformers(scaler, label_encoder)
    print("Data transformation completed")
    return X_train_scaled, X_test_scaled, y_train, y_test

# def transform_data(df):
    print("Starting data transformation...")
    
    df_features = create_churn_features(df)
    df_encoded = encode_categorical_features(df_features)
    
    X_train, X_test, y_train, y_test, label_encoder, X, y = split_data(df_encoded)

    # Decide numerical columns from training set
    numerical_cols = ['age', 'tenure', 'monthly_charges', 'total_charges']
    
    X_train_scaled, scaler = scale_numerical_features(X_train, numerical_cols, fit_scaler=True)
    X_test_scaled = scale_numerical_features(X_test, numerical_cols, fit_scaler=False, scaler=scaler)

    
    # Save processed train and test data
    output_dir = "../data/processed"
    os.makedirs(output_dir, exist_ok=True)
    
    X_train_scaled.to_csv(os.path.join(output_dir, "X_train.csv"), index=False)
    X_test_scaled.to_csv(os.path.join(output_dir, "X_test.csv"), index=False)
    y_train_saved = pd.DataFrame(y_train)
    y_train_saved.to_csv(os.path.join(output_dir, "y_train.csv"), index=False)
    y_test_saved = pd.DataFrame(y_test)
    y_test_saved.to_csv(os.path.join(output_dir, "y_test.csv"), index=False)

    # Save features and target name
    col_features = pd.DataFrame(list(X.columns))
    col_features.to_csv('../features.csv', header=False, index=False)
    target_col = pd.DataFrame([y.name])
    target_col.to_csv('../target.csv', header=False, index=False)
    
    save_transformers(scaler, label_encoder)
    print("Data transformation completed")
    return X_train_scaled, X_test_scaled, y_train, y_test
    


