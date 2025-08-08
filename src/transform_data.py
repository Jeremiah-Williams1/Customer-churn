import pandas as pd
import numpy as np
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

def scale_numerical_features(df, fit_scaler=True, scaler=None):
    """Scale numerical features using StandardScaler"""
    df_scaled = df.copy()
    numerical_cols = ['age', 'tenure', 'monthly_charges', 'total_charges']
    
    if fit_scaler:
        scaler = StandardScaler()
        for col in numerical_cols:
            if col in df_scaled.columns:
                df_scaled[col] = scaler.fit_transform(df_scaled[[col]])
        return df_scaled, scaler
    else:
        for col in numerical_cols:
            if col in df_scaled.columns:
                df_scaled[col] = scaler.transform(df_scaled[[col]])
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
    
    return X_train, X_test, y_train, y_test, le

def save_transformers(scaler, label_encoder, filepath="models/transformers.pkl"):
    """Save transformers for later use"""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    
    transformers = {
        'scaler': scaler,
        'label_encoder': label_encoder
    }
    joblib.dump(transformers, filepath)
    print(f"Transformers saved to {filepath}")

def load_transformers(filepath="models/transformers.pkl"):
    """Load saved transformers"""
    transformers = joblib.load(filepath)
    print(f"Transformers loaded from {filepath}")
    return transformers['scaler'], transformers['label_encoder']

def transform_data(df):
    """Complete data transformation pipeline"""
    print("Starting data transformation...")
    
    # Create features
    df_features = create_churn_features(df)
    
    # Encode categorical features
    df_encoded = encode_categorical_features(df_features)
    
    # Split data
    X_train, X_test, y_train, y_test, label_encoder = split_data(df_encoded)
    
    # Scale numerical features
    X_train_scaled, scaler = scale_numerical_features(X_train, fit_scaler=True)
    X_test_scaled = scale_numerical_features(X_test, fit_scaler=False, scaler=scaler)
    
    # Save transformers
    save_transformers(scaler, label_encoder)
    
    print("Data transformation completed")
    return X_train_scaled, X_test_scaled, y_train, y_test