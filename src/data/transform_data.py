import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
from pathlib import Path

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
    
    if 'monthly_charges' in df_features.columns:
        threshold = df_features['monthly_charges'].quantile(0.75)
        df_features['high_value_customer'] = (df_features['monthly_charges'] > threshold).astype(int)
    
    # Service count
    service_cols = ['online_security', 'tech_support', 'streaming_tv']
    df_features['service_count'] = 0
    for col in service_cols:
        if col in df_features.columns:
            df_features['service_count'] += (df_features[col] == 'Yes').astype(int)
    
    return df_features

def encode_categorical_features(df, categorical_encoders=None, fit_encoders=True):

    df_encoded = df.copy()
    
    categorical_cols = ['gender', 'contract', 'internet_service', 'online_security',
                       'tech_support', 'streaming_tv', 'payment_method', 'paperless_billing']
    
    if fit_encoders:
        encoders = {}
        all_encoded_columns = []
        
        for col in categorical_cols:
            if col in df_encoded.columns:
                dummies = pd.get_dummies(df_encoded[col], prefix=col, drop_first=True)
                # Store the column names for this categorical variable
                encoders[col] = list(dummies.columns)
                all_encoded_columns.extend(dummies.columns)
                df_encoded = pd.concat([df_encoded.drop(col, axis=1), dummies], axis=1)
                print(f"Fitted encoder for {col}")
        
        return df_encoded, encoders
    
    else:
        # Prediction mode - use saved encoders
        for col in categorical_cols:
            if col in df_encoded.columns and col in categorical_encoders:
                dummies = pd.get_dummies(df_encoded[col], prefix=col, drop_first=True)
                df_encoded = df_encoded.drop(col, axis=1)
                
                # Ensure all expected columns are present
                for encoded_col in categorical_encoders[col]:
                    if encoded_col in dummies.columns:
                        df_encoded[encoded_col] = dummies[encoded_col]
                    else:
                        # Add missing columns as 0 (this category wasn't seen in new data)
                        df_encoded[encoded_col] = 0
                
                print(f"Applied saved encoder for {col}")
        
        return df_encoded

def scale_numerical_features(df, numerical_cols, fit_scaler=True, scaler=None):
    """Scale numerical features using StandardScaler"""
    df_scaled = df.copy()
    
    # Ensure all expected numerical columns exist
    for col in numerical_cols:
        if col not in df_scaled.columns:
            print(f"Warning: {col} not found, adding with 0 values")
            df_scaled[col] = 0
    
    if fit_scaler:
        scaler = StandardScaler()
        df_scaled[numerical_cols] = scaler.fit_transform(df_scaled[numerical_cols])
        return df_scaled, scaler
    else:
        df_scaled[numerical_cols] = scaler.transform(df_scaled[numerical_cols])
        return df_scaled

def split_data(df, target_column='churn', test_size=0.2):
    """Split data into train and test sets"""
    if 'customer_id' in df.columns:
        df = df.drop('customer_id', axis=1)
    
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Encode target variable
    le = LabelEncoder()
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    y_train = le.fit_transform(y_train)
    y_test = le.transform(y_test)
    
    print(f"Train set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    return X_train, X_test, y_train, y_test, le, X, y

def save_transformers(scaler, label_encoder, categorical_encoders, feature_columns, 
                     high_value_threshold, numerical_cols, filepath="Models/transformers.joblib"):

    script_dir = Path(__file__).parent
    main_folder = script_dir.parent.parent
    full_filepath = main_folder / filepath
    full_filepath.parent.mkdir(parents=True, exist_ok=True)
    
    # Save everything needed for preprocessing
    transformers = {
        'scaler': scaler,
        'label_encoder': label_encoder,
        'categorical_encoders': categorical_encoders, 
        'feature_columns': feature_columns,           
        'high_value_threshold': high_value_threshold, 
        'numerical_cols': numerical_cols              
    }
    
    joblib.dump(transformers, full_filepath)
    print(f"All transformers saved to {full_filepath}")

def load_transformers(filepath="Models/transformers.joblib"):

    script_dir = Path(__file__).parent    
    main_folder = script_dir.parent.parent
    full_filepath = main_folder / filepath
    
    transformers = joblib.load(full_filepath)
    print(f"All transformers loaded from {full_filepath}")
    return transformers


def transform_data(df):
    print("Starting data transformation...")
    
    # Create features and save threshold
    df_features = create_churn_features(df)
    high_value_threshold = df_features['monthly_charges'].quantile(0.75) if 'monthly_charges' in df_features.columns else None
    
    # Encode categorical features and save encoders
    df_encoded, categorical_encoders = encode_categorical_features(df_features, fit_encoders=True)
    
    # Split data
    X_train, X_test, y_train, y_test, label_encoder, X, y = split_data(df_encoded)

    # Scale numerical features
    numerical_cols = ['age', 'tenure', 'monthly_charges', 'total_charges', 'avg_monthly_charges', 'tenure_group']
    X_train_scaled, scaler = scale_numerical_features(X_train, numerical_cols, fit_scaler=True)
    X_test_scaled = scale_numerical_features(X_test, numerical_cols, fit_scaler=False, scaler=scaler)

    # Save processed data
    script_dir = Path(__file__).parent
    main_folder = script_dir.parent.parent
    output_dir = main_folder / "data" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    X_train_scaled.to_csv(output_dir / "X_train.csv", index=False)
    X_test_scaled.to_csv(output_dir / "X_test.csv", index=False)
    pd.DataFrame(y_train).to_csv(output_dir / "y_train.csv", index=False)
    pd.DataFrame(y_test).to_csv(output_dir / "y_test.csv", index=False)

    # Save metadata
    pd.DataFrame(list(X.columns)).to_csv(main_folder / 'features.csv', header=False, index=False)
    pd.DataFrame([y.name]).to_csv(main_folder / 'target.csv', header=False, index=False)
    
    # Save all transformers with enhanced metadata
    save_transformers(
        scaler=scaler,
        label_encoder=label_encoder, 
        categorical_encoders=categorical_encoders,
        feature_columns=list(X_train_scaled.columns),
        high_value_threshold=high_value_threshold,
        numerical_cols=numerical_cols
    )
    
    print("Data transformation completed")
    return X_train_scaled, X_test_scaled, y_train, y_test



def create_churn_features_with_threshold(df, high_value_threshold=None):

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
    
    # High value customer flag with saved threshold
    if 'monthly_charges' in df_features.columns:
        if high_value_threshold is not None:
            df_features['high_value_customer'] = (df_features['monthly_charges'] > high_value_threshold).astype(int)
        else:
            # Fallback for training
            threshold = df_features['monthly_charges'].quantile(0.75)
            df_features['high_value_customer'] = (df_features['monthly_charges'] > threshold).astype(int)
    
    # Service count
    service_cols = ['online_security', 'tech_support', 'streaming_tv']
    df_features['service_count'] = 0
    for col in service_cols:
        if col in df_features.columns:
            df_features['service_count'] += (df_features[col] == 'Yes').astype(int)
    
    return df_features

def preprocess_new_data(df_new, transformers_path="Models/transformers.joblib"):

    print("Loading transformers...")
    transformers = load_transformers(transformers_path)
    
    print("Starting preprocessing of new data...")
    
    # Step 1: Create features using saved threshold
    df_features = create_churn_features_with_threshold(
        df_new, 
        high_value_threshold=transformers['high_value_threshold']
    )
    
    # Step 2: Encode categorical features using saved encoders
    df_encoded = encode_categorical_features(
        df_features, 
        categorical_encoders=transformers['categorical_encoders'], 
        fit_encoders=False
    )
    
    # Step 3: Remove customer_id if present
    if 'customer_id' in df_encoded.columns:
        df_encoded = df_encoded.drop('customer_id', axis=1)
    
    # Step 4: Ensure all expected columns are present and in correct order
    expected_columns = transformers['feature_columns']
    for col in expected_columns:
        if col not in df_encoded.columns:
            print(f"Adding missing column: {col}")
            df_encoded[col] = 0
    
    # Reorder columns to match training data
    df_encoded = df_encoded[expected_columns]
    
    # Step 5: Scale numerical features
    df_scaled = scale_numerical_features(
        df_encoded, 
        transformers['numerical_cols'], 
        fit_scaler=False, 
        scaler=transformers['scaler']
    )
    
    print("Preprocessing completed!")
    return df_scaled