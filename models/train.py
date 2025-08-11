import sys
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
import joblib
from pathlib import Path

def get_model_configs():
    """Get model configurations"""
    configs = {
        'logistic_regression': {
            'model': LogisticRegression(random_state=42, max_iter=1000),
            'params': {
                'C': [0.1, 1, 10],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear']
            }
        },
        'random_forest': {
            'model': RandomForestClassifier(random_state=42),
            'params': {
                'n_estimators': [100, 200],
                'max_depth': [10, 20],
                'min_samples_split': [2, 5]
            }
        }
    }
    return configs

def train_model_with_config(model_name, X_train, y_train):
    """Train a model using its configuration"""
    configs = get_model_configs()
    
    if model_name not in configs:
        raise ValueError(f"Unknown model: {model_name}")
    
    config = configs[model_name]
    print(f"Training {model_name}...")
    
    # Hyperparameter tuning
    grid_search = GridSearchCV(
        config['model'], 
        config['params'], 
        cv=5, 
        scoring='roc_auc', 
        n_jobs=-1
    )
    
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_

    parameters = grid_search.best_params_
    
    print(f"Best CV score: {grid_search.best_score_:.4f}")
    
    return best_model, parameters

def train_all_models(X_train, y_train):
    """Train all configured models"""
    configs = get_model_configs()
    trained_models = {}
    reports = []

    for model_name in configs.keys():
        try:
            model, parameters  = train_model_with_config(model_name, X_train, y_train)
            trained_models[model_name] = model
            reports.append({'Model_name':model_name,'Model':model, 'Parameters':parameters})

            # Save model
            save_model(model, f"models/{model_name}.pkl")

            # Save metrics
            path = Path('../metrics')
            path.mkdir(parents=True, exist_ok=True) 
            joblib.dump(reports, path / "Report.joblib")


            
        except Exception as e:
            print(f"Error training {model_name}: {str(e)}")
    
    return trained_models

def save_model(model, filepath):
    """Save trained model"""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, filepath)
    print(f"Model saved to {filepath}")

def load_model(filepath):
    """Load trained model"""
    model = joblib.load(filepath)
    print(f"Model loaded from {filepath}")
    return model
