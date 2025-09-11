# tests/test_data_validation.py
import sys
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.data.load_data import load_data
from src.data.clean_data import clean_data
from src.data.transform_data import transform_data


def test_transform_data():
    # Load data tests
    df = load_data("tests/sample.csv")
    assert df.shape[0] > 0, "Dataset should not be empty"
    assert "churn" in df.columns, "Target column missing"

    clean = clean_data(df)
    X_train, X_test, y_train, y_test = transform_data(clean, training=False)


    # 1. Assert shapes
    assert X_train.shape[1] == X_test.shape[1], "Train/Test column mismatch"
    
    # 2. Assert expected features exist
    expected_features = ["age", "tenure", "monthly_charges", "total_charges", "tenure_group", 
                "avg_monthly_charges", "high_value_customer", "service_count", "gender_Male", 
                "contract_One year", "contract_Two year", "internet_service_Fiber optic", 
                "internet_service_No", "online_security_No internet service", "online_security_Yes", 
                "tech_support_No internet service", "tech_support_Yes", "streaming_tv_No internet service", 
                "streaming_tv_Yes", "payment_method_Credit card", "payment_method_Electronic check", 
                "payment_method_Mailed check", "paperless_billing_Yes"]

    assert all(f in X_train.columns for f in expected_features), "Missing features"


    features = ['customer_id','age','gender','tenure','monthly_charges','total_charges',
                'contract','internet_service','online_security','tech_support','streaming_tv',
                'payment_method','paperless_billing','churn'
]
    for i in features:
        assert clean[i].isnull().sum() == 0, f"Missing values found in column: {i}"


