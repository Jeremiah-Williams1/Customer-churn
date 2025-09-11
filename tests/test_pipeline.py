# tests/test_pipeline.py
import sys
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent))


from sklearn.metrics import accuracy_score
from src.data.load_data import load_data
from src.data.clean_data import clean_data
from src.data.transform_data import transform_data
from model_configs.config import train_model_with_config

def test_end_to_end_pipeline():
    data = load_data('tests/sample.csv') 
    clean = clean_data(data)
    X_transformed, X_test, y_transformed, y_test = transform_data(clean, training=False)

    model, _ = train_model_with_config('logistic_regression', X_transformed, y_transformed)
    preds = model.predict(X_test[:5])
    acc = accuracy_score(y_test[:5], preds)

    assert acc > 0.6 

