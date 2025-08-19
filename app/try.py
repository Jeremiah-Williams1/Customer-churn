# import sys
# import os
# from pathlib import Path

# sys.path.append(str(Path(__file__).resolve().parent.parent))

# from src.mlflow.load_registered_model import get_model
# from src.mlflow.mlflow_setup import setup_mlflow

# model = get_model(model_name='Random Forest')
# print('Model_loaded sucessfully')

import sys
from pathlib import Path
import pandas as pd


# Add src to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from Models import load_file
from src.data.transform_data import preprocess_new_data

# preprocesor = load_file('transformers.pkl')
model = load_file('random_forest.joblib')
print('Loaded successfully')


data = {
    'age': 42,
    'gender': 'Male',
    'tenure': 4,
    'monthly_charges': 42.06,
    'total_charges': 166.75,
    'contract': 'Month-to-month',
    'internet_service': 'DSL',
    'online_security': 'No',
    'tech_support': 'No',
    'streaming_tv': 'Yes',
    'payment_method': 'Credit card',
    'paperless_billing': 'No',
}

# Convert dict → DataFrame (1 row)
df = pd.DataFrame([data])

# Transform with your preprocessor
X_transformed = preprocess_new_data(df)

# Predict with your model
prediction = model.predict(X_transformed)

if prediction == 1:
    print("Would Churn")
    
else:
    print("won't churn")