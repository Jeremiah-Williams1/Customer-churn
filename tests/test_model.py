import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
# from src.mlflow.load_registered_model import get_model
from src.data.load_data import load_data
from src.data.transform_data import preprocess_new_data
from Models import load_file


def test_model_loads_and_predicts():

    data = load_data('tests/sample.csv') 
    processed = preprocess_new_data(data)

    model = load_file('random_forest.joblib')

    # model = get_model("Random Forest")  
    # seem to have an issue loading the model from dagshub-mlflow 

    preds = model.predict(processed[:5])
    assert preds.shape[0] == 5

