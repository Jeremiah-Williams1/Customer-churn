import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
from src.mlflow.load_registered_model import get_model
from src.data.load_data import load_data
from src.data.transform_data import preprocess_new_data

def test_model_loads_and_predicts():

    data = load_data('data/test/sample.csv') 
    processed = preprocess_new_data(data)

    model = get_model("Random Forest")  

    preds = model.predict(processed[:5])
    assert preds.shape[0] == 5

# test_model_loads_and_predicts()
