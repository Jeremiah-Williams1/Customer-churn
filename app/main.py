import sys
from pathlib import Path
from fastapi import FastAPI
from .schema_model import Churn_Input
import pandas as pd
from prometheus_fastapi_instrumentator import Instrumentator

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.data.transform_data import preprocess_new_data
from Models import load_file

app = FastAPI()

Instrumentator().instrument(app).expose(app)

model = load_file('random_forest.joblib')

@app.get('/')
def home():
    return{'Welcome': 'Churn Prediction'}


@app.post("/predict")
def predict(data: Churn_Input):
    input_df = pd.DataFrame([data.dict()])  
    
    processed = preprocess_new_data(input_df)
    prediction = model.predict(processed)[0]  
    # proba = model.predict_proba(processed)[0]
    
    if prediction == 1:
        return {
            "prediction": "This customer would churn",
            # "confidence": round(proba[1], 3)   
        }
    else:
        return {
            "prediction": "Customer would not churn",
            # "confidence": round(proba[0], 3)  
        }