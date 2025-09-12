import sys
from pathlib import Path
from fastapi import FastAPI
from .schema_model import Churn_Input
import pandas as pd
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Counter, Histogram, Summary
import time

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.data.transform_data import preprocess_new_data
from Models import load_file

app = FastAPI()
Instrumentator().instrument(app).expose(app)
model = load_file('random_forest.joblib')



REQUEST_COUNT = Counter('request_count', 'Total number of request maade')
NO_CHURN = Counter('no_churn', 'Number of customers predicted to not churn')
CHURN = Counter('churn', 'Number of customers predicted to churn')
INFERENCE_TIME = Histogram('inference_time', 'Time spent for inference')
REQUEST_LATENCY = Summary('request_latency_seconds', 'Latency per request in seconds')



@app.get('/')
def home():
    return{'Welcome': 'Churn Prediction'}


@app.post("/predict")
def predict(data: Churn_Input):

    REQUEST_COUNT.inc()
    latency_time = time.time()

    input_df = pd.DataFrame([data.dict()])  
    processed = preprocess_new_data(input_df)
    
    inference_start = time.time()
    prediction = model.predict(processed)[0]  
    # proba = model.predict_proba(processed)[0]
    INFERENCE_TIME.observe(time.time() - inference_start)


    
    if prediction == 1:
        CHURN.inc()
        REQUEST_LATENCY.observe(time.time() - latency_time)
        return {
            "prediction": "This customer would churn",
            # "confidence": round(proba[1], 3)   
        }
    else:
        NO_CHURN.inc()
        REQUEST_LATENCY.observe(time.time() - latency_time)
        return {
            "prediction": "Customer would not churn",
            # "confidence": round(proba[0], 3)  
        }