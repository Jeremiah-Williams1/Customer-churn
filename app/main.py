from fastapi import FastAPI

app = FastAPI()

@app.get('/')
def home():
    return {'Message': 'Welcome to Churn Prediction',
            'Goto': '/predict root for prediction'
            }