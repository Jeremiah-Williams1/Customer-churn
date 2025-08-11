from fastapi import FastAPI

app = FastAPI()

@app.get('/')
def home():
    return {'Message': 'Welcome to Churn Prediction',
            'Goto': '/predict root for prediction'
            }

@app.post('/predict')
def predict():
    pass
    # loads the encoders
    # set the schema for the input to be set
    # pass the input to the encoder to encdde it
    # load the model
    # pass the encoded input to the model
    # get a prediction 
