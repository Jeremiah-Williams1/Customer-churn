import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient
from app.main import app   

client = TestClient(app)

def test_home():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"Welcome": "Churn Prediction"}

def test_predict():
    sample_input = {
        "age": 45,
        "tenure": 24,
        "monthly_charges": 70.5,
        "total_charges": 1680.0,
        "gender": "Male",                     
        "online_security": "Yes",             
        "contract": "Month-to-month",         
        "internet_service": "Fiber optic",    
        "tech_support": "No",                 
        "streaming_tv": "Yes",              
        "payment_method": "Electronic check",
        "paperless_billing": "Yes"            
    }


    response = client.post("/predict", json=sample_input)
    # print("Status:", response.status_code)
    # print("Response JSON:", response.json())
    assert response.status_code == 200
    assert "prediction" in response.json()
