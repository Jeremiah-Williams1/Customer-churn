# registry/load_registered_model.py
import mlflow
import sklearn
from utils.mlflow_setup import setup_mlflow

def load_registered_model(model_name: str, model_version: str):
    """
    Loads a registered model from MLflow Model Registry for inference.
    """
    # Set up MLflow connection
    setup_mlflow(experiment_name="Customer Churn Prediction")

    load_uri = f"models:/{model_name}/{model_version}"
    model = mlflow.sklearn.load_model(load_uri)

    print(f"Loaded model '{model_name}' version {model_version} from registry.")
    return model
