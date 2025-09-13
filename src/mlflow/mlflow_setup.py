import os
import mlflow
from mlflow.tracking import MlflowClient


def setup_mlflow(experiment_name: str):
    """
    MLflow tracking with Dagshub credentials using token-based auth.
    """
    tracking_uri = "https://dagshub.com/jerremiahwilly/Customer-churn.mlflow"
    
    # Set environment variables for authentication
    os.environ["MLFLOW_TRACKING_USERNAME"] = "jerremiahwilly"
    os.environ["MLFLOW_TRACKING_PASSWORD"] = "6bf1a4070c641409d85a62e55a66ae963eaaf4d5"
    os.environ["MLFLOW_TRACKING_URI"] = tracking_uri

    # Point MLflow to the tracking server
    mlflow.set_tracking_uri(tracking_uri)

    # Set experiment
    try:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            mlflow.create_experiment(experiment_name)
        mlflow.set_experiment(experiment_name)
    except Exception as e:
        print(f"Error setting up MLflow: {e}")
