import os
import mlflow
import dagshub
from mlflow.tracking import MlflowClient


def setup_mlflow(experiment_name: str):
    """
    Sets up MLflow tracking with Dagshub credentials and experiment.
    Credentials are hardcoded (not recommended for production).
    """
    repo_owner = "jerremiahwilly"
    repo_name = "Customer-churn"
    tracking_username = "jerremiahwilly"
    tracking_password = "6bf1a4070c641409d85a62e55a66ae963eaaf4d5"
    tracking_uri = "https://dagshub.com/jerremiahwilly/Customer-churn.mlflow"

    # Initialize Dagshub MLflow connection
    dagshub.init(
        repo_owner=repo_owner,
        repo_name=repo_name,
        mlflow=True
    )

    os.environ["MLFLOW_TRACKING_USERNAME"] = tracking_username
    os.environ["MLFLOW_TRACKING_PASSWORD"] = tracking_password
    os.environ["MLFLOW_TRACKING_URI"] = tracking_uri

    # Point MLflow to the correct tracking server
    mlflow.set_tracking_uri(tracking_uri)

    client = MlflowClient()

    # Set experiment
    mlflow.set_experiment(experiment_name)
