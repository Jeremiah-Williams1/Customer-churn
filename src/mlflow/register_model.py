# registry/register_model.py
import mlflow
from utils.mlflow_setup import setup_mlflow

def register_model():
    """
    Registers a model from MLflow artifacts based on run ID and model path.
    """
    # Set up MLflow connection (same as logging function)
    setup_mlflow(experiment_name="Customer Churn Prediction")

    model_name = input("Enter Registered Model Name: ").strip()
    run_id = input("Enter Run ID: ").strip()
    artifact_path = input("Enter artifact path used in logging (e.g., model name in log_model): ").strip()

    model_uri = f"runs:/{run_id}/{artifact_path}"
    result = mlflow.register_model(model_uri, model_name)

    print(f"Model registered: {result.name}, version: {result.version}")
    return result

# the artifct path is set in the Experiment_tracking but a code snippet is shown below
'''
                mlflow.sklearn.log_model(
                    sk_model=model_report['Model'],
                    artifact_path=model_report['Model_name'],
                )

'''