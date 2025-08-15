import os
import sys
import mlflow
import dagshub
import joblib
import mlflow.sklearn
from pathlib import Path
import sys
from pathlib import Path


sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.data.load_data import load_data


reports = joblib.load('../metrics/Report.joblib')

from utils.mlflow_setup import setup_mlflow
import mlflow

def log_experiment_tracking(reports, X_train):
    """
    Logs each model's metrics and parameters to MLflow.
    """
    # Setup MLflow with credentials & experiment name
    setup_mlflow(experiment_name="Customer Churn Prediction")

    for i, model_report in enumerate(reports):
        try:
            with mlflow.start_run(run_name=model_report['Model_name']):
                mlflow.set_tag("model_name", model_report['Model_name'])

                # Log hyperparameters
                for p, v in model_report["Parameters"].items():
                    mlflow.log_param(p, v)

                # Log metrics
                for m, v in model_report["Metrics"].items():
                    mlflow.log_metric(m, v)

                mlflow.log_param('Model name', model_report['Model_name'])

                # Log the model artifact
                mlflow.sklearn.log_model(
                    sk_model=model_report['Model'],
                    artifact_path=model_report['Model_name'],
                )

                print(f"Successfully logged experiment for {model_report['Model_name']}")

        except Exception as e:
            print(f"Error tracking {model_report['Model_name']}: {e}")

    print(f"Successfully logged {i + 1} experiments")


 

def main():
    """Main Experiement tracking function"""
    print('\n Logging the Experiements')

    X_train = load_data('../data/processed/X_train.csv')
    reports = joblib.load('../metrics/Report.joblib')

    log_experiement_tracking(reports, X_train)

if __name__ == "__main__":
    main()