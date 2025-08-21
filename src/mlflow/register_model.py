import mlflow
from mlflow import MlflowClient
from mlflow_setup import setup_mlflow

def register_model(run_id, artifact_path, model_name):

    setup_mlflow(experiment_name="Customer Churn Prediction")
    
    try:
        model_uri = f"runs:/{run_id}/{artifact_path}"
        print(f"Registering model from: {model_uri}")
        
        # Register the model (
        result = mlflow.register_model(model_uri, model_name)
        
        print(f"  Model registered successfully!")
        print(f"   Model Name: {result.name}")
        print(f"   Version: {result.version}")
        print(f"   Run ID: {run_id}")
        return result
        
    except Exception as e:
        print(f" Registration failed: {e}")
        raise

def register_latest_run(model_name, artifact_path="model"):
    setup_mlflow(experiment_name="Customer Churn Prediction")
    
    # Get the latest run
    client = MlflowClient()
    experiment = client.get_experiment_by_name("Customer Churn Prediction")
    
    if not experiment:
        raise ValueError("Experiment 'Customer Churn Prediction' not found")
    
    # Get the most recent successful run
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="status = 'FINISHED'",
        max_results=1,
        order_by=["start_time DESC"]
    )
    
    if not runs:
        raise ValueError("No completed runs found in experiment")
    
    latest_run = runs[0]
    run_id = latest_run.info.run_id
    
    print(f"Found latest run: {run_id}")
    return register_model(run_id, artifact_path, model_name)



if __name__ == "__main__":
    print("🚀 MLflow Model Registration")
    print("-" * 40)
    
    choice = input("Register (1) specific run or (2) latest run? Enter 1 or 2: ").strip()
    
    if choice == "1":
        # Manual registration
        run_id = input("Enter Run ID: ").strip()
        artifact_path = input("Enter artifact path (default: 'model'): ").strip() or "model"
        model_name = input("Enter model name: ").strip()
        
        register_model(run_id, artifact_path, model_name)
        
    elif choice == "2":
        # Register latest run
        model_name = input("Enter model name: ").strip()
        artifact_path = input("Enter artifact path (default: 'model'): ").strip() or "model"
        
        register_latest_run(model_name, artifact_path)
    
    else:
        print("Invalid choice. Please run again.")


