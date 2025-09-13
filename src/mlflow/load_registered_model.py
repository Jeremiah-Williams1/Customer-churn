import mlflow
import mlflow.pyfunc
from .mlflow_setup import setup_mlflow


def load_model_by_version(model_name, version):

    setup_mlflow(experiment_name="Customer Churn Prediction")
    
    try:
        model_uri = f"models:/{model_name}/{version}"
        print(f"Loading model: {model_uri}")
        
        model = mlflow.pyfunc.load_model(model_uri)
        print(f" Model loaded successfully!")
        return model
        
    except Exception as e:
        print(f" Failed to load model: {e}")
        raise


def load_latest_model(model_name):
    setup_mlflow(experiment_name="Customer Churn Prediction")
    
    try:
        # Use "latest" to get the most recent version
        model_uri = f"models:/{model_name}/latest"
        print(f"Loading latest model: {model_uri}")
        
        model = mlflow.pyfunc.load_model(model_uri)
        print(f"Latest model loaded successfully!")
        return model
        
    except Exception as e:
        print(f"Failed to load latest model: {e}")
        raise


# Simple wrapper function - use this in your FastAPI app
def get_model(model_name, version=None):
    """
    use this in your FastAPI app
    """
    if version:
        return load_model_by_version(model_name, version)
    else:
        return load_latest_model(model_name)



# Testing the func
if __name__ == "__main__":
    print(" Test Model Loading")
    print("-" * 30)
    
    model_name = input("Enter model name to test: ").strip()
    
    try:
        print("\n1. Testing latest version...")
        model = get_model(model_name,version=2)
        print(f"Model type: {type(model)}")
     
    except Exception as e:
        print(f"Test failed: {e}")