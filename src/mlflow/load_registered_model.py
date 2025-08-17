import mlflow
import mlflow.pyfunc
from mlflow_setup import setup_mlflow

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
    """
    Load the latest version of a registered model
    
    Args:
        model_name (str): Name of the registered model
    
    Returns:
        Loaded model object
    """
    setup_mlflow(experiment_name="Customer Churn Prediction")
    
    try:
        # Use "latest" to get the most recent version
        model_uri = f"models:/{model_name}/latest"
        print(f"Loading latest model: {model_uri}")
        
        model = mlflow.pyfunc.load_model(model_uri)
        print(f"✅ Latest model loaded successfully!")
        return model
        
    except Exception as e:
        print(f"❌ Failed to load latest model: {e}")
        raise

def load_production_model(model_name):
    """
    Load model from Production stage
    
    Args:
        model_name (str): Name of the registered model
    
    Returns:
        Loaded model object
    """
    setup_mlflow(experiment_name="Customer Churn Prediction")
    
    try:
        model_uri = f"models:/{model_name}/Production"
        print(f"Loading production model: {model_uri}")
        
        model = mlflow.pyfunc.load_model(model_uri)
        print(f"✅ Production model loaded successfully!")
        return model
        
    except Exception as e:
        print(f"❌ Failed to load production model: {e}")
        print("Tip: Make sure a model version is promoted to Production stage")
        raise

# Simple wrapper function - use this in your FastAPI app
def get_model(model_name, version=None):
    """
    Simple function to get a model - use this in your FastAPI app
    
    Args:
        model_name (str): Name of the registered model
        version (int/str, optional): Specific version. If None, loads latest.
    
    Returns:
        Loaded model object
    """
    if version:
        return load_model_by_version(model_name, version)
    else:
        return load_latest_model(model_name)



# For testing the functions
if __name__ == "__main__":
    print("🔍 Test Model Loading")
    print("-" * 30)
    
    model_name = input("Enter model name to test: ").strip()
    
    try:
        # Test loading latest version
        print("\n1. Testing latest version...")
        model = load_latest_model(model_name)
        print(f"Model type: {type(model)}")
        
        # Test prediction (assuming your model has predict method)
        print(f"Model ready for predictions!")
        
    except Exception as e:
        print(f"Test failed: {e}")