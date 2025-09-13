import sys
from pathlib import Path
from prometheus_client import CollectorRegistry, Gauge, Counter, push_to_gateway
import time

# Add src to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.data.load_data import load_data
from src.data.clean_data import clean_data
from src.data.transform_data import transform_data

from model_configs.config import train_all_models


def main():
    """Main training pipeline"""
    print("=== CUSTOMER CHURN PREDICTION TRAINING ===")
    
    # Initialize metrics
    registry, training_duration, data_samples, model_count = create_metrics()
    start_time = time.time()
    
    try:
        # Load data
        print("\n1. Loading data...")
        df = load_data("data/raw/customer_churn.csv")
        
        # Clean data
        print("\n2. Cleaning data...")
        df_clean = clean_data(df)
        
        # Transform data
        print("\n3. Transforming data...")
        X_train, X_test, y_train, y_test = transform_data(df_clean)
        
        # Record data samples metric
        data_samples.set(len(X_train))
        
        # Train models
        print("\n4. Training models...")
        models = train_all_models(X_train, y_train)
        
        # Record metrics
        model_count.inc(len(models))
        training_duration.set(time.time() - start_time)
        
        print(f"\nTraining completed! {len(models)} models trained and saved.")
        
        # Push metrics to Pushgateway
        push_metrics(registry)
        
    except Exception as e:
        print(f"Error during training: {e}")
        # Still try to push metrics even if training fails
        training_duration.set(time.time() - start_time)
        push_metrics(registry)
        raise

def create_metrics():
    registry = CollectorRegistry()
    
    # Define metrics
    training_duration = Gauge('model_training_duration_seconds', 
                            'Time taken for model training',
                            registry=registry)
    
    data_samples = Gauge('training_data_samples_total',
                        'Number of training samples',
                        registry=registry)
    
    model_count = Counter('models_trained_total',
                         'Number of models trained',
                         registry=registry)
    
    return registry, training_duration, data_samples, model_count

def push_metrics(registry, job_name='customer_churn_training'):
    """Push metrics to Pushgateway"""
    try:
        push_to_gateway('localhost:9091', job=job_name, registry=registry)
        print("Metrics pushed to Pushgateway successfully")
    except Exception as e:
        print(f"Failed to push metrics: {e}")

if __name__ == "__main__":
    main()