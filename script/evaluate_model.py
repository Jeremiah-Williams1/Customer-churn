import sys
from pathlib import Path
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Add src to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.data.load_data import load_data
from src.data.clean_data import clean_data
from src.data.transform_data import transform_data
from models.train import load_model

def evaluate_single_model(model, X_test, y_test, model_name):
    """Evaluate a single model"""
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        'Model': model_name,
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1_Score': f1_score(y_test, y_pred),
        'ROC_AUC': roc_auc_score(y_test, y_pred_proba)
    }
    
    return metrics

def main():
    """Main evaluation pipeline"""
    print("=== MODEL EVALUATION ===")
    
    # Load and prepare data
    print("\n1. Preparing data...")
    df = load_data("../data/raw/customer_churn.csv")
    df_clean = clean_data(df)
    X_train, X_test, y_train, y_test = transform_data(df_clean)
    
    # Load and evaluate models
    print("\n2. Evaluating models...")
    model_files = {
        'Logistic Regression': 'models/logistic_regression.pkl',
        'Random Forest': 'models/random_forest.pkl'
    }
    
    results = []
    for model_name, filepath in model_files.items():
        try:
            model = load_model(filepath)
            metrics = evaluate_single_model(model, X_test, y_test, model_name)
            results.append(metrics)
            print(f"\n{model_name} Results:")
            for metric, value in metrics.items():
                if metric != 'Model':
                    print(f"  {metric}: {value:.4f}")
        except Exception as e:
            print(f"Error evaluating {model_name}: {str(e)}")
    
    # Summary
    if results:
        results_df = pd.DataFrame(results)
        print("\n=== SUMMARY ===")
        print(results_df.round(4))
        
        # Best model
        best_model = results_df.loc[results_df['ROC_AUC'].idxmax()]
        print(f"\nBest model: {best_model['Model']} (ROC-AUC: {best_model['ROC_AUC']:.4f})")

if __name__ == "__main__":
    main()