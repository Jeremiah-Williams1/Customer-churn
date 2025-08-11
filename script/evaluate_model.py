import sys
import os
import mlflow
import joblib
from pathlib import Path
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,classification_report

# Add src to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.data.load_data import load_data
from src.data.clean_data import clean_data
from src.data.transform_data import transform_data
from models.train import load_model
from src.mlflow.Experiement_tracking import log_experiement_tracking

def evaluate_single_model(model, X_test, y_test):
    """Evaluate a single model"""
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    classification_results = classification_report(y_test,y_pred,output_dict=True)
    recall_class_0 = classification_results['0']['recall']
    recall_class_1 = classification_results['0']['recall']

    metrics = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'recall_for_class_0': recall_class_0,
        'recall_for_class_1': recall_class_1,
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
    
    results = []
    reports = joblib.load('../metrics/Report.joblib')
    

    for report in reports:
        model_name = report['Model_name']
        try:
            model = report['Model']
            metrics = evaluate_single_model(model,X_test,y_test)
            report['Metrics'] = metrics


            results.append(metrics)
            print(f"\n{model_name} Results:")
            for metric, value in metrics.items():
                print(f"  {metric}: {value:.4f}")


        except Exception as e:
            print(f"Error evaluating {model_name}: {str(e)}")

    joblib.dump(reports, "../metrics/Report.joblib")

    # Summary
    if results:
        results_df = pd.DataFrame(results)
        print("\n=== SUMMARY ===")
        print(results_df.round(4))
        

if __name__ == "__main__":
    main()