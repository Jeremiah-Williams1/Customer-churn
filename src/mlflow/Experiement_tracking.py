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


def log_experiement_tracking(reports, X_train):
    """
    Takes in the report and then logs each of the metrics to mlflow for online tracking
    """
    # Login credentials for mlflow tracking on dagshub
    import dagshub
    dagshub.init(repo_owner='jerremiahwilly', repo_name='Customer-churn', mlflow=True)
    os.environ['MLFLOW_TRACKING_USERNAME'] = 'jerremiahwilly'
    os.environ['MLFLOW_TRACKING_PASSWORD'] = '6bf1a4070c641409d85a62e55a66ae963eaaf4d5'
    os.environ['MLFLOW_TRACKING_URI'] = 'https://dagshub.com/jerremiahwilly/Customer-churn.mlflow'

    # Experiement Name and uri
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    mlflow.set_experiment('Customer Churn Predictio')  # The name of the Project

    # Logging metrics and model 
    for i, model_report in enumerate(reports):
        try:    
            with mlflow.start_run(run_name=  model_report['Model_name']):

                mlflow.set_tag("model_name",  model_report['Model_name']) # adds metadata tages to model 
                
                # saves the hyperparameters used
                for p, v in model_report["Parameters"].items():
                    mlflow.log_param(p, v) 

                # save the metrics
                for m, v in model_report["Metrics"].items():
                    mlflow.log_metric(m, v) 

                mlflow.log_param('Model name',  model_report['Model_name']) # a good pratice, usefull when trying to compare 

                # logs the model 
                model_info = mlflow.sklearn.log_model(
                        sk_model=model_report['Model'],
                        artifact_path= model_report['Model_name'],
                        # consider adding an input example to show snippet of the datset used
                        )
                
                print(f'successfully logged Experiment for {model_report['Model_name']}')

        except Exception as e:
            print(f"Error tracking {model_report['Model_name']}")
    
    print(f'successfully logged {i+1} experiments')

    


def register_model():
    """
    Registers a model that has been logged based on it id
    """
    model_name = 'RandomForest Model'
    run_id = input('Enter Run ID')

    model_uri = f'run:/{run_id}/{model_name}'
    result = mlflow.register_model(model_uri, model_name)  # my prefered choice based on the metrics 
    
    return result


def load_registered_model(model_name,  model_version):
    """
    loads one of the registered model based on some id 
    """
    load_uri = f"models:/{model_name}/{model_version}"

    # loaded_model = mlflow.sklearn.load_model(load_uri)
    loaded_ml = mlflow.sklearn.load_model(load_uri)


def main():
    """Main Experiement tracking function"""
    print('\n Logging the Experiements')

    X_train = load_data('../data/processed/X_train.csv')
    reports = joblib.load('../metrics/Report.joblib')

    log_experiement_tracking(reports, X_train)


if __name__ == "__main__":
    main()