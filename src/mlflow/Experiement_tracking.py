import os
import mlflow
import dagshub
import joblib
import mlflow.sklearn



# use the function inside the evaluate model cause all the report necessary for tracking are developed there
def log_experiement_tracking(reports, X_train):
    """
    Takes in the report and then logs each of the metrics to mlflow for online tracking
    don't forget the login credentials with dagshub
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
    for model_report in reports:
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
                        input_example=X_train,
                        )
                
                print('Experiement tracked successfully')

        except Exception as e:
            print(f"Error tracking {model_report['Model_name']}")
    

    


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