from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.models import Variable 
from dotenv import load_dotenv
load_dotenv()
import os


# Azure imports
from azure.ai.ml import MLClient, command
from azure.identity import ClientSecretCredential
from azure.ai.ml.entities import Workspace, CommandJob, AmlCompute, Environment

# Azure Credentials - Put your actual values here
subscription_id = os.getenv("SUBSCRIPTION_ID")
resource_group = "rg-training"
workspace_name = "my_ml_workspace"

# Authentication credentials - Replace with your actual values
tenant_id = os.getenv("TENANT_ID")
client_secret = os.getenv("CLIENT_SECRET")
client_id = os.getenv("CLIENT_ID")

# Create credential object
credential = ClientSecretCredential(
    tenant_id=tenant_id,
    client_id=client_id,
    client_secret=client_secret
)

# Default arguments for the DAG
default_args = {
    'owner': 'Jerremiah-Williams',
    'start_date': datetime(2025, 1, 1),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Create the DAG
dag = DAG(
    'azure_pipeline',
    default_args=default_args,
    description='Simple ML Pipeline: Setup -> Train -> Evaluate -> Log',
    schedule_interval=timedelta(days=1),
    catchup=False,
)

# Task 1: Create workspace (check if exists first)
def prepare_AML_workspace():
    try:
        ml_client = MLClient(credential, subscription_id, resource_group)
        
        # Check if workspace exists
        try:
            existing_workspace = ml_client.workspaces.get(workspace_name)
            print(f"Workspace {workspace_name} already exists!")
            return
        except:
            print("Workspace doesn't exist, creating new one...")

        workspace = Workspace(
            name=workspace_name,
            description="ML workspace for training",
            location='southafricanorth', 
        )
        
        ml_client.workspaces.begin_create(workspace).result()
        print("Workspace created successfully!")
        
    except Exception as e:
        print(f"Error with workspace: {e}")
        raise

prepare_workspace_task = PythonOperator(
    task_id='prepare_workspace',
    python_callable=prepare_AML_workspace,
    dag=dag,
)

# Task 2: Create compute (check if exists first)
def create_compute():
    try:
        ml_client = MLClient(credential, subscription_id, resource_group, workspace_name)
        
        # Check if compute exists
        try:
            existing_compute = ml_client.compute.get("my-compute")
            print("Compute already exists!")
            return
        except:
            print("Creating new compute...")

        compute_instance = AmlCompute(
            name="my-compute",
            size="Standard_DS3_v2",
            description="Compute for ML jobs",
            min_instances=0,
            max_instances=4,
            idle_time_before_scale_down=120,
        )

        ml_client.compute.begin_create_or_update(compute_instance).result()
        print("Compute created successfully!")
        
    except Exception as e:
        print(f"Error creating compute: {e}")
        raise

create_compute_task = PythonOperator(
    task_id='create_compute',
    python_callable=create_compute,
    dag=dag,
)

# Task 3: Create environment (always update to get latest Docker image)
def create_environment():
    try:
        ml_client = MLClient(credential, subscription_id, resource_group, workspace_name)
        
        environment = Environment(
            name="my-environment",
            description="Environment with latest Docker image",
            conda_file=None,
            image="972111245465383674563253/prediction-apps:latest",
        )

        ml_client.environments.create_or_update(environment)
        print("Environment created/updated successfully!")
        
    except Exception as e:
        print(f"Error creating environment: {e}")
        raise

create_environment_task = PythonOperator(
    task_id='create_environment',
    python_callable=create_environment,
    dag=dag,
)

# Simple helper function for running jobs
def run_job(job_name, Command):
    try:
        ml_client = MLClient(credential, subscription_id, resource_group, workspace_name)
        
        # Add timestamp to make job name unique
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_job_name = f"{job_name}_{timestamp}"

        job = command(
            display_name=unique_job_name,
            command=Command,
            environment="my-environment@latest",
            compute="my-compute",
            experiment_name="azure_orchestration_experiment" 
        )

        ml_client.create_or_update(job)
        print(f"Job {unique_job_name} submitted successfully!")
        
    except Exception as e:
        print(f"Error running {job_name}: {e}")
        raise

# Task 4: Train model
train_model_task = PythonOperator(
    task_id='train_model',
    python_callable=run_job,
    op_kwargs={
        "job_name": "train_job",
        "Command": "python /app/script/train.py"
    },
    dag=dag,
)   

# Task 5: Evaluate model
evaluate_model_task = PythonOperator(
    task_id='evaluate_model',
    python_callable=run_job,
    op_kwargs={
        "job_name": "evaluate_job", 
        "Command": "python /app/script/evaluate.py"
    },
    dag=dag,
)   

# Task 6: Log experiments
log_experiments_task = PythonOperator(
    task_id='log_experiments',
    python_callable=run_job,
    op_kwargs={
        "job_name": "log_job",
        "Command": "python /app/src/mlflow/log_experiments.py"
    },
    dag=dag,
)

# Set task order
prepare_workspace_task >> create_compute_task >> create_environment_task >> train_model_task >> evaluate_model_task >> log_experiments_task