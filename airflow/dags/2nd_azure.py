from datetime import datetime, timedelta
from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from airflow.models import Variable 
import os

# Azure import
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from azure.ai.ml.entities import Workspace, CommandJob, AmlCompute, Environment

# Azure Credientials 
subscription_id = Variable.get("subscription_id", default_var="aaa50594-b0f4-4cd7-b433-a6f952685c32")
resource_group = Variable.get("resource_group", default_var="rg-training")
workspace_name = Variable.get("workspace", default_var="my_ml_workspace")


# Default arguments for the DAG
default_args = {
    'owner': 'Itoro-williams',
    'start_date': datetime(2025, 1, 1),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Create the DAG
dag = DAG(
    'orchestration_with_aml',
    default_args=default_args,
    description='ML Pipeline: Train -> Evaluate -> Log Experiments',
    schedule_interval=timedelta(days=1),
    catchup=False,
)

# Task 1: Prepare AML workspace
def prepare_AML_workspace():
    try:
        ml_client = MLClient(
            DefaultAzureCredential(), subscription_id, resource_group
        )

        print('Connected to Azure ML Successfully')
        print('Creating Workspace...\n')

        workspace = Workspace(
            name = workspace_name,
            description = "created a workspace using the nano inside a python sdk",
            location = 'southafrica north',
        )
        ml_client.workspace.begin_create(workspace).result()
        print("Workspace created successfully.")
    except Exception as e:
        print(f"Error creating workspace: {e}")

prepare_workspace_task = PythonOperator(
    task_id='prepare_AML_workspace',
    python_callable=prepare_AML_workspace,
    dag=dag,
)


# Task 2: Create Compute Instance
def create_compute():
    try:
        ml_client = MLClient(
            DefaultAzureCredential(), subscription_id, resource_group, workspace_name
        )

        compute_instance = AmlCompute(
            name="my-compute",
            size="Standard_DS3_v2",
            description="Compute cluster for running dags",
            min_instances=0,
            max_instances=4,
            idle_time_before_scale_down=120,
        )

        ml_client.compute.begin_create_or_update(compute_instance).result()
        print("Compute Instance created successfully.")
    except Exception as e:
        print(f"Error creating Compute Instance: {e}")

create_compute_instance_task = PythonOperator(
    task_id='create_compute_cluster',
    python_callable=create_compute,
    dag=dag,
)

# Task 3: Create the environment
def create_environment():
    try:
        ml_client = MLClient(
            DefaultAzureCredential(), subscription_id, resource_group, workspace_name
        )

        environment = Environment(
            name="my-environment",
            description="Environment for training and evaluation",
            conda_file=None,
            image="972111245465383674563253/prediction-apps:latest",
        )

        ml_client.environments.create_or_update(environment)
        print("Environment created successfully.")
    except Exception as e:
        print(f"Error creating environment: {e}")

create_environment_task = PythonOperator(
    task_id='create_environment',
    python_callable=create_environment,
    dag=dag,
)

# Helper function
def train_job():
    try:
        ml_client = MLClient(
            DefaultAzureCredential(), subscription_id, resource_group, workspace_name
        )

        job = CommandJob(
            name='train-job',
            command='python script/train.py',
            environment="my-environment:1",
            compute="my-compute",
        )

        ml_client.jobs.create_or_update(job)
        print(f" submitted successfully.")
    except Exception as e:
        print(f"Error submitting: {e}")  

# Task 4: Train the model
train_model_task = PythonOperator(
    task_id='train_model_using_AML',
    python_callable=train_job,
    dag=dag,
)   


# Task 5: Evaluate the model
def evaluate_job():
    try:
        ml_client = MLClient(
            DefaultAzureCredential(), subscription_id, resource_group, workspace_name
        )

        job = CommandJob(
            name='train-job',
            command='python script/evaluate.py',
            environment="my-environment:1",
            compute="my-compute",
        )

        ml_client.jobs.create_or_update(job)
        print(f" submitted successfully.")
    except Exception as e:
        print(f"Error submitting: {e}")  

evaluate_model_task = PythonOperator(
    task_id='evaluate_model_using_AML',
    python_callable=evaluate_job,
    dag=dag,
)   


# Define task dependencies
prepare_workspace_task >> create_compute_instance_task >> train_model_task >> evaluate_model_task 