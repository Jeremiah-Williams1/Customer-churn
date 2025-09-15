from datetime import datetime, timedelta
from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from airflow.models import Variable
from docker.types import Mount
from pathlib import Path
import os

# Default arguments for the DAG
default_args = {
    'owner': 'williams',
    'start_date': datetime(2025, 1, 1),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Create the DAG
dag = DAG(
    'just_orchestration',
    default_args=default_args,
    description='ML Pipeline: Train -> Evaluate -> Log Experiments',
    schedule_interval=timedelta(days=1),
    catchup=False,
)

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
MODELS_PATH = f'{PROJECT_ROOT}/Models'
DATA_DIR = f'{PROJECT_ROOT}/data' 
METRICS_PATH = f'{PROJECT_ROOT}/metrics'
SRC_PATH = f'{PROJECT_ROOT}/src'

print(PROJECT_ROOT)

ML_DOCKER_IMAGE = Variable.get("ml_docker_image", default_var="972111245465383674563253/prediction-apps:latest")
DOCKER_NETWORK = Variable.get("docker_network", default_var="mlops-net")

# Task 1: Prepare workspace (ensure directories exist)
prepare_workspace = BashOperator(
    task_id='prepare_workspace',
    bash_command=f'''
    mkdir -p {MODELS_PATH}
    mkdir -p {DATA_DIR}
    mkdir -p {METRICS_PATH}
    echo "Pipeline started at $(date)" > {MODELS_PATH}/pipeline_log.txt
    ''',
    dag=dag,
)

# Define mount configurations
docker_mounts = [
    Mount(
        source=str(Path(__file__).resolve().parent.parent.parent / 'Models'),
        target='/app/Models',
        type='bind',
        read_only=False
    ),
    Mount(
        source=str(Path(__file__).resolve().parent.parent.parent / 'script'/'train.py'),
        target='/app/script/train.py',
        type='bind',
        read_only=False
    ),
    Mount(
        source=str(Path(__file__).resolve().parent.parent.parent / 'metrics'),
        target='/app/metrics',
        type='bind',
        read_only=False
    ),
    Mount(
        source=str(Path(__file__).resolve().parent.parent.parent / 'src'),
        target='/app/src',
        type='bind',
        read_only=False
    )
]

# Task 2: Train the model
train_model = DockerOperator(
    task_id='train_model',
    image=ML_DOCKER_IMAGE,
    command=['python', 'script/train.py'],
    api_version='auto',
    auto_remove=True,
    docker_url='unix://var/run/docker.sock',
    network_mode=DOCKER_NETWORK,
    mounts=docker_mounts,
    working_dir='/app',
    environment={'PYTHONPATH': '/app'},
    dag=dag,
)

# Task 3: Evaluate the model  
evaluate_model = DockerOperator(
    task_id='evaluate_model',
    image=ML_DOCKER_IMAGE,
    command='python script/evaluate.py',
    api_version='auto',
    auto_remove=True,
    docker_url='unix://var/run/docker.sock',
    network_mode=DOCKER_NETWORK,
    mount_tmp_dir=False,
    container_name='ml_evaluate_{{ ds_nodash }}_{{ ts_nodash }}',
    mounts=docker_mounts,  # Use mounts instead of volumes
    working_dir='/app',
    dag=dag,
)

# Task 4: Log experiments
log_experiments = DockerOperator(
    task_id='log_experiments',
    image=ML_DOCKER_IMAGE,
    command='python src/mlflow/log_experiments.py',
    api_version='auto',
    auto_remove=True,
    docker_url='unix://var/run/docker.sock',
    network_mode=DOCKER_NETWORK,
    mount_tmp_dir=False,
    container_name='ml_log_{{ ds_nodash }}_{{ ts_nodash }}',
    mounts=docker_mounts,  # Use mounts instead of volumes
    working_dir='/app',
    dag=dag,
)

def summarize_results():
    """Python function to summarize the pipeline results"""
    import os
    import json
    
    summary = {
        "pipeline_completed": True,
        "timestamp": datetime.now().isoformat(),
        "data_available": os.path.exists(f"{DATA_DIR}"),
        "metrics_available": os.path.exists(f"{METRICS_PATH}"),
        "Models_available": os.path.exists(f"{MODELS_PATH}")
    }
    
    with open(f"{MODELS_PATH}/pipeline_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print("Pipeline summary created successfully!")
    return summary

summarize_pipeline = PythonOperator(
    task_id='summarize_pipeline',
    python_callable=summarize_results,
    dag=dag,
)

# Define task dependencies
prepare_workspace >> train_model >> evaluate_model >> log_experiments >> summarize_pipeline