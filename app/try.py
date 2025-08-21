import sys
import os
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.mlflow.load_registered_model import get_model
from src.mlflow.mlflow_setup import setup_mlflow

model = get_model(model_name='Random Forest')
print('Model_loaded sucessfully')
