import os 
import joblib

file_path = os.path.dirname(__file__)

def load_file(filename):
    path =os.path.join(file_path,filename)
    return joblib.load(path)