import os
import json
import mlflow
from mlflow.tracking import MlflowClient

def load_model_info(file_path: str) -> dict:
    """Load the model info from a JSON file."""
    try:
        with open(file_path, 'r') as file:
            model_info = json.load(file)
        return model_info
    except Exception as e:
        print(e)
        raise

if __name__ == "__main__":
    mlflow.set_tracking_uri('http://localhost:8888')
    try:
        model_info = load_model_info(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../data/experiment-info.json'))
        model_uri = f"runs:/{model_info['run_id']}/{model_info['model_path']}"

        model_ver = mlflow.register_model(model_uri, model_info['model_path'])
    
        client = MlflowClient()
        client.transition_model_version_stage(model_info['model_path'], model_ver.version, "Staging")
    except Exception as e:
        print(e)
        raise