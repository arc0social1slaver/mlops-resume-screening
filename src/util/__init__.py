import yaml
import pandas as pd
import os
import pickle
from dotenv import load_dotenv

load_dotenv()

def get_mlflow_url():
    return os.getenv('MLFLOW_ENDPOINT_URL')

def load_params(param_path: str) -> dict:
    try:
        with open(param_path, 'r') as file:
            params = yaml.safe_load(file)
        return params
    except Exception as e:
        print(e)
        raise

def load_data(data_url: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    try:
        df = pd.read_csv(data_url)
        return df
    except Exception as e:
        print(e)
        raise

def save_data(data : pd.DataFrame, data_path: str, file_name : str) -> None:
    """Load data from a CSV file."""
    try:
        os.makedirs(data_path, exist_ok=True)
        raw_data_path = os.path.join(data_path, file_name)        

        data.to_csv(raw_data_path, index=False)
    except Exception as e:
        print(e)
        raise

def save_train_test_data(train_data: pd.DataFrame, test_data: pd.DataFrame, data_path: str, mode: str) -> None:
    """Save the train and test datasets, creating the folder depending on mode if it doesn't exist."""
    try:
        raw_data_path = os.path.join(data_path, mode)
        
        os.makedirs(raw_data_path, exist_ok=True)

        # Save the train and test data
        train_data.to_csv(os.path.join(raw_data_path, "train.csv"), index=False)
        test_data.to_csv(os.path.join(raw_data_path, "test.csv"), index=False)

    except Exception as e:
        print(e)
        raise

def save_object(obj, file_path: str) -> None:
    """
    Save object; e.g, model, encoder or vectorize tool to specific file path
    """
    try:
        with open(file_path, 'wb') as file:
            pickle.dump(obj, file)
    except Exception as e:
        print(e)
        raise

def load_object(file_path: str):
    """
    Load object; e.g, model, encoder or vectorize tool from specific file path
    """
    try:
        with open(file_path, 'rb') as file:
            return pickle.load(file)
    except Exception as e:
        print(e)
        raise