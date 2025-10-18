import yaml
import pandas as pd
import os


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

def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, data_path: str, mode: str) -> None:
    """Save the train and test datasets, creating the raw folder if it doesn't exist."""
    try:
        raw_data_path = os.path.join(data_path, mode)
        
        os.makedirs(raw_data_path, exist_ok=True)

        # Save the train and test data
        train_data.to_csv(os.path.join(raw_data_path, "train.csv"), index=False)
        test_data.to_csv(os.path.join(raw_data_path, "test.csv"), index=False)

    except Exception as e:
        print(e)
        raise