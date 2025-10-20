import os
import pandas as pd
from src.util import load_object


class ProcessingData:
    encoder_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "../../data/res/encoder.pkl"
    )
    tfidf_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "../../data/res/tfidf.pkl"
    )
    data: pd.DataFrame

    def __init__(self, data: pd.DataFrame) -> None:
        self.data = data
        self.tfidf = load_object(self.tfidf_path)
        self.label_encoder = load_object(self.encoder_path)

    def standardize_data(self) -> tuple:
        try:
            self.encode_label()
            X_data, y_data = self.vectorization()

            # Ensure that X_data are dense if they are sparse
            X_data = X_data.toarray() if hasattr(X_data, "toarray") else X_data
            return X_data, y_data
        except Exception as e:
            print(e)
            raise

    def encode_label(self) -> None:
        try:
            self.data["Category"] = self.label_encoder.transform(self.data["Category"])
        except Exception as e:
            print(e)
            raise

    def vectorization(self) -> tuple:
        try:
            requiredTxt = self.tfidf.transform(self.data["Resume"])
            return requiredTxt, self.data["Category"]
        except Exception as e:
            print(e)
            raise
