import re
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from src.util import load_data, save_train_test_data, load_params, save_object


def cleanResume(txt):
    cleanText = re.sub("http\S+\s", " ", txt)
    cleanText = re.sub("RT|cc", " ", cleanText)
    cleanText = re.sub("#\S+\s", " ", cleanText)
    cleanText = re.sub("@\S+", "  ", cleanText)
    cleanText = re.sub(
        "[%s]" % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), " ", cleanText
    )
    cleanText = re.sub(r"[^\x00-\x7f]", " ", cleanText)
    cleanText = re.sub("\s+", " ", cleanText)
    return cleanText


def encode_label(train_data: pd.DataFrame):
    try:
        label_encoder = LabelEncoder()
        label_encoder.fit(train_data["Category"])
        save_object(
            label_encoder,
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "../../data/res/encoder.pkl"
            ),
        )
        return train_data
    except Exception as e:
        print(e)
        raise


def vectorization(train_data: pd.DataFrame) -> pd.DataFrame:
    try:
        tfidf = TfidfVectorizer(stop_words="english")
        tfidf.fit(train_data["Resume"])
        save_object(
            tfidf,
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "../../data/res/tfidf.pkl"
            ),
        )
        return train_data
    except Exception as e:
        print(e)
        raise


def normalize_text(df: pd.DataFrame) -> pd.DataFrame:
    try:
        os.makedirs(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../data/res"),
            exist_ok=True,
        )
        df["Resume"] = df["Resume"].apply(lambda x: cleanResume(x))
        df = encode_label(df)
        df = vectorization(df)
        return df
    except Exception as e:
        print(e)
        raise


if __name__ == "__main__":
    params = load_params(
        param_path=os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "../../params.yaml"
        )
    )
    raw_data = load_data(
        data_url=os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "../../data/raw/resume.csv"
        )
    )

    processed_data = normalize_text(raw_data)
    train_data, test_data = train_test_split(
        processed_data,
        test_size=params["data_preprocessing"]["test_size"],
        random_state=params["data_preprocessing"]["random_state"],
    )

    save_train_test_data(
        train_data,
        test_data,
        data_path=os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "../../data"
        ),
        mode="processed",
    )
