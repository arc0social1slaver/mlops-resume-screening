import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multiclass import OneVsRestClassifier
from src.util import save_object, load_data
from src.model import ProcessingData


def model_training(X_train, y_train) -> None:
    try:
        knn_model = OneVsRestClassifier(KNeighborsClassifier())
        knn_model.fit(X_train, y_train)
        save_object(
            knn_model,
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "../../data/res/clf.pkl"
            ),
        )
    except Exception as e:
        print(e)
        raise


if __name__ == "__main__":
    train_data = load_data(
        data_url=os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "../../data/processed/train.csv"
        )
    )
    X_train, y_train = ProcessingData(train_data).standardize_data()
    model_training(X_train, y_train)
