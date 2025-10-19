import os
import json
import pandas as pd
import mlflow
import mlflow.sklearn as mlf_sklearn
import matplotlib.pyplot as plt
import seaborn as sns
from mlflow.models import infer_signature
from src.util import load_data, load_object, get_mlflow_url
from src.model import ProcessingData
from sklearn.metrics import classification_report, confusion_matrix


def model_evaluation(X_test, y_test, processor: ProcessingData) -> None:
    mlflow.set_tracking_uri(get_mlflow_url())
    mlflow.set_experiment('DVC pipeline')
    with mlflow.start_run() as run:
        try:
            model = load_object(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../data/res/clf.pkl'))
            ip_example = pd.DataFrame(X_test[:5], columns=processor.tfidf.get_feature_names_out())
            sign = infer_signature(ip_example, model.predict(X_test[:5]))
            mlf_sklearn.log_model(model, 'KNeigbours', signature=sign, input_example=ip_example)
            
            with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../data/experiment-info.json'), 'w') as file:
                model_info = {
                    'model_path': 'KNeigbours',
                    'run_id': run.info.run_id
                }
                json.dump(model_info, file, indent=4)
            
            mlflow.log_artifact(processor.tfidf_path)
            mlflow.log_artifact(processor.encoder_path)

            y_pred_knn = model.predict(X_test)

            classification_rp = classification_report(y_test, y_pred_knn, output_dict=True)
            for label, metrics in classification_rp.items():
                if isinstance(metrics, dict):
                    mlflow.log_metrics({
                        f"test_{label}_precision": metrics['precision'],
                        f"test_{label}_recall": metrics['recall'],
                        f"test_{label}_f1-score": metrics['f1-score']
                    })

            conf_matrix = confusion_matrix(y_test, y_pred_knn)
            plt.figure(figsize=(8, 6))
            sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            plt.title("Confusion Matric")
            plt.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../data/confusion_matrix.png'))
            mlflow.log_artifact(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../data/confusion_matrix.png'))
        except Exception as e:
            print(e)
            raise

if __name__ == "__main__":
    test_data = load_data(data_url=os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../data/processed/test.csv'))
    processor = ProcessingData(test_data)
    X_test, y_test = processor.standardize_data()
    model_evaluation(X_test, y_test, processor)