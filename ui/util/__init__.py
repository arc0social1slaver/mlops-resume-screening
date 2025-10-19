import re
import os
import mlflow
import json
import mlflow.sklearn as mlf_sklearn
from src.util import load_object, get_mlflow_url

def load_model_info(file_path: str) -> dict:
    """Load the model info from a JSON file."""
    try:
        with open(file_path, 'r') as file:
            model_info = json.load(file)
        return model_info
    except Exception as e:
        print(e)
        raise

def cleanResume(txt):
    cleanText = re.sub('http\S+\s', ' ', txt)
    cleanText = re.sub('RT|cc', ' ', cleanText)
    cleanText = re.sub('#\S+\s', ' ', cleanText)
    cleanText = re.sub('@\S+', '  ', cleanText)  
    cleanText = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', cleanText)
    cleanText = re.sub(r'[^\x00-\x7f]', ' ', cleanText) 
    cleanText = re.sub('\s+', ' ', cleanText)
    return cleanText

# Function to predict the category of a resume
def pred(input_resume):
    
    mlflow.set_tracking_uri(get_mlflow_url())
    my_tfidf = load_object(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../data/res/tfidf.pkl'))
    my_le = load_object(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../data/res/encoder.pkl'))
    model_info = load_model_info(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../data/experiment-info.json'))
    
    model_uri = f"models:/{model_info["model_path"]}/1"
    my_model = mlf_sklearn.load_model(model_uri)

    # Preprocess the input text (e.g., cleaning, etc.)

    cleaned_text = cleanResume(input_resume) 

    # Vectorize the cleaned text using the same TF-IDF vectorizer used during training
    vectorized_text = my_tfidf.transform([cleaned_text])
    
    # Convert sparse matrix to dense
    vectorized_text = vectorized_text.toarray()

    # Prediction
    predicted_category = my_model.predict(vectorized_text)

    # get name of predicted category
    predicted_category_name = my_le.inverse_transform(predicted_category)

    return predicted_category_name[0]  # Return the category name