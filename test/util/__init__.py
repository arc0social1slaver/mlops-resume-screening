import os
from ui.util import cleanResume
from src.util import load_object


def prediction(fileName: str):
    my_tfidf = load_object(
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "../artifact/tfidf.pkl"
        )
    )
    my_le = load_object(
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "../artifact/encoder.pkl"
        )
    )

    my_model = load_object(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "../artifact/clf.pkl")
    )

    # Preprocess the input text (e.g., cleaning, etc.)

    with open(
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)), f"../input/{fileName}"
        ),
        "r",
    ) as file:
        input_resume = file.read()

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
