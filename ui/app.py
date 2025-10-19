from flask import Flask, request, render_template, jsonify
from flask_cors import CORS
from ui.util import pred

app = Flask(__name__)
CORS(app)

@app.route("/")
def home():
    return render_template("index.html")

@app.post("/predict")
def predictable():
    if request.files and request.method == 'POST':

        if 'file' not in request.files:
            return jsonify({"message": "No file part in the request"}), 400
        
        user_file = request.files['file']

        content = user_file.read().decode('utf-8')

        prediction = pred(content)

        return jsonify({"message": prediction})
    return jsonify({"message": "Invalid request"}), 403