from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
from preprocessing import Preprocessing

# Load the trained model
model_path = 'model.pkl'
with open(model_path, 'rb') as file:
    model = pickle.load(file)

app = Flask(__name__)

# Serve the HTML form
@app.route('/')
def home():
    return render_template('index.html')

# Handle form submission and prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        question1 = request.form.get('question1', "")
        question2 = request.form.get('question2', "")

        if not question1 or not question2:
            return render_template('index.html', result="Both questions are required!")

        # Preprocess and prepare the input data
        preprocessor = Preprocessing()
        processed_data = preprocessor.query_point_creator(question1, question2)

        # Perform prediction
        prediction = model.predict(processed_data)
        result = "Duplicate" if prediction[0] else "Not Duplicate"

        return render_template('index.html', result=result)
    except Exception as e:
        return render_template('index.html', result=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
