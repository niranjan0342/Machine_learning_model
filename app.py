from flask import Flask, request, jsonify
import joblib
import pandas as pd    
import numpy as np

app = Flask(__name__)

# Load the trained model and scaler
model = joblib.load('logistic_regression_model.joblib')
scaler = joblib.load('scalar.joblib')

@app.route('/')
def home():
    return "Flask app is running. Use /predict for predictions."

@app.route('https://machine-learning-model-5.onrender.com//predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        input_df = pd.DataFrame(data, index=[0])

        # Scale the input data
        input_scaled = scaler.transform(input_df)

        # Make prediction
        prediction = model.predict(input_scaled)
        prediction_proba = model.predict_proba(input_scaled)

        return jsonify({
            'prediction': int(prediction[0]),
            'prediction_proba_class_0': float(prediction_proba[0][0]),
            'prediction_proba_class_1': float(prediction_proba[0][1])
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000) 

