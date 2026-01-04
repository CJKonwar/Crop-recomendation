# backend/app.py
import os
import urllib.request
import joblib
import xgboost as xgb
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS

# --- Initialize Flask App ---
app = Flask(__name__)
# CORS allows our React frontend to make requests to our Flask backend
CORS(app)

# --- Model and Encoder Loading ---
# URLs for the model and label encoder files
model_url = "https://raw.githubusercontent.com/CJKonwar/Crop-recomendation/main/xgboost_model.json"
label_encoder_url = "https://raw.githubusercontent.com/CJKonwar/Crop-recomendation/main/label_encoder.pkl"

# Local file paths
model_file = "xgboost_model.json"
label_encoder_file = "label_encoder.pkl"

# Download the XGBoost model if it doesn't exist
if not os.path.exists(model_file):
    print("Downloading XGBoost model...")
    urllib.request.urlretrieve(model_url, model_file)
    print("Model downloaded.")

# Download the LabelEncoder if it doesn't exist
if not os.path.exists(label_encoder_file):
    print("Downloading Label Encoder...")
    urllib.request.urlretrieve(label_encoder_url, label_encoder_file)
    print("Label Encoder downloaded.")

# Load the trained XGBoost model and LabelEncoder
try:
    model = xgb.Booster()
    model.load_model(model_file)
    label_encoder = joblib.load(label_encoder_file)
    print("Model and Label Encoder loaded successfully.")
except Exception as e:
    print(f"Error loading model or encoder: {e}")
    model = None
    label_encoder = None

# --- Prediction Logic (from your Streamlit app) ---
def predict_top_crops(input_data, model, label_encoder):
    """
    Predicts the top 5 crops based on input data.
    """
    dmatrix = xgb.DMatrix(input_data)  # Convert input data to DMatrix
    preds = model.predict(dmatrix)  # Predict probabilities

    # Handle multi-dimensional output from prediction
    if preds.ndim > 1:
        preds = preds[0]

    # Get indices and scores of the top 5 predictions
    top_5_indices = np.argsort(preds)[::-1][:5]
    top_5_crops = label_encoder.inverse_transform(top_5_indices)
    top_5_scores = preds[top_5_indices]

    # Format the results into a list of dictionaries
    results = []
    for crop, score in zip(top_5_crops, top_5_scores):
        results.append({"crop": crop, "score": float(score)})
        
    return results

# --- API Endpoint ---
@app.route('/predict', methods=['POST'])
def handle_prediction():
    """
    This function is triggered when the frontend sends a POST request to /predict.
    """
    if not model or not label_encoder:
        # Return an error if the model isn't loaded
        return jsonify({'error': 'Model or encoder not loaded properly.'}), 500

    try:
        # Get the JSON data sent from the React frontend
        data = request.get_json(force=True)

        # Convert the incoming JSON data into a pandas DataFrame
        # The model expects the columns in a specific order.
        input_data = pd.DataFrame({
            'N': [data['N']],
            'P': [data['P']],
            'K': [data['K']],
            'temperature': [data['temperature']],
            'humidity': [data['humidity']],
            'ph': [data['ph']],
            'rainfall': [data['rainfall']]
        })

        # Get the top 5 crop predictions
        top_crops = predict_top_crops(input_data, model, label_encoder)

        # Return the results as a JSON response
        return jsonify({'predictions': top_crops})

    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({'error': 'An error occurred during prediction.'}), 400

# --- Run the App ---
if __name__ == '__main__':
    # Use port 5000 for the backend server
    app.run(host='0.0.0.0', port=5000, debug=True)
