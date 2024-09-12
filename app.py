import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import xgboost as xgb
import numpy as np
import joblib

# Load the trained XGBoost model and LabelEncoder
model = xgb.Booster()
model.load_model(r"F:\MLPROJECTS\SIH\xgboost_model.json")  # Load the model
label_encoder = joblib.load(r"F:\MLPROJECTS\SIH\label_encoder.pkl")  # Load label encoder

# Streamlit app title and input form
st.title("Crop Recommendation System")
st.write("Provide the following inputs to get crop recommendations")

# Create input fields for the crop features with default values and buttons for adjustment
def increment_value(value):
    return value + 1

def decrement_value(value):
    return max(value - 1, 0)

col1, col2 = st.columns(2)

with col1:
    nitrogen = st.number_input('Nitrogen', min_value=0, max_value=100, value=0, step=1)
    phosphorus = st.number_input('Phosphorus', min_value=0, max_value=100, value=0, step=1)
    potassium = st.number_input('Potassium', min_value=0, max_value=100, value=0, step=1)
    temperature = st.number_input('Temperature (C)', min_value=0, max_value=50, value=0, step=1)
    humidity = st.number_input('Humidity (%)', min_value=0, max_value=100, value=0, step=1)
    ph = st.number_input('pH', min_value=0.0, max_value=14.0, value=0.0, step=0.1)
    rainfall = st.number_input('Rainfall (mm)', min_value=0, max_value=500, value=0, step=1)

# Prepare input data as a pandas DataFrame
input_data = pd.DataFrame({
    'N': [nitrogen],
    'P': [phosphorus],
    'K': [potassium],
    'temperature': [temperature],
    'humidity': [humidity],
    'ph': [ph],
    'rainfall': [rainfall]
})

# Prediction function to get the top 5 crops
def predict_top_crops(input_data, model, label_encoder):
    dmatrix = xgb.DMatrix(input_data)  # Convert input data to DMatrix for XGBoost
    preds = model.predict(dmatrix)  # Predict using the model
    
    # If preds has more than one dimension, handle it properly
    if preds.ndim > 1:
        preds = preds[0]
    
    # Get indices of top 5 predictions (sorted by highest probabilities)
    top_5_indices = np.argsort(preds)[::-1][:5]  # Get top 5 predictions
    top_5_crops = label_encoder.inverse_transform(top_5_indices)  # Decode crop names
    top_5_scores = preds[top_5_indices]  # Get prediction scores for the top 5
    
    return top_5_crops, top_5_scores

# When the user clicks the Predict button
if st.button('Predict Best Crops'):
    top_crops, top_scores = predict_top_crops(input_data, model, label_encoder)
    st.write("Top 5 Best-Suited Crops: ", top_crops)

    # Plot the top 5 crops and their prediction scores using Plotly
    st.write("Crop Prediction Results")
    fig = go.Figure(data=[go.Bar(x=top_crops, y=top_scores)])
    fig.update_layout(
        title="Top 5 Crop Recommendations",
        xaxis_title="Crops",
        yaxis_title="Prediction Score"
    )
    st.plotly_chart(fig)
