import streamlit as st
import numpy as np
import tensorflow as tf
import joblib
from sklearn.preprocessing import StandardScaler

# Load the trained model
model = tf.keras.models.load_model("C:\\Users\\pc\\Downloads\\crop_model.h5")
crop_labels = [
    "apple", "banana", "Black-gram", "chickpea", "coconut", "coffee",
    "cotton", "grapes", "jute", "kidney-beans", "lentil", "maize",
    "mango", "moth-beans", "mung-bean", "muskmelon", "orange",
    "papaya", "pigeon-peas", "pomegranate", "rice", "watermelon"
]
scaler = joblib.load('C:\\Users\\pc\\Downloads\\scaler.pkl')

# Function to predict crop
def predict_crop(n, k, p, temperature, humidity, ph, rainfall):
    inputs = np.array([n, k, p, temperature, humidity, ph, rainfall]).reshape(1, -1)
    inputs_scaled = scaler.transform(inputs)
    prediction = model.predict(inputs_scaled)
    predicted_crop_index = np.argmax(prediction, axis=1)[0]
    return crop_labels[predicted_crop_index]

# Page title
st.title("Crop Prediction")
st.write("Enter the environmental factors below to get the recommended crop.")

# Input fields for prediction
col1, col2 = st.columns(2)
with col1:
    n = st.number_input("N (Nitrogen)", min_value=0.0, value=0.0)
    k = st.number_input("K (Potassium)", min_value=0.0, value=0.0)
    p = st.number_input("P (Phosphorus)", min_value=0.0, value=0.0)
    temperature = st.number_input("Temperature (°C)", min_value=0.0, value=0.0)
with col2:
    humidity = st.number_input("Humidity (%)", min_value=0.0, value=0.0)
    ph = st.number_input("pH", min_value=0.0, value=0.0)
    rainfall = st.number_input("Rainfall (mm)", min_value=0.0, value=0.0)

# Prediction button
if st.button("Predict Crop"):
    prediction = predict_crop(n, k, p, temperature, humidity, ph, rainfall)
    st.success(f"**Predicted Crop:** {prediction}")

# Displaying additional information or suggestions (optional)
st.write("Based on the prediction, you may consider checking specific requirements for the recommended crop.")
