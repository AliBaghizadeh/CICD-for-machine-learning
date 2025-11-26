import gradio as gr
import skops.io as sio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Define the model path relative to the root of the application (Hugging Face space)
MODEL_PATH = "./Model/energy_forecast_pipeline.skops"

# Define the trusted types for skops v0.10+ security update (includes final fixes)
TRUSTED_TYPES = [
    'numpy.ndarray', 
    'sklearn.preprocessing._data.StandardScaler', 
    'sklearn.pipeline.Pipeline', 
    'xgboost.sklearn.XGBRegressor',
    'xgboost.core.Booster', # CRITICAL FIX
    'pandas._libs.tslibs.timestamps.Timestamp', 
    'sklearn.impute._base.SimpleImputer' 
]

# Function to engineer time features for a target time
def engineer_time_features():
    """Calculates time features for the next hour (t+1)."""
    
    # 1. Determine the target time (t + 1 hour)
    target_time = datetime.now() + timedelta(hours=1)
    
    # 2. Extract standard time features (ensure these match your train.py logic!)
    hour = target_time.hour
    day_of_week = target_time.dayofweek  # Monday=0, Sunday=6
    month = target_time.month
    
    # Example: If your model used an 'Is_Weekend' feature
    is_weekend = 1 if day_of_week >= 5 else 0 
    
    # Return a list of all engineered time features
    # NOTE: The order and number must match the sequence used in train.py!
    return [
        hour, 
        day_of_week, 
        month,
        is_weekend,
        # ... add all other time/calendar features here ...
    ]

# --- 1. Load the Model Pipeline ---
try:
    # Use the explicit list of trusted types
    pipeline = sio.load(MODEL_PATH, trusted=TRUSTED_TYPES) 
    FEATURE_COUNT = pipeline['model'].n_features_in_
except Exception as e:
    print(f"Error loading model: {e}")
    print("NOTE: Model file missing locally. Proceeding with default settings.")
    pipeline = None
    FEATURE_COUNT = 45 # Default count based on standard feature engineering

# --- 2. Prediction Function (Simplified for Demo) ---
def predict_load_simplified(last_load: float, current_temp: float, country_id: str):
    """
    Predicts energy load based on a simplified set of dynamic features.
    """
    if pipeline is None:
        return "ERROR: Model Pipeline failed to load. Please ensure the model file exists."

    # Encode Country ID (must match the encoding used in train.py)
    id_map = {"AT": 0, "DE": 1, "FR": 2, "IT": 3, "BE": 4, "CH": 5, "NL": 6, "PL": 7, "CZ": 8, "ES": 9}
    id_encoded = id_map.get(country_id, 0)

    # Calculate future time features
    time_features = engineer_time_features()
    num_time_features = len(time_features)
    
    # Create the feature array with the expected number of features (e.g., 45)
    input_features = np.zeros((1, FEATURE_COUNT))
    
    # Place the user inputs into the array at arbitrary (but consistent) locations
    input_features[0, 0] = last_load     
    input_features[0, 1] = current_temp 
    input_features[0, 2] = id_encoded    

    # Inject the calculated time features into the array
    # (Assuming they start at index 3)
    input_features[0, 3:3 + num_time_features] = time_features

    try:
        predicted_load = pipeline.predict(input_features)[0]
    except Exception as e:
        return f"Prediction Error: {e}. Check feature count ({FEATURE_COUNT})"

    label = f"Predicted Energy Load: {predicted_load:,.2f} MW"
    return label

# --- 3. Gradio Interface Setup ---
inputs = [
    gr.Slider(0, 10000, step=1, label="Last Known Load (MW)", value=5000),
    gr.Slider(-20, 40, step=0.1, label="Current Temperature (°C)", value=15.0),
    gr.Radio(["AT", "DE", "FR", "IT", "BE", "CH", "NL", "PL", "CZ", "ES"], label="Country ID", value="DE"),
]

outputs = [gr.Label(label="Energy Load Forecast")]

examples = [
    [5000, 15.0, "DE"],
    [3000, 5.5, "FR"],
    [7000, 30.0, "AT"],
]

title = "⚡ Energy Load Forecast: Production Ready CI/CD Demo"

description = (
    "This application forecasts short-term energy consumption (MW) for the **next hour (t+1)**. "
    "The forecast is based on: Last Known Load, Expected Temperature, and Country context."
    )
article = (
    "### Model Details and Prediction Logic\n"
    "The core of this application is a **XGBoost Regressor** trained on historical time-series data.\n\n"
    "**Feature Engineering:** To ensure an accurate $t+1$ forecast, the model dynamically calculates "
    "essential features such as the **Hour of Day, Day of Week, and Month** based on the current time + 1 hour. "
    "The user only provides the dynamic inputs (Load, Temp, Country); the rest is automated by the pipeline.\n\n"
    "**Deployment:** This entire application, including the XGBoost model and results, is deployed "
    "automatically via **GitHub Actions** and **Continuous Deployment (CD)** to Hugging Face Spaces."
    )

gr.Interface(
    fn=predict_load_simplified,
    inputs=inputs,
    outputs=outputs,
    examples=examples,
    title=title,
    description=description,
    article=article,
).launch(server_name="0.0.0.0", server_port=7860)