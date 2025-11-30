import gradio as gr
import joblib
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

# Model paths
MODEL_DIR = Path("./Model")
MODELS = {
    "XGBoost": MODEL_DIR / "xgboost_model.pkl",
    "LightGBM": MODEL_DIR / "lightgbm_model.pkl",
    "CatBoost": MODEL_DIR / "catboost_model.pkl"
}
SCALER_PATH = MODEL_DIR / "scaler.pkl"

# Model performance (from HPO)
MODEL_PERFORMANCE = {
    "XGBoost": 92.81,
    "LightGBM": 93.42,
    "CatBoost": 124.74
}

# Load models and scaler
print("Loading models...")
loaded_models = {}
for name, path in MODELS.items():
    try:
        loaded_models[name] = joblib.load(path)
        print(f"✅ {name} loaded")
    except Exception as e:
        print(f"❌ Failed to load {name}: {e}")
        loaded_models[name] = None

try:
    scaler = joblib.load(SCALER_PATH)
    print("✅ Scaler loaded")
except Exception as e:
    print(f"❌ Failed to load scaler: {e}")
    scaler = None

# Feature engineering function
def engineer_time_features():
    """Calculates time features for the next hour (t+1)."""
    target_time = datetime.now() + timedelta(hours=1)
    
    hour = target_time.hour
    day_of_week = target_time.weekday()
    month = target_time.month
    is_weekend = 1 if day_of_week >= 5 else 0
    
    return [hour, day_of_week, month, is_weekend]

# Prediction function
def predict_all_models(last_load: float, current_temp: float, country_id: str):
    """
    Predicts energy load using all 3 models and returns comparison.
    """
    if scaler is None:
        return "❌ Error: Scaler not loaded", "❌ Error: Scaler not loaded", "❌ Error: Scaler not loaded"
    
    # Encode Country ID
    id_map = {"AT": 0, "DE": 1, "FR": 2, "IT": 3, "BE": 4, "CH": 5, "NL": 6, "PL": 7, "CZ": 8, "ES": 9}
    id_encoded = id_map.get(country_id, 0)
    
    # Calculate time features
    time_features = engineer_time_features()
    
    # Create feature array (simplified - adjust based on your actual features)
    # This is a placeholder - you'll need to match your actual feature engineering
    num_features = 45  # Adjust based on your actual feature count
    input_features = np.zeros((1, num_features))
    
    input_features[0, 0] = last_load
    input_features[0, 1] = current_temp
    input_features[0, 2] = id_encoded
    input_features[0, 3:3 + len(time_features)] = time_features
    
    # Scale features
    try:
        input_scaled = scaler.transform(input_features)
    except Exception as e:
        return f"❌ Scaling error: {e}", f"❌ Scaling error: {e}", f"❌ Scaling error: {e}"
    
    # Get predictions from all models
    results = {}
    for name, model in loaded_models.items():
        if model is None:
            results[name] = "❌ Model not loaded"
        else:
            try:
                prediction = model.predict(input_scaled)[0]
                mae = MODEL_PERFORMANCE[name]
                results[name] = f"**{prediction:,.2f} MW**\n\n📊 Model MAE: {mae:.2f} MW"
            except Exception as e:
                results[name] = f"❌ Prediction error: {e}"
    
    return results["XGBoost"], results["LightGBM"], results["CatBoost"]

# Gradio Interface
with gr.Blocks(title="⚡ Energy Load Forecast - Multi-Model Comparison") as demo:
    gr.Markdown(
        """
        # ⚡ Energy Load Forecast: Multi-Model Comparison
        
        This application forecasts short-term energy consumption (MW) for the **next hour (t+1)** using **3 optimized models**:
        - **XGBoost** (MAE: 92.81 MW) - Best performer
        - **LightGBM** (MAE: 93.42 MW)
        - **CatBoost** (MAE: 124.74 MW)
        
        All models were optimized using Optuna with 50 trials each on GPU (RTX 5080).
        """
    )
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### Input Parameters")
            last_load = gr.Slider(0, 10000, step=1, label="Last Known Load (MW)", value=5000)
            current_temp = gr.Slider(-20, 40, step=0.1, label="Current Temperature (°C)", value=15.0)
            country_id = gr.Radio(
                ["AT", "DE", "FR", "IT", "BE", "CH", "NL", "PL", "CZ", "ES"], 
                label="Country ID", 
                value="DE"
            )
            predict_btn = gr.Button("🔮 Predict with All Models", variant="primary")
    
    gr.Markdown("### 📊 Model Predictions Comparison")
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("#### 🥇 XGBoost (Best)")
            xgb_output = gr.Markdown()
        
        with gr.Column():
            gr.Markdown("#### 🥈 LightGBM")
            lgb_output = gr.Markdown()
        
        with gr.Column():
            gr.Markdown("#### 🥉 CatBoost")
            cat_output = gr.Markdown()
    
    # Examples
    gr.Examples(
        examples=[
            [5000, 15.0, "DE"],
            [3000, 5.5, "FR"],
            [7000, 30.0, "AT"],
        ],
        inputs=[last_load, current_temp, country_id],
    )
    
    gr.Markdown(
        """
        ---
        ### 🚀 Model Details
        
        **Feature Engineering:** The models use advanced time-series features including:
        - Cyclical encoding (hour, day of week, month)
        - Lagged features (15min, 1hr, 1day, 1week)
        - Weather data (temperature, radiation)
        - Holiday indicators
        
        **Deployment:** This application is deployed automatically via **GitHub Actions CI/CD** 
        to Hugging Face Spaces. Models are tracked and versioned using **MLflow Model Registry**.
        
        **Training:** All models were trained on 800K+ rows of European energy data with 
        GPU acceleration and hyperparameter optimization.
        """
    )
    
    # Connect button to prediction function
    predict_btn.click(
        fn=predict_all_models,
        inputs=[last_load, current_temp, country_id],
        outputs=[xgb_output, lgb_output, cat_output]
    )

# Launch
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)