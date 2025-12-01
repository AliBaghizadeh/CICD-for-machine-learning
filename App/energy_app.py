import gradio as gr
import joblib
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import os
import csv

# Model paths
MODEL_DIR = Path("./Model")
MODELS = {
    "XGBoost": MODEL_DIR / "xgboost_model.pkl",
    "LightGBM": MODEL_DIR / "lightgbm_model.pkl",
    "CatBoost": MODEL_DIR / "catboost_model.pkl"
}
SCALER_PATH = MODEL_DIR / "scaler.pkl"

# Logging setup
LOG_FILE = Path("logs/inference_log.csv")
LOG_FILE.parent.mkdir(exist_ok=True)

if not LOG_FILE.exists():
    with open(LOG_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "last_load", "current_temp", "country_id", "pred_xgboost", "pred_lightgbm", "pred_catboost"])

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
    try:
        df = pd.read_csv(LOG_FILE)
        return df.tail(20).iloc[::-1]  # Show last 20, newest first
    except:
        return pd.DataFrame()

def plot_load_dist():
    df = get_recent_logs()
    if df.empty:
        return None
    return gr.BarPlot(
        df,
        x="last_load",
        y="pred_xgboost",
        title="Input Load vs Predicted Load (XGBoost)",
        tooltip=["last_load", "pred_xgboost", "timestamp"],
        width=400,
        height=300
    )

def plot_temp_dist():
    df = get_recent_logs()
    if df.empty:
        return None
    return gr.LinePlot(
        df,
        x="timestamp",
        y="current_temp",
        title="Temperature Trend",
        tooltip=["timestamp", "current_temp"],
        width=400,
        height=300
    )

# Gradio Interface
with gr.Blocks(title="⚡ Energy Load Forecast - Multi-Model Comparison") as demo:
    gr.Markdown(
        """
        # ⚡ Energy Load Forecast: Multi-Model Comparison
        
        This application forecasts short-term energy consumption (MW) for the **next hour (t+1)** using **3 optimized models**.
        """
    )
    
    with gr.Tabs():
        # Tab 1: Prediction Interface
        with gr.TabItem("🔮 Forecast"):
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

        # Tab 2: Monitoring Dashboard
        with gr.TabItem("📊 Monitoring"):
            gr.Markdown("### 📈 Live Model Monitoring")
            refresh_btn = gr.Button("🔄 Refresh Data")
            
            with gr.Row():
                load_plot = gr.BarPlot(label="Load Distribution")
                temp_plot = gr.LinePlot(label="Temperature Trend")
            
            gr.Markdown("### 📝 Recent Inference Logs")
            log_table = gr.DataFrame(headers=["timestamp", "last_load", "current_temp", "country_id", "pred_xgboost"])
            
            # Refresh logic
            refresh_btn.click(plot_load_dist, outputs=load_plot)
            refresh_btn.click(plot_temp_dist, outputs=temp_plot)
            refresh_btn.click(get_recent_logs, outputs=log_table)
            
            # Auto-load on tab select (simulated by button click for now)
            demo.load(plot_load_dist, outputs=load_plot)
            demo.load(plot_temp_dist, outputs=temp_plot)
            demo.load(get_recent_logs, outputs=log_table)

    
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