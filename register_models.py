"""
Register optimized models in MLflow Model Registry
This script loads the trained models and registers them in MLflow for version control
"""

import mlflow
import mlflow.sklearn
import mlflow.xgboost
import mlflow.lightgbm
import mlflow.catboost
import joblib
from pathlib import Path
import json

# Set MLflow tracking URI
mlflow.set_tracking_uri("file:./mlruns")

# Model directory
MODEL_DIR = Path("Model")

# Load models and scaler
print("Loading models...")
xgb_model = joblib.load(MODEL_DIR / "xgboost_model.pkl")
lgb_model = joblib.load(MODEL_DIR / "lightgbm_model.pkl")
cat_model = joblib.load(MODEL_DIR / "catboost_model.pkl")
scaler = joblib.load(MODEL_DIR / "scaler.pkl")

# Load best parameters and parse MAE values
mae_values = {}
with open(MODEL_DIR / "best_params.txt", "r") as f:
    content = f.read()

    # Parse each model section
    sections = content.split(
        "------------------------------------------------------------"
    )
    for section in sections:
        if "XGBOOST:" in section:
            for line in section.split("\n"):
                if "MAE:" in line:
                    mae_values["xgboost"] = float(line.split("MAE:")[1].strip())
        elif "LIGHTGBM:" in section:
            for line in section.split("\n"):
                if "MAE:" in line:
                    mae_values["lightgbm"] = float(line.split("MAE:")[1].strip())
        elif "CATBOOST:" in section:
            for line in section.split("\n"):
                if "MAE:" in line:
                    mae_values["catboost"] = float(line.split("MAE:")[1].strip())

print(f"MAE values: {mae_values}")

# ============================================================================
# Register XGBoost Model
# ============================================================================
print("\n" + "=" * 60)
print("Registering XGBoost model...")
print("=" * 60)

mlflow.set_experiment("XGBoost_Production")
with mlflow.start_run(run_name="xgboost_optimized") as run:
    # Log metrics
    mlflow.log_metric("mae", mae_values.get("xgboost", 0))
    mlflow.log_metric("mae_cv", mae_values.get("xgboost", 0))

    # Log model
    mlflow.xgboost.log_model(
        xgb_model, artifact_path="model", registered_model_name="energy-xgboost"
    )

    # Log scaler as artifact
    mlflow.log_artifact(str(MODEL_DIR / "scaler.pkl"))

    print(f"✅ XGBoost registered with MAE: {mae_values.get('xgboost', 0):.2f}")

# ============================================================================
# Register LightGBM Model
# ============================================================================
print("\n" + "=" * 60)
print("Registering LightGBM model...")
print("=" * 60)

mlflow.set_experiment("LightGBM_Production")
with mlflow.start_run(run_name="lightgbm_optimized") as run:
    # Log metrics
    mlflow.log_metric("mae", mae_values.get("lightgbm", 0))
    mlflow.log_metric("mae_cv", mae_values.get("lightgbm", 0))

    # Log model
    mlflow.lightgbm.log_model(
        lgb_model, artifact_path="model", registered_model_name="energy-lightgbm"
    )

    # Log scaler as artifact
    mlflow.log_artifact(str(MODEL_DIR / "scaler.pkl"))

    print(f"✅ LightGBM registered with MAE: {mae_values.get('lightgbm', 0):.2f}")

# ============================================================================
# Register CatBoost Model
# ============================================================================
print("\n" + "=" * 60)
print("Registering CatBoost model...")
print("=" * 60)

mlflow.set_experiment("CatBoost_Production")
with mlflow.start_run(run_name="catboost_optimized") as run:
    # Log metrics
    mlflow.log_metric("mae", mae_values.get("catboost", 0))
    mlflow.log_metric("mae_cv", mae_values.get("catboost", 0))

    # Log model
    mlflow.catboost.log_model(
        cat_model, artifact_path="model", registered_model_name="energy-catboost"
    )

    # Log scaler as artifact
    mlflow.log_artifact(str(MODEL_DIR / "scaler.pkl"))

    print(f"✅ CatBoost registered with MAE: {mae_values.get('catboost', 0):.2f}")

# ============================================================================
# Set Production Aliases (for best model)
# ============================================================================
print("\n" + "=" * 60)
print("Setting production aliases...")
print("=" * 60)

from mlflow import MlflowClient

client = MlflowClient()

# Get latest versions
for model_name in ["energy-xgboost", "energy-lightgbm", "energy-catboost"]:
    try:
        latest_versions = client.get_latest_versions(model_name, stages=["None"])
        if latest_versions:
            version = latest_versions[0].version
            # Set alias for production
            client.set_registered_model_alias(model_name, "production", version)
            print(f"✅ Set 'production' alias for {model_name} version {version}")
    except Exception as e:
        print(f"⚠️  Could not set alias for {model_name}: {e}")

print("\n" + "=" * 60)
print("✅ All models registered successfully!")
print("=" * 60)
print("\nTo view registered models:")
print("  mlflow ui")
print("  Then navigate to 'Models' tab")
print("\nRegistered models:")
print("  - energy-xgboost")
print("  - energy-lightgbm")
print("  - energy-catboost")
