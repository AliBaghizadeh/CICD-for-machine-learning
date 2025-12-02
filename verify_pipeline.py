import os
import warnings

os.environ["BOOST_COMPUTE_USE_OFFLINE_CACHE"] = "0"
warnings.filterwarnings("ignore")

import mlflow
import mlflow.sklearn
import requests
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from train import load_and_preprocess_data, engineer_features, DATA_PATH, target_col

# Apply Host header patch
original_request = requests.Session.request


def patched_request(self, method, url, **kwargs):
    if "headers" not in kwargs:
        kwargs["headers"] = {}
    kwargs["headers"]["Host"] = "127.0.0.1:5001"
    return original_request(self, method, url, **kwargs)


requests.Session.request = patched_request

# Set MLflow tracking
MLFLOW_TRACKING_URI = "http://18.153.53.234:5000"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

print(f"🔗 MLflow Tracking URI: {mlflow.get_tracking_uri()}")
print("📊 Loading data...")

# Load data
df = engineer_features(load_and_preprocess_data(DATA_PATH))
X = df.drop(columns=[target_col])
y = df[target_col]

# Quick train/test split (no CV for speed)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("🚀 Training quick XGBoost model...")
model = XGBRegressor(n_estimators=50, max_depth=5, learning_rate=0.1, random_state=42)
model.fit(X_train, y_train)

preds = model.predict(X_test)
mae = mean_absolute_error(y_test, preds)
print(f"✅ Model trained! MAE: {mae:.2f}")

# Log to MLflow
print(f"\n📤 Logging to MLflow at: {mlflow.get_tracking_uri()}")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)  # Force remote

try:
    mlflow.set_experiment("Pipeline_Verification")
    with mlflow.start_run(run_name="Quick_Test_XGBoost"):
        mlflow.log_param("n_estimators", 50)
        mlflow.log_param("max_depth", 5)
        mlflow.log_metric("mae", mae)
        mlflow.sklearn.log_model(model, "model")
        print("✅ Successfully logged model to AWS MLflow!")
        print(f"🔗 Check: http://18.153.53.234:5000/#/experiments")
except Exception as e:
    print(f"❌ Failed to log: {e}")
    import traceback

    traceback.print_exc()

print("\n✅ Verification complete!")
print("If you see '✅ Successfully logged model to AWS MLflow!' above,")
print("then train_hpo.py will work correctly too.")
