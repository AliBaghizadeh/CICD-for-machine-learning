import pandas as pd
import numpy as np
import skops.io as sio
from pathlib import Path

import pandas as pd
import numpy as np
import skops.io as sio
from pathlib import Path

# Scikit-learn and XGBoost imports
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Matplotlib for plotting
import matplotlib.pyplot as plt

# MLflow
import mlflow
import mlflow.sklearn

# -----------------------
# --- 1. SETUP AND DATA LOADING ---
# -----------------------
PROJECT_ROOT = Path.cwd()
DATA_PATH = PROJECT_ROOT / "Data" / "train-00000-of-00001.parquet"
target_col = "target"
lags = [1, 4, 96, 672]  # 15min, 1hr, 1 day, 1 week


def load_and_preprocess_data(path):
    """Loads the Parquet file, explodes list-columns, and preprocesses data."""
    try:
        df = pd.read_parquet(path)

        # 1. Define columns to explode (Must correspond to the time series lists)
        # CRITICAL FIX: Removed 'id' from this list because it is a static value, not a list.
        time_series_cols = [
            "timestamp",
            "solar_generation_actual",
            "wind_onshore_generation_actual",
            "target",
            "temperature",
            "radiation_direct_horizontal",
            "radiation_diffuse_horizontal",
        ]

        # 2. Force List Conversion and Flattening
        # We only apply this to the columns that are actually time series lists
        for col in time_series_cols:
            if col in df.columns:
                # Use np.ravel to flatten and ensure it's a simple list
                df[col] = df[col].apply(
                    lambda x: (
                        list(np.ravel(x)) if isinstance(x, (np.ndarray, list)) else x
                    )
                )

        # 3. Explode Operation
        # We only explode the time series columns.
        # Pandas will automatically repeat the static 'id' column for every row.
        df_exploded = df.explode(column=time_series_cols, ignore_index=True)

        # 4. Convert Types and Set Final Index
        # 'timestamp' is now a column of individual values
        df_exploded["timestamp"] = pd.to_datetime(df_exploded["timestamp"])

        for col in [
            "solar_generation_actual",
            "wind_onshore_generation_actual",
            "target",
            "temperature",
            "radiation_direct_horizontal",
            "radiation_diffuse_horizontal",
        ]:
            df_exploded[col] = pd.to_numeric(df_exploded[col], errors="coerce")

        # Set the final DatetimeIndex and sort
        df_final = df_exploded.set_index("timestamp").sort_index()

        # 5. Imputation
        df_with_id_index = df_final.set_index("id", append=True)
        df_imputed = df_with_id_index.groupby(level="id").ffill()
        df_imputed = df_imputed.reset_index(level="id")
        df_imputed = df_imputed.fillna(0)

        return df_imputed

    except Exception as e:
        print(f"Error loading or preprocessing data: {e}")
        return None


# --------------------------------
# --- 2. FEATURE ENGINEERING ---
# --------------------------------


def engineer_features(df_imputed):
    """Creates temporal, circular, and lagged features."""
    df_feat = df_imputed.copy()

    # Temporal Features
    df_feat["hour"] = df_feat.index.hour
    df_feat["quarter"] = df_feat.index.quarter
    df_feat["dayofweek"] = df_feat.index.dayofweek
    df_feat["dayofyear"] = df_feat.index.dayofyear
    df_feat["weekofyear"] = df_feat.index.isocalendar().week.astype(
        int
    )  # Ensure int type
    df_feat["month"] = df_feat.index.month
    df_feat["quarter_start"] = df_feat.index.is_quarter_start.astype(int)
    df_feat["is_weekend"] = (df_feat.index.dayofweek >= 5).astype(int)

    # Circular Encoding
    df_feat["hour_sin"] = np.sin(2 * np.pi * df_feat["hour"] / 24)
    df_feat["hour_cos"] = np.cos(2 * np.pi * df_feat["hour"] / 24)
    df_feat["dayofweek_sin"] = np.sin(2 * np.pi * df_feat["dayofweek"] / 7)
    df_feat["dayofweek_cos"] = np.cos(2 * np.pi * df_feat["dayofweek"] / 7)
    df_feat["month_sin"] = np.sin(
        2 * np.pi * df_feat["month"] / 12
    )  # Use 12 for month period
    df_feat["month_cos"] = np.cos(2 * np.pi * df_feat["month"] / 12)

    # Drop original integer features to avoid redundancy/confusion
    df_feat = df_feat.drop(columns=["hour", "dayofweek", "month"])

    # Lagged Features (Endogenous)
    print("--- Creating Lagged Features ---")
    for lag in lags:
        df_feat[f"{target_col}_lag_{lag}"] = df_feat.groupby("id")[target_col].shift(
            lag
        )

    # Encode 'id' and drop string 'id'
    df_feat["id_encoded"] = df_feat["id"].astype("category").cat.codes
    df_feat = df_feat.drop(columns=["id"])

    # Drop NaNs created by Lags
    df_energy = df_feat.dropna()

    return df_energy


# -----------------------------------------------
# --- 3. TRAIN/TEST SPLIT AND MODEL TRAINING ---
# -----------------------------------------------
if __name__ == "__main__":
    df_energy = engineer_features(load_and_preprocess_data(DATA_PATH))

    # Separate Target (y) and Features (X)
    X = df_energy.drop(columns=[target_col])
    y = df_energy[target_col]

    # Temporal Split (80% train, 20% test)
    split_point = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_point], X.iloc[split_point:]
    y_train, y_test = y.iloc[:split_point], y.iloc[split_point:]

    print(f"X_train size: {len(X_train)} rows")
    print(f"X_test size: {len(X_test)} rows")

    # -----------------------------------------------
    # --- 3. TRAIN/TEST SPLIT AND MODEL TRAINING ---
    # -----------------------------------------------

    # Set MLflow Experiment
    mlflow.set_experiment("Energy_Load_Forecasting")

    with mlflow.start_run():
        print("Starting MLflow run...")

        # Log Parameters
        params = {
            "n_estimators": 1000,
            "learning_rate": 0.1,
            "max_depth": 7,
            "objective": "reg:squarederror",
            "random_state": 42,
            "device": "cuda",  # Enable GPU support
        }
        mlflow.log_params(params)

        # Define and Train the Pipeline
        pipeline = Pipeline(
            [("scaler", StandardScaler()), ("model", XGBRegressor(**params, n_jobs=-1))]
        )

        print("\nTraining XGBoost Pipeline...")
        pipeline.fit(X_train, y_train)
        print("Training Complete.")

        # -----------------------
        # --- 4. EVALUATION ---
        # -----------------------
        predictions = pipeline.predict(X_test)

        # Calculate Regression Metrics
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)

        print("\n--- Model Performance ---")
        print("Mean absolute error :", round(mae, 2), "MW")
        print("R2 Score:", round(r2, 2))

        # Log Metrics
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)

        # Save Metrics to file
        (PROJECT_ROOT / "Results").mkdir(exist_ok=True)
        with open(PROJECT_ROOT / "Results/metrics.txt", "w") as outfile:
            outfile.write(
                f"Mean Absolute Error = {round(mae, 2)}, R2 Score = {round(r2, 2)}.\n"
            )

        # Save Plot to file
        plt.figure(figsize=(10, 6))
        plt.plot(
            y_test.index[:100], y_test.iloc[:100], label="Actual Load", color="blue"
        )
        plt.plot(
            y_test.index[:100],
            predictions[:100],
            label="XGBoost Forecast",
            color="red",
            linestyle="--",
        )
        plt.xlabel("Time")
        plt.ylabel("Load (MW)")
        plt.title("Forecast vs Actual (First 100 Test Steps)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plot_path = PROJECT_ROOT / "Results/model_results.png"
        plt.savefig(plot_path, dpi=120)
        print(f"Plot saved to {plot_path}")

        # Log Artifacts
        mlflow.log_artifact(plot_path)

        # -----------------------------------------------------
        # --- 5. SAVING THE MODEL (The Course's Next Step) ---
        # -----------------------------------------------------
        print("\n--- Saving Model using skops ---")
        MODEL_PATH = PROJECT_ROOT / "Model"
        MODEL_PATH.mkdir(exist_ok=True)  # Ensure the Model directory exists

        # Save the Pipeline object to the specified path
        sio.dump(pipeline, MODEL_PATH / "energy_forecast_pipeline.skops")

        print(
            f"Pipeline successfully saved to {MODEL_PATH / 'energy_forecast_pipeline.skops'}"
        )

        # Log Model to MLflow
        mlflow.sklearn.log_model(pipeline, "model")
        print("Model logged to MLflow.")
