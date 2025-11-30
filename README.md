# ⚡ End-to-End Energy Load Forecasting Pipeline (Professional MLOps)

## 🎯 Overview

This project implements a professional, end-to-end MLOps pipeline for energy load forecasting using modern tools and practices. The system provides highly accurate hourly energy demand predictions by leveraging advanced feature engineering, powerful XGBoost models, and robust MLflow-based lifecycle management.

The core solution focuses on solving a challenging time-series regression problem. The pipeline is designed for high reliability, using a dedicated feature engineering layer, automated Hyperparameter Optimization (HPO), and strict model versioning to ensure that only verified models are promoted to the deployment environment.

The workflow follows a standard MLOps pattern: code changes trigger the CI pipeline to run training and evaluation, and upon success, the CD pipeline handles the deployment of the validated model.

## 🧠 Model and Feature Engineering

The primary model used is an **XGBoost Regressor**, chosen for its speed, performance, and robustness with structured data and complex feature interactions. Model training is handled by `train.py`, which is responsible for invoking the HPO process to find the optimal combination of parameters (e.g., learning rate, max depth). All HPO trials and the final model configuration are meticulously logged to MLflow.

Crucial to the model's performance is the quality of the input data. This is managed by the external Python package, `src/features.py`. This module encapsulates all logic for creating sophisticated time-series features, including cyclical encoding (sin/cos transformations of the hour and day-of-year), holiday flags, and the integration and alignment of external weather data (e.g., temperature and humidity). Separating this logic ensures that the features used during model training are identical to those used during live inference in the deployed application.

The project uses two dedicated files for data management and exploration. The `Data/data_import.py` script is responsible for the critical first step: reliably downloading and setting up the initial Parquet file (`Data/train-00000-of-00001.parquet`) from an external source, ensuring data provenance. Furthermore, the `Results/notebook.ipynb` file contains the initial exploratory data analysis (EDA) and prototyping work, serving as a transparent record of the feature engineering decisions and data understanding that informed the final `train.py` script.

## ⚙️ CI/CD and Model Management

The CI/CD pipeline, orchestrated by GitHub Actions workflows (`.github/workflows/`), provides full automation and quality gates. The `ci.yml` workflow first executes the training and evaluation steps, including running a Performance Baseline (a simple statistical model) to ensure the advanced XGBoost model provides a significant performance lift. All metrics and artifacts from this run are logged to MLflow Experiment Tracking. The evaluation step relies heavily on CML (Continuous Machine Learning) to post performance reports (from `Results/metrics.txt` and `model_results.png`) directly to GitHub pull requests, enabling governance-approved model promotion.

Upon successful completion of the CI job, the deployment workflow (`cd.yml`) triggers. This job authenticates with MLflow Model Registry, retrieves the latest, approved model version, and deploys the entire application stack—including the Gradio app code (`App/energy_app.py`) and required dependencies—to the Hugging Face Space. This practice ensures that the deployed service always uses the highest-quality, governance-approved model artifact.

## ✨ Core MLOps & ML Features

| Category | Feature | Description |
|----------|---------|-------------|
| **Model Lifecycle** | MLflow Integration | All runs, metrics (MAE, R²), and artifacts are logged and versioned in MLflow Tracking and Model Registry. |
| **Data & Features** | Advanced Feature Engineering | Features are generated via a versioned Python package (`src/`) and include: external weather data, cyclical time encoding, and holiday indicators. |
| **Monitoring** | Data Drift Detection | The deployed application logs inputs/outputs for external monitoring services, preparing the app for Continuous Model Monitoring. |
| **Quality Control** | Performance Baselines | CI automatically runs and logs a simple statistical baseline model for comparison. |
| **Model Packaging** | Skops Serialization | The trained pipeline is saved using skops for secure and robust deployment. |

## 🗂️ Automation and Makefile Targets

The entire ML lifecycle is streamlined using the Makefile. This ensures consistency between local development and the automated CI/CD environment. By executing simple, descriptive targets, developers can manage the pipeline without needing to know specific Python commands.

| Target | Purpose | Description |
|--------|---------|-------------|
| `make install` | Environment Setup | Installs all required dependencies from `requirements.txt`. |
| `make train` | Training & Artifact Generation | Executes `train.py`, which handles data prep, feature engineering, model training, and saves the final model (`.skops`), metrics, and plot. |
| `make eval` | Quality Reporting (CML) | Generates the `report.md` from the `Results/` artifacts and uses CML to post the performance report as a comment in the GitHub PR. |
| `make deploy` | Continuous Deployment | Runs the necessary Hugging Face CLI login and then executes the push command to deploy the model and Gradio app to the target Space. |
| `make format` | Code Quality | Applies the black code formatter to enforce consistent style across the project. |

## 📦 Project Structure

The repository is structured for clear separation of ML stages:
```
├── .github/
│   └── workflows/
│       ├── ci.yml              # CI: Training, Evaluation, Baseline, MLflow Logging.
│       └── cd.yml              # CD: Fetches model from MLflow Registry, deploys to Hugging Face Space.
├── src/                        # Placeholder for reusable ML logic
│   └── features.py             # Feature engineering logic (time/weather features).
├── App/
│   └── energy_app.py           # Gradio application code for deployment.
├── Model/
│   └── energy_forecast_pipeline.skops # Saved trained scikit-learn pipeline (via skops).
├── Data/
│   ├── data_import.py          # Script to download and verify the raw Parquet data.
│   └── train-00000-of-00001.parquet # Raw input energy data file.
├── Results/
│   ├── metrics.txt             # Final performance metrics (MAE, R2) for CML report.
│   └── notebook.ipynb          # Jupyter Notebook for exploratory data analysis (EDA).
├── train.py                    # Main script: Data Prep, Training, Evaluation, and Model Saving.
├── Makefile                    # Automation targets for local and CI/CD execution.
└── requirements.txt            # Python dependencies (incl. pandas, xgboost, skops, mlflow).
```

## 💻 Getting Started

### Installation

Clone the repository and install the dependencies:
```bash
git clone <your-repo-url>
cd <project-name>
pip install -r requirements.txt
```

### Local Execution

To run the full training pipeline and launch the MLflow UI locally:
```bash
# Run training, logging results to MLflow
python train.py

# Launch the MLflow UI to inspect runs and register the best model
mlflow ui
```