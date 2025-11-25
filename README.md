# ⚡ Energy Load Forecasting Pipeline (CICD MLOps Project)

## 🎯 Overview

This project implements a robust **Continuous Integration/Continuous Delivery (CI/CD)** pipeline for a Machine Learning model that forecasts energy load consumption. The goal is to automate the entire process from data processing and model training to evaluation and deployment, ensuring high confidence in model quality with every code change.

The core of the system is an **XGBoost Regressor** trained on a large-scale time series dataset of aggregated energy demand and generation metrics.

## 🚀 Key Features

* **Automated Training & Evaluation:** A GitHub Actions workflow automatically trains the model, evaluates its performance (MAE, R²), and generates visual reports on every push to the main branch.
* **Time Series Feature Engineering:** Includes sophisticated steps for handling high-frequency time series data, such as:
    * Data explosion and imputation.
    * Creation of temporal features (hour, day of week, month).
    * Circular encoding for cyclical features.
    * Lagged features for capturing temporal dependencies.
* **Model Packaging:** The final trained scikit-learn pipeline (including the `StandardScaler` and `XGBRegressor`) is saved using the `skops` library for easy, secure, and future-proof deployment.
* **Reporting:** Uses **Continuous Machine Learning (CML)** to post model performance metrics and a forecast vs. actual plot directly to the GitHub pull request/commit status for easy review.

## ⚙️ Project Structure

The repository is structured to separate code, data, and artifacts clearly:

´´´
├── .github/

│   └── workflows/

│       └── ci.yml             # GitHub Actions workflow for automated pipeline runs.

├── Data/

│   └── train-00000-of-00001.parquet # Input energy data file (Parquet format).

├── Model/

│   └── energy_forecast_pipeline.skops # Saved trained model pipeline (using skops).

├── Results/

│   ├── metrics.txt            # Final performance metrics (MAE, R2).

│   └── model_results.png      # Visualization of Forecast vs. Actual load.

├── Makefile                   # Automation targets for local and CI/CD execution.

├── requirements.txt           # Python dependencies (incl. pandas, xgboost, dvc, skops).

└── train.py                   # Main script for data preprocessing, feature engineering, training, and saving.
´´´


## 💻 Getting Started

### Prerequisites

You need **Python 3.9+** and `pip` installed.

### Installation

Clone the repository and install the dependencies:

```bash
git clone <your-repo-url>
cd <project-name>
pip install -r requirements.txt