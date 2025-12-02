"""
Forecast evaluation datasets
This repository contains time series datasets that can be used for evaluation of univariate & multivariate forecasting models.

The main focus of this repository is on datasets that reflect real-world forecasting scenarios, such as those involving covariates, missing values, and other practical complexities.

The datasets follow a format that is compatible with the fev package.
"""

import requests

url = "https://huggingface.co/datasets/autogluon/fev_datasets/resolve/main/entsoe/15T/train-00000-of-00001.parquet"
output_path = "Data/train-00000-of-00001.parquet"

print(f"Downloading data from {url}...")
r = requests.get(url, allow_redirects=True)
with open(output_path, "wb") as f:
    f.write(r.content)
print(f"Download complete. File saved to {output_path}")


import pandas as pd
import numpy as np

# Define the file path
DATA_PATH = "Data/train-00000-of-00001.parquet"


# --- Loading the Dataset ---
def load_and_preprocess_data(path):
    """Loads the Parquet file and performs initial time series setup."""
    try:
        df = pd.read_parquet(path)
    except Exception as e:
        print(f"Error reading Parquet file: {e}")
        return None

    # Assuming the first column is the datetime index and the second is the target ('load' or similar)
    # The fev_datasets are generally well-structured, but we need to ensure the time index is correct.

    # 1. Convert the index to datetime if it isn't already (common step for time series)
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    # 2. Rename the target column for consistency (assuming the load data column is unnamed or generic)
    # You will need to check the actual column names in your notebook. For now, we'll assume a 'target' column.
    # Note: For ENTSO-e, the target is typically the 'load' column.

    # For initial experimentation, let's keep only the target column and ensure correct frequency
    # We will skip shuffling for time series data, as it destroys the temporal dependency!

    # Display the top 3 rows (as requested by the course instruction)
    print("\n--- Loaded Data (Top 3 Rows) ---")
    print(df.head())
    print("---------------------------------")

    return df


if __name__ == "__main__":
    energy_df = load_and_preprocess_data(DATA_PATH)
    if energy_df is not None:
        print(f"Data loaded successfully. Total rows: {len(energy_df)}")
        # Next steps will go here: feature engineering, pipeline, training, etc.
