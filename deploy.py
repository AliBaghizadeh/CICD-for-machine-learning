import os
from huggingface_hub import HfApi

# 1. Setup
# Get the token and your specific Space ID from environment variables
TOKEN = os.environ.get("HF_TOKEN")
REPO_ID = "alibaghizade/time_series_energy" 

if not TOKEN:
    raise ValueError("HF_TOKEN environment variable is missing!")

# Initialize the API with the token (Handles login automatically)
api = HfApi(token=TOKEN)

print(f"Starting deployment to {REPO_ID} using Python API...")

# 2. Upload the Application File (Renaming energy_app.py to app.py)
api.upload_file(
    path_or_fileobj="App/energy_app.py",
    path_in_repo="app.py",
    repo_id=REPO_ID,
    repo_type="space",
    commit_message="Deploy App (Python API CD)"
)

# 3. Upload Requirements
api.upload_file(
    path_or_fileobj="requirements.txt",
    path_in_repo="requirements.txt",
    repo_id=REPO_ID,
    repo_type="space",
    commit_message="Sync Requirements (Python API CD)"
)

# 4. Upload Model Folder
api.upload_folder(
    folder_path="Model",
    path_in_repo="Model",
    repo_id=REPO_ID,
    repo_type="space",
    commit_message="Sync Model (Python API CD)"
)

# 5. Upload Results Folder (Mapped to Metrics)
api.upload_folder(
    folder_path="Results",
    path_in_repo="Metrics",
    repo_id=REPO_ID,
    repo_type="space",
    commit_message="Sync Metrics (Python API CD)"
)

print("✅ Deployment Complete via Python API!")