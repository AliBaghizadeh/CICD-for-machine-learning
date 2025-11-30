# Define the project's default python environment
PYTHON = python

# -----------------
# Core ML Targets
# -----------------

# 1. install: Installs/updates necessary Python packages
install:
	pip install --upgrade pip &&\
	pip install -r requirements.txt

# 2. train: Runs the main training script
# This command executes your train.py, which loads, processes, trains,
# saves metrics/plot, and saves the final model pipeline.
train:
	$(PYTHON) train.py

# 3. eval: Generates the CML markdown report using the saved metrics and plot
eval:
	echo "## ⚡ Energy Load Forecast Model Report" > report.md
	echo "---" >> report.md
	echo "### 📊 Performance Metrics (Time Series Regression)" >> report.md
	# This command grabs the metrics you saved in train.py (MAE and R2)
	cat ./Results/metrics.txt >> report.md
	
	echo '\n### 📉 Forecast Visualization' >> report.md
	# CML displays the image saved by train.py in the GitHub comment
	echo '![Forecast vs Actual](./Results/model_results.png)' >> report.md
	
	# Uses the CML command to post the report.md content as a comment on the GitHub commit
	cml comment create report.md

# -----------------
# Utility Targets
# -----------------

# format: For code quality (requires 'black' in requirements.txt)
format:
	black .


# --- New Target for CI/CD Commit ---
# This explicitly stages the generated Model and Results files before committing.
update-branch:
	# Configure Git with secrets
	git config --global user.name $(USER_NAME)
	git config --global user.email $(USER_EMAIL)

	# FIX: Ensure Model and Results directories exist before running git add
	mkdir -p Model Results

	# CRITICAL FIX: Explicitly stage all generated files
	git add Model Results 

	# Commit the changes (using --allow-empty to avoid failure)
	git commit -m "Update with new results from $(GITHUB_SHA)" --allow-empty

	# Push the committed changes to the 'update' branch
	git push --force origin HEAD:update

# --- Continuous Deployment Targets ---

# hf-login: Logs in using the token. Dependencies must be installed in cd.yml step.
hf-login:
	huggingface-cli login --token $(HF_TOKEN) --add-to-git-credential

# push-hub: Uploads specific files/folders to the Space root, renaming the app file.
push-hub:
	huggingface-cli upload alibaghizade/time_series_energy ./App/energy_app.py app.py --repo-type=space --commit-message="Deploy App"
	huggingface-cli upload alibaghizade/time_series_energy ./requirements.txt requirements.txt --repo-type=space --commit-message="Sync Requirements"
	huggingface-cli upload alibaghizade/time_series_energy ./Model /Model --repo-type=space --commit-message="Sync Model"
	huggingface-cli upload alibaghizade/time_series_energy ./Results /Metrics --repo-type=space --commit-message="Sync Metrics and Report"

# deploy: Runs the login followed by the push.
# Define the project's default python environment
PYTHON = python

# -----------------
# Core ML Targets
# -----------------

# 1. install: Installs/updates necessary Python packages
install:
	pip install --upgrade pip &&\
	pip install -r requirements.txt

# 2. train: Runs the main training script
# This command executes your train.py, which loads, processes, trains,
# saves metrics/plot, and saves the final model pipeline.
train:
	$(PYTHON) train.py

# 3. eval: Generates the CML markdown report using the saved metrics and plot
eval:
	echo "## ⚡ Energy Load Forecast Model Report" > report.md
	echo "---" >> report.md
	echo "### 📊 Performance Metrics (Time Series Regression)" >> report.md
	# This command grabs the metrics you saved in train.py (MAE and R2)
	cat ./Results/metrics.txt >> report.md
	
	echo '\n### 📉 Forecast Visualization' >> report.md
	# CML displays the image saved by train.py in the GitHub comment
	echo '![Forecast vs Actual](./Results/model_results.png)' >> report.md
	
	# Uses the CML command to post the report.md content as a comment on the GitHub commit
	cml comment create report.md

# -----------------
# Utility Targets
# -----------------

# format: For code quality (requires 'black' in requirements.txt)
format:
	black .


# --- New Target for CI/CD Commit ---
# This explicitly stages the generated Model and Results files before committing.
update-branch:
	# Configure Git with secrets
	git config --global user.name $(USER_NAME)
	git config --global user.email $(USER_EMAIL)

	# FIX: Ensure Model and Results directories exist before running git add
	mkdir -p Model Results

	# CRITICAL FIX: Explicitly stage all generated files
	git add Model Results 

	# Commit the changes (using --allow-empty to avoid failure)
	git commit -m "Update with new results from $(GITHUB_SHA)" --allow-empty

	# Push the committed changes to the 'update' branch
	git push --force origin HEAD:update

# --- Continuous Deployment Targets ---

# hf-login: Logs in using the token. Dependencies must be installed in cd.yml step.
hf-login:
	huggingface-cli login --token $(HF_TOKEN) --add-to-git-credential

# push-hub: Uploads specific files/folders to the Space root, renaming the app file.
push-hub:
	huggingface-cli upload alibaghizade/time_series_energy ./App/energy_app.py app.py --repo-type=space --commit-message="Deploy App"
	huggingface-cli upload alibaghizade/time_series_energy ./requirements.txt requirements.txt --repo-type=space --commit-message="Sync Requirements"
	huggingface-cli upload alibaghizade/time_series_energy ./Model /Model --repo-type=space --commit-message="Sync Model"
	huggingface-cli upload alibaghizade/time_series_energy ./Results /Metrics --repo-type=space --commit-message="Sync Metrics and Report"

# deploy: Runs the login followed by the push.
deploy: hf-login push-hub

# -----------------
# MLflow Targets
# Define the project's default python environment
PYTHON = python

# -----------------
# Core ML Targets
# -----------------

# 1. install: Installs/updates necessary Python packages
install:
	pip install --upgrade pip &&\
	pip install -r requirements.txt

# 2. train: Runs the main training script
# This command executes your train.py, which loads, processes, trains,
# saves metrics/plot, and saves the final model pipeline.
train:
	$(PYTHON) train.py

# 3. eval: Generates the CML markdown report using the saved metrics and plot
eval:
	echo "## ⚡ Energy Load Forecast Model Report" > report.md
	echo "---" >> report.md
	echo "### 📊 Performance Metrics (Time Series Regression)" >> report.md
	# This command grabs the metrics you saved in train.py (MAE and R2)
	cat ./Results/metrics.txt >> report.md
	
	echo '\n### 📉 Forecast Visualization' >> report.md
	# CML displays the image saved by train.py in the GitHub comment
	echo '![Forecast vs Actual](./Results/model_results.png)' >> report.md
	
	# Uses the CML command to post the report.md content as a comment on the GitHub commit
	cml comment create report.md

# -----------------
# Utility Targets
# -----------------

# format: For code quality (requires 'black' in requirements.txt)
format:
	black .


# --- New Target for CI/CD Commit ---
# This explicitly stages the generated Model and Results files before committing.
update-branch:
	# Configure Git with secrets
	git config --global user.name $(USER_NAME)
	git config --global user.email $(USER_EMAIL)

	# FIX: Ensure Model and Results directories exist before running git add
	mkdir -p Model Results

	# CRITICAL FIX: Explicitly stage all generated files
	git add Model Results 

	# Commit the changes (using --allow-empty to avoid failure)
	git commit -m "Update with new results from $(GITHUB_SHA)" --allow-empty

	# Push the committed changes to the 'update' branch
	git push --force origin HEAD:update

# --- Continuous Deployment Targets ---

# hf-login: Logs in using the token. Dependencies must be installed in cd.yml step.
hf-login:
	huggingface-cli login --token $(HF_TOKEN) --add-to-git-credential

# push-hub: Uploads specific files/folders to the Space root, renaming the app file.
push-hub:
	huggingface-cli upload alibaghizade/time_series_energy ./App/energy_app.py app.py --repo-type=space --commit-message="Deploy App"
	huggingface-cli upload alibaghizade/time_series_energy ./requirements.txt requirements.txt --repo-type=space --commit-message="Sync Requirements"
	huggingface-cli upload alibaghizade/time_series_energy ./Model /Model --repo-type=space --commit-message="Sync Model"
	huggingface-cli upload alibaghizade/time_series_energy ./Results /Metrics --repo-type=space --commit-message="Sync Metrics and Report"

# deploy: Runs the login followed by the push.
# Define the project's default python environment
PYTHON = python

# -----------------
# Core ML Targets
# -----------------

# 1. install: Installs/updates necessary Python packages
install:
	pip install --upgrade pip &&\
	pip install -r requirements.txt

# 2. train: Runs the main training script
# This command executes your train.py, which loads, processes, trains,
# saves metrics/plot, and saves the final model pipeline.
train:
	$(PYTHON) train.py

# 3. eval: Generates the CML markdown report using the saved metrics and plot
eval:
	echo "## ⚡ Energy Load Forecast Model Report" > report.md
	echo "---" >> report.md
	echo "### 📊 Performance Metrics (Time Series Regression)" >> report.md
	# This command grabs the metrics you saved in train.py (MAE and R2)
	cat ./Results/metrics.txt >> report.md
	
	echo '\n### 📉 Forecast Visualization' >> report.md
	# CML displays the image saved by train.py in the GitHub comment
	echo '![Forecast vs Actual](./Results/model_results.png)' >> report.md
	
	# Uses the CML command to post the report.md content as a comment on the GitHub commit
	cml comment create report.md

# -----------------
# Utility Targets
# -----------------

# format: For code quality (requires 'black' in requirements.txt)
format:
	black .


# --- New Target for CI/CD Commit ---
# This explicitly stages the generated Model and Results files before committing.
update-branch:
	# Configure Git with secrets
	git config --global user.name $(USER_NAME)
	git config --global user.email $(USER_EMAIL)

	# FIX: Ensure Model and Results directories exist before running git add
	mkdir -p Model Results

	# CRITICAL FIX: Explicitly stage all generated files
	git add Model Results 

	# Commit the changes (using --allow-empty to avoid failure)
	git commit -m "Update with new results from $(GITHUB_SHA)" --allow-empty

	# Push the committed changes to the 'update' branch
	git push --force origin HEAD:update

# --- Continuous Deployment Targets ---

# hf-login: Logs in using the token. Dependencies must be installed in cd.yml step.
hf-login:
	huggingface-cli login --token $(HF_TOKEN) --add-to-git-credential

# push-hub: Uploads specific files/folders to the Space root, renaming the app file.
push-hub:
	huggingface-cli upload alibaghizade/time_series_energy ./App/energy_app.py app.py --repo-type=space --commit-message="Deploy App"
	huggingface-cli upload alibaghizade/time_series_energy ./requirements.txt requirements.txt --repo-type=space --commit-message="Sync Requirements"
	huggingface-cli upload alibaghizade/time_series_energy ./Model /Model --repo-type=space --commit-message="Sync Model"
	huggingface-cli upload alibaghizade/time_series_energy ./Results /Metrics --repo-type=space --commit-message="Sync Metrics and Report"

# deploy: Runs the login followed by the push.
deploy: hf-login push-hub

# -----------------
# MLflow Targets
# -----------------

# mlflow-ui: Launches the MLflow UI to view experiments
mlflow-ui:
	mlflow ui

# hpo: Runs the Hyperparameter Optimization script
hpo:
	$(PYTHON) train_hpo.py

# register-models: Registers optimized models in MLflow Model Registry
register-models:
	$(PYTHON) register_models.py