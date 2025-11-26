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

# deploy: Installs the library and runs the python deployment script
deploy:
	# 1. Install the Hugging Face library
	pip install huggingface_hub
	
	# CRITICAL FIX: Use the 'env' command to explicitly set the environment variable 
    # for the python process, using the value passed as an argument to 'make'.
	env HF_TOKEN=$(HF_TOKEN) python deploy.py