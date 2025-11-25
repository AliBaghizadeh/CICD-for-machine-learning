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