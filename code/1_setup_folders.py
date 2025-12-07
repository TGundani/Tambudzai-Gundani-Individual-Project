"""
========================================================================================================================
RegPolicyBot – FOLDER SETUP
========================================================================================================================

Purpose: Initializing the directory structure for the RegPolicyBot project. The folders reflect distinct components of
         the project workflow:
        - Code:     All Python scripts and notebooks for data collection, modeling, and deployment
        - data_raw: Stores unprocessed text retrieved from the Regulatory Studies Center (RSC)
        - data_clean: Contains cleaned and preprocessed text ready for modeling
        - models:   Serialized model files and embeddings
        - results:  Evaluation metrics, plots, and explainability visuals
        - reports:  Final report, presentation slides, and supporting documentation
========================================================================================================================
"""
# ======================================================================================================================
# CREATING FOLDER STRUCTURE
# ======================================================================================================================
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.join(BASE_DIR, "code")
DATA_RAW_DIR = os.path.join(BASE_DIR, "data_raw")
DATA_CLEAN_DIR = os.path.join(BASE_DIR, "data_clean")
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")

os.makedirs(CODE_DIR, exist_ok=True)
os.makedirs(DATA_RAW_DIR, exist_ok=True)
os.makedirs(DATA_CLEAN_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

print("Project directories are set up correctly.")
# ======================================================================================================================