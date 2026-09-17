"""
run_all.py
----------
Runs the full analysis pipeline end-to-end, in order:
    1. Data cleaning (reconciles the messy raw Excel workbook)
    2. Exploratory data analysis (demographics + churn drivers)
    3. Geography deep-dive (France vs Germany vs Spain)
    4. Customer segmentation (K-Means)
    5. Predictive modeling (Logistic Regression / Random Forest / XGBoost)

Usage:
    python run_all.py

All outputs (figures, csvs, trained model) are written to outputs/.
"""

import subprocess
import sys

STEPS = [
    "src/01_data_cleaning.py",
    "src/02_eda.py",
    "src/03_geography_analysis.py",
    "src/04_customer_segmentation.py",
    "src/05_churn_prediction.py",
]

if __name__ == "__main__":
    for step in STEPS:
        print(f"\n{'='*70}\nRunning {step}\n{'='*70}")
        result = subprocess.run([sys.executable, step])
        if result.returncode != 0:
            print(f"Step {step} failed — stopping pipeline.")
            sys.exit(1)
    print("\nPipeline complete. See outputs/ for all results.")
