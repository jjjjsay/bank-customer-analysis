# Bank-customer-analysis

Predicting and explaining customer attrition for a European retail bank using Python, statistical testing, unsupervised segmentation, and supervised machine learning.

## Repository Structure

```
bank-churn-analysis/
├── README.md                          <- this file
├── requirements.txt
├── run_all.py                         <- runs the entire pipeline end-to-end
├── data/
│   ├── raw/                           <- original messy + reference files
│   └── processed/                     <- cleaned, merged, and segmented output
├── src/
│   ├── 01_data_cleaning.py
│   ├── 02_eda.py
│   ├── 03_geography_analysis.py
│   ├── 04_customer_segmentation.py
│   └── 05_churn_prediction.py
└── outputs/
    ├── figures/                       <- all 10 charts (PNG)
    ├── models/                        <- saved best model (.pkl)
    ├── eda_summary.csv
    ├── eda_categorical_churn_rates.csv
    ├── geography_means.csv
    ├── geography_behavior.csv
    ├── segment_profiles.csv
    ├── model_comparison.csv
    ├── feature_importance.csv
    └── model_metrics.json
```

## How to Run

```bash
git clone <your-repo-url>
cd bank-churn-analysis
pip install -r requirements.txt
python run_all.py
```

This regenerates every figure, csv, and the trained model from the raw data in `data/raw/`.

## Data Source

10,000-customer European bank churn dataset (`Bank_Churn.csv`), with a companion intentionally-messy two-sheet Excel extract (`Bank_Churn_Messy.xlsx`) used to demonstrate the data cleaning workflow. See `data/raw/Bank_Churn_Data_Dictionary.csv` for field definitions.
