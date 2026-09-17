# Executive Summary

This project analyses account and demographic data for 10,000 customers of a European retail bank operating in **France, Germany, and Spain** to understand why customers churn and to identify which customers are most at risk.

**Key findings:**

| Finding | Detail |
|---|---|
| **Overall churn rate** | 20.4% of customers have exited the bank |
| **Germany is the highest-risk market** | 32.4% churn rate — 2x France (16.2%) and Spain (16.7%) |
| **Inactivity is the strongest behavioral flag** | Inactive members churn at 26.9% vs. 14.3% for active members |
| **Older customers churn more** | Churned customers average 44.8 years old vs. 37.4 for retained customers (+20%) |
| **Higher balances correlate with churn** | Churned customers hold 25% higher average balances (\$91,109 vs \$72,745) |
| **Churn is predictable** | An XGBoost model reaches **86.5% ROC-AUC** and correctly flags **73% of churners** in the test set |
| **4 distinct customer segments identified** | Ranging from a low-risk "Active Affluent" segment (13% churn) to a high-risk "Senior" segment (36% churn) |


The recommendations below focus on Germany-specific retention programs, activity-reactivation campaigns, and using the trained model to power a proactive, risk-scored retention workflow.

---

## Business Problem

Customer acquisition in retail banking is expensive, and churn directly erodes lifetime value and market share. The bank needs to answer four questions to run a more targeted retention program instead of a one-size-fits-all approach:

1. **What attributes are more common among churners than non-churners, and can churn be predicted?**
2. **What do the overall demographics of the customer base look like?**
3. **Is there a meaningful difference in account behavior between German, French, and Spanish customers?**
4. **What natural customer segments exist, and how should the bank treat each differently?**

A secondary, real-world complication tackled in this project: the bank's actual source data arrives as **two separate, messy Excel extracts** (customer records and account records) that must be cleaned, standardized, and merged before any of the above can be answered — mirroring the kind of raw data an analyst receives in practice, rather than a pre-cleaned CSV.

---

## Methodology

### 1. Data Cleaning & Integration (`src/01_data_cleaning.py`)
- Loaded two raw sheets (`Customer_Info`, `Account_Info`) from `Bank_Churn_Messy.xlsx`
- Fixed data quality issues found during profiling:
  - Currency-formatted strings (e.g. `€101348.88`) parsed to numeric
  - A `-€999999` sentinel value used for missing `EstimatedSalary` entries, converted to `NaN` and median-imputed
  - Inconsistent country labels (`France` / `French` / `FRA`) standardized to one canonical value per country
  - `Yes`/`No` text fields converted to boolean 0/1
  - Exact duplicate rows removed from both sheets
  - A handful of missing `Age` and `Surname` values imputed/flagged
- Merged both sheets into one customer-level table on `CustomerId`
- **Validated** the cleaned output against the bank's reference clean export (`Bank_Churn.csv`) — achieved a **100% match rate** across all numeric fields for all 10,000 customers, confirming the cleaning logic is correct

### 2. Exploratory Data Analysis (`src/02_eda.py`)
- Profiled customer demographics (age, geography, gender, salary distributions)
- Compared churners vs. non-churners across every numeric and categorical attribute

### 3. Geographic Deep-Dive (`src/03_geography_analysis.py`)
- Compared account balance, credit score, product holdings, and activity rates across the three countries
- Ran one-way ANOVA tests on continuous variables and a chi-square test of independence between geography and churn to confirm differences are statistically significant, not noise

### 4. Customer Segmentation (`src/04_customer_segmentation.py`)
- Standardized 7 behavioral/financial features and applied **K-Means clustering**
- Used the elbow method and silhouette score to sanity-check cluster count, then selected **k=4** for business interpretability
- Profiled each segment's demographics, financial behavior, and churn rate
- Visualized segments via PCA projection

### 5. Predictive Modeling (`src/05_churn_prediction.py`)
- Engineered features: balance-to-salary ratio, zero-balance flag, Germany flag, tenure-to-age ratio
- One-hot encoded categorical variables; stratified 80/20 train/test split
- Trained and compared three models: **Logistic Regression** (interpretable baseline), **Random Forest**, and **XGBoost**
- Evaluated with accuracy, precision, recall, F1, and ROC-AUC (ROC-AUC and recall on the minority "churned" class matter most for a retention use case, since missing a churner is costlier than a false alarm)
- Extracted feature importances from the winning model

---
