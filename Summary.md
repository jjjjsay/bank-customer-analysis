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

## Skills Demonstrated

- **Data wrangling**: multi-sheet Excel ingestion, currency/text parsing, sentinel-value handling, deduplication, data validation against a ground-truth source
- **Statistical inference**: ANOVA, chi-square tests of independence
- **Exploratory data analysis & data visualization**: `pandas`, `matplotlib`, `seaborn`
- **Unsupervised learning**: feature scaling, K-Means clustering, elbow/silhouette model selection, PCA for visualization
- **Supervised machine learning**: classification model comparison (Logistic Regression, Random Forest, XGBoost), class imbalance handling, feature engineering, feature importance interpretation
- **Business communication**: translating model output and statistical results into segment-specific, actionable retention recommendations

---

## Results

### Demographics
The customer base skews slightly male (54.6%), averages ~39 years old, and is split roughly 50/25/25 across France/Germany/Spain (France holds twice the customer count of Germany or Spain individually).

### Churn Drivers (Churners vs. Non-Churners)

| Attribute | Stayed | Churned | Difference |
|---|---|---|---|
| Age | 37.4 | 44.8 | **+19.9%** |
| Balance | $72,745 | $91,109 | **+25.2%** |
| Credit Score | 651.9 | 645.4 | -1.0% |
| Tenure | 5.03 yrs | 4.93 yrs | -2.0% |

Categorical churn rates: **Inactive members (26.9%) churn nearly 2x active members (14.3%)**; **Female customers (25.1%) churn more than male customers (16.5%)**; customers with **3-4 products churn dramatically more** than those with 1-2 (a strong red flag for over-cross-sold accounts).

### Geographic Differences (statistically significant, p < 0.001)
- **Germany**: 32.4% churn rate (vs. ~16% for France/Spain), and *every* German customer maintains a non-zero balance — German customers carry meaningfully higher balances on average, making their departure more costly per customer.
- **France & Spain**: near-identical behavior on churn, balance, and product mix — France and Spain can likely share a retention playbook, while Germany needs its own strategy.

### Customer Segments

| Segment | Size | Profile | Churn Rate |
|---|---|---|---|
| **0 — Active Affluent Loyalists** | 28.7% | Young (35), high balance (~$108k), fully active, single product | **13%** (lowest risk) |
| **1 — Multi-Product Value Customers** | 27.6% | Low balance (~$9.6k), highest product count (2.1), moderate activity | **12%** (lowest risk) |
| **2 — Disengaged At-Risk** | 32.5% | High balance (~$106k), **0% active membership**, single product | **29%** (high risk) |
| **3 — Senior High-Value** | 11.2% | Oldest group (60 yrs), active (83%), moderate balance | **36%** (highest risk) |

Segment 2 is the single largest group (32.5% of the base) and combines high account value with zero engagement — the clearest reactivation target. Segment 3, while smaller, churns at the highest rate despite being active, suggesting age-related life-stage churn (retirement, relocation, competitor offers) rather than a satisfaction problem.

### Predictive Model Performance

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 71.2% | 38.6% | 70.8% | 50.0% | 77.5% |
| Random Forest | 82.8% | 56.1% | 70.0% | 62.3% | 86.1% |
| **XGBoost (best)** | 80.9% | 52.1% | **73.5%** | 61.0% | **86.5%** |

**Top predictive features (XGBoost):** Number of Products, Zero-Balance Flag, Age, Has Credit Card, Germany Flag — confirming the patterns found in EDA and the geographic analysis.

XGBoost was selected as the production candidate because it achieves the best ROC-AUC and the highest recall on churners — for a retention use case, catching more true churners (even at some cost to precision) is the right trade-off, since a missed churner is a lost customer while a false alarm just costs a discount offer.

---

## Business Recommendations

1. **Launch a Germany-specific retention program.** Germany churns at 2x the rate of France/Spain despite carrying the bank's highest average balances — this is the single highest-value, highest-urgency segment to address. Investigate local competitive pressure and pricing.
2. **Build an inactivity early-warning trigger.** Since inactive members churn at nearly 2x the rate of active ones, flag any customer who goes inactive for 60-90 days and route them into a re-engagement campaign (personal outreach, fee waivers, product bundling) before they churn.
3. **Re-examine multi-product cross-sell practices.** Customers holding 3-4 products churn at a much higher rate than those with 1-2 — this points to over-selling or bundling that isn't landing well, rather than "more products = more loyalty."
4. **Prioritize Segment 2 ("Disengaged At-Risk") for win-back campaigns.** This is the largest segment (32.5% of customers) and combines high balances with zero activity — the highest expected-value reactivation target.
5. **Design an age-aware retention track for Segment 3 ("Senior High-Value").** Their churn appears life-stage driven rather than a satisfaction issue; consider retirement-planning products, loyalty perks, or dedicated relationship managers rather than generic discounts.
6. **Operationalize the XGBoost model as a monthly churn-risk score.** Score the full active customer base monthly and route the top-risk decile to the retention team — at 73.5% recall, the model would catch roughly 3 in 4 customers who are about to churn.
7. **Fix the data pipeline at the source.** The messy raw export contained a fake `-999999` missing-value sentinel and three spellings of "France" — these should be fixed upstream in the source system, not patched downstream in every analysis.

---

## Next Steps

- **Cost-sensitive threshold tuning**: work with the retention team to estimate the cost of a false positive (unnecessary retention offer) vs. false negative (lost customer) and tune the model's decision threshold accordingly, rather than using the default 0.5 cutoff.
- **A/B test retention interventions**: run controlled experiments on the top-risk decile (e.g., proactive call vs. email offer vs. control) to measure actual causal impact on retention, since correlation-based recommendations above are hypotheses, not proven interventions.
- **Add tenure-stage and transaction-level data**: this dataset only has a snapshot; monthly transaction trends, complaint/support-ticket history, and product usage over time would likely sharpen both the segmentation and the model considerably.
- **Model monitoring**: set up a simple monthly re-training and performance-drift check, since customer behavior and market conditions (interest rates, competitor offers) will shift the churn drivers over time.
- **Explainability layer**: add SHAP values on top of the XGBoost model so retention agents can see *why* a specific customer was flagged, not just that they were.
- **Survival analysis**: model *time-to-churn* (e.g. Cox proportional hazards) rather than a binary snapshot, to estimate not just who will churn but roughly when — useful for timing interventions.

---
