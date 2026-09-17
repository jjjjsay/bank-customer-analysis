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
