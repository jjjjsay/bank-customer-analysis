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
