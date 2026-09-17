"""
03_geography_analysis.py
-------------------------
Step 3: Geographic Deep-Dive

Answers:
    - Is there a difference between German, French, and Spanish customers
      in terms of account behavior?

Runs group comparisons + ANOVA/chi-square significance tests and saves a
summary chart.
"""

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")
DATA_PATH = "data/processed/bank_churn_clean.csv"
FIG_DIR = "outputs/figures"

df = pd.read_csv(DATA_PATH)

numeric_cols = ["CreditScore", "Age", "Tenure", "Balance", "NumOfProducts", "EstimatedSalary"]

print("=== Mean account metrics by Geography ===")
geo_means = df.groupby("Geography")[numeric_cols].mean().round(2)
print(geo_means)

print("\n=== Churn rate & product/activity mix by Geography ===")
geo_behavior = df.groupby("Geography").agg(
    ChurnRate=("Exited", "mean"),
    PctActive=("IsActiveMember", "mean"),
    PctHasCrCard=("HasCrCard", "mean"),
    PctZeroBalance=("Balance", lambda x: (x == 0).mean()),
    AvgProducts=("NumOfProducts", "mean"),
).round(3)
print(geo_behavior)

# --- Statistical significance tests ---
print("\n=== Significance tests across the 3 geographies ===")
france = df[df.Geography == "France"]
germany = df[df.Geography == "Germany"]
spain = df[df.Geography == "Spain"]

for col in ["Balance", "CreditScore", "Age", "NumOfProducts"]:
    f_stat, p_val = stats.f_oneway(france[col], germany[col], spain[col])
    sig = "significant" if p_val < 0.05 else "not significant"
    print(f"  ANOVA on {col}: F={f_stat:.2f}, p={p_val:.4f} -> {sig}")

contingency = pd.crosstab(df["Geography"], df["Exited"])
chi2, p_chi, _, _ = stats.chi2_contingency(contingency)
print(f"  Chi-square (Geography vs Churn): chi2={chi2:.2f}, p={p_chi:.6f} "
      f"-> {'significant' if p_chi < 0.05 else 'not significant'}")

# --- Visualization ---
fig, axes = plt.subplots(2, 2, figsize=(13, 10))

sns.boxplot(data=df, x="Geography", y="Balance", hue="Geography", ax=axes[0, 0], legend=False)
axes[0, 0].set_title("Account Balance by Geography")

sns.barplot(x=geo_behavior.index, y=geo_behavior["ChurnRate"], hue=geo_behavior.index,
            ax=axes[0, 1], legend=False)
axes[0, 1].set_title("Churn Rate by Geography")

sns.countplot(data=df, x="NumOfProducts", hue="Geography", ax=axes[1, 0])
axes[1, 0].set_title("Number of Products by Geography")

sns.barplot(x=geo_behavior.index, y=geo_behavior["PctActive"], hue=geo_behavior.index,
            ax=axes[1, 1], legend=False)
axes[1, 1].set_title("% Active Members by Geography")

plt.tight_layout()
plt.savefig(f"{FIG_DIR}/04_geography_comparison.png", dpi=150)
plt.close()

geo_means.to_csv("outputs/geography_means.csv")
geo_behavior.to_csv("outputs/geography_behavior.csv")
print("\nGeography analysis complete. Figures saved to outputs/figures/")
