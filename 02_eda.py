"""
02_eda.py
---------
Step 2: Exploratory Data Analysis

Answers:
    - What do the overall demographics of the bank's customers look like?
    - What attributes are more common among churners than non-churners?

Outputs charts to outputs/figures/ and prints a summary table to console
(also saved to outputs/eda_summary.csv).
"""

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", palette="viridis")
DATA_PATH = "data/processed/bank_churn_clean.csv"
FIG_DIR = "outputs/figures"

df = pd.read_csv(DATA_PATH)
overall_churn_rate = df["Exited"].mean()
print(f"Overall churn rate: {overall_churn_rate:.1%}")

# ---------------------------------------------------------------
# 1. Overall demographics
# ---------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(13, 10))

sns.histplot(df["Age"], bins=30, kde=True, ax=axes[0, 0], color="#2c7fb8")
axes[0, 0].set_title("Age Distribution")

df["Geography"].value_counts().plot(kind="bar", ax=axes[0, 1], color="#41b6c4")
axes[0, 1].set_title("Customers by Geography")
axes[0, 1].tick_params(axis="x", rotation=0)

df["Gender"].value_counts().plot(kind="bar", ax=axes[1, 0], color="#a1dab4")
axes[1, 0].set_title("Customers by Gender")
axes[1, 0].tick_params(axis="x", rotation=0)

sns.histplot(df["EstimatedSalary"], bins=30, kde=True, ax=axes[1, 1], color="#253494")
axes[1, 1].set_title("Estimated Salary Distribution")

plt.tight_layout()
plt.savefig(f"{FIG_DIR}/01_demographics_overview.png", dpi=150)
plt.close()

# ---------------------------------------------------------------
# 2. Churn rate by key categorical attributes
# ---------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(13, 10))

geo_churn = df.groupby("Geography")["Exited"].mean().sort_values(ascending=False)
geo_churn.plot(kind="bar", ax=axes[0, 0], color="#e34a33")
axes[0, 0].set_title("Churn Rate by Geography")
axes[0, 0].set_ylabel("Churn Rate")
axes[0, 0].tick_params(axis="x", rotation=0)

gender_churn = df.groupby("Gender")["Exited"].mean().sort_values(ascending=False)
gender_churn.plot(kind="bar", ax=axes[0, 1], color="#fc8d59")
axes[0, 1].set_title("Churn Rate by Gender")
axes[0, 1].set_ylabel("Churn Rate")
axes[0, 1].tick_params(axis="x", rotation=0)

product_churn = df.groupby("NumOfProducts")["Exited"].mean()
product_churn.plot(kind="bar", ax=axes[1, 0], color="#fdcc8a")
axes[1, 0].set_title("Churn Rate by Number of Products")
axes[1, 0].set_ylabel("Churn Rate")
axes[1, 0].tick_params(axis="x", rotation=0)

active_churn = df.groupby("IsActiveMember")["Exited"].mean()
active_churn.index = active_churn.index.map({0: "Inactive", 1: "Active"})
active_churn.plot(kind="bar", ax=axes[1, 1], color="#d7301f")
axes[1, 1].set_title("Churn Rate by Membership Activity")
axes[1, 1].set_ylabel("Churn Rate")
axes[1, 1].tick_params(axis="x", rotation=0)

plt.tight_layout()
plt.savefig(f"{FIG_DIR}/02_churn_by_category.png", dpi=150)
plt.close()

# ---------------------------------------------------------------
# 3. Churners vs non-churners on numeric attributes
# ---------------------------------------------------------------
numeric_cols = ["CreditScore", "Age", "Tenure", "Balance", "EstimatedSalary"]
fig, axes = plt.subplots(1, 5, figsize=(22, 4.5))
for i, col in enumerate(numeric_cols):
    sns.boxplot(data=df, x="Exited", y=col, hue="Exited", ax=axes[i],
                palette=["#2c7fb8", "#e34a33"], legend=False)
    axes[i].set_xticks([0, 1])
    axes[i].set_xticklabels(["Stayed", "Churned"])
    axes[i].set_title(col)
    axes[i].set_xlabel("")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/03_numeric_attributes_by_churn.png", dpi=150)
plt.close()

# ---------------------------------------------------------------
# 4. Summary table: churner vs non-churner profile
# ---------------------------------------------------------------
summary = df.groupby("Exited")[numeric_cols].mean().T
summary.columns = ["Stayed (Exited=0)", "Churned (Exited=1)"]
summary["Difference (%)"] = (
    (summary["Churned (Exited=1)"] - summary["Stayed (Exited=0)"])
    / summary["Stayed (Exited=0)"] * 100
).round(1)
print("\n=== Numeric attribute comparison: churners vs non-churners ===")
print(summary.round(2))
summary.to_csv("outputs/eda_summary.csv")

cat_summary = pd.DataFrame({
    "Churn Rate": [
        overall_churn_rate,
        *geo_churn.values,
        *gender_churn.values,
        *active_churn.values,
    ]
}, index=[
    "Overall",
    *[f"Geography: {g}" for g in geo_churn.index],
    *[f"Gender: {g}" for g in gender_churn.index],
    *[f"Active: {a}" for a in active_churn.index],
])
print("\n=== Categorical churn rates ===")
print(cat_summary.round(3))
cat_summary.to_csv("outputs/eda_categorical_churn_rates.csv")

print("\nEDA complete. Figures saved to outputs/figures/")
