"""
05_churn_prediction.py
------------------------
Step 5: Predictive Modeling

Answers:
    - Can churn be predicted using the variables in the data?

Approach:
    - Feature engineering (encode categoricals, derived features)
    - Train/test split (stratified, 80/20)
    - Compare Logistic Regression (baseline/interpretable), Random Forest,
      and XGBoost (best performer expected on tabular data)
    - Evaluate with accuracy, precision, recall, F1, ROC-AUC
    - Extract feature importance from the best model
    - Save the trained model + metrics
"""

import pandas as pd
import numpy as np
import json
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)
from xgboost import XGBClassifier

sns.set_theme(style="whitegrid")
DATA_PATH = "data/processed/bank_churn_clean.csv"
FIG_DIR = "outputs/figures"
MODEL_DIR = "outputs/models"

df = pd.read_csv(DATA_PATH)

# ---------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------
df["BalanceSalaryRatio"] = df["Balance"] / (df["EstimatedSalary"] + 1)
df["IsGermany"] = (df["Geography"] == "Germany").astype(int)
df["IsZeroBalance"] = (df["Balance"] == 0).astype(int)
df["TenureByAge"] = df["Tenure"] / df["Age"]

df_model = pd.get_dummies(
    df, columns=["Geography", "Gender"], drop_first=True
)

drop_cols = ["CustomerId", "Surname", "Exited"]
feature_cols = [c for c in df_model.columns if c not in drop_cols]
X = df_model[feature_cols]
y = df_model["Exited"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train size: {X_train.shape}, Test size: {X_test.shape}")
print(f"Train churn rate: {y_train.mean():.1%}, Test churn rate: {y_test.mean():.1%}")

# Scale for logistic regression only
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------------------------------------------------------
# Train models
# ---------------------------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=300, max_depth=8, class_weight="balanced", random_state=42),
    "XGBoost": XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
        eval_metric="logloss", random_state=42
    ),
}

results = {}
roc_data = {}

for name, model in models.items():
    if name == "Logistic Regression":
        model.fit(X_train_scaled, y_train)
        pred = model.predict(X_test_scaled)
        proba = model.predict_proba(X_test_scaled)[:, 1]
    else:
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        proba = model.predict_proba(X_test)[:, 1]

    results[name] = {
        "accuracy": round(accuracy_score(y_test, pred), 4),
        "precision": round(precision_score(y_test, pred), 4),
        "recall": round(recall_score(y_test, pred), 4),
        "f1": round(f1_score(y_test, pred), 4),
        "roc_auc": round(roc_auc_score(y_test, proba), 4),
    }
    fpr, tpr, _ = roc_curve(y_test, proba)
    roc_data[name] = (fpr, tpr, results[name]["roc_auc"])
    print(f"\n{name}")
    print(classification_report(y_test, pred, target_names=["Stayed", "Churned"]))

results_df = pd.DataFrame(results).T
print("\n=== Model Comparison ===")
print(results_df)
results_df.to_csv("outputs/model_comparison.csv")

best_model_name = results_df["roc_auc"].idxmax()
best_model = models[best_model_name]
print(f"\nBest model by ROC-AUC: {best_model_name}")

# ---------------------------------------------------------------
# ROC curve comparison chart
# ---------------------------------------------------------------
plt.figure(figsize=(7, 6))
for name, (fpr, tpr, auc) in roc_data.items():
    plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
plt.plot([0, 1], [0, 1], linestyle="--", color="grey")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/08_roc_curves.png", dpi=150)
plt.close()

# ---------------------------------------------------------------
# Confusion matrix for best model
# ---------------------------------------------------------------
if best_model_name == "Logistic Regression":
    best_pred = best_model.predict(X_test_scaled)
else:
    best_pred = best_model.predict(X_test)

cm = confusion_matrix(y_test, best_pred)
plt.figure(figsize=(5.5, 4.5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Stayed", "Churned"], yticklabels=["Stayed", "Churned"])
plt.title(f"Confusion Matrix — {best_model_name}")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/09_confusion_matrix.png", dpi=150)
plt.close()

# ---------------------------------------------------------------
# Feature importance (tree-based best model)
# ---------------------------------------------------------------
if hasattr(best_model, "feature_importances_"):
    importances = pd.Series(best_model.feature_importances_, index=feature_cols).sort_values(ascending=False)
    print("\n=== Top 10 Feature Importances ===")
    print(importances.head(10))
    importances.to_csv("outputs/feature_importance.csv")

    plt.figure(figsize=(8, 6))
    importances.head(10).sort_values().plot(kind="barh", color="#2c7fb8")
    plt.title(f"Top 10 Feature Importances — {best_model_name}")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/10_feature_importance.png", dpi=150)
    plt.close()

# ---------------------------------------------------------------
# Save best model + metadata
# ---------------------------------------------------------------
joblib.dump(best_model, f"{MODEL_DIR}/best_churn_model_{best_model_name.replace(' ', '_').lower()}.pkl")
with open("outputs/model_metrics.json", "w") as f:
    json.dump({"results": results, "best_model": best_model_name}, f, indent=2)

print("\nModeling complete. Model + metrics saved to outputs/")
