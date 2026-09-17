"""
04_customer_segmentation.py
-----------------------------
Step 4: Customer Segmentation

Answers:
    - What types of segments exist within the bank's customers?

Approach:
    - Select behavioral/financial features
    - Standardize them
    - Use the elbow method + silhouette score to choose k
    - Fit K-Means and profile each resulting segment
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

sns.set_theme(style="whitegrid")
DATA_PATH = "data/processed/bank_churn_clean.csv"
FIG_DIR = "outputs/figures"

df = pd.read_csv(DATA_PATH)

features = ["CreditScore", "Age", "Tenure", "Balance", "NumOfProducts", "EstimatedSalary", "IsActiveMember"]
X = df[features].copy()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- Choose k: elbow method + silhouette ---
inertias, sil_scores = [], []
k_range = range(2, 9)
for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    inertias.append(km.inertia_)
    sil_scores.append(silhouette_score(X_scaled, labels))

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
axes[0].plot(list(k_range), inertias, marker="o")
axes[0].set_title("Elbow Method")
axes[0].set_xlabel("k")
axes[0].set_ylabel("Inertia")

axes[1].plot(list(k_range), sil_scores, marker="o", color="darkorange")
axes[1].set_title("Silhouette Score by k")
axes[1].set_xlabel("k")
axes[1].set_ylabel("Silhouette Score")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/05_kmeans_k_selection.png", dpi=150)
plt.close()

best_k = list(k_range)[int(np.argmax(sil_scores))]
print(f"Silhouette-optimal k: {best_k} (scores: {dict(zip(k_range, np.round(sil_scores,3)))})")

# Business-driven choice: use 4 segments for interpretability unless silhouette
# strongly favors something else
K_FINAL = 4
km_final = KMeans(n_clusters=K_FINAL, random_state=42, n_init=10)
df["Segment"] = km_final.fit_predict(X_scaled)

# --- Profile segments ---
profile = df.groupby("Segment")[features + ["Exited"]].mean().round(2)
profile["CustomerCount"] = df["Segment"].value_counts().sort_index()
profile["PctOfBase"] = (profile["CustomerCount"] / len(df) * 100).round(1)
print("\n=== Segment Profiles ===")
print(profile)

profile.to_csv("outputs/segment_profiles.csv")

# --- Visualize segments via PCA ---
pca = PCA(n_components=2)
pca_coords = pca.fit_transform(X_scaled)
df["PCA1"], df["PCA2"] = pca_coords[:, 0], pca_coords[:, 1]

plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x="PCA1", y="PCA2", hue="Segment", palette="viridis", alpha=0.6, s=25)
plt.title(f"Customer Segments (K-Means, k={K_FINAL}) — PCA Projection")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/06_segments_pca.png", dpi=150)
plt.close()

# Segment churn rate chart
plt.figure(figsize=(7, 5))
seg_churn = df.groupby("Segment")["Exited"].mean().sort_values(ascending=False)
sns.barplot(x=seg_churn.index.astype(str), y=seg_churn.values, hue=seg_churn.index.astype(str), legend=False)
plt.title("Churn Rate by Customer Segment")
plt.ylabel("Churn Rate")
plt.xlabel("Segment")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/07_segment_churn_rate.png", dpi=150)
plt.close()

df.drop(columns=["PCA1", "PCA2"]).to_csv("data/processed/bank_churn_with_segments.csv", index=False)
print("\nSegmentation complete. Figures saved to outputs/figures/")
print("Segmented dataset saved to data/processed/bank_churn_with_segments.csv")
