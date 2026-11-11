# South German Credit: PCA Dimensionality Reduction & Cluster Evaluation Pipeline

This repository contains a Python script for exploratory data analysis, cluster validation metrics evaluation, and analyzing the impact of dimensionality reduction (PCA) on data structure using the **South German Credit** dataset.

## 📋 Pipeline Overview

1. **Data Standardization**:
   * Drops the target column (`kredit`, if present) and normalizes features using `StandardScaler` to ensure uniform scale contribution during distance-based modeling.

2. **Optimal Cluster Number Selection ($k$)**:
   * Iterates through cluster counts ranging from 2 to 10 for both **K-Means** and **Agglomerative Clustering** algorithms.
   * Evaluates clustering quality using **Silhouette Score** (maximization) and **Davies-Bouldin Score** (minimization).

3. **Information Loss Analysis (PCA)**:
   * Computes cumulative explained variance to determine information loss percentage relative to the number of retained principal components.
   * Plots the information loss curve.

4. **Dimensionality Reduction & Re-clustering**:
   * Identifies the component threshold required to preserve at least 70% of information (yielding 11 principal components).
   * Performs K-Means clustering on the reduced feature space and compares the partitions against the full dataset using the **Adjusted Rand Index (ARI)**.

---

## 🛠️ Code Implementation

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, adjusted_rand_score

df = pd.read_csv('SouthGermanCredit.asc', sep=r'\s+')

# Standardization and preparation
X = df.drop(columns=['kredit']) if 'kredit' in df.columns else df
X_scaled = StandardScaler().fit_transform(X)

# Cluster range iteration
k_range = range(2, 11)
km_sil, km_db = [], []
agg_sil, agg_db = [], []

for k in k_range:
    # K-Means
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km_labels = km.fit_predict(X_scaled)
    km_sil.append(silhouette_score(X_scaled, km_labels))
    km_db.append(davies_bouldin_score(X_scaled, km_labels))

    # Agglomerative Clustering
    agg = AgglomerativeClustering(n_clusters=k)
    agg_labels = agg.fit_predict(X_scaled)
    agg_sil.append(silhouette_score(X_scaled, agg_labels))
    agg_db.append(davies_bouldin_score(X_scaled, agg_labels))

print("Max silhouette index for K-Means:", k_range[np.argmax(km_sil)])
print("Min Davies-Bouldin index for K-Means:", k_range[np.argmin(km_db)])

# Information loss curve visualization
pca_full = PCA().fit(X_scaled)
cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)
information_loss = (1 - cumulative_variance) * 100

plt.figure(figsize=(8, 5))
plt.plot(range(1, len(information_loss) + 1), information_loss, marker='o', color='b')
plt.title("Information Loss vs. Number of Principal Components (PCA)")
plt.xlabel("Number of Principal Components")
plt.ylabel("Information Loss (%)")
plt.grid(True)
plt.show()

# Transform data keeping 70% information
n_components_70 = np.argmax(cumulative_variance >= 0.70) + 1
pca_reduced = PCA(n_components=n_components_70)
X_reduced = pca_reduced.fit_transform(X_scaled)

# Optimal clustering on reduced dataset
best_k_reduced = 3
optimal_model = KMeans(n_clusters=best_k_reduced, random_state=42, n_init=10)
clusters_reduced = optimal_model.fit_predict(X_reduced)

# Optimal clustering on full dataset
optimal_model_full = KMeans(n_clusters=best_k_reduced, random_state=42, n_init=10)
clusters_full = optimal_model_full.fit_predict(X_scaled)

# Partition comparison
ari_score = adjusted_rand_score(clusters_full, clusters_reduced)
print(f"ARI between full and reduced dataset: {ari_score:.4f}")
```

---

## 📊 Results & Conclusions

* Principal Component Analysis (PCA) revealed that retaining **11 principal components** keeps information loss within the 30% threshold limit.
* The resulting **ARI (Adjusted Rand Index) score of 0.3594** indicates a moderate similarity between cluster structures. This demonstrates that while dimensionality reduction slightly modifies fine-grained partitioning details, the overall distribution patterns and macro-structures of the data remain intact.