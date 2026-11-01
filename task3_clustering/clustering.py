import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

K = 5  # set after checking elbow.png

# 1. Load data
if os.path.exists("Mall_Customers.csv"):
    df = pd.read_csv("Mall_Customers.csv")
    df = df.drop(columns=[c for c in df.columns if "id" in c.lower()])
    df = pd.get_dummies(df, drop_first=True, dtype=int)
else:
    rng = np.random.default_rng(42)
    centers = [(25, 30, 80, 12, 20), (45, 90, 85, 10, 70), (55, 85, 15, 2, 90),
               (30, 25, 20, 3, 15), (40, 55, 50, 6, 40)]
    parts = [rng.normal(c, [4, 8, 8, 1.5, 8], size=(200, 5)) for c in centers]
    df = pd.DataFrame(np.vstack(parts),
                      columns=["age", "income", "spending_score",
                               "visits_per_month", "avg_basket"])
print(df.describe())

# 2. Standardize 
X = StandardScaler().fit_transform(df)

# 3. PCA 
pca = PCA().fit(X)
print("Explained variance:", pca.explained_variance_ratio_.round(3))
X_pca = PCA(n_components=2).fit_transform(X)

# 4. Elbow method 
ks = range(1, 11)
inertia = [KMeans(n_clusters=k, n_init=10, random_state=42).fit(X_pca).inertia_ for k in ks]
plt.figure(figsize=(6, 4))
plt.plot(ks, inertia, "o-")
plt.xlabel("Number of clusters (k)")
plt.ylabel("Inertia")
plt.title("Elbow Method")
plt.savefig("elbow.png", dpi=150, bbox_inches="tight")
plt.close()

# 5. Fit K-Means 
km = KMeans(n_clusters=K, n_init=10, random_state=42).fit(X_pca)
df["segment"] = km.labels_
print("Silhouette score:", round(silhouette_score(X_pca, km.labels_), 3))

# 6. Visualize
plt.figure(figsize=(7, 5))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=km.labels_, cmap="tab10", s=25)
plt.scatter(*km.cluster_centers_.T, c="black", marker="X", s=200, label="Centroids")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title(f"Customer Segments (K={K})")
plt.legend()
plt.savefig("segments.png", dpi=150, bbox_inches="tight")
plt.close()

# 7. Segment profile 
profile = df.groupby("segment").mean().round(1)
profile["size"] = df["segment"].value_counts().sort_index()
print(profile)
profile.to_csv("segment_profile.csv")
df.to_csv("customers_segmented.csv", index=False)