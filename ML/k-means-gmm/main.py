"""
P0 - Track G (Unsupervised Learning) - K-Means & Gaussian Mixture Models (GMM)
Library: scikit-learn (KMeans, GaussianMixture, metrics) + matplotlib for one plot.

Goal: cluster the same data with K-Means (hard assignments, round clusters) and a GMM
(soft assignments, elliptical clusters), compare them, and learn how to pick the number
of clusters. No true "from scratch" code: the library does the work in a few lines.

Setup:  pip install -r requirements.txt
Run:    python main.py        (saves clusters.png)
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.mixture import GaussianMixture

SEED = 42

# Synthetic data (designed for this demo): 3 overlapping groups of different spread, stretched into ellipses.
X, y_true = make_blobs(n_samples=600, centers=3, cluster_std=[1.6, 3.2, 1.0], random_state=SEED)
X = X @ np.array([[0.6, -0.8], [-0.1, 1.0]])

# 1) Fit both models with k=3
kmeans = KMeans(n_clusters=3, n_init=10, random_state=SEED).fit(X)
gmm = GaussianMixture(n_components=3, covariance_type="full", n_init=5, random_state=SEED).fit(X)
km_labels, gmm_labels = kmeans.labels_, gmm.predict(X)

print("=== 1) K-Means vs GMM (k = 3) ===")
for name, labels in [("K-Means", km_labels), ("GMM", gmm_labels)]:
    print(f"{name:8} adjusted Rand index vs true groups: {adjusted_rand_score(y_true, labels):.3f}   "
          f"silhouette: {silhouette_score(X, labels):.3f}")
print("(ARI: 1.0 = perfect match with the true groups, ~0 = random. Silhouette needs no labels.)\n")

# 2) Hard vs soft assignment: the GMM tells you how unsure it is
probs = gmm.predict_proba(X)
most_uncertain = np.argsort(probs.max(axis=1))[:3]
print("=== 2) Soft assignments (GMM) ===")
for i in most_uncertain:
    print(f"point {i}: membership probabilities = {np.round(probs[i], 2)}  (K-Means would just say cluster {km_labels[i]})")
print(f"Points with max probability below 0.9: {(probs.max(axis=1) < 0.9).sum()} of {len(X)}\n")

# 3) Choosing k
print("=== 3) Choosing the number of clusters ===")
print(f"{'k':>2} {'inertia':>9} {'silhouette':>11} {'GMM BIC':>9}")
bics = {}
for k in range(2, 7):
    km = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(X)
    g = GaussianMixture(n_components=k, n_init=5, random_state=SEED).fit(X)
    bics[k] = g.bic(X)
    print(f"{k:>2} {km.inertia_:>9.0f} {silhouette_score(X, km.labels_):>11.3f} {bics[k]:>9.0f}")
print(f"Lowest BIC (better) at k = {min(bics, key=bics.get)}. Inertia always falls as k grows, so look for an 'elbow'.\n")

# 4) Plot
fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharex=True, sharey=True)
for ax, labels, title in [(axes[0], y_true, "True groups"), (axes[1], km_labels, "K-Means"), (axes[2], gmm_labels, "GMM")]:
    ax.scatter(X[:, 0], X[:, 1], c=labels, s=10, cmap="viridis")
    ax.set_title(title)
axes[1].scatter(*kmeans.cluster_centers_.T, c="red", marker="x", s=120)
plt.tight_layout()
plt.savefig("clusters.png", dpi=110)
print("Saved clusters.png (true groups | K-Means | GMM)")
