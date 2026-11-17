# P0 — Track G (Unsupervised Learning): K-Means & Gaussian Mixture Models

**Library:** scikit-learn (`KMeans`, `GaussianMixture`, metrics) + matplotlib. A few lines each, no from-scratch code.

## What this project does
Clusters the same synthetic 2-D data with K-Means and with a Gaussian Mixture Model (GMM), compares them, shows the GMM's "soft" probabilities, and compares ways to choose the number of clusters.

- **Input:** none to provide (data is generated, seed 42).
- **Output:** printed scores and tables, plus `clusters.png` (true groups | K-Means | GMM).

## Run
```bash
pip install -r requirements.txt
python main.py
```

## Theory
**Clustering** groups unlabeled data so that similar points end up together. No answers are given in advance (unsupervised learning).

**K-Means** (Lloyd's algorithm): pick k, place k centroids, then repeat: (1) assign every point to its nearest centroid, (2) move each centroid to the mean of its points, until nothing changes. It minimizes **inertia** (total squared distance to the assigned centroid).
- Strengths: fast, simple, scales well.
- Weaknesses: you must choose k; results depend on the starting centroids (use `k-means++` and several `n_init`); it implicitly assumes round, similar-sized clusters; each point gets exactly one cluster; features should be scaled.

**GMM:** assumes the data came from a mix of k Gaussian (bell-shaped) blobs, each with its own mean, shape (covariance) and weight. It is fitted with the **EM algorithm**: the E-step computes each point's probability of belonging to each blob, and the M-step updates the blobs from those probabilities. It can model **elliptical** clusters of different sizes and gives **soft assignments** (`predict_proba`). `covariance_type` (`full`, `tied`, `diag`, `spherical`) controls how flexible the shapes are. K-Means behaves like a GMM restricted to equal round blobs with hard assignments.

**Choosing k:**
- *Elbow (inertia):* always falls as k grows; look for where the drop slows.
- *Silhouette score* (-1 to 1): how well points fit their own cluster vs the nearest other one. Needs no labels, but favours compact, well-separated clusters.
- *BIC/AIC (GMM only):* likelihood with a penalty for complexity; lower is better.
- Also ask whether the clusters are *useful* for your task.

**Where this shows up in AI work:**
- *Vector quantization:* K-Means centroids act as a "codebook" (the idea behind VQ-VAE codebooks and compression).
- *Vector search:* IVF-style indexes in vector databases cluster embeddings with K-Means to narrow the search.
- *Data cleaning:* cluster embeddings to find near-duplicates and balance a dataset.

## What the output shows (one run)
| Model | Adjusted Rand index vs true groups | Silhouette |
|---|---|---|
| K-Means | 0.718 | 0.614 |
| GMM | 0.942 | 0.612 |

- On these overlapping, stretched, unequal-sized groups the GMM recovered the true groups much better than K-Means.
- **Silhouette was almost identical for both**, so it did not reveal the gap. It is not a measure of agreement with the truth.
- 14 of 600 points had a GMM membership probability below 0.9. K-Means hides that uncertainty.
- Choosing k: BIC was lowest at k = 3 (the true number), but silhouette scored k = 2 highest (0.659 vs 0.614). Different criteria can disagree.

**Be careful:** the data is synthetic and I designed it (overlap and stretching) to show where K-Means struggles. Real data won't behave this neatly, and the Adjusted Rand index needs true labels, which you don't have in real clustering. One seed is a small experiment.

## Try these
1. Change `cluster_std` or the stretching matrix back to well-separated groups. Does K-Means catch up?
2. Try `covariance_type="spherical"` and compare with K-Means.
3. Scale your own data with `StandardScaler` before clustering and see what changes.
4. Cluster the digits from `sklearn.datasets.load_digits` (k = 10) and score with ARI.
5. Try `MiniBatchKMeans` for large data.

## Likely interview questions
- How does K-Means work, and what are its limits? How do you pick k?
- How does a GMM differ from K-Means? What is EM?
- When would you prefer soft over hard clustering?

## One thing learned
K-Means is a GMM with the flexibility switched off: when clusters are not round and equal-sized, the extra flexibility (and the soft probabilities) of a GMM can matter a lot.
