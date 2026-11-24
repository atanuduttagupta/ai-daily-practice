"""
P0 case study - K-Nearest Neighbors (KNN): "who does this customer resemble?" and "find similar items fast"
Libraries: scikit-learn (KNeighborsClassifier, pipelines) + FAISS (Meta's open-source vector search).
Data: IBM Telco Customer Churn (7,043 customers, open dataset, downloaded once and cached).

Business problems:
  Part 1  Predict churn by looking at the most similar past customers. KNN is only as good as its
          notion of "similar", so scaling, k and the distance metric all matter.
  Part 2  Recommendation / semantic search needs the nearest items among MILLIONS of vectors.
          Exact search gets slow; vector databases use approximate nearest neighbor (ANN) search.

Setup:  pip install -r requirements.txt
Run:    python main.py        (needs internet the first time to download the CSV)
"""

import os
import time

import faiss
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEED = 42
URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
CACHE = "telco-churn.csv"

# ======================================================================= Part 1
if not os.path.exists(CACHE):
    pd.read_csv(URL).to_csv(CACHE, index=False)
df = pd.read_csv(CACHE)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
y = (df.pop("Churn") == "Yes").astype(int)
X = df.drop(columns="customerID")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
numeric = X.select_dtypes("number").columns
categorical = X.select_dtypes("object").columns
cv = StratifiedKFold(5, shuffle=True, random_state=SEED)


def knn_pipeline(k=25, scale=True, metric="euclidean"):
    num_steps = [SimpleImputer(strategy="median")] + ([StandardScaler()] if scale else [])
    prep = ColumnTransformer([("num", make_pipeline(*num_steps), numeric),
                              ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical)])
    return make_pipeline(prep, KNeighborsClassifier(n_neighbors=k, metric=metric, weights="distance"))


def score(pipe):
    cv_auc = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc").mean()
    pipe.fit(X_train, y_train)
    return cv_auc, roc_auc_score(y_train, pipe.predict_proba(X_train)[:, 1]), roc_auc_score(y_test, pipe.predict_proba(X_test)[:, 1])


print(f"{len(df)} customers, {y.mean():.0%} churned.\n")
print("=== Part 1a) Does feature scaling matter? (k=25) ===")
for label, pipe in [("no scaling", knn_pipeline(scale=False)), ("with StandardScaler", knn_pipeline(scale=True))]:
    cv_auc, _, test_auc = score(pipe)
    print(f"{label:<22} CV AUC {cv_auc:.3f}   test AUC {test_auc:.3f}")
print("Without scaling, big-number columns (TotalCharges, MonthlyCharges, tenure) swamp the 0/1 columns in the distance.\n")

print("=== Part 1b) Choosing k (bias-variance trade-off) ===")
print(f"{'k':>4} {'train AUC':>10} {'CV AUC':>8} {'test AUC':>9}")
for k in (1, 5, 15, 25, 51, 101, 201):
    cv_auc, train_auc, test_auc = score(knn_pipeline(k=k))
    print(f"{k:>4} {train_auc:>10.3f} {cv_auc:>8.3f} {test_auc:>9.3f}")
print("k=1 memorizes (CV far below the others). Train AUC is ~1.0 for every k because each training point is its own")
print("nearest neighbor (distance 0), so ignore it. Here CV AUC kept rising up to k=201; try larger k to find where it turns down.\n")

print("=== Part 1c) Distance metric (k=25) ===")
for metric in ("euclidean", "manhattan", "cosine"):
    cv_auc, _, test_auc = score(knn_pipeline(metric=metric))
    print(f"{metric:<10} CV AUC {cv_auc:.3f}   test AUC {test_auc:.3f}")

lr = make_pipeline(ColumnTransformer([("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), numeric),
                                      ("cat", OneHotEncoder(handle_unknown="ignore"), categorical)]),
                   LogisticRegression(max_iter=1000)).fit(X_train, y_train)
print(f"\nFor context, logistic regression test AUC: {roc_auc_score(y_test, lr.predict_proba(X_test)[:, 1]):.3f}\n")

# ======================================================================= Part 2
print("=== Part 2) Exact vs approximate nearest neighbors at scale ===")
N, D, NQ = 100_000, 128, 200
rng = np.random.default_rng(SEED)
centers = rng.normal(size=(200, D)).astype("float32")                    # 200 "product families"
items = (centers[rng.integers(0, 200, N)] + 0.6 * rng.normal(size=(N, D))).astype("float32")
queries = (items[rng.integers(0, N, NQ)] + 0.1 * rng.normal(size=(NQ, D))).astype("float32")
print(f"{N:,} item vectors of {D} dimensions (stand-ins for embeddings), {NQ} queries, top-10 neighbors\n")


def search_one_by_one(index, queries, k=10):
    """Online search handles one request at a time, so measure latency per query."""
    start = time.perf_counter()
    ids = np.vstack([index.search(q[None, :], k)[1] for q in queries])
    return ids, (time.perf_counter() - start) / len(queries) * 1000


exact = faiss.IndexFlatL2(D)
exact.add(items)
truth, exact_ms = search_one_by_one(exact, queries)

print(f"{'method':<22}{'build s':>9}{'ms/query':>10}{'recall@10':>11}")
print(f"{'exact (brute force)':<22}{'0.0':>9}{exact_ms:>10.2f}{1.0:>11.3f}")
start = time.perf_counter()
hnsw = faiss.IndexHNSWFlat(D, 16)          # HNSW: a layered "small-world" graph of neighbors
hnsw.hnsw.efConstruction = 100
hnsw.add(items)
build = time.perf_counter() - start
for ef in (8, 32, 128):
    hnsw.hnsw.efSearch = ef                # how widely the graph is explored per query
    ids, ms = search_one_by_one(hnsw, queries)
    recall = np.mean([len(set(a) & set(b)) / 10 for a, b in zip(ids, truth)])
    print(f"{'HNSW efSearch=' + str(ef):<22}{build:>9.1f}{ms:>10.2f}{recall:>11.3f}")
print("\nrecall@10 = share of the true 10 nearest neighbors that the approximate search found.")
