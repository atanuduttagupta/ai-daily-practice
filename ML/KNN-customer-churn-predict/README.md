# P0 case study — K-Nearest Neighbors (KNN): "who does this customer resemble?" and "find similar items fast"

**Libraries:** scikit-learn (`KNeighborsClassifier`, pipelines) and FAISS (Meta's open-source vector search library).
**Data:** IBM Telco Customer Churn, 7,043 customers (open dataset, downloaded once from GitHub and cached as `telco-churn.csv`). Part 2 uses generated vectors.

## The business problems
1. **Churn prediction by similarity.** Predict whether a customer will leave by looking at the most similar past customers. KNN is only as good as its idea of "similar", so we test scaling, the number of neighbors k, and the distance metric.
2. **Fast similar-item search.** Recommendations ("customers also viewed") and RAG retrieval need the nearest items among millions of vectors. We compare exact search with approximate nearest neighbor (ANN) search.

- **Input:** none to provide (the CSV downloads automatically; internet needed the first time).
- **Output:** tables for scaling, k, distance metric, and exact vs approximate search speed and accuracy.

## Run
```bash
pip install -r requirements.txt
python main.py
```
It takes about a minute (most of it building the ANN index).

## Theory
**KNN** is "lazy" or instance-based learning: there is no training step. It stores all the examples, and to predict for a new point it finds the **k closest** stored points and takes a vote (classification) or an average (regression). With `weights="distance"`, closer neighbors count more.

**Distance metrics:** *Euclidean* (straight-line), *Manhattan* (sum of absolute differences), *cosine* (angle between vectors, ignores length; the usual choice for text and embeddings).

**Things that decide whether KNN works:**
- **Feature scaling is essential.** A column measured in thousands dominates a 0/1 column in the distance. Always scale numeric features (or use embeddings already on a common scale).
- **Choosing k (bias-variance):** k=1 follows every point, including noise (high variance, overfits). Larger k smooths the decision (more bias). Pick k with cross-validation.
- **Curse of dimensionality:** in many dimensions, distances between points become similar, so "nearest" means less. Use fewer, more meaningful features, PCA, or learned embeddings.
- **Cost:** every prediction compares against all N stored points, so query time grows with N and the whole dataset sits in memory.

**Approximate nearest neighbor (ANN):** exact search is O(N x dimensions) per query. ANN gives up a little accuracy for a big speed-up. Common methods:
- **HNSW:** a layered graph where each point links to near neighbors; a query hops greedily through the graph (used here).
- **IVF:** cluster the vectors with K-Means, then search only the few nearest clusters.
- **PQ / quantization:** compress vectors to save memory.

Vector databases (Chroma, Qdrant, Milvus, Pinecone) and libraries like FAISS are built on these ideas. The quality measure is **recall@k**: what share of the true k nearest neighbors the approximate search returned. You tune the trade-off: more search effort means higher recall but slower queries.

**Where this shows up in AI work:** RAG retrieval is KNN in embedding space: embed the question, fetch the top-k nearest chunks (usually by cosine similarity or inner product). Recommenders and duplicate detection work the same way.

## What the output shows (one run, single CPU core)
**Part 1: churn (AUC: 0.5 = random, 1.0 = perfect)**
| Setting | CV AUC | Test AUC |
|---|---|---|
| no scaling, k=25 | 0.781 | 0.791 |
| scaled, k=25 | 0.814 | 0.808 |
| scaled, k=1 | 0.643 | 0.663 |
| scaled, k=101 | 0.825 | 0.818 |
| scaled, k=201 | 0.827 | 0.818 |
| metric: Euclidean / Manhattan / cosine (k=25) | 0.814 / 0.811 / 0.787 | 0.808 / 0.803 / 0.790 |

- Scaling helped, and k=1 was clearly the worst. Cross-validation AUC kept rising up to k=201, so the best k is large here and could be larger still.
- Train AUC is about 1.0 for every k. With distance weighting each training point is its own nearest neighbor, so that number is meaningless. Trust the cross-validation and test scores.
- Logistic regression (test AUC 0.842) beat the best KNN (0.818). KNN is a strong idea for search, but not always the best classifier.

**Part 2: 100,000 vectors, 128 dimensions, one query at a time**
| Method | ms per query | recall@10 |
|---|---|---|
| exact (brute force) | 2.87 | 1.000 |
| HNSW, efSearch=8 | 0.03 | 0.853 |
| HNSW, efSearch=32 | 0.06 | 0.979 |
| HNSW, efSearch=128 | 0.14 | 0.999 |

HNSW took about 10 seconds to build. At efSearch=32 it was roughly 50x faster than brute force while finding 98% of the true neighbors.

**Be careful:** Part 2 uses synthetic clustered vectors, not real embeddings, and timings depend on your hardware. At 100,000 vectors brute force is already only a few milliseconds, and batching many queries together makes exact search much faster too, so ANN matters most at millions of vectors and strict latency limits. HNSW also uses extra memory and must be rebuilt or updated as data changes. Part 1 is one dataset and one split.

## Try these
1. Add `PCA` (e.g. 10 components) before KNN and see how AUC and speed change.
2. Try `k` = 301, 501 to find where CV AUC turns down.
3. Use `weights="uniform"` and compare train AUC with the distance-weighted run.
4. In Part 2, change `M` (graph connections) from 16 to 32 and `efConstruction`; watch build time and recall.
5. Try `faiss.IndexIVFFlat` (clustering-based ANN) instead of HNSW.

## Likely interview questions
- How does KNN work? Why must you scale features? How do you choose k?
- What is the curse of dimensionality? Why is KNN slow at scale?
- What is ANN search, how does HNSW work, and what does recall@k mean?

## One thing learned
KNN's whole model is its definition of "similar", so scaling and metric matter more than the algorithm. At scale, you accept approximate answers (here 98% recall) to make that search fast.
