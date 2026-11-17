"""
P0 - Track G: K-Means & Gaussian Mixture Models
Run: python -m streamlit run app.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.mixture import GaussianMixture

st.set_page_config(page_title="K-Means vs GMM Lab", page_icon="🔬", layout="wide")
st.title("🔬 K-Means & Gaussian Mixture Models")
st.caption("Unsupervised Learning Lab · Hard assignments, soft probabilities and model selection")

with st.sidebar:
    st.header("Experiment settings")
    seed = st.number_input("Random seed", 0, 99999, 42)
    n_samples = st.slider("Number of samples", 200, 1500, 600, 100)
    true_k = st.slider("Generated groups", 2, 6, 3)
    model_k = st.slider("Clusters / components to fit", 2, 6, 3)
    covariance_type = st.selectbox("GMM covariance type", ["full", "tied", "diag", "spherical"])
    st.divider()
    st.markdown("**Synthetic data spread**")
    spreads = [
        st.slider("Group 1 spread", 0.5, 4.0, 1.6, 0.1),
        st.slider("Group 2 spread", 0.5, 5.0, 3.2, 0.1),
        st.slider("Group 3 spread", 0.5, 4.0, 1.0, 0.1),
    ]
    stretch = st.checkbox("Apply elliptical stretching", True)

stds = spreads[:true_k] + ([2.0] * max(0, true_k - 3))
X, y_true = make_blobs(n_samples=int(n_samples), centers=true_k,
                       cluster_std=stds, random_state=int(seed))
if stretch:
    X = X @ np.array([[0.6, -0.8], [-0.1, 1.0]])

kmeans = KMeans(n_clusters=model_k, n_init=10, random_state=int(seed)).fit(X)
gmm = GaussianMixture(n_components=model_k, covariance_type=covariance_type,
                      n_init=5, random_state=int(seed), reg_covar=1e-6).fit(X)
km_labels = kmeans.labels_
gmm_labels = gmm.predict(X)
probs = gmm.predict_proba(X)

st.subheader("1. Model comparison")
st.markdown("K-Means gives each point one cluster (**hard assignment**). GMM estimates a probability for each component (**soft assignment**).")
ari_km = adjusted_rand_score(y_true, km_labels)
ari_gmm = adjusted_rand_score(y_true, gmm_labels)
sil_km = silhouette_score(X, km_labels)
sil_gmm = silhouette_score(X, gmm_labels)

a, b, c, d = st.columns(4)
a.metric("Observations", f"{len(X):,}")
b.metric("Generated groups", true_k)
c.metric("Fitted clusters", model_k)
d.metric("GMM covariance", covariance_type)
m1, m2 = st.columns(2)
m1.metric("K-Means · ARI", f"{ari_km:.3f}")
m1.metric("K-Means · Silhouette", f"{sil_km:.3f}")
m2.metric("GMM · ARI", f"{ari_gmm:.3f}")
m2.metric("GMM · Silhouette", f"{sil_gmm:.3f}")

st.subheader("2. Visual comparison")
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharex=True, sharey=True)
for ax, labels, title in [
    (axes[0], y_true, "True generated groups"),
    (axes[1], km_labels, "K-Means assignments"),
    (axes[2], gmm_labels, "GMM assignments"),
]:
    ax.scatter(X[:, 0], X[:, 1], c=labels, s=13, cmap="viridis", alpha=0.8)
    ax.set_title(title)
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
axes[1].scatter(*kmeans.cluster_centers_.T, c="red", marker="x", s=130, linewidths=3, label="Centroids")
axes[1].legend()
fig.tight_layout()
st.pyplot(fig, use_container_width=True)
plt.close(fig)

st.subheader("3. GMM soft assignments")
st.write("Observations with the lowest maximum membership probability are shown first.")
uncertain = np.argsort(probs.max(axis=1))[:min(10, len(X))]
prob_df = pd.DataFrame(probs[uncertain], columns=[f"Component {i}" for i in range(model_k)])
prob_df.insert(0, "Observation", uncertain)
prob_df["Predicted component"] = gmm_labels[uncertain]
prob_df["Maximum probability"] = probs.max(axis=1)[uncertain]
st.dataframe(prob_df.round(3), use_container_width=True)
st.info(f"{int((probs.max(axis=1) < 0.9).sum())} of {len(X)} observations have maximum GMM membership probability below 0.90.")

st.subheader("4. Choosing the number of clusters")
st.write("Compare K-Means inertia and silhouette with GMM BIC. Inertia falls as k increases; use it with other evidence.")
candidate_ks = range(2, min(7, len(X)))
rows = []
for k in candidate_ks:
    km = KMeans(n_clusters=k, n_init=10, random_state=int(seed)).fit(X)
    gm = GaussianMixture(n_components=k, covariance_type=covariance_type,
                         n_init=5, random_state=int(seed), reg_covar=1e-6).fit(X)
    rows.append({"k": k, "K-Means inertia": km.inertia_,
                 "K-Means silhouette": silhouette_score(X, km.labels_),
                 "GMM BIC (lower is better)": gm.bic(X)})
kdf = pd.DataFrame(rows)
st.dataframe(kdf.style.format({"K-Means inertia": "{:,.1f}",
                               "K-Means silhouette": "{:.3f}",
                               "GMM BIC (lower is better)": "{:,.1f}"}),
             use_container_width=True)
best_bic = int(kdf.loc[kdf["GMM BIC (lower is better)"].idxmin(), "k"])
best_sil = int(kdf.loc[kdf["K-Means silhouette"].idxmax(), "k"])
st.write(f"**Lowest GMM BIC:** k = {best_bic}  ·  **Highest K-Means silhouette:** k = {best_sil}")
st.caption("Criteria can disagree because they measure different properties. Generated labels are known here only because this is a synthetic teaching dataset.")

with st.expander("5. Concepts and interview reminders"):
    st.markdown("""
    **K-Means**
    - Hard assignment: one cluster label per point.
    - Alternates nearest-centroid assignment and mean-centroid updates.
    - Minimizes within-cluster squared distances (inertia).
    - Often works best for compact, roughly spherical groups.

    **Gaussian Mixture Model**
    - `predict_proba()` returns component membership probabilities.
    - Uses EM: E-step estimates responsibilities; M-step updates parameters.
    - Covariance controls component shape, orientation and spread.
    - `predict()` returns the most likely component.

    **Evaluation**
    - ARI compares predictions with known reference labels.
    - Silhouette measures geometric cohesion and separation without true labels.
    - BIC balances GMM likelihood and complexity; lower is preferred.
    - No single metric should decide the final clustering.
    """)

st.divider()
st.caption("Synthetic experiment for learning. Results are specific to the selected data and settings, not a universal benchmark.")
