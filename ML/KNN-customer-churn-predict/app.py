"""KNN & Similarity Search Lab — Streamlit UI for main.py's case study.
Run: python -m streamlit run app.py
Place telco-churn.csv beside this file, or allow the app to download it once.
"""
from __future__ import annotations

import time
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEED = 42
DATA_URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
DATA_FILE = Path(__file__).resolve().parent / "telco-churn.csv"

st.set_page_config(page_title="KNN & Similarity Search Lab", page_icon="🔎", layout="wide")
st.title("🔎 KNN & Similarity Search Lab")
st.caption("P0 case study | Customer churn prediction and fast nearest-neighbor retrieval")
st.markdown("Explore two related ideas: **KNN classification** for customer churn and **FAISS vector retrieval** for fast similarity search.")


@st.cache_data(show_spinner="Loading Telco Churn data...")
def load_data():
    if DATA_FILE.exists():
        return pd.read_csv(DATA_FILE)
    try:
        frame = pd.read_csv(DATA_URL)
        frame.to_csv(DATA_FILE, index=False)
        return frame
    except Exception as exc:
        raise RuntimeError("telco-churn.csv was not found and could not be downloaded. Place it beside this app or check internet access.") from exc


@st.cache_data(show_spinner=False)
def split_data():
    frame = load_data().copy()
    frame["TotalCharges"] = pd.to_numeric(frame["TotalCharges"], errors="coerce")
    y = (frame.pop("Churn") == "Yes").astype(int)
    customer_ids = frame.pop("customerID")
    return train_test_split(frame, y, customer_ids, test_size=0.2, stratify=y, random_state=SEED)


def preprocessor(X, scale=True):
    numeric = X.select_dtypes(include="number").columns
    categorical = X.select_dtypes(include="object").columns
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        numeric_steps.append(("scaler", StandardScaler()))
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("num", Pipeline(numeric_steps), numeric),
        ("cat", categorical_pipe, categorical),
    ])


def make_knn(X, k, metric, scale, weights, pca_components):
    steps = [("preprocessor", preprocessor(X, scale))]
    if pca_components is not None:
        steps.append(("pca", PCA(n_components=pca_components, random_state=SEED)))
    steps.append(("knn", KNeighborsClassifier(n_neighbors=k, metric=metric, weights=weights, n_jobs=-1)))
    return Pipeline(steps)


@st.cache_data(show_spinner="Fitting KNN and Logistic Regression...")
def evaluate_knn(k, metric, scale, weights, pca_components, run_cv):
    X_train, X_test, y_train, y_test, id_train, id_test = split_data()
    model = make_knn(X_train, k, metric, scale, weights, pca_components)
    cv_auc = None
    if run_cv:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        cv_auc = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=1).mean()
    model.fit(X_train, y_train)
    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    lr = Pipeline([
        ("preprocessor", preprocessor(X_train, True)),
        ("classifier", LogisticRegression(max_iter=1000)),
    ])
    lr.fit(X_train, y_train)
    lr_probabilities = lr.predict_proba(X_test)[:, 1]
    return {
        "model": model, "lr": lr, "X_train": X_train, "X_test": X_test,
        "y_train": y_train, "y_test": y_test, "id_train": id_train,
        "id_test": id_test, "knn_auc": roc_auc_score(y_test, probabilities),
        "lr_auc": roc_auc_score(y_test, lr_probabilities),
        "accuracy": accuracy_score(y_test, predictions),
        "matrix": confusion_matrix(y_test, predictions), "cv_auc": cv_auc,
    }


@st.cache_resource(show_spinner="Building FAISS indexes...")
def build_faiss():
    n, dimensions, n_queries = 100_000, 128, 200
    rng = np.random.default_rng(SEED)
    centers = rng.normal(size=(200, dimensions)).astype("float32")
    items = (centers[rng.integers(0, 200, n)] + 0.6 * rng.normal(size=(n, dimensions))).astype("float32")
    queries = (items[rng.integers(0, n, n_queries)] + 0.1 * rng.normal(size=(n_queries, dimensions))).astype("float32")
    exact = faiss.IndexFlatL2(dimensions)
    exact.add(items)
    hnsw = faiss.IndexHNSWFlat(dimensions, 16)
    hnsw.hnsw.efConstruction = 100
    start = time.perf_counter()
    hnsw.add(items)
    build_seconds = time.perf_counter() - start
    return exact, hnsw, queries, build_seconds


def timed_search(index, queries, k=10):
    start = time.perf_counter()
    ids = np.vstack([index.search(q[None, :], k)[1] for q in queries])
    return ids, (time.perf_counter() - start) / len(queries) * 1000


home, churn, search, revise = st.tabs(["🏠 Overview", "👥 Churn Explorer", "⚡ FAISS Search Lab", "📘 Learn & Revise"])

with home:
    st.subheader("Two business problems")
    a, b, c = st.columns(3)
    a.metric("Customers", "7,043")
    b.metric("Churned", "1,869")
    c.metric("Vector dimensions", "128")
    left, right = st.columns(2)
    with left:
        st.markdown("**1. Customer churn prediction**\n\nUse similar past customers to estimate whether a customer may leave. Explore scaling, k, distance metrics and PCA.")
    with right:
        st.markdown("**2. Fast similar-item search**\n\nCompare exact nearest-neighbor retrieval with FAISS HNSW using query latency and recall@10.")
    st.info("The FAISS experiment uses synthetic clustered vectors as stand-ins for embeddings, not real customer or product embeddings.")
    st.markdown("**Suggested path:** Start with Churn Explorer, change one setting at a time, then investigate the speed-versus-recall trade-off in FAISS Search Lab.")

with churn:
    st.subheader("Part 1 — Customer churn prediction")
    try:
        raw = load_data()
        X_train, X_test, y_train, y_test, id_train, id_test = split_data()
    except Exception as exc:
        st.error(str(exc))
        st.stop()
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Customers", f"{len(raw):,}")
    m2.metric("Churned", f"{(raw.Churn == 'Yes').sum():,}")
    m3.metric("Churn rate", f"{(raw.Churn == 'Yes').mean():.1%}")
    m4.metric("Train / test", f"{len(X_train):,} / {len(X_test):,}")
    with st.expander("Explore the dataset"):
        st.bar_chart(raw["Churn"].value_counts().rename(index={"No": "Stayed", "Yes": "Churned"}))
        st.dataframe(raw.head(10), use_container_width=True)
        st.write("Blank TotalCharges values:", int(pd.to_numeric(raw.TotalCharges, errors="coerce").isna().sum()))

    st.markdown("### Configure your KNN experiment")
    c1, c2, c3 = st.columns(3)
    with c1:
        k = st.slider("Number of neighbors (k)", 1, 301, 101, step=2)
        metric = st.selectbox("Distance metric", ["euclidean", "manhattan", "cosine"])
    with c2:
        scale = st.checkbox("Scale numeric features", value=True)
        weights = st.selectbox("Neighbor weighting", ["distance", "uniform"])
    with c3:
        pca_label = st.selectbox("Dimensionality reduction", ["None", "PCA — 10 components", "PCA — 20 components"])
        run_cv = st.checkbox("Run 5-fold cross-validation", value=False, help="Adds computation time.")
    pca_n = None if pca_label == "None" else (10 if "10" in pca_label else 20)

    if st.button("Run churn experiment", type="primary", use_container_width=True):
        try:
            st.session_state["knn_result"] = evaluate_knn(k, metric, scale, weights, pca_n, run_cv)
        except Exception as exc:
            st.error(f"Experiment failed: {exc}")
    result = st.session_state.get("knn_result")
    if result:
        st.markdown("### Results")
        r1, r2, r3 = st.columns(3)
        r1.metric("KNN test AUC", f"{result['knn_auc']:.3f}")
        r2.metric("Logistic Regression AUC", f"{result['lr_auc']:.3f}")
        r3.metric("KNN accuracy", f"{result['accuracy']:.3f}")
        if result["cv_auc"] is not None:
            st.metric("KNN 5-fold CV AUC", f"{result['cv_auc']:.3f}")
        st.bar_chart(pd.DataFrame({"Test AUC": [result["knn_auc"], result["lr_auc"]]}, index=["KNN", "Logistic Regression"]))
        st.markdown("**KNN confusion matrix**")
        st.dataframe(pd.DataFrame(result["matrix"], index=["Actual stayed", "Actual churned"], columns=["Predicted stayed", "Predicted churned"]), use_container_width=True)
        if result["knn_auc"] > result["lr_auc"]:
            st.success("KNN has the higher test AUC in this run. Check cross-validation before concluding it generalizes better.")
        else:
            st.info("Logistic Regression has the higher or equal test AUC in this run. This can indicate that a global linear model fits this representation well.")
        st.caption("AUC measures ranking quality across thresholds. Accuracy uses a 0.5 threshold. The split is stratified with random_state=42.")

        st.markdown("### Inspect a test customer and its neighbors")
        test_indices = result["X_test"].index
        pos = st.number_input("Test-row position", min_value=0, max_value=len(test_indices)-1, value=0, step=1)
        row_id = test_indices[int(pos)]
        customer = result["X_test"].loc[[row_id]]
        actual = int(result["y_test"].loc[row_id])
        probability = float(result["model"].predict_proba(customer)[0, 1])
        p1, p2, p3 = st.columns(3)
        p1.metric("Actual outcome", "Churn" if actual else "Stayed")
        p2.metric("Predicted churn probability", f"{probability:.1%}")
        p3.metric("Predicted class", "Churn" if probability >= 0.5 else "Stayed")
        with st.expander("Customer profile"):
            st.dataframe(customer, use_container_width=True)
        fitted = result["model"]
        transformed = fitted.named_steps["preprocessor"].transform(customer)
        if "pca" in fitted.named_steps:
            transformed = fitted.named_steps["pca"].transform(transformed)
        distances, positions = fitted.named_steps["knn"].kneighbors(transformed, n_neighbors=min(k, len(result["X_train"])))
        neighbor_rows = result["X_train"].iloc[positions[0]].copy()
        neighbor_rows.insert(0, "Customer ID", result["id_train"].loc[neighbor_rows.index].values)
        neighbor_rows.insert(1, "Actual churn", result["y_train"].loc[neighbor_rows.index].map({0: "No", 1: "Yes"}).values)
        neighbor_rows.insert(2, "Distance", distances[0])
        st.dataframe(neighbor_rows.head(10), use_container_width=True)
        st.caption("Shows the 10 nearest training customers. Distances reflect the selected preprocessing and metric.")
    else:
        st.info("Choose settings and click **Run churn experiment**.")

with search:
    st.subheader("Part 2 — Exact vs approximate nearest-neighbor search")
    st.write("The experiment creates 100,000 synthetic vectors with 128 dimensions and 200 queries. Exact search supplies the ground truth; HNSW approximates the top 10 neighbors.")
    ef = st.select_slider("HNSW efSearch", options=[8, 16, 32, 64, 128, 256], value=32, help="Higher values explore more candidates and usually improve recall at a latency cost.")
    if st.button("Build indexes and run search", type="primary", use_container_width=True):
        try:
            exact, hnsw, queries, build_seconds = build_faiss()
            with st.spinner("Running exact and HNSW searches..."):
                truth, exact_ms = timed_search(exact, queries)
                hnsw.hnsw.efSearch = int(ef)
                found, hnsw_ms = timed_search(hnsw, queries)
                recall = float(np.mean([len(set(x) & set(y)) / 10 for x, y in zip(truth, found)]))
            st.session_state["faiss_result"] = (exact_ms, hnsw_ms, recall, build_seconds, ef)
        except Exception as exc:
            st.error(f"FAISS experiment failed: {exc}")
    faiss_result = st.session_state.get("faiss_result")
    if faiss_result:
        exact_ms, hnsw_ms, recall, build_seconds, used_ef = faiss_result
        f1, f2, f3 = st.columns(3)
        f1.metric("Exact search", f"{exact_ms:.3f} ms/query")
        f2.metric("HNSW search", f"{hnsw_ms:.3f} ms/query")
        f3.metric("Recall@10", f"{recall:.3f}")
        st.metric("Approximate-search speed-up", f"{exact_ms / hnsw_ms:.1f}×" if hnsw_ms else "N/A")
        st.write(f"HNSW build time: **{build_seconds:.2f} seconds** | efSearch: **{used_ef}**")
        st.caption("Timings vary by hardware and system load. Index construction is measured separately from query latency.")
    else:
        st.info("Click the button to generate vectors, build the indexes and compare search.")

with revise:
    st.subheader("Learn & revise")
    topics = {
        "How KNN works": "KNN is a lazy, instance-based method. It stores examples and finds the k closest examples for a new observation. Classification uses a vote; distance weighting gives closer neighbors more influence.",
        "Why scaling matters": "Distance is affected by feature magnitude. A large-valued column can dominate binary or small-valued columns. Scaling numeric features helps make distance calculations more balanced.",
        "Choosing k": "Small k can be sensitive to noise (high variance). Large k smooths predictions but can miss local structure (higher bias). Use validation or cross-validation to select k.",
        "Curse of dimensionality": "As dimensions increase, distances can become less discriminative. Feature selection, dimensionality reduction or useful learned embeddings may help.",
        "HNSW": "HNSW is a hierarchical graph-based approximate nearest-neighbor method. Search navigates graph links to find promising candidates. efSearch controls how widely the graph is explored.",
        "Recall@k": "Recall@k is the fraction of the true k nearest neighbors returned by approximate search. Recall@10 of 0.9 means it found, on average, 9 of the exact top 10 neighbors.",
    }
    for heading, explanation in topics.items():
        with st.expander(heading):
            st.write(explanation)
    st.markdown("### Interview questions")
    for i, question in enumerate([
        "How does KNN work, and why is it called a lazy learner?",
        "Why must numeric features be scaled for distance-based learning?",
        "How do you select k?",
        "What is the curse of dimensionality?",
        "How does approximate nearest-neighbor search differ from exact search?",
        "How does HNSW work, and what does efSearch control?",
        "What does recall@10 measure?",
        "Why can Logistic Regression outperform KNN on a churn dataset?",
    ], 1):
        st.write(f"{i}. {question}")
    st.markdown("### Suggested experiments")
    st.markdown("1. Try larger k values and observe cross-validation AUC.\n2. Compare uniform and distance weighting.\n3. Add PCA and compare AUC.\n4. Compare Euclidean, Manhattan and cosine distance.\n5. Change HNSW efSearch and observe recall versus latency.")

st.divider()
st.caption("Educational case study | scikit-learn + FAISS + Streamlit | Results vary with data and hardware.")
