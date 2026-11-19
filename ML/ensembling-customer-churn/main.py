"""
P0 case study - Bagging & Ensembling: predicting telecom customer churn
Library: scikit-learn (BaggingClassifier, RandomForest, HistGradientBoosting, Voting, Stacking).
Data: IBM Telco Customer Churn (7,043 customers, open dataset, downloaded once and cached).

Business problem: a telecom company can only call a limited number of customers. Which
customers are most likely to leave, and does combining models find them better than one model?

Setup:  pip install -r requirements.txt
Run:    python main.py        (needs internet the first time to download the CSV)
"""

import os

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (BaggingClassifier, HistGradientBoostingClassifier,
                              RandomForestClassifier, StackingClassifier, VotingClassifier)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
CACHE = "telco-churn.csv"
SEED = 42

# ---- data: minimal cleaning, then a stratified train/test split
if not os.path.exists(CACHE):
    pd.read_csv(URL).to_csv(CACHE, index=False)
df = pd.read_csv(CACHE)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")   # blanks -> NaN
y = (df.pop("Churn") == "Yes").astype(int)
X = df.drop(columns="customerID")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
print(f"{len(df)} customers, {y.mean():.0%} churned. Train {len(X_train)} / test {len(X_test)}\n")

numeric = X.select_dtypes("number").columns
categorical = X.select_dtypes("object").columns
prep = ColumnTransformer([
    ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), numeric),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
])

# ---- the models. Every one sits in a pipeline so preprocessing is learned inside each CV fold.
logreg = LogisticRegression(max_iter=1000)
forest = RandomForestClassifier(n_estimators=200, min_samples_leaf=3, random_state=SEED, n_jobs=-1)
boost = HistGradientBoostingClassifier(random_state=SEED)
models = {
    "single decision tree": DecisionTreeClassifier(random_state=SEED),
    "logistic regression": logreg,
    "bagged trees (100)": BaggingClassifier(DecisionTreeClassifier(), n_estimators=100, random_state=SEED, n_jobs=-1),
    "random forest": forest,
    "gradient boosting": boost,
    "voting (LR+RF+GB)": VotingClassifier([("lr", logreg), ("rf", forest), ("gb", boost)], voting="soft"),
    "stacking (LR+RF+GB)": StackingClassifier([("lr", logreg), ("rf", forest), ("gb", boost)],
                                              final_estimator=LogisticRegression(max_iter=1000), cv=3),
}

def top20_capture(y_true, scores):
    """Share of all churners found among the 20% of customers ranked riskiest."""
    top = np.argsort(scores)[::-1][: int(0.2 * len(scores))]
    return y_true.iloc[top].sum() / y_true.sum()

cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
rows = []
for name, model in models.items():
    pipe = make_pipeline(prep, model)
    auc = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc")     # 5-fold CV on train only
    pipe.fit(X_train, y_train)
    p_train = pipe.predict_proba(X_train)[:, 1]
    p_test = pipe.predict_proba(X_test)[:, 1]
    rows.append({"model": name, "CV AUC": auc.mean(), "CV std": auc.std(),
                 "train AUC": roc_auc_score(y_train, p_train), "test AUC": roc_auc_score(y_test, p_test),
                 "top-20% capture": top20_capture(y_test, p_test)})
    print(f"done: {name}")

result = pd.DataFrame(rows).sort_values("test AUC", ascending=False)
print("\n=== Results (AUC: 0.5 = random, 1.0 = perfect) ===")
print(result.round(3).to_string(index=False))

# ---- bagging's free validation: out-of-bag score
oob = make_pipeline(prep, BaggingClassifier(DecisionTreeClassifier(), n_estimators=100, oob_score=True,
                                            random_state=SEED, n_jobs=-1)).fit(X_train, y_train)
print(f"\nBagging out-of-bag accuracy (needs no separate validation set): {oob[-1].oob_score_:.3f}")
best = result.iloc[0]
print(f"Business view: calling a random 20% of customers finds about 20% of churners. "
      f"'{best['model']}' finds {best['top-20% capture']:.0%} (about {best['top-20% capture'] / 0.2:.1f}x better).")
