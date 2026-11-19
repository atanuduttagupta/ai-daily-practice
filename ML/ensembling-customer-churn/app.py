from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (BaggingClassifier, RandomForestClassifier,
    HistGradientBoostingClassifier, VotingClassifier, StackingClassifier)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")
URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
CSV = Path(__file__).with_name("telco-churn.csv")

st.set_page_config(page_title="Telecom Churn Ensemble Lab", page_icon="📡", layout="wide")
st.title("📡 Telecom Customer Churn — Bagging & Ensembling Lab")
st.caption("P0 Case Study · Predicting which customers are most likely to leave")
st.write("A telecom company can only call a limited number of customers with retention offers. Can machine-learning models rank customers by churn risk, and do ensembles improve on individual models?")

@st.cache_data(show_spinner=False)
def load_data():
    raw = pd.read_csv(CSV if CSV.exists() else URL)
    if not CSV.exists():
        raw.to_csv(CSV, index=False)
    raw = raw.copy()
    raw["TotalCharges"] = pd.to_numeric(raw["TotalCharges"], errors="coerce")
    y = (raw.pop("Churn").astype(str).str.strip() == "Yes").astype(int)
    X = raw.drop(columns=["customerID"])
    return X, y, raw

try:
    with st.spinner("Loading IBM Telco churn data..."):
        X, y, raw = load_data()
except Exception as e:
    st.error(f"Dataset loading failed: {e}")
    st.info("The CSV downloads on first run. Check internet access or put telco-churn.csv beside app.py.")
    st.stop()

st.sidebar.header("Experiment settings")
seed = st.sidebar.number_input("Random seed", 0, 99999, 42)
test_size = st.sidebar.slider("Test set fraction", .15, .35, .20, .05)
folds = st.sidebar.selectbox("Stratified CV folds", [3, 4, 5, 6, 7, 8], index=2)
top_fraction = st.sidebar.slider("Retention team contact capacity", .05, .40, .20, .05)
n_bag = st.sidebar.selectbox("Bagging trees", [10, 50, 100, 200, 300], index=2)
n_rf = st.sidebar.selectbox("Random forest trees", [50, 100, 200, 300], index=2)

st.subheader("1. Dataset overview")
a,b,c,d = st.columns(4)
a.metric("Customers", f"{len(X):,}")
b.metric("Churned", f"{int(y.sum()):,}")
c.metric("Churn rate", f"{y.mean():.1%}")
d.metric("Input features", str(X.shape[1]))
left,right = st.columns(2)
with left:
    counts = y.map({0:"Stayed",1:"Churned"}).value_counts().reindex(["Stayed","Churned"])
    fig,ax = plt.subplots(figsize=(5,3))
    ax.bar(counts.index, counts.values)
    ax.set_ylabel("Customers"); ax.set_title("Churn distribution"); ax.grid(axis="y",alpha=.2)
    st.pyplot(fig,use_container_width=True); plt.close(fig)
with right:
    sample = raw.head(7).copy()
    sample["Churn"] = y.head(7).map({0:"No",1:"Yes"}).values
    st.markdown("**Sample records**")
    st.dataframe(sample,use_container_width=True,hide_index=True)

st.info("Business goal: identify a high share of future churners within the limited group of customers the company can contact.")

def make_preprocessor(frame):
    nums = frame.select_dtypes(include=["number"]).columns
    cats = frame.select_dtypes(include=["object","category","bool"]).columns
    num_pipe = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
    cat_pipe = make_pipeline(SimpleImputer(strategy="most_frequent"),
                             OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    return ColumnTransformer([("num",num_pipe,nums),("cat",cat_pipe,cats)])

def make_models(seed, n_bag, n_rf):
    lr = LogisticRegression(max_iter=1000)
    rf = RandomForestClassifier(n_estimators=n_rf,min_samples_leaf=3,random_state=seed,n_jobs=-1)
    gb = HistGradientBoostingClassifier(random_state=seed)
    return {
        "single decision tree": DecisionTreeClassifier(random_state=seed),
        "logistic regression": lr,
        f"bagged trees ({n_bag})": BaggingClassifier(estimator=DecisionTreeClassifier(random_state=seed),n_estimators=n_bag,random_state=seed,n_jobs=-1),
        "random forest": rf,
        "gradient boosting": gb,
        "voting (LR+RF+GB)": VotingClassifier([("lr",lr),("rf",rf),("gb",gb)],voting="soft"),
        "stacking (LR+RF+GB)": StackingClassifier([("lr",lr),("rf",rf),("gb",gb)],final_estimator=LogisticRegression(max_iter=1000),cv=3),
    }

def capture_rate(y_true, probs, fraction):
    yt = np.asarray(y_true)
    n = max(1,int(np.ceil(len(yt)*fraction)))
    idx = np.argsort(np.asarray(probs))[::-1][:n]
    return float(yt[idx].sum()/yt.sum()) if yt.sum() else np.nan

st.subheader("2. Train and compare seven models")
st.markdown("Preprocessing is kept inside each scikit-learn Pipeline, so imputation, scaling and one-hot encoding are learned from each CV training fold. Cross-validation uses the training split only; the test set is reserved for final evaluation.")
settings = (int(seed),float(test_size),int(folds),float(top_fraction),int(n_bag),int(n_rf))
if st.button("▶ Run model comparison",type="primary"):
    try:
        Xtr,Xte,ytr,yte = train_test_split(X,y,test_size=float(test_size),stratify=y,random_state=int(seed))
        models = make_models(int(seed),int(n_bag),int(n_rf))
        cv = StratifiedKFold(n_splits=int(folds),shuffle=True,random_state=int(seed))
        rows, fitted, probs_by_model = [], {}, {}
        progress = st.progress(0)
        status = st.empty()
        for i,(name,model) in enumerate(models.items()):
            status.write(f"Training and cross-validating: {name}")
            pipe = make_pipeline(make_preprocessor(Xtr),model)
            aucs = cross_val_score(pipe,Xtr,ytr,cv=cv,scoring="roc_auc",n_jobs=1)
            pipe.fit(Xtr,ytr)
            ptr = pipe.predict_proba(Xtr)[:,1]
            pte = pipe.predict_proba(Xte)[:,1]
            rows.append({"Model":name,"CV AUC":aucs.mean(),"CV std":aucs.std(),
                         "Train AUC":roc_auc_score(ytr,ptr),"Test AUC":roc_auc_score(yte,pte),
                         "Top capture":capture_rate(yte,pte,float(top_fraction))})
            fitted[name]=pipe
            probs_by_model[name]=pte
            progress.progress((i+1)/len(models))
        scores = pd.DataFrame(rows).sort_values("Test AUC",ascending=False).reset_index(drop=True)

        bag_name=f"bagged trees ({n_bag})"
        oob_pipe=make_pipeline(make_preprocessor(Xtr),BaggingClassifier(
            estimator=DecisionTreeClassifier(random_state=int(seed)),n_estimators=int(n_bag),
            oob_score=True,random_state=int(seed),n_jobs=-1))
        oob_pipe.fit(Xtr,ytr)
        st.session_state["churn_result"]={"scores":scores,"Xte":Xte,"yte":yte,
            "probs":probs_by_model,"oob":float(oob_pipe[-1].oob_score_)}
        st.session_state["churn_settings"]=settings
        status.success("Experiment completed.")
    except Exception as e:
        st.error(f"Experiment failed: {e}")
        st.exception(e)

res=st.session_state.get("churn_result")
if res is None:
    st.warning("Run the experiment to see cross-validation and test results.")
elif st.session_state.get("churn_settings") != settings:
    st.warning("Settings changed. Run the experiment again to refresh results.")
else:
    scores=res["scores"]
    st.markdown("### Model scorecard")
    st.caption("ROC-AUC: higher is better. Top capture: share of all churners found in the highest-risk customer group.")
    pretty=scores.copy()
    for col in ["CV AUC","CV std","Train AUC","Test AUC"]:
        pretty[col]=pretty[col].map(lambda x:f"{x:.3f}")
    pretty["Top capture"]=pretty["Top capture"].map(lambda x:f"{x:.1%}")
    st.dataframe(pretty,use_container_width=True,hide_index=True)
    best=scores.iloc[0]
    c1,c2,c3,c4=st.columns(4)
    c1.metric("Best test AUC",f"{best['Test AUC']:.3f}",best["Model"])
    c2.metric(f"Top {top_fraction:.0%} capture",f"{best['Top capture']:.1%}")
    c3.metric("Lift vs random",f"{best['Top capture']/top_fraction:.2f}×")
    c4.metric("Bagging OOB accuracy",f"{res['oob']:.3f}")

    fig,ax=plt.subplots(figsize=(10,4))
    order=scores.sort_values("Test AUC")
    ax.barh(order["Model"],order["Test AUC"]); ax.set_xlim(.5,1)
    ax.set_xlabel("Test ROC-AUC"); ax.set_title("Test-set ranking performance"); ax.grid(axis="x",alpha=.2)
    st.pyplot(fig,use_container_width=True); plt.close(fig)

    st.markdown("### Who should the retention team call?")
    model_name=st.selectbox("Inspect risk ranking for",scores["Model"].tolist())
    ptest=res["probs"][model_name]
    yt=res["yte"].reset_index(drop=True)
    ranked=res["Xte"].reset_index(drop=True).copy()
    ranked.insert(0,"Actual churn",yt.map({0:"Stayed",1:"Churned"}))
    ranked.insert(1,"Predicted churn probability",ptest)
    ranked["Risk rank"]=pd.Series(ptest).rank(ascending=False,method="first").astype(int)
    cutoff=max(1,int(np.ceil(len(ranked)*top_fraction)))
    ranked["Contact priority"]=np.where(ranked["Risk rank"]<=cutoff,f"Top {top_fraction:.0%} — call first","Remaining")
    ranked=ranked.sort_values("Risk rank")
    st.dataframe(ranked.head(30).style.format({"Predicted churn probability":"{:.1%}"}),use_container_width=True,hide_index=True)
    st.download_button("⬇ Download full test risk ranking",ranked.to_csv(index=False).encode(),
                       "telecom_churn_risk_ranking.csv","text/csv")

    with st.expander("ROC curve and confusion matrix"):
        fpr,tpr,_=roc_curve(yt,ptest)
        fig,ax=plt.subplots(figsize=(6,4))
        ax.plot(fpr,tpr,label=f"AUC {roc_auc_score(yt,ptest):.3f}")
        ax.plot([0,1],[0,1],"--",label="Random")
        ax.set(xlabel="False positive rate",ylabel="True positive rate",title="ROC curve")
        ax.legend(); ax.grid(alpha=.2); st.pyplot(fig); plt.close(fig)
        cm=confusion_matrix(yt,(ptest>=.5).astype(int))
        fig,ax=plt.subplots(figsize=(5,3))
        ax.imshow(cm)
        ax.set_xticks([0,1],["Predicted stayed","Predicted churn"])
        ax.set_yticks([0,1],["Actually stayed","Actually churned"])
        for (i,j),v in np.ndenumerate(cm): ax.text(j,i,str(v),ha="center",va="center")
        ax.set_title("Confusion matrix at 0.5 threshold")
        st.pyplot(fig); plt.close(fig)

st.subheader("3. Learn the concepts")
t1,t2,t3,t4,t5=st.tabs(["Bagging","Boosting","Voting & Stacking","Metrics","Experiments"])
with t1:
    st.markdown("**Bagging (Bootstrap Aggregating)** trains many copies of a model on random samples drawn *with replacement*, then combines their votes. It mainly reduces **variance**: averaging unstable trees makes the overall prediction less sensitive to one particular training sample. A **Random Forest** also selects a random subset of features at each split, making trees less alike.")
with t2:
    st.markdown("**Boosting** trains models one after another, with later models focusing on patterns or errors earlier models did not handle well. Gradient boosting often performs strongly on tabular data and mainly aims to reduce bias.")
with t3:
    st.markdown("- **Voting:** combines different models' predictions; soft voting averages predicted probabilities.\\n- **Stacking:** trains a final model to learn how to combine predictions from base models.\\n- Ensembles may add little when models make similar mistakes or a simple model is already stable.")
with t4:
    st.markdown("- **ROC-AUC:** probability that a randomly chosen churner is ranked above a randomly chosen non-churner. 0.5 is random; 1.0 is perfect.\\n- **Top-fraction capture:** share of all actual churners found among the highest-risk fraction the company can contact. Random ranking captures roughly that same fraction.\\n- **OOB score:** evaluates each bootstrap tree on training rows it did not see, giving a built-in validation estimate.")
with t5:
    st.markdown("1. Change bagging trees (10, 50, 100, 300): where do gains flatten?\\n2. Try class weighting and observe top-fraction capture.\\n3. Remove StandardScaler: which models change?\\n4. Add random feature selection to bagged trees and compare with Random Forest.\\n5. Try LightGBM or XGBoost.\\n6. Add campaign economics: call cost, offer cost, customer value, and probability that an offer prevents churn.")

st.divider()
st.caption("One public dataset and one train/test split. A high AUC does not by itself prove campaign profitability; call costs, saved-customer value and intervention effectiveness matter.")
