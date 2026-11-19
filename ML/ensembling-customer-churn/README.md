# P0 case study — Bagging & Ensembling: predicting telecom customer churn

**Library:** scikit-learn (`BaggingClassifier`, `RandomForestClassifier`, `HistGradientBoostingClassifier`, `VotingClassifier`, `StackingClassifier`).
**Data:** IBM Telco Customer Churn, 7,043 customers (open dataset, downloaded once from GitHub and cached as `telco-churn.csv`).

## The business problem
A telecom company loses customers ("churn") and can only call a limited number of them with a retention offer. Which customers should it call first, and does combining several models find churners better than one model?

- **Input:** none to provide (the CSV downloads automatically; internet needed the first time).
- **Output:** a table comparing 7 models, plus the out-of-bag score and a business summary.

## Run
```bash
pip install -r requirements.txt
python main.py
```
It takes a minute or two on a laptop CPU.

## Theory
**Ensemble:** combine several models so that their mistakes partly cancel. Three main ideas:

- **Bagging (bootstrap aggregating):** train many copies of a model, each on a random sample of the data drawn *with replacement* (a bootstrap sample), then average their votes. It mainly reduces **variance**: a single deep decision tree memorizes its training data and changes a lot if the data changes, and averaging many such trees smooths that out. A **random forest** is bagged trees that also use a random subset of features at each split, which makes the trees less alike.
- **Boosting:** train models one after another, each focusing on the mistakes of the previous ones (gradient boosting, used by XGBoost, LightGBM and scikit-learn's `HistGradientBoosting`). It mainly reduces **bias** and is often the strongest option on tabular data.
- **Voting and stacking:** combine *different kinds* of models. Voting averages their predicted probabilities; stacking trains a final model to learn how to weigh them.

**Out-of-bag (OOB) score:** each bootstrap sample leaves out about a third of the data. Testing each tree on the rows it never saw gives a free validation estimate without a separate validation set.

**When ensembles help less:** if the models make similar mistakes, or if one simple model is already stable and well-suited (logistic regression here), averaging adds little. Ensembles also cost more to train and run, and are harder to explain.

**Metrics used:**
- **ROC-AUC:** the chance that a random churner is scored higher than a random non-churner (0.5 = coin flip, 1.0 = perfect).
- **Top-20% capture:** of all customers who actually churned, what share is found among the 20% the model ranks riskiest? Random ranking gives 20%. This matches how a retention team works.

**Evaluation hygiene:** preprocessing (imputing, scaling, one-hot encoding) sits inside a scikit-learn `Pipeline`, so in cross-validation it is learned from each training fold only. Cross-validation uses the training set only; the test set is touched once at the end.

**Where this shows up in AI work:** the same "many votes beat one" idea appears as *self-consistency* (sample several LLM answers and take the majority), *model merging/averaging*, and Mixture-of-Experts models, which route inputs among specialist sub-networks (related to, though not the same as, a classic ensemble).

## What the output shows (one run)
| Model | CV AUC (±std) | Train AUC | Test AUC | Top-20% capture |
|---|---|---|---|---|
| stacking (LR+RF+GB) | 0.848 ± 0.010 | 0.917 | 0.844 | 50% |
| voting (LR+RF+GB) | 0.848 ± 0.009 | 0.936 | 0.843 | 50% |
| logistic regression | 0.846 ± 0.013 | 0.849 | 0.842 | 50% |
| random forest | 0.841 ± 0.009 | 0.962 | 0.837 | 49% |
| gradient boosting | 0.838 ± 0.007 | 0.951 | 0.835 | 49% |
| bagged trees (100) | 0.821 ± 0.008 | 1.000 | 0.812 | 46% |
| single decision tree | 0.658 ± 0.012 | 1.000 | 0.657 | 35% |

- **Bagging fixed the overfit tree:** one tree scored perfectly on training data (AUC 1.0) but only 0.66 on test data. 100 bagged trees reached 0.81. That jump is variance reduction.
- **A simple model matched the ensembles:** plain logistic regression (0.842) was within noise of stacking (0.844). The top four models differ by less than the cross-validation spread (about 0.01), so don't read a winner into them.
- **Business value:** calling the 20% of customers the best model ranks riskiest finds about 50% of churners, around 2.5x better than calling at random.
- Out-of-bag accuracy for bagging was 0.784, close to what a held-out set would show.

**Be careful:** this is one dataset and one split, and results can shift with other seeds or library versions. Accuracy-type scores here also ignore the cost of a call versus the value of a saved customer, which is what really decides a campaign.

## Try these
1. Change `n_estimators` for bagging (10, 50, 300). Where does the gain flatten?
2. Add `max_features="sqrt"` to the bagged trees and compare with the random forest.
3. Use `class_weight="balanced"` and see how top-20% capture changes.
4. Remove `StandardScaler`. Which models care?
5. Try the `lightgbm` or `xgboost` libraries as extra models.

## Likely interview questions
- How does bagging reduce variance? What is the difference between bagging and boosting?
- Why does a random forest use a random subset of features at each split?
- What is out-of-bag error? When might stacking not beat a simple model?

## One thing learned
Averaging many overfit models can rescue them (single tree 0.66 to bagged trees 0.81), but a good simple baseline can still keep up, so always compare against one.
