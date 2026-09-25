import sys
import pickle
import joblib
import warnings
import pandas as pd
import numpy as np
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)
from sklearn.exceptions import ConvergenceWarning, UndefinedMetricWarning
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
warnings.filterwarnings(
    "ignore",
    message="'penalty' was deprecated.*",
    category=FutureWarning,
)
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, accuracy_score, balanced_accuracy_score, confusion_matrix, classification_report, brier_score_loss
from sklearn.linear_model import LogisticRegression

thread = 50
max_iter = 100000 # maximum solver iterations

print("All arguments:", sys.argv)
chr = sys.argv[1]
print("First argument:", chr)

missingrate = sys.argv[2]
print("Second argument (missing rate):", missingrate)

diff_methyl = pd.read_csv(f"{chr}.missing_rate_{missingrate}.createDiffDatasetFromMethylDataset.out", sep='\t', header = 0)
print(f"left : {diff_methyl['LeftOpHapLabel'].unique()}")

trainingset = diff_methyl

# Training set
training_data = diff_methyl.copy()
training_data = training_data.reset_index().drop("index", axis = 1)
training_data_a = training_data.iloc[:,:3].copy()
training_data_a['LeftOpHapLabel'] = "Paternal"
training_data_a['RightOpHapLabel'] = "Maternal"
training_data_b = training_data.iloc[:,3:].copy()
training_data_b= training_data_b *-1
training_data_flip = pd.concat([training_data_a, training_data_b], axis=1).copy()
training_data = pd.concat([training_data, training_data_flip], axis = 0).copy()
group=training_data[["Sample"]].to_numpy().flatten()
training_data = training_data.reset_index().drop("index", axis = 1)
train_X = training_data.iloc[:, 3:]
yd_train = pd.get_dummies(training_data, columns=['RightOpHapLabel'], prefix='', prefix_sep='')
yd_train = yd_train['Paternal'].to_numpy()
print(f"group : {group}")
print(f"training_data : {training_data.shape}")
print(f"train : {yd_train}")

if training_data.shape[0] == 0:
    raise ValueError("No training rows matched trainingset after filtering; check sample lists and input file")
if train_X.shape[0] == 0:
    raise ValueError("train_X is empty after preprocessing; cannot fit LogisticRegression")
if len(training_data['Sample'].to_numpy()) != len(train_X):
    raise ValueError(f"Group length mismatch: train_groups={len(training_data['Sample'].to_numpy())} train_X={len(train_X)}")

from sklearn.feature_selection import SelectFdr, f_classif
## Fitting
pipe = Pipeline([
    ("select", SelectFdr(score_func=f_classif, alpha=0.05)),
    ("model", LogisticRegression(
        penalty="elasticnet",
        solver="saga",
        max_iter=max_iter,
        random_state=42,
        fit_intercept=False
    ))
])

# ----------------------------
# Hyperparameter grid
# ----------------------------

param_grid = {
    "model__C" : [0.01, 0.1, 1, 10, 100, 1000],   # inverse of regularization strength
    "model__l1_ratio": [0, 0.25, 0.5, 0.75, 1]  # 0=ridge, 1=lasso
}

inner_cv = GroupKFold(n_splits=5)
outer_cv = GroupKFold(n_splits=5)

search = GridSearchCV(
    estimator=pipe,
    param_grid=param_grid,
    scoring=["neg_brier_score", "accuracy"],
    cv=inner_cv,
    n_jobs=-1,
    refit="neg_brier_score"
)

# ----------------------------
# Nested CV estimate
# ----------------------------
# This gives a less biased estimate of performance
cv_pred = np.zeros(len(train_X))

for train_idx, valid_idx in outer_cv.split(
    train_X,
    yd_train,
    groups=group
):

    X_train = train_X.iloc[train_idx]
    X_valid = train_X.iloc[valid_idx]

    y_train = yd_train[train_idx]

    group_train = group[train_idx]


    search.fit(
        X_train,
        y_train,
        groups=group_train
    )


    cv_pred[valid_idx] = (
        search.best_estimator_
        .predict_proba(X_valid)[:,1]
    )


cv_auc = roc_auc_score(
    yd_train,
    cv_pred
)

print(
    f"Nested CV AUC: {cv_auc:.4f}"
)

# cv_auc = roc_auc_score(yd_train, cv_pred)
print(f"Nested CV AUC (train only): {cv_auc:.4f}")




# ----------------------------
# Fit final model on full training data
# ----------------------------
search.fit(train_X, yd_train, groups=group)
best_model = search.best_estimator_

mask = best_model.named_steps["select"].get_support()
selected_features = train_X.columns[mask]

select = best_model['select']
model = best_model['model']
best_model['model'].feature_names_in_ = selected_features
k = len(selected_features)
l1_ratio = best_model['model'].l1_ratio
c = best_model['model'].C
coefs = best_model['model'].coef_.flatten()

# save the best model
import joblib
import pickle

# Save best model
with open(f"elasticnet_bestmodel_{chr}.pkl", "wb") as f:
    pickle.dump(best_model, f)

with open(f"elasticnet_training_{chr}.pkl", "wb") as f:
    pickle.dump(model, f)

# Save full grid search (for cv_results_)
joblib.dump(search, f"elasticnet_model_gridsearch_{chr}.pkl")

feature_names = selected_features
if len(feature_names) != len(coefs):
    print(
        f"Warning: feature name count ({len(feature_names)}) != coef count ({len(coefs)}). "
        "Falling back to train_X columns."
    )
    feature_names = train_X.columns

selected = pd.Series(coefs, index=feature_names)
selected = selected[selected != 0].sort_values(key=np.abs, ascending=False)

print(f"\nOriginal CpGs: {len(coefs)}")
print(f"\nSelected CpGs: {len(selected)}")
print(selected.head(20))

# make one table
selected_df = pd.DataFrame([
    {   "original_CpG": train_X.shape[1],
        # "total_CpG": len(coefs),
        "selected_CpG": len(selected),
        "cv_auc": f"{cv_auc:.4f}",
        "best_c": c,
        "best_l1_ratio": l1_ratio,
        #"Test_Accuracy": accuracy_score(yd_test, test_pred),
        #"Test_Balanced_Accuracy": balanced_accuracy_score(yd_test, test_pred),
        #"Test_Brier_Score": brier_score_loss(yd_test, test_prob),
    }
])
selected_df.to_csv(f"elasticnet_selected_cpgs_{chr}.csv", index=False)