#!/usr/bin/env python3
"""
model_diagnostics.py

Usage:
    python model_diagnostics.py \
        --pkg train.model-pkg.pickle \
        --X X.csv --y y.csv \
        --outdir diagnostics_out \
        [--bootstrap 200] \
        [--threads 4]

Inputs:
 - --pkg  : path to ModelPackage pickle created by your training script
 - --X    : path to features (CSV). Should match what your xform_pipeline expects
 - --y    : path to labels (CSV or 1-col). For convenience, if omitted and your
            package contains training labels, you can reuse them, but recommended
            to pass explicit validation set or full dataset for OOF evaluation.
 - --outdir : directory to write plots/tables
 - --bootstrap : optional integer >0 to compute bootstrap 95% CI for coefficients
 - --threads : number of jobs for cross_val_predict (default: 1)

Notes:
 - The script constructs a Pipeline([('xform', xform_pipeline), ('clf', LogisticRegression(...))])
   and uses the model.C_ (if available) and model.solver, etc., to reconstruct the final classifier
   for OOF predictions. If model.C_ is not present (split-mode), it uses model.C (or args).
"""

import argparse
import pickle
import pathlib as pl
import os
import sys
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import (
    roc_curve, roc_auc_score,
    precision_recall_curve, average_precision_score,
    accuracy_score, precision_score, recall_score, f1_score, auc
)
from sklearn.utils import resample

sns.set_style("whitegrid")
plt.rcParams.update({'figure.max_open_warning': 0})

def load_pkg(path):
    with open(path, "rb") as fd:
        pkg = pickle.load(fd)
    # Expect pkg.model and pkg.xform_pipeline
    model = getattr(pkg, "model", None)
    xform = getattr(pkg, "xform_pipeline", None)
    if model is None:
        raise ValueError("Loaded package has no attribute 'model'.")
    return pkg, model, xform

def load_X_y(x_path, y_path):
    X = pd.read_csv(x_path, index_col=None)
    y_series = pd.read_csv(y_path, index_col=None, header=None).squeeze()
    # if y is a single column with header, try to handle it
    if isinstance(y_series, pd.DataFrame):
        # fallback: try first column
        y_series = y_series.iloc[:, 0].squeeze()
    return X, y_series

def make_full_estimator(xform, model):
    # Reconstruct a LogisticRegression matching the final refit parameters.
    # Prefer model.C_ (best C from LR-CV) else model.C or fallback to 1.0
    C_val = getattr(model, "C_", None)
    if C_val is None:
        C_val = getattr(model, "C", None)
    if C_val is None:
        # fallback numeric or list
        if hasattr(model, "Cs"):
            # choose median
            try:
                C_val = np.median(model.Cs)
            except Exception:
                C_val = 1.0
        else:
            C_val = 1.0

    # if C_ is array-like (multiclass), use first element
    if isinstance(C_val, (list, np.ndarray)) and np.size(C_val) > 1:
        C_val = float(np.asarray(C_val).flatten()[0])

    clf_kwargs = {}
    # solver, penalty, max_iter, tol, random_state if available
    if hasattr(model, "solver"):
        clf_kwargs["solver"] = getattr(model, "solver")
    if hasattr(model, "penalty"):
        clf_kwargs["penalty"] = getattr(model, "penalty")
    if hasattr(model, "max_iter"):
        clf_kwargs["max_iter"] = int(getattr(model, "max_iter"))
    if hasattr(model, "tol"):
        clf_kwargs["tol"] = getattr(model, "tol")
    if hasattr(model, "random_state"):
        clf_kwargs["random_state"] = getattr(model, "random_state")
    clf = LogisticRegression(C=float(C_val), **clf_kwargs)
    if xform is None:
        return Pipeline([("clf", clf)])
    else:
        return Pipeline([("xform", xform), ("clf", clf)])

def plot_cv_score_vs_C(model, outdir):
    # Only valid if model is LogisticRegressionCV and has scores_ and Cs
    if not hasattr(model, "scores_") or not hasattr(model, "Cs"):
        print("No scores_ or Cs on model; skipping CV score vs C plot.")
        return
    Cs = np.array(model.Cs)
    # binary: scores_ keyed by class label. use positive class index of classes_[1]
    classes = getattr(model, "classes_", None)
    # pick pos label index: usually classes_[1] is positive
    if classes is None:
        print("No classes_ attribute on model; skipping CV score vs C plot.")
        return
    pos_label = classes[1] if len(classes) == 2 else classes[0]
    scores = model.scores_[pos_label]  # shape (n_folds, n_C)
    mean_scores = np.mean(scores, axis=0)
    std_scores  = np.std(scores, axis=0)

    plt.figure(figsize=(7,5))
    plt.semilogx(Cs, mean_scores, marker='o', label='mean CV score')
    plt.fill_between(Cs, mean_scores - std_scores, mean_scores + std_scores, alpha=0.25)
    # mark best C if present
    if hasattr(model, "C_"):
        bestC = model.C_
        # if array-like, take first
        if isinstance(bestC, (list, np.ndarray)):
            bestC = float(np.asarray(bestC).flatten()[0])
        # find index
        try:
            idx = np.where(Cs == bestC)[0][0]
            plt.scatter([bestC], [mean_scores[idx]], color='red', zorder=5, label=f"selected C={bestC:.3g}")
        except Exception:
            pass
    plt.xlabel("C (inverse regularization)")
    plt.ylabel("CV score (scoring used during CV)")
    plt.title("CV mean score vs C")
    plt.legend()
    plt.grid(True, which='both', linestyle='--', alpha=0.4)
    fn = outdir / "cv_score_vs_C.png"
    plt.tight_layout()
    plt.savefig(fn, dpi=150)
    plt.close()
    print(f"Wrote {fn}")

def plot_coef_paths(model, xform, outdir, max_features_legend=15):
    if not hasattr(model, "coefs_paths_"):
        print("No coefs_paths_ on model; skipping coefficient paths plot.")
        return
    Cs = np.array(model.Cs)
    classes = getattr(model, "classes_", None)
    pos_label = classes[1] if len(classes) == 2 else classes[0]
    paths = model.coefs_paths_[pos_label]  # (n_folds, n_C, n_features)
    mean_paths = np.mean(paths, axis=0)    # (n_C, n_features)
    n_features = mean_paths.shape[1]

    # try to get feature names from xform or saved model information
    feature_names = None
    if xform is not None:
        feature_names = getattr(xform, "feature_names_in_", None)
    if feature_names is None and hasattr(model, "feature_names_in_"):
        feature_names = getattr(model, "feature_names_in_")
    if feature_names is None:
        feature_names = [f"f{i}" for i in range(n_features)]

    plt.figure(figsize=(9,6))
    # Plot all, but highlight top-k by abs at best C
    for j in range(n_features):
        plt.semilogx(Cs, mean_paths[:, j], alpha=0.5, linewidth=0.9)
    # highlight top features at best C
    try:
        bestC = model.C_
        if isinstance(bestC, (list, np.ndarray)):
            bestC = float(np.asarray(bestC).flatten()[0])
        idxC = int(np.where(Cs == bestC)[0][0]) if bestC in Cs else -1
    except Exception:
        idxC = -1
    if idxC >= 0:
        abs_at_best = np.abs(mean_paths[idxC, :])
        top_idx = np.argsort(abs_at_best)[-max_features_legend:]
        for i in top_idx:
            plt.semilogx(Cs, mean_paths[:, i], linewidth=2.2, label=feature_names[i])
        plt.legend(fontsize='small', ncol=2)
    plt.xlabel("C")
    plt.ylabel("Coefficient value")
    plt.title("Coefficient paths (mean across folds)")
    plt.grid(True, which='both', linestyle='--', alpha=0.3)
    fn = outdir / "coef_paths.png"
    plt.tight_layout()
    plt.savefig(fn, dpi=150)
    plt.close()
    print(f"Wrote {fn}")

def compute_oof_and_plots(full_estimator, X, y, cv, outdir, n_jobs=1):
    # cross_val_predict with method='predict_proba' to get OOF probabilities
    print("Computing out-of-fold predicted probabilities (this fits models on each fold)...")
    y_proba_oof = cross_val_predict(full_estimator, X, y, cv=cv, method='predict_proba', n_jobs=n_jobs)[:, 1]
    # ROC
    fpr, tpr, _ = roc_curve(y, y_proba_oof)
    roc_auc = roc_auc_score(y, y_proba_oof)
    plt.figure()
    plt.plot(fpr, tpr, label=f'OOF ROC (AUC={roc_auc:.3f})')
    plt.plot([0,1],[0,1],'--', alpha=0.5)
    plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
    plt.title('Out-of-Fold ROC Curve')
    plt.legend()
    fn = outdir / "oof_roc.png"
    plt.tight_layout(); plt.savefig(fn, dpi=150); plt.close()
    print(f"Wrote {fn}")

    # PR
    prec, rec, _ = precision_recall_curve(y, y_proba_oof)
    ap = average_precision_score(y, y_proba_oof)
    plt.figure()
    plt.plot(rec, prec, label=f'OOF PR (AP={ap:.3f})')
    plt.xlabel('Recall'); plt.ylabel('Precision')
    plt.title('Out-of-Fold Precision-Recall Curve')
    plt.legend()
    fn = outdir / "oof_pr.png"
    plt.tight_layout(); plt.savefig(fn, dpi=150); plt.close()
    print(f"Wrote {fn}")

    # scalar metrics (threshold 0.5)
    y_pred_oof = (y_proba_oof >= 0.5).astype(int)
    metrics = {
        "AUC": roc_auc_score(y, y_proba_oof),
        "AveragePrecision(AP)": ap,
        "Accuracy": accuracy_score(y, y_pred_oof),
        "Precision": precision_score(y, y_pred_oof),
        "Recall": recall_score(y, y_pred_oof),
        "F1": f1_score(y, y_pred_oof)
    }
    metrics_df = pd.DataFrame(list(metrics.items()), columns=["metric", "value"])
    metrics_df.to_csv(outdir / "oof_metrics_summary.tsv", sep='\t', index=False)
    print(f"Wrote {outdir / 'oof_metrics_summary.tsv'}")
    return y_proba_oof, metrics

def per_fold_roc(full_estimator, X, y, cv, outdir):
    print("Computing per-fold ROC curves (fits per fold individually)...")
    if isinstance(cv, int):
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=0)
    else:
        skf = cv
    plt.figure(figsize=(8,6))
    aucs = []
    for fold, (tr, te) in enumerate(skf.split(X, y), 1):
        est = full_estimator
        # We must instantiate a fresh estimator per fold to avoid carrying fitted state
        # So create new estimator by cloning pipeline components where possible
        from sklearn.base import clone
        est_clone = clone(full_estimator)
        est_clone.fit(X.iloc[tr], y.iloc[tr])
        y_score = est_clone.predict_proba(X.iloc[te])[:,1]
        fpr, tpr, _ = roc_curve(y.iloc[te], y_score)
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        plt.plot(fpr, tpr, lw=1.2, alpha=0.7, label=f'fold {fold} (AUC={roc_auc:.3f})')
    mean_auc = np.mean(aucs)
    plt.plot([0,1],[0,1],'--', color='black', alpha=0.5)
    plt.xlabel('FPR'); plt.ylabel('TPR')
    plt.title(f'Per-fold ROC curves (mean AUC={mean_auc:.3f})')
    plt.legend(fontsize='small', ncol=2)
    fn = outdir / "per_fold_roc.png"
    plt.tight_layout(); plt.savefig(fn, dpi=150); plt.close()
    print(f"Wrote {fn}")
    # write aucs
    pd.DataFrame({"fold": list(range(1, len(aucs)+1)), "auc": aucs}).to_csv(outdir / "per_fold_auc.tsv", sep='\t', index=False)
    print(f"Wrote {outdir / 'per_fold_auc.tsv'}")

def bootstrap_coefs(full_estimator, X, y, n_boot, outdir):
    print(f"Running bootstrap for coefficients (n_boot={n_boot})... this may take time.")
    coefs = []
    from sklearn.base import clone
    for b in range(n_boot):
        Xb, yb = resample(X, y, replace=True)
        est = clone(full_estimator)
        est.fit(Xb, yb)
        # get coef from inner logistic (clf)
        # Pipeline -> get last step
        clf = est.named_steps['clf']
        coefs.append(clf.coef_.flatten())
    coefs = np.vstack(coefs)  # (n_boot, n_features)
    ci_low = np.percentile(coefs, 2.5, axis=0)
    ci_high = np.percentile(coefs, 97.5, axis=0)
    df = pd.DataFrame({
        "feature": list(X.columns),
        "coef_boot_mean": coefs.mean(axis=0),
        "ci_low_2.5": ci_low,
        "ci_high_97.5": ci_high
    })
    df.to_csv(outdir / "bootstrap_coef_cis.tsv", sep='\t', index=False)
    print(f"Wrote {outdir / 'bootstrap_coef_cis.tsv'}")
    # plot top features by absolute mean
    topk = min(20, df.shape[0])
    df['abs_mean'] = np.abs(df['coef_boot_mean'])
    df_top = df.sort_values('abs_mean', ascending=False).head(topk)
    plt.figure(figsize=(8, max(4, topk*0.3)))
    plt.errorbar(x=df_top['coef_boot_mean'], y=df_top['feature'], xerr=[df_top['coef_boot_mean']-df_top['ci_low_2.5'], df_top['ci_high_97.5']-df_top['coef_boot_mean']], fmt='o')
    plt.axvline(0, color='black', linestyle='--', alpha=0.5)
    plt.xlabel("Coefficient (bootstrap mean and 95% CI)")
    plt.title("Bootstrap coefficient estimates")
    fn = outdir / "bootstrap_coef_cis.png"
    plt.tight_layout(); plt.savefig(fn, dpi=150); plt.close()
    print(f"Wrote {fn}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pkg", required=True, help="ModelPackage pickle path")
    parser.add_argument("--X", required=True, help="CSV of features (rows correspond to samples)")
    parser.add_argument("--y", required=True, help="CSV (or single-column) of labels (binary: 0/1 or equivalent)")
    parser.add_argument("--outdir", required=True, help="Directory to write results")
    parser.add_argument("--bootstrap", type=int, default=0, help="Number of bootstrap resamples for coefficient CIs (optional, slow)")
    parser.add_argument("--threads", type=int, default=1, help="Number of threads (n_jobs) for cross_val_predict")
    args = parser.parse_args()

    outdir = pl.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    pkg, model, xform = load_pkg(args.pkg)
    print(f"Loaded package: model type = {type(model)}, xform = {type(xform)}")

    X, y = load_X_y(args.X, args.y)
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)
    if not isinstance(y, pd.Series):
        y = pd.Series(y).reset_index(drop=True)

    # Build full estimator (xform+predictor)
    full_est = make_full_estimator(xform, model)

    # If model is LogisticRegressionCV: plot CV score vs C and coef paths
    try:
        plot_cv_score_vs_C(model, outdir)
    except Exception as e:
        print("plot_cv_score_vs_C failed:", e)

    try:
        plot_coef_paths(model, xform, outdir)
    except Exception as e:
        print("plot_coef_paths failed:", e)

    # OOF predictions and plots
    # Determine cv splitter: if model.cv is int, use that; else if model.cv is splitter, use it; fallback to 5
    cv = getattr(model, "cv", None)
    if cv is None:
        cv = 5
    # cross_val_predict expects estimator that will be refit each fold; full_est uses default C computed earlier
    y_proba_oof, metrics = compute_oof_and_plots(full_est, X, y, cv=cv, outdir=outdir, n_jobs=args.threads)

    # per-fold ROC
    try:
        per_fold_roc(full_est, X, y, cv=cv, outdir=outdir)
    except Exception as e:
        print("per_fold_roc failed:", e)

    # bootstrap coefficient CI (optional)
    if args.bootstrap and args.bootstrap > 1:
        try:
            bootstrap_coefs(full_est, X, y, n_boot=args.bootstrap, outdir=outdir)
        except Exception as e:
            print("bootstrap_coefs failed:", e)

    # Write a short run info file
    meta = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "pkg": str(args.pkg),
        "X": str(args.X),
        "y": str(args.y),
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "metrics": metrics
    }
    pd.Series(meta).to_frame("value").to_csv(outdir / "run_metadata.tsv", sep='\t')
    print("Done. Outputs in", outdir)

if __name__ == "__main__":
    main()