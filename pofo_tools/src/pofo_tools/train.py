#! /usr/bin/env python3

#__author__ == "Brandon Pickett"

#---------------------- IMPORTS ---------------------------------------------||
import sys
import re
import pickle
import pandas as pd
import numpy as np
#from subprocess import Popen, PIPE
import matplotlib.pyplot as plt 
import seaborn as sns
from collections import defaultdict
from timeit import default_timer as timer
from datetime import timedelta
import argparse
import pathlib as pl
import logging
#import statsmodels.api as sm
from scipy import stats
#import hashlib
from packaging import version
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline

from pofo_tools.utils import *

#---------------------- Misc Global Settings --------------------------------||
pd.options.mode.copy_on_write = True
logger = logging.getLogger(__name__)
#__all__ = ["blah_blah_blah"] # prevent unlisted functions from being imported via `from ... import *`
plt.set_loglevel("warning")
sns.set_style("white")

#---------------------- Command-line Args -----------------------------------||
def _processInverseRegularization(values, mode):
    # expected all elements of values to be int or float. values is a list unless it is None
    assert mode in ["cv", "split"] # i.e., can't be "auto" at the time this function is called
    if not values: # i.e., use the defaults for args.C because the user didn't specify anything
        if args.mode == 'cv': return 10
        return 1. # <-- mode must be "split"
    if len(values) == 1: # <-- values is a list instead of None if we get here
        if isinstance(values[0], float):
            if mode == "cv":
                return values
            return values[0] # <-- when mode is "split"
        if mode == "split": # <-- values[0] must be an int if we get htere
            return float(values[0])
        return values[0] # <-- mode must be "cv" and values[0] must be an int
    return list(map(float, values)) # <-- values is a list with >1 items, return them all as floats
    
def _parseArgs():
    '''
    Helper function to _main to parse the arguments provided when running this
    file directly instead of as a module.
    '''

    parser = argparse.ArgumentParser(prog=sys.argv[0], description="Train a LogisticRegression or LogisticRegressionCV model on the input dataset subject to various parameters.")
    parser.add_argument("-c", "--chr", metavar="STR", type=str, action="store", dest="chrom", help="The chromosome to run. Technically, this could be any sequence identifier from a fasta file. It should match the sequence ID. Thus, if your fasta file has \"chrX\", provide \"chrX\", not \"X\". A special value of \"auto\" (case insensitive) may be specified to infer the chromosome from the dataset filename (provided to -d|--dataset); it will use the rightmost occurance of the following regex: `chr[0-9A-Za-z]+`. [default: auto]", default="auto", required=False)
    parser.add_argument("-i", "--input-dataset", metavar="FILE", type=pl.Path, action="store", dest="dataset_fn", help="The train/dev set TSV.", required=True)
    parser.add_argument("-l", "--log-file", metavar="FILE", type=pl.Path, action="store", dest="logfn", help="The output file to write logging output to. [default: no log file]", default=None, required=False)
    parser.add_argument("-m", "--mode", metavar="STR", type=str, action="store", dest="mode", choices=["auto", "split", "cv"], help="The training mode or method: LogisticRegression or LogisticRegressionCV. They are respectively selected with the keywords 'split' (think train/test split) or 'cv' (think cv=cross validation). To have the mode inferred from the input data, use the keyword 'auto'. In 'auto' mode, 'split' will be selected if a 'Dataset' column is present in the input data and it has a 2 (or more) categories with 'Train' and 'Dev' specifically having >=1 row each. In this case, any other categories in that column will be ignored. Otherwise (e.g., no Dataset column or no Dev rows), 'cv' will be selected. [default: auto]", default="auto", required=True)
    #parser.add_argument("-o", "--output-dir", metavar="STR", type=pl.Path, action="store", dest="outdir", help="The output directory to write output files to. [default: $PWD]", default=".", required=False)
    parser.add_argument("-o", "--output-prefix", metavar="STR", type=pl.Path, action="store", dest="outpfx", help="The prefix to which additional naming will be added when writing output files. [default: -i|--input-dataset, except replacing any suffix with .train]", default=None, required=False)
    parser.add_argument("-t", "--threads", metavar="INT", type=int, action="store", dest="threads", help="The number of threads to use at once. [default: 1]", default=1, required=False)

    classify_grp = parser.add_argument_group("Classifiers options", "The following options control _both_ the LogisticRegression (w/o cross validation) and LogisticRegressionCV classifiers. As needed, the differences between the two classifiers for any given option will be noted here.")
    classify_grp.add_argument("-C", "--inv-reg-strength", metavar="NUM", type=floatOrInt, action="store", dest="C", nargs='+', help="Inverse of regularization strength. Like in support vector machines, smaller values specify stronger regularization. For -m|--mode 'split', provide a single positive float; this single value will the the inverse regularization strength specified to the LogisticRegression constructor. For -m|--mode 'cv', provide either (a) a list of positive floats for the classifier to optimize over or (b) a single positive integer that will be used to construct a grid of inverse regularization strengths to optimize over; see the parameter 'Cs' in the LogisticRegressionCV constructor for additional details. [default: 1 for 'split' mode, 10 for 'cv' mode]", default=None, required=False)
    classify_grp.add_argument("-M", "--max-iter", metavar="INT", type=sciInt, action="store", dest="max_iter", help="The maximum number of iterations for the solvers to converge [default: 1e3]", default=1e3, required=False)
    classify_grp.add_argument("-T", "--tolerance", metavar="FLOAT", type=float, action="store", dest="tol", help="Tolerance for stopping criteria. [default 1e-4]", default=1e-4, required=False)

    #logreg_grp   = parser.add_argument_group("LogisticRegression (no CV)", "The following options control the LogisticRegression classifier (w/o cross validation). These options take effect only when -m|--mode is 'split'.")

    logregcv_grp = parser.add_argument_group("LogisticRegressionCV options", "The following options control the LogisticRegressionCV classifier. These options take effect only when -m|--mode is 'cv'.")
    logregcv_grp.add_argument("-N", "--n-fold-cv", metavar="INT", type=sciInt, action="store", dest="n_folds", help="Number of folds or partitions (e.g., 10) for cross validation (CV). [default: 10]", default=10, required=False)

    #parser.add_argument("-R", "--random-seed", metavar="INT", type=int, action="store", dest="random_seed", help="Provide a positive integer to use as a random seed for train_test_split [default: None]", default=None, required=False)

    if len(sys.argv) == 1: sys.argv.append("-h")
    args = parser.parse_args()

    args.chrom = processChromosomeArg(args.dataset_fn, args.chrom)
    if not args.outpfx: args.outpfx = args.dataset_fn.with_suffix(".train")

    return args

#---------------------- Classes ---------------------------------------------||

#---------------------- Functions -------------------------------------------||
def _main():
    # parse args
    args = _parseArgs()

    # setup logging
    setupLogging(logfn=args.logfn)

    # create output directories
    args.outpfx.parent.mkdir(mode=0o2775, parents=True, exist_ok=True)

    logger.info(f"Using up to {args.threads} CPUs")
    logger.info(f"Loading input data")
    logger.debug(f"Chromosome is {args.chrom}")

    # load in the dataset
    data = parseDiffDatasetFile(args.dataset_fn)

    # add "Dataset" column if needed
    if not "Dataset" in data.columns:
        # die if mode split
        if args.mode == "split":
            msg = f"Mode 'split' was requested, but no dataset partitions (i.e., Train vs Dev) were provided in the input file because the Dataset column was missing."
            logger.critical(msg)
            raise DatasetCompositionError(msg)

        # if mode auto, set to cv
        if args.mode == "auto":
            args.mode = "cv"
            logger.info(f"Mode 'cv' autoselected because no 'Dataset' column was present in the input")
        data["Dataset"] = "Train" # actually add the Dataset column

    # Determine mode feasibility now that we are guaranteed to have a Dataset
    # column now.
    dataset_partitions = list(frozenset(data["Dataset"].to_list()))
    if "Train" in dataset_partitions and "Dev" in dataset_partitions:
        if args.mode == "auto":
            args.mode = "split"
            logger.info(f"Mode 'split' autoselected because 'Train' and 'Dev' were each present 1+ times in column 'Dataset'")
            other_ds_partition_counts = data["Dataset"].value_counts(dropna=False).drop(["Train", "Dev"], inplace=False)
            if len(other_ds_partition_counts):
                logger.warning(f"Partitions other than 'Train' and 'Dev' present in column 'Dataset'. Training will ignore the rows in the other partition(s). Other partition(s):\n{other_ds_partition_counts}")
        elif args.mode == "cv":
            logger.warning(f"Ignoring Train and Dev (and, if present, any other labels) partitioning in column 'Dataset' since mode 'cv' was requested")
            data["Dataset"] = "Train" # reset all dataset labels to Train for the purposes of this run
    else:
        if args.mode == "split":
            msg = f"Mode 'split' was requested, but column 'Dataset' lacked rows with Train and/or Dev."
            logger.critical(msg)
            raise DatasetCompositionError(msg)
        if args.mode == "auto":
            args.mode = "cv"
            logger.info(f"Mode 'cv' autoselected because 'Dataset' column lacked Train and/or Dev rows")
        if args.mode == "cv" and len(data) != data["Dataset"].value_counts().get("Train", 0):
            logger.warning(f"Ignoring all labels) in partitioning column 'Dataset' since mode 'cv' was requested")
            data["Dataset"] = "Train" # reset all dataset labels to Train for the purposes of this run

    # now that mode is definitely set, let's handle args.C
    args.C = _processInverseRegularization(args.C, args.mode)

    # extract the samples, labels, other categoricals, and X (the training data) into
    # separate data containers by train/dev set
    samples        = data["Sample"].cat.categories.to_list()
    data_grpbyDs   = data.groupby("Dataset") # does not copy everything, just a view-like reference object
    data_train     = data_grpbyDs.get_group("Train")
    #data_dev       = data_grpbyDs.get_group("Dev")
    data_dev       = data.loc[data_grpbyDs.groups.get("Dev", [])]
    D_train        = data_train["Dataset"]
    D_test         = data_dev["Dataset"]
    S_train        = data_train["Sample"]
    S_test         = data_dev["Sample"]
    L_train        = data_train["LeftOpHapLabel"]
    L_test         = data_dev["LeftOpHapLabel"]
    R_train        = data_train["RightOpHapLabel"]
    R_test         = data_dev["RightOpHapLabel"]

    if args.mode == "split":
        inferred_split_thresh = len(data_train) / ( len(data_train) + len(data_dev) )
        logger.info(f"This training dataset is split at the following percentage: {int(inferred_split_thresh * 100)}%. That's a {int(inferred_split_thresh * 100)}/{int((1-inferred_split_thresh) * 100)} split.")
    if args.mode == "cv":
        num_T = int(len(data_train) / args.n_folds) # num rows in any given training   partition
        num_V = len(data_train) - num_T             # num rows in any given validation partition
        logger.info(f"N-fold cross-validation will proceed with N={args.n_folds}, approximately {num_T}:{num_V} (training:validation) rows per fold.")

    arbitrary_hap1 = L_train.iat[0]
    arbitrary_hap2 = R_train.iat[0]

    cat_cols       = frozenset([col_name for col_name in data.columns if re.fullmatch(r"[0-9]+", col_name) is None])
    data_cols      = [col_name for col_name in data.columns if not col_name in cat_cols]
    O_train        = data_train.drop(columns=data_cols, inplace=False).drop(columns=["Sample", "LeftOpHapLabel", "RightOpHapLabel", "Dataset"], inplace=False)
    O_test         = data_dev.drop(columns=data_cols,   inplace=False).drop(columns=["Sample", "LeftOpHapLabel", "RightOpHapLabel", "Dataset"], inplace=False)

    cat_cols       = list(cat_cols)
    X_train        = data_train.drop(columns=cat_cols, inplace=False)
    X_test         = data_dev.drop(columns=cat_cols,   inplace=False)
    logger.debug(f"X_train (shape: {X_train.shape}):\n{X_train}")
    logger.debug(f"X_dev   (shape: {X_test.shape}):\n{X_test}")
    logger.debug(f"L_train (shape: {L_train.shape}):\n{L_train}")
    logger.debug(f"L_dev   (shape: {L_test.shape}):\n{L_test}")
    logger.debug(f"R_train (shape: {R_train.shape}):\n{R_train}")
    logger.debug(f"R_dev   (shape: {R_test.shape}):\n{R_test}")
    logger.debug(f"S_train (shape: {S_train.shape}):\n{S_train}")
    logger.debug(f"S_dev   (shape: {S_test.shape}):\n{S_test}")
    logger.debug(f"D_train (shape: {D_train.shape}):\n{D_train}")
    logger.debug(f"D_dev   (shape: {D_test.shape}):\n{D_test}")
    logger.debug(f"O_train (shape: {O_train.shape}):\n{O_train}")
    logger.debug(f"O_dev   (shape: {O_test.shape}):\n{O_test}")

    # create an initial y (output labels) to show which haplotype was first in
    # the difference subtraction
    logger.info(f"Creating y_train and y_dev")
    y_train = list(map(lambda a, b: f"{a}-{b}", L_train.to_list(), R_train.to_list()))
    y_test  = list(map(lambda a, b: f"{a}-{b}", L_test.to_list(),  R_test.to_list() ))
    #y_cats = pd.CategoricalDtype(categories=list(frozenset(y_train + y_test)), ordered=False)
    y_cats = pd.CategoricalDtype(categories=[f"{arbitrary_hap1}-{arbitrary_hap2}", f"{arbitrary_hap2}-{arbitrary_hap1}"], ordered=False)
    logger.debug(f"Here's what the y categories looks like:\n{y_cats.categories.to_list()}")
    y_train = pd.Series(y_train, name="Label", dtype=y_cats)
    y_test  = pd.Series(y_test,  name="Label", dtype=y_cats)
    logger.debug(f"y_train (shape: {y_train.shape}):\n{y_train}")
    logger.debug(f"y_dev   (shape: {y_test.shape}):\n{y_test}")

    # replace the y with "dummies" (simple boolean instead of Categorical (even
    # though the 2-value Categorical is functionally boolean))
    logger.info(f"Replace y (labels; both train and dev (if applicable)) with True/False values (converted from existing y (labels)). This will make {arbitrary_hap1}-{arbitrary_hap2} == False (0)  and {arbitrary_hap2}-{arbitrary_hap1} == True (1)")

    logger.debug(f"undropped yd_train:\n{pd.get_dummies(y_train, columns=['Label'], prefix='', prefix_sep='')}")
    logger.debug(f"dropped yd_train:\n{pd.get_dummies(y_train, columns=['Label'], drop_first=True, prefix='', prefix_sep='')}")
    yd_train = pd.get_dummies(y_train, columns=["Label"], drop_first=True, prefix='', prefix_sep='')
    yd_train.rename(columns={f"{arbitrary_hap2}-{arbitrary_hap1}": f"is{arbitrary_hap2}First"}, inplace=True)

    yd_test = pd.get_dummies(y_test, columns=["Label"], drop_first=True, prefix='', prefix_sep='')
    yd_test.rename(columns={f"{arbitrary_hap2}-{arbitrary_hap1}": f"is{arbitrary_hap2}First"}, inplace=True)

    logger.debug(f"Dummified y_train (shape: {yd_train.shape}):\n{yd_train}")
    logger.debug(f"Dummified y_dev   (shape: {yd_test.shape}):\n{yd_test}")

    # create LogisticRegression or LogisticRegressionCV instance
    logreg = None
    if args.mode == "split":
        logger.info(f"Creating LogisticRegression instance with n_jobs={args.threads}, max_iter={args.max_iter}, solver=lbgfs, penalty=l2, C={args.C}, tol={args.tol}")
        #logreg = LogisticRegression(n_jobs=args.threads, max_iter=args.max_iter, solver="saga", penalty="elasitcnet", C=0.1, l1_ratio=0.25)
        logreg = LogisticRegression(n_jobs=args.threads, max_iter=args.max_iter, solver="lbfgs", penalty="l2", C=args.C, tol=args.tol)
    if args.mode == "cv":
        logger.info(f"Creating LogisticRegressionCV instance with n_jobs={args.threads}, max_iter={args.max_iter}, solver=lbgfs, penalty=l2, Cs={args.C}, tol={args.tol}, cv={args.n_folds}, refit=True")
        #logreg = LogisticRegressionCV(n_jobs=args.threads, max_iter=args.max_iter, solver="liblinear", dual=True, penalty="l2", Cs=args.C, tol=args.tol, cv=args.n_folds, refit=True)
        logreg = LogisticRegressionCV(n_jobs=args.threads, max_iter=args.max_iter, solver="lbfgs", penalty="l2", Cs=args.C, tol=args.tol, cv=args.n_folds, refit=True)
        

    # fit
    logger.info("Fitting")
    start = timer()
    logreg.fit(X_train, yd_train.to_numpy().ravel())
    end = timer()
    logger.info(f"Elapsed time: {timedelta(seconds=end-start)}")

    # report the iterations, coefficients, & intercept
    if logreg.n_iter_.size == 1:
        logger.info(f"The logistic regression used {logreg.n_iter_.item()} iterations (of {args.max_iter} allowed)")
        with open(f"{args.outpfx}.model.iters.txt", 'w') as iters_fd:
            print(f"{logreg.n_iter_[0]}",         file=iters_fd)
    else:
        logger.info(f"The logistic regression used {logreg.n_iter_.min()}-{logreg.n_iter_.max()} iterations across the hyperparameter search space ({args.max_iter} max iterations allowed):\n{logreg.n_iter_}")
        with open(f"{args.outpfx}.model.iters.txt", 'w') as iters_fd:
            print(f"{logreg.n_iter_}",            file=iters_fd)
            print(f"MIN\t{logreg.n_iter_.min()}", file=iters_fd)
            print(f"MAX\t{logreg.n_iter_.max()}", file=iters_fd)

    logger.info(f"The following is the logistic regression intercept: {logreg.intercept_}\n")
    with open(f"{args.outpfx}.model.intercept.txt", 'w') as intercept_fd: print(f"{logreg.intercept_[0]}", file=intercept_fd)

    logger.info("The following are the logistic regression coefficients:\n" + '\n'.join(minifyLongListForReporting(list(map(str, logreg.coef_[0])))) + '\n')
    #pd.DataFrame(zip(columns, np.transpose(logreg.coef_)), columns=['features', 'coef']).to_csv(f"{args.outpfx}.model.coefficients.tsv", sep='\t', header=True, na_rep="NA")
    pd.DataFrame(zip(data_cols, logreg.coef_[0]), columns=['features', 'coef']).to_csv(f"{args.outpfx}.model.coefficients.tsv", sep='\t', header=True, na_rep="NA")

    # save the model (via pickling)
    logger.info(f"Pickling LogReg model into {args.outpfx}.model.pickle")
    with open(f"{args.outpfx}.model.pickle", "wb") as pickle_fd: pickle.dump(logreg, pickle_fd, protocol=5)
    md5SumFileAndWriteMd5(f"{args.outpfx}.model.pickle")

    # predict on the training set based on fitted model and report various
    # things (e.g., accuracy, conf. matrix, etc.)
    logger.info("Predicting on the training set")
    y_pred_train = logreg.predict(X_train)
    logger.debug(f"y_pred_train (shape: {y_pred_train.shape}):\n{y_pred_train}")

    ## report accuracy
    logger.info("Accuracy of logistic regression classifier on training set: {:.2f}".format(logreg.score(X_train, yd_train)))
    with open(f"{args.outpfx}.model.accuracy-train.txt", 'w') as ofd:
            print(f"{logreg.score(X_train, yd_train):.2f}", file=ofd)

    ## report confusion matrix
    confusion_matrix_train = confusion_matrix(yd_train, y_pred_train)
    logger.info(f"confusion_matrix (training):\n{confusion_matrix_train}")
    with open(f"{args.outpfx}.model.confusion-matrix-train.txt", 'w') as ofd:
        print(confusion_matrix_train, file=ofd)

    ## report prediction probabilities
    logger.info(f"Calculating prediciton probabilities for X_train")
    pred_probs_train = logreg.predict_proba(X_train) # shape n_samples (i.e., len(X_train)), n_classes (i.e., 2: True/False)
    pp_df_train = pd.DataFrame(pred_probs_train, columns=[f"PredictProb_{arbitrary_hap1}-{arbitrary_hap2}", f"PredictProb_{arbitrary_hap2}-{arbitrary_hap1}"])
    logger.debug(f"Prediciton probabilities for X_train ({pp_df_train.shape}):\n{pp_df_train}")
    pp_df_train = pd.concat (   [   S_train.to_frame(),
                                    L_train.to_frame(),
                                    R_train.to_frame(),
                                    O_train,
                                    pp_df_train,
                                ],
                                axis="columns",
                                ignore_index=False
                            )
    pp_df_train.to_csv(f"{args.outpfx}.model.probs-train.tsv", sep='\t', index=False, header=True, na_rep="NA", float_format="{:g}".format)

    # If a test set exists, predict on the test set based on the fitted model
    # and report various thing (e.g., acuracy, conf. matrix, etc.).
    if len(X_test):
        logger.info("Predicting on the test set")
        y_pred_test = logreg.predict(X_test)
        logger.debug(f"y_pred_test (shape: {y_pred_test.shape}):\n{y_pred_test}")

        ## report accuracy
        logger.info("Accuracy of logistic regression classifier on test set: {:.2f}".format(logreg.score(X_test, yd_test)))
        with open(f"{args.outpfx}.model.accuracy-test.txt", 'w') as ofd:
            print(f"{logreg.score(X_test, yd_test):.2f}", file=ofd)

        ## report confusion matrix
        confusion_matrix_test = confusion_matrix(yd_test, y_pred_test)
        logger.info(f"confusion_matrix (test):\n{confusion_matrix_test}")
        with open(f"{args.outpfx}.model.confusion-matrix-test.txt", 'w') as ofd:
            print(confusion_matrix_test, file=ofd)

        ## report prediction probabilities
        logger.info(f"Calculating prediciton probabilities for X_test")
        pred_probs_test = logreg.predict_proba(X_test) # shape n_samples (i.e., len(X_test)), n_classes (i.e., 2: True/False)
        pp_df_test = pd.DataFrame(pred_probs_test, columns=[f"PredictProb_{arbitrary_hap1}-{arbitrary_hap2}", f"PredictProb_{arbitrary_hap2}-{arbitrary_hap1}"])
        logger.debug(f"Prediciton probabilities for X_test ({pp_df_test.shape}):\n{pp_df_test}")
        pp_df_test = pd.concat  (   [   S_test.to_frame(),
                                        L_test.to_frame(),
                                        R_test.to_frame(),
                                        O_test,
                                        pp_df_test,
                                    ],
                                    axis="columns",
                                    ignore_index=False
                                )
        pp_df_test.to_csv(f"{args.outpfx}.model.probs-test.tsv", sep='\t', index=False, header=True, na_rep="NA", float_format="{:g}".format)

    # test for significance and report (no statsmodel) on the training set
    logger.info("Testing for significance (training data only)")
    coef = logreg.coef_[0]
    denom = (X_train.to_numpy().T @ X_train.to_numpy()).diagonal()
    denom = np.copy(denom)[denom == 0] = 1e-6 # change 0 to small number to avoid division by zero below. copy is needed bc it was read-only.
    std_err = np.sqrt( 1 / denom )
    z_scores = coef / std_err
    p_values = 2 * ( 1 - stats.norm.cdf(np.abs(z_scores))  )

    with open(f"{args.outpfx}.model.significance.txt", 'w') as ofd:
        for feature, coef, p_value in zip(X_train.columns, coef, p_values):
            print(f"{feature}: coef = {coef:.4f}, p-value = {p_value:.4f}", file=ofd)

    # report completion
    logger.info("Completed.")

# ------------- MAIN ----------------------------- ||
if __name__ == "__main__":
    _main()

