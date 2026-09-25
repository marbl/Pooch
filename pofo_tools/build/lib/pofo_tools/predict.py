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
def _parseArgs():
    '''
    Helper function to _main to parse the arguments provided when running this
    file directly instead of as a module.
    '''

    parser = argparse.ArgumentParser(prog=sys.argv[0], description="Predict parent of origin given a pre-trained model and methylation differences between two haplotypes on one chromosome. If the true labels are known and listed in the LeftOpHapLabel and RightOpHapLabel columns of the input dataset and you want the logs to report accuracy, specify -m and -p; otherwise, it is assumed the the truth is not known. If -m and -p are specified, all labels in the appropriate columns of the input dataset must match one of the labels (i.e., no labels are allowed other than those two).")
    parser.add_argument("-c", "--chr", metavar="STR", type=str, action="store", dest="chrom", help="The chromosome to run. Technically, this could be any sequence identifier from a fasta file. It should match the sequence ID. Thus, if your fasta file has \"chrX\", provide \"chrX\", not \"X\". A special value of \"auto\" (case insensitive) may be specified to infer the chromosome from the dataset filename (provided to -d|--dataset); it will use the rightmost occurance of the following regex: `chr[0-9A-Za-z]+`. [default: auto]", default="auto", required=False)
    parser.add_argument("-i", "--input-dataset", metavar="FILE", type=pl.Path, action="store", dest="dataset_fn", help="The input diff dataset TSV. It should have >=(N+3) columns, where N is the number of CpG sites in the dataset and the 3 extra are for the sample (e.g., HG01234) and left and right operand haplotype labels (e.g., mat & pat or hap1 & hap2). It may have extra categorical columns. It should have M+1 rows, where M is the number of samples and 1 is for the header row. Other than the 3 expected categorical columns (and any extra categorical columns) that contain strings, all other values should be numeric. No NAs are expected. The column order doesn't matter as long as there is a header with the non-position columns named as \"Sample\", \"LeftOpHapLabel\", and \"RightOpHapLabel\" (and whatever any extra categorical columns are called) and the rest as integers representing the position of the CpG site in the reference sequence (e.g., 13859).", required=True)
    parser.add_argument("-l", "--log-file", metavar="FILE", type=pl.Path, action="store", dest="logfn", help="The output file to write logging output to. [default: no log file]", default=None, required=False)
    parser.add_argument("-M", "--model-pkg", metavar="FILE", type=pl.Path, action="store", dest="model_pkg_file", help="File (pickled) of packaged trained model and data transformation Pipeline.", required=True)
    parser.add_argument("-m", "--maternal-label", metavar="STR", type=str, action="store", dest="mat_lab", help="Maternal label as listed in the input dataset. Useful for reporting prediction success when the answer is known.", default=None, required=False)
    #parser.add_argument("-o", "--output-prefix", metavar="STR", type=pl.Path, action="store", dest="outpfx", help="The prefix to which additional naming will be added when writing output files. [default: -i|--input-dataset, except replacing any suffix with .train]", default=None, required=False)
    parser.add_argument("-o", "--output-predictions", metavar="FILE", type=pl.Path, action="store", dest="outfn", help="The output predictions in TSV format. It will copy the categorical columns from the input file, adding four categorical columns: LeftHapPredictedOrigin and RightHapPredictedOrigin with values being Maternal and Paternal and PredictProb_Maternal-Paternal and PredictProb_Paternal-Maternal with values being floats in the range [0,1] referencing the probability of each predicted class. If -m|--maternal_label and -p|--paternal_label are provided, a fifth extra column is added: PredictionCorrect, with values being True/False. [default: -i|--dataset_fn except with suffix .prediction.tsv]", default=None, required=False)
    parser.add_argument("-p", "--paternal-label", metavar="STR", type=str, action="store", dest="pat_lab", help="Paternal label as listed in the input dataset. Useful for reporting prediction success when the answer is known.", default=None, required=False)
    parser.add_argument("-X", "--output-xformed", action="store_true", dest="output_xformed", help="In addition to the primary output file (-o|--output-predictions), also write an intermediate dataset created after performing transormations on the differences. The filename will match that provided to -o|--output-predictions except with the suffix '.intermXformed.tsv'). [default: do not output this file]", required=False)
    #parser.add_argument("-t", "--threads", metavar="INT", type=int, action="store", dest="threads", help="The number of threads to use at once. [default: 1]", default=1, required=False)
    if len(sys.argv) == 1: sys.argv.append("-h")
    args = parser.parse_args()
    args.chrom = processChromosomeArg(args.dataset_fn, args.chrom)
    if not args.outfn: args.outfn = args.dataset_fn.with_suffix(".predictions.tsv")
    if bool(args.mat_lab) != bool(args.pat_lab):
        raise ValueError(
            "You must provide both -m/--maternal-label and -p/--paternal-label, or neither."
        )
    args.y_known = args.mat_lab and args.pat_lab
    return args

#---------------------- Classes ---------------------------------------------||

#---------------------- Functions -------------------------------------------||
def _main():
    # parse args
    args = _parseArgs()

    # setup logging
    setupLogging(logfn=args.logfn)

    # report system arguments
    logger.info(f"Arguments: {' '.join(sys.argv)}")

    # create output directories
    args.outfn.parent.mkdir(mode=0o2775, parents=True, exist_ok=True)

    logger.info(f"Loading input data")
    logger.debug(f"Chromosome is {args.chrom}")

    # load in the dataset
    data = parseDiffDatasetFile(args.dataset_fn)

    # extract the samples, labels, other categoricals, and X (the training data) into
    # separate data containers by train/dev set
    samples        = data["Sample"].cat.categories.to_list()
    S              = data["Sample"]
    L              = data["LeftOpHapLabel"]
    R              = data["RightOpHapLabel"]

    if args.y_known:
        # sanity check that the mat/pat labels provided match the actual labels
        # in the data. We can trust from the parsing function that there will
        # only be two labels total and that any given row will have both labels
        # in the relevant columns.
        arbitrary_hap1 = L.iat[0]
        arbitrary_hap2 = R.iat[0]
        if ( arbitrary_hap1 != args.mat_lab and arbitrary_hap1 != args.pat_lab ) or ( arbitrary_hap2 != args.mat_lab and arbitrary_hap2 != args.pat_lab ):
            msg = f"one or both of the mat/pat labels provided to -m/-p ({args.mat_lab}/{args.pat_lab}) do not match the haplotype labels found in the input dataset: {arbitrary_hap1}, {arbitrary_hap2}"
            logger.critical(msg)
            raise argparse.ArgumentError(msg)

    cat_cols       = frozenset([col_name for col_name in data.columns if re.fullmatch(r"chr[0-9X]+_[0-9]+_[0-9]+|[0-9]+", col_name) is None]) #juhyun
    data_cols      = [col_name for col_name in data.columns if not col_name in cat_cols] #juhyun
    O              = data.drop(columns=data_cols, inplace=False).drop(columns=["Sample", "LeftOpHapLabel", "RightOpHapLabel"], inplace=False)

    cat_cols       = list(cat_cols)
    X              = data.drop(columns=cat_cols, inplace=False)
    X = X.astype(float)
    logger.debug(f"X (shape: {X.shape}):\n{X}")
    logger.debug(f"L (shape: {L.shape}):\n{L}")
    logger.debug(f"R (shape: {R.shape}):\n{R}")
    logger.debug(f"S (shape: {S.shape}):\n{S}")
    logger.debug(f"O (shape: {O.shape}):\n{O}")

    # load the ModelPackage (pickled)
    logger.info(f"Unpickling ModelPackage from {args.model_pkg_file}")
    # model, xform_pipeline = ModelPackage.unPickle(args.model_pkg_file).unpack()

    # fix -3 : 3 scaling for test set (note that the training set was scaled to -3 : 3 in the pipeline that was fitted on the training set, so we need to do the same thing to the test set before predicting)
    # from sklearn.preprocessing import MinMaxScaler
    # scaler = MinMaxScaler(feature_range=(-3, 3))
    # xform_pipeline = scaler.fit_transform(test_X)
   
    with open(args.model_pkg_file, "rb") as f:
            pipe = pickle.load(f)
    # model = pipe.named_steps["logisticregression"]
    model = pipe.named_steps['model']

    # transform X based on the transformation pipeline that had been "fitted"
    # on the original training set
    # logger.info(f"Transforming using the Pipeline unpacked from ModelPackage")
    # X = scaler.fit_transform(X)
    # logger.debug(f"X (shape: {X.shape}):\n{X}")

    # output intermediate file, if requested
    if args.output_xformed:
        outfn = args.outfn.with_suffix(f".intermXformed{args.outfn.suffix}")
        logger.info(f"Writing intermediate output becuase of -X|--output-xformed to {outfn}")
        writeXasDiffDataset (
                                X,
                                S.to_frame(),
                                L.to_frame(),
                                R.to_frame(),
                                outfn,
                                **O
                            )

    # predict on the training set based on fitted model and report various
    # things (e.g., accuracy, conf. matrix, etc.)
    logger.info("Predicting using the model unpacked from ModelPackage")
    y_pred = model.predict(X)
    logger.debug(f"y_pred (shape: {y_pred.shape}):\n{y_pred}")

    # unpack y_pred into LeftHapPredictedOrigin and RightHapPredictedOrigin
    # we assume that True/False output from the model is answering the question
    # isPaternalFirst? Therefore, the following table applies:
    #     Prediction  Left_label  Right_label
    #     -----------------------------------
    #           True    Paternal     Maternal
    #          False    Maternal     Paternal
    # pred_mapping = {True: "Paternal", False: "Maternal"}
    pred_mapping = {True: "Maternal", False: "Paternal"}
    yL = pd.Series([pred_mapping[    pred] for pred in y_pred], name="LeftHapPredictedOrigin" )
    yR = pd.Series([pred_mapping[not pred] for pred in y_pred], name="RightHapPredictedOrigin")
    is_correct = pd.Series(["Unknown"] * len(yR),               name="PredictionCorrect"      )
    logger.debug(f"yL (shape: {yL.shape}):\n{yL}")
    logger.debug(f"yR (shape: {yR.shape}):\n{yR}")
    logger.debug(f"is_correct (shape: {is_correct.shape}):\n{is_correct}")

    mp_addtnl = '' # addtnl = additional
    pm_addtnl = ''

    # if we know the truth (via -m and -p), let's report accuracy of the predictions
    if args.y_known:
        logger.info("Establishing truth based on values provided to -m and -p")
        logger.debug(f"Maternal haplotype label (assumed true for all rows with this label): {args.mat_lab}")
        logger.debug(f"Paternal haplotype label (assumed true for all rows with this label): {args.pat_lab}")

        # create an initial y (output labels) to show which haplotype was first in
        # the difference subtraction
        logger.debug(f"Creating y")
        y = list(map(lambda a, b: f"{a}-{b}", L.to_list(), R.to_list()))
        y_cats = pd.CategoricalDtype(categories=[f"{args.mat_lab}-{args.pat_lab}", f"{args.pat_lab}-{args.mat_lab}"], ordered=False)
        logger.debug(f"Here's what the y categories looks like:\n{y_cats.categories.to_list()}")
        y = pd.Series(y, name="Label", dtype=y_cats)
        logger.debug(f"y (shape: {y.shape}):\n{y}")

        # replace the y with "dummies" (simple boolean instead of Categorical (even
        # though the 2-value Categorical is functionally boolean))
        mp_extra = ''
        pm_extra = ''
        dummies_extra = ''
        if re.match(r'^mat(ernal)?$', args.mat_lab.lower()) is None or re.match(r'^pat(ernal)?$', args.pat_lab.lower()) is None:
            mp_addtnl = f"_{args.mat_lab}-{args.pat_lab})"
            pm_addtnl = f"_{args.pat_lab}-{args.mat_lab})"
            mp_extra = f" (i.e., {args.mat_lab}-{args.pat_lab})"
            pm_extra = f" (i.e., {args.pat_lab}-{args.mat_lab})"
            dummies_extra = f"_{args.pat_lab}=Paternal"

        logger.info(f"Replace y (labels) with True/False values (converted from existing y (labels)). This will make Maternal-Paternal{mp_extra} == False (0)  and Paternal-Maternal{pm_extra} == True (1)")

        yd = pd.get_dummies(y, columns=["Label"], drop_first=True, prefix='', prefix_sep='')
        yd.rename(columns={f"{args.pat_lab}-{args.mat_lab}": f"is{args.pat_lab}First{dummies_extra}"}, inplace=True)

        logger.debug(f"Dummified y (shape: {yd.shape}):\n{yd}")

        # determine the per-row correctness of the predictions w.r.t. the
        # presumed truth
        #logger.debug(f"yd (type: {type(yd)}, shape: {yd.shape}):\n{yd}")
        #logger.debug(f"y_pred (type: {type(y_pred)}, shape: {y_pred.shape}):\n{y_pred}")
        #print(f"yd (type: {type(yd)}, shape: {yd.shape}):\n{yd}", file=sys.stderr)
        #print(f"y_pred (type: {type(y_pred)}, shape: {y_pred.shape}):\n{y_pred}", file=sys.stderr)
        is_correct = pd.Series(yd[f"is{args.pat_lab}First{dummies_extra}"] == y_pred, name="PredictionCorrect")
        logger.debug(f"is_correct (shape: {is_correct.shape}):\n{is_correct}")

        # Now we can compare yd (presumed truth) with y_pred (model's predictions)
        ## report accuracy
        logger.info("Accuracy of predictions based on presumed truth as provided by -m and -p: {:.2f}".format(model.score(X, yd)))
        ## report confusion matrix
        conf_matrix = confusion_matrix(yd, y_pred, labels=(False, True))
        logger.info(f"confusion_matrix:\n{conf_matrix}")
        ## report the correctness for each row
        logger.info(f"Row-wise correctness ({is_correct.shape}):\n{is_correct}")

    # report prediction probabilities
    logger.info(f"Calculating probabilities for predictions")
    pred_probs = model.predict_proba(X) # shape n_samples (i.e., len(X)), n_classes (i.e., 2: True/False)
    # pp_df = pd.DataFrame(pred_probs, columns=[f"PredictProb_Maternal-Paternal{mp_addtnl}", f"PredictProb_Paternal-Maternal{pm_addtnl}"])
    pp_df = pd.DataFrame(pred_probs, columns=[f"PredictProb_Paternal-Maternal{pm_addtnl}", f"PredictProb_Maternal-Paternal{mp_addtnl}"])
    pp_df = pp_df[[f"PredictProb_Maternal-Paternal{mp_addtnl}", f"PredictProb_Paternal-Maternal{pm_addtnl}"]]
    logger.debug(f"Prediciton probabilities for X ({pp_df.shape}):\n{pp_df}")

    # write the predictions to the output file
    logger.info(f"Writing predictions to {args.outfn}")
    out_df = pd.concat([S, L, R, O, yL, yR, pp_df, is_correct], axis="columns", ignore_index=False)
    out_df.to_csv(args.outfn, sep='\t', header=True, na_rep="NA", index=False, float_format="{:g}".format)

    # report completion
    logger.info("Completed.")

# ------------- MAIN ----------------------------- ||
if __name__ == "__main__":
    _main()

