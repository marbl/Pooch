#! /usr/bin/env python3

#__author__ == "Brandon Pickett"

#---------------------- IMPORTS ---------------------------------------------||
import sys
import re
import pickle
import pandas as pd
#import numpy as np
from collections import defaultdict
#from timeit import default_timer as timer
#from datetime import timedelta
import argparse
import pathlib as pl
import logging
#from packaging import version
#from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
#from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline

from pofo_tools.utils import *

#---------------------- Misc Global Settings --------------------------------||
pd.options.mode.copy_on_write = True
logger = logging.getLogger(__name__)
#__all__ = ["blah_blah_blah"] # prevent unlisted functions from being imported via `from ... import *`

#---------------------- Command-line Args -----------------------------------||
def _parseArgs():
    '''
    Helper function to _main to parse the arguments provided when running this
    file directly instead of as a module.
    '''

    parser = argparse.ArgumentParser(prog=sys.argv[0], description="Create a training dataset from a \"diff\" dataset. The output training set is technically a diff dataset in terms of internal DataFrame structure (just with possible extra categorical columns), but it will be have twice the number of rows (duplicated and inverted) and may have transformations applied. The output TSV file will match the input except for the following differences: (1) transformations applied to the differences, (2) twice the number of rows (as already mentioned), and (3) an extra 'Dataset' column specifying train vs dev set, if -T|--split-threshold <1.")
    parser.add_argument("-C", "--emit-chr", action="store_true", dest="emit_chrom", help="Emit a 'Chromosome' column with the values matching the chromosome specified with -c|--chr. If such a column already exists in the input methyl dataset, it will be output regardless of whether this option is specified (it won't be duplicated if this option is specified). Adding this option when such a column is not already present in the input will ensure it is added to the output. [default: emit only if already in input methyl dataset]", required=False)
    parser.add_argument("-c", "--chr", metavar="STR", type=str, action="store", dest="chrom", help="The chromosome to run. Technically, this could be any sequence identifier from a fasta file. It should match the sequence ID. Thus, if your fasta file has \"chrX\", provide \"chrX\", not \"X\". A special value of \"auto\" (case insensitive) may be specified to infer the chromosome from the dataset filename (provided to -i|--input-methyl-dataset); it will use the rightmost occurance of the following regex: `chr[0-9A-Za-z]+`. [default: auto]", default="auto", required=False)
    parser.add_argument("-D", "--output-dup", action="store_true", dest="output_dupped", help="In addition to the primary output file (-o|--output-file), also write an intermediate dataset created duplicating+inverting the differences to account for the opposite operand order in the difference calculation. The data will already be split into Train/Dev sets in accordance with -T|--split-threshold. This intermediate output comes after the intermediate step available with -S|--output-split. The filename will match that provided to -o|--output-file except that '.intermSplitDup' will be appended to the name (before any file extension, if present). [default: do not output this file]", required=False)
    parser.add_argument("-i", "--input-diff-dataset", metavar="FILE", type=pl.Path, action="store", dest="input_diff_dataset_fn", help="The input diff dataset TSV. It should have >=(N+3) columns, where N is the number of CpG sites in the dataset and the 3 extra are for the sample (e.g., HG01234) and left and right operant haplotype labels (e.g., mat & pat or hap1 & hap2). It may have extra categorical columns. It should have M+1 rows, where M is the number of samples and 1 is for the header row. Other than the 3 expected categorical columns (and any extra categorical columns) that contain strings, all other values should be numeric. No NAs are expected. The column order doesn't matter as long as there is a header with the non-position columns named as \"Sample\", \"LeftOpHapLabel\", and \"RightOpHapLabel\" (and whatever any extra categorical columns are called) and the rest as integers representing the position of the CpG site in the reference sequence (e.g., 13859).", required=True)
    parser.add_argument("-l", "--log-file", metavar="FILE", type=pl.Path, action="store", dest="logfn", help="The output file to write logging output to. [default: no log file]", default=None, required=False)
    parser.add_argument("-o", "--output-file", metavar="FILE", type=pl.Path, action="store", dest="outfn", help="The output training dataset. [default: same as -i|--input-diff-dataset but with suffix .trainSet.tsv]", default=None, required=False)
    #parser.add_argument("-o", "--output-prefix", metavar="STR", type=pl.Path, action="store", dest="outpfx", help="The prefix to which additional naming will be added when writing output files. [default: $PWD/<samples-set-id>.<chrom>.diffDataset]", default=None, required=False)
    parser.add_argument("-P", "--output-xform-pipe", metavar="FILE", type=pl.Path, action="store", dest="pipelinefn", help="The filename where the transformation Pipeline should be stored. This is needed to properly transform data during prediction. [default: same as -i|--input-diff-dataset but with suffix .xformPipeline.pickle]", default=None, required=False)
    parser.add_argument("-R", "--random-seed", metavar="INT", type=int, action="store", dest="random_seed", help="Provide a positive integer to use as a random seed for train_test_split [default: None]", default=None, required=False)
    parser.add_argument("-T", "--split-threshold", metavar="FLOAT", type=floatOrInt, action="store", dest="split_thresh", help="The threshold for creating a training and dev set. FLOAT proportion will go into the training set, and 1-FLOAT proportion will be in the dev set. Thus, for a 70/30 split, you would provide 0.7. FLOAT must be >0 and <=1. If 1, no dev set will be created, which is useful if doing cross-validation. [default: 0.7]", default=0.7, required=False)
    parser.add_argument("-S", "--output-split", action="store_true", dest="output_split", help="In addition to the primary output file (-o|--output-file), also write an intermediate dataset created by applying the requested -T|--split-threshold to create a Training (and, if requested, Dev) set. Functionally, this just adds a 'Dataset' column with labels. If the split threshold is <1, it will also have the rows re-ordered. This intermediate output will not yet have rows duplicated/inverted and will not yet have any transformations applied. The filename will match that provided to -o|--output-file except that '.intermSplit' will be appended to the name (before any file extension, if present). [default: do not output this file]", required=False)
    if len(sys.argv) == 1: sys.argv.append("-h")
    args = parser.parse_args()
    args.chrom = processChromosomeArg(args.input_diff_dataset_fn, args.chrom)
    if args.split_thresh <= 0 or args.split_thresh > 1:
        raise argparse.ArgumentError(f"-T|--split-threshold must be in the range (0,1], {args.split_thresh} provided.")
    if not args.outfn: args.outfn = args.input_diff_dataset_fn.with_suffix(".trainSet.tsv")
    if not args.pipelinefn: args.pipelinefn = args.input_diff_dataset_fn.with_suffix(".xformPipeline.pickle")
    return args

#---------------------- Classes ---------------------------------------------||

#---------------------- Functions -------------------------------------------||
def _main():
    # parse args
    args = _parseArgs()

    # setup logging
    setupLogging(logfn=args.logfn)

    # create output directory
    args.outfn.parent.mkdir(mode=0o2775, parents=True, exist_ok=True)

    logger.info(f"Loading input data")
    logger.debug(f"Chromosome is {args.chrom}")

    # load in the dataset
    diffs = parseDiffDatasetFile(args.input_diff_dataset_fn)

    # extract the samples, labels, other categoricals, and X (the diffs) into
    # separate data containers
    samples      = diffs["Sample"].cat.categories.to_list()
    left_label   = diffs["LeftOpHapLabel"].cat.categories.to_list()[0]
    right_label  = diffs["RightOpHapLabel"].cat.categories.to_list()[0]
    S            = diffs["Sample"]
    left_labels  = diffs["LeftOpHapLabel"]
    right_labels = diffs["RightOpHapLabel"]

    cat_cols     = frozenset([col_name for col_name in diffs.columns if re.fullmatch(r"[0-9]+", col_name) is None])
    diff_cols    = [col_name for col_name in diffs.columns if not col_name in cat_cols]
    other_cats   = diffs.drop(columns=diff_cols, inplace=False).drop(columns=["Sample", "LeftOpHapLabel", "RightOpHapLabel"], inplace=False)

    cat_cols     = list(cat_cols)
    X            = diffs.drop(columns=cat_cols, inplace=False)
    logger.debug(f"Here's what X looks like (shape: {X.shape}):\n{X}")

    # add Chromosome to other_cats if outputting Chromosome is desired and it
    # is not already there
    if args.emit_chrom and not "Chromosome" in other_cats.columns:
        other_cats["Chromosome"] = args.chrom

    ## create an initial y (label) to show which haplotype (i.e., hap1 vs hap2)
    ## was first in the subtraction for the difference calculation. For now, we
    ## have only a one-way difference, so we'll have to add the other way later.
    #logger.info(f"Creating y (initially only one category ({hap1_label}-{hap2_label}); second category ({hap2_label}-{hap1_label}) to be added later)")
    #y = list(map(lambda a, b: f"{a}-{b}", labels_hap1["Label"].to_list(), labels_hap2["Label"].to_list()))
    ##y.extend(list(map(lambda a, b: f"{a}-{b}", y_hap2["Label"].to_list(), y_hap1["Label"].to_list())))
    #y_cats = pd.CategoricalDtype(categories=[f"{hap1_label}-{hap2_label}", f"{hap2_label}-{hap1_label}"], ordered=False)
    #logger.debug(f"Here's what the y categories looks like:\n{y_cats.categories.to_list()}")
    #y = pd.DataFrame(y, columns=["Label"], dtype=y_cats)
    #logger.debug(f"Here's what y looks like (shape: {y.shape}):\n{y}")

    # do the train/test split. we'll still need to duplicate everything to
    # consider things both directions. NOTE: "test" set in this case is really
    # more like a "dev" or "validation" set. A separate dataset is assumed to
    # be held out as a true test set.
    logger.info(f"Splitting into training/dev sets with {int(args.split_thresh*100)}/{int((1-args.split_thresh)*100)} split")
    ## first set the train datasets to the entire datasets and create
    ## placeholders for the test datasets
    ## first create placeholders for the train/test dataframes
    #X_train, X_test, L_train, L_test, R_train, R_test, S_train, S_test, other_cats_train, other_cats_test, D_train, D_test = (pd.DataFrame() for _ in range(12))
    #X_train, X_test, L_train, L_test, R_train, R_test, S_train, S_test, other_cats_train, other_cats_test, D_train, D_test = (None for _ in range(12))
    X_train = X_test = L_train = L_test = R_train = R_test = S_train = S_test = other_cats_train = other_cats_test = D_train = D_test = None
    
    ## fill them differently if we have a real split vs. 100% training (e.g.,
    ## for CV) via split_thresh == 1
    if args.split_thresh == 1:
        X_train          = X
        L_train          = left_labels.to_frame()
        R_train          = right_labels.to_frame()
        S_train          = S.to_frame()
        other_cats_train = other_cats
        X_test           = pd.DataFrame(columns=X_train.columns)
        L_test           = pd.DataFrame(columns=L_train.columns)
        R_test           = pd.DataFrame(columns=R_train.columns)
        S_test           = pd.DataFrame(columns=S_train.columns)
        other_cats_test  = pd.DataFrame(columns=other_cats_train.columns)
        D_test           = pd.DataFrame(columns=["Dataset"])
    else: # if args.split_thresh is >0 and <1
        X_train, X_test, L_train, L_test, R_train, R_test, S_train, S_test, other_cats_train, other_cats_test = train_test_split(X, left_labels.to_frame(), right_labels.to_frame(), S.to_frame(), other_cats, train_size=args.split_thresh, random_state=args.random_seed, shuffle=True)
        D_test  = pd.DataFrame({"Dataset": ["Dev"]   * len(X_test) })
    D_train = pd.DataFrame({"Dataset": ["Train"] * len(X_train)})

    columns = X_train.columns
    logger.debug(f"X_train (shape: {X_train.shape}):\n{X_train}")
    logger.debug(f"L_train (shape: {L_train.shape}):\n{L_train}")
    logger.debug(f"R_train (shape: {R_train.shape}):\n{R_train}")
    logger.debug(f"S_train (shape: {S_train.shape}):\n{S_train}")
    logger.debug(f"D_train (shape: {D_train.shape}):\n{D_train}")
    logger.debug(f"other_cats_train (shape: {other_cats_train.shape}):\n{other_cats_train}")
    logger.debug(f"X_test (shape: {X_test.shape}):\n{X_test}")
    logger.debug(f"L_test (shape: {L_test.shape}):\n{L_test}")
    logger.debug(f"R_test (shape: {R_test.shape}):\n{R_test}")
    logger.debug(f"S_test (shape: {S_test.shape}):\n{S_test}")
    logger.debug(f"D_test (shape: {D_test.shape}):\n{D_test}")
    logger.debug(f"other_cats_test (shape: {other_cats_test.shape}):\n{other_cats_test}")

    #train_set = pd.concat([S_train, L_train, R_train, other_cats_train, D_train, X_train], axis="columns", ignore_index=True)
    #dev_set   = pd.concat([S_test,  L_test,  R_test,  other_cats_test,  D_test,  X_test],  axis="columns", ignore_index=True)

    # output intermediate file, if requested
    if args.output_split:
        outfn = args.outfn.with_suffix(f".intermSplit{args.outfn.suffix}")
        logger.info(f"Writing train/dev split (unduplicated, untransformed) intermediate output becuase of -S|--output-split to {outfn}")
        writeXasDiffDataset (
                                pd.concat([X_train,            X_test],           axis="index", ignore_index=True),
                                pd.concat([S_train,            S_test],           axis="index", ignore_index=True),
                                pd.concat([L_train,            L_test],           axis="index", ignore_index=True),
                                pd.concat([R_train,            R_test],           axis="index", ignore_index=True),
                                outfn,
                                Dataset=pd.concat([D_train,    D_test],           axis="index", ignore_index=True),
                                **pd.concat([other_cats_train, other_cats_test],  axis="index", ignore_index=True)
                            )
                            

    # create second half of X (and categorical columns) to consider the other
    # direction (i.e., right-left).

    ## X
    logger.info(f"Creating the second half of X (both train and test) with inverted (i.e., multiplied by -1) diffs (thus, these diffs become transformed from {left_label}-{right_label} to {right_label}-{left_label}). The rows will be renumbered.")
    X_train_second_half = X_train * -1
    X_train = pd.concat([X_train, X_train_second_half], axis="index", ignore_index=True)
    X_test_second_half  = X_test  * -1
    X_test = pd.concat([X_test,   X_test_second_half],  axis="index", ignore_index=True)
    logger.debug(f"Updated (dup+invert) X_train (shape: {X_train.shape}):\n{X_train}")
    logger.debug(f"Updated (dup+invert) X_dev   (shape: {X_test.shape}):\n{X_test}")

    ## Left & Right labels
    logger.info(f"Creating the second half of L & R (both train and test) with the other category, i.e., R & L. This functionally labels the order of the difference from {left_label}-{right_label} to {right_label}-{left_label}. The rows will be renumbered.")
    tmp_train = L_train.copy(deep=True)
    tmp_test  = L_test.copy(deep=True)
    L_train = pd.concat([L_train, R_train.rename  (columns={R_train.columns[0]:   L_train.columns[0]}, inplace=False)], axis="index", ignore_index=True)
    L_test  = pd.concat([L_test,  R_test.rename   (columns={R_test.columns[0]:    L_test.columns[0] }, inplace=False)], axis="index", ignore_index=True)
    R_train = pd.concat([R_train, tmp_train.rename(columns={tmp_train.columns[0]: R_train.columns[0]}, inplace=False)], axis="index", ignore_index=True)
    R_test  = pd.concat([R_test,  tmp_test.rename (columns={tmp_test.columns[0]:  R_test.columns[0] }, inplace=False)], axis="index", ignore_index=True)
    logger.debug(f"Duplicated L_train (shape: {L_train.shape}):\n{L_train}")
    logger.debug(f"Duplicated L_dev   (shape: {L_test.shape}):\n{L_test}")
    logger.debug(f"Duplicated R_train (shape: {R_train.shape}):\n{R_train}")
    logger.debug(f"Duplicated R_dev   (shape: {R_test.shape}):\n{R_test}")

    ## S
    logger.info(f"Creating the second half of S (both train and test) with an exact copy. The rows will be renumbered.")
    S_train = pd.concat([S_train, S_train], axis="index", ignore_index=True)
    S_test  = pd.concat([S_test,  S_test],  axis="index", ignore_index=True)
    logger.debug(f"Duplicated S_train (shape: {S_train.shape}):\n{S_train}")
    logger.debug(f"Duplicated S_dev   (shape: {S_test.shape}):\n{S_test}")

    ## D
    logger.info(f"Creating the second half of D (both train and test) with an exact copy. The rows will be renumbered.")
    D_train = pd.concat([D_train, D_train], axis="index", ignore_index=True)
    D_test  = pd.concat([D_test,  D_test],  axis="index", ignore_index=True)
    logger.debug(f"Duplicated D_train (shape: {D_train.shape}):\n{D_train}")
    logger.debug(f"Duplicated D_dev   (shape: {D_test.shape}):\n{D_test}")

    ## other_cats
    logger.info(f"Creating the second half of other_cats (both train and test) with an exact copy. The rows will be renumbered.")
    other_cats_train = pd.concat([other_cats_train, other_cats_train], axis="index", ignore_index=True)
    other_cats_test  = pd.concat([other_cats_test,  other_cats_test],  axis="index", ignore_index=True)
    logger.debug(f"Duplicated other_cats_train (shape: {other_cats_train.shape}):\n{other_cats_train}")
    logger.debug(f"Duplicated other_cats_dev   (shape: {other_cats_test.shape}):\n{other_cats_test}")

    # output intermediate file, if requested
    if args.output_dupped:
        outfn = args.outfn.with_suffix(f".intermSplitDup{args.outfn.suffix}")
        logger.info(f"Writing duplicated (not transformed) intermediate output becuase of -D|--output-dup to {outfn}")
        writeXasDiffDataset (
                                pd.concat([X_train,            X_test],           axis="index", ignore_index=True),
                                pd.concat([S_train,            S_test],           axis="index", ignore_index=True),
                                pd.concat([L_train,            L_test],           axis="index", ignore_index=True),
                                pd.concat([R_train,            R_test],           axis="index", ignore_index=True),
                                outfn,
                                Dataset=pd.concat([D_train,    D_test],           axis="index", ignore_index=True),
                                **pd.concat([other_cats_train, other_cats_test],  axis="index", ignore_index=True)
                            )


    # "learn" (on the training set) and apply transformations to all sets
    logger.info(f"Creating a pipeline: StandardScaler | MinMaxScaler")
    logger.debug(f"StandardScaler will adjust all values: mean->0 (already 0 bc of dup+invert) with unit std. dev. MinMaxScaler will squish or expand all values until min/max reach provided min/max.")
    scalerS = StandardScaler(with_mean=True, with_std=True)
    scalerS.set_output(transform="pandas")
    scalerM = MinMaxScaler((-3, 3), clip=False)
    scalerM.set_output(transform="pandas")
    transformer_pipeline = Pipeline([
                                        ("ScalerS", scalerS),
                                        ("ScalerM", scalerM)
                                    ])
    logger.info(f"\"Fitting\" pipeline (StandardScaler | MinMaxScaler) w/ X_train")
    transformer_pipeline.fit(X_train)
    logger.info(f"Pickling 'fitted' (on training set) transformation (StandardScaler | MinMaxScaler) Pipeline to {args.pipelinefn}")
    with open(args.pipelinefn, "wb") as pickle_fd: pickle.dump(transformer_pipeline, pickle_fd, protocol=5)
    logger.info(f"Transforming X_train w/ pipeline (StandardScaler | MinMaxScaler) \"fitted\" on X_train")
    X_train = transformer_pipeline.transform(X_train)
    logger.info(f"Transforming X_dev w/ pipeline (StandardScaler | MinMaxScaler) \"fitted\" on X_train")
    if len(X_test): X_test = transformer_pipeline.transform(X_test)
    logger.debug(f"Transformed X_train (shape: {X_train.shape}):\n{X_train}")
    logger.debug(f"Transformed X_dev (shape: {X_test.shape}):\n{X_test}")

    # write duplicated and transformed train/dev sets to file
    logger.info(f"Writing duplicated/inverted and transformed output to {args.outfn}")
    writeXasDiffDataset (
                            pd.concat([X_train,            X_test],           axis="index", ignore_index=True),
                            pd.concat([S_train,            S_test],           axis="index", ignore_index=True),
                            pd.concat([L_train,            L_test],           axis="index", ignore_index=True),
                            pd.concat([R_train,            R_test],           axis="index", ignore_index=True),
                            args.outfn,
                            Dataset=pd.concat([D_train,    D_test],           axis="index", ignore_index=True),
                            **pd.concat([other_cats_train, other_cats_test],  axis="index", ignore_index=True)
                        )
    logger.info(f"Finished writing output to {args.outfn}. main complete.")

# ------------- MAIN ----------------------------- ||
if __name__ == "__main__":
    _main()

