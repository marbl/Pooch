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
#from sklearn.model_selection import train_test_split
#from sklearn.metrics import confusion_matrix
#from sklearn.preprocessing import StandardScaler
#from sklearn.preprocessing import MinMaxScaler
#from sklearn.preprocessing import RobustScaler
#from sklearn.pipeline import Pipeline

from pofo_tools.utils import *

#---------------------- Misc Global Settings --------------------------------||
pd.options.mode.copy_on_write = True
logger = logging.getLogger(__name__)
#__all__ = ["blah_blah_blah"] # prevent unlisted functions from being imported via `from ... import *`
plt.set_loglevel("warning")
sns.set_style("white")

#---------------------- Command-line Args -----------------------------------||
def parseArgs():
    # TODO - remove the things we don't need, add the things we do
    parser = argparse.ArgumentParser(prog=sys.argv[0], description="Train a LogisticRegression model on the input dataset subject to various parameters. See -d|--dataset for a description of the dataset.")
    parser.add_argument("-C", "--inv-reg-strength", metavar="FLOAT", type=float, action="store", dest="C", help="Inverse of regularization strength; must be a positive float. Like in support vector machines, smaller values specify stronger regularization. This value is passed directly to sklearn's LogisticRegression constructor. [default 1]", default=1.0, required=False)
    parser.add_argument("-c", "--chr", metavar="STR", type=str, action="store", dest="chrom", help="The chromosome to run. Technically, this could be any sequence identifier from a fasta file. It should match the sequence ID. Thus, if your fasta file has \"chrX\", provide \"chrX\", not \"X\". A special value of \"auto\" (case insensitive) may be specified to infer the chromosome from the dataset filename (provided to -d|--dataset); it will use the rightmost occurance of the following regex: `chr[0-9A-Za-z]+`. [default: auto]", default="auto", required=False)
    parser.add_argument("-d", "--dataset", metavar="FILE", type=pl.Path, action="store", dest="dataset_fn", help="The dataset TSV. It should have N+2 columns, where N is the number of CpG sites in the dataset and the 2 extra are for the sample (e.g., HG01234) and label (e.g., mat/pat or hap1/hap2). It should have 2M+1 rows, where M is the number of samples, 2 is the number of haplotypes, and 1 is for the header row. Other than the sample and label columns that contain strings, all other values should be numeric. Represent NAs, if any, with \"NA\".", required=True)
    parser.add_argument("-l", "--log-file", metavar="FILE", type=pl.Path, action="store", dest="logfn", help="The output file to write logging output to. [default: no log file]", default=None, required=False)
    parser.add_argument("-M", "--max-iter", metavar="INT", type=sciInt, action="store", dest="max_iter", help="The maximum number of iterations when calling functions (e.g., fit) on LogisticRegression instances [default: 1,000]", default=1000, required=False)
    parser.add_argument("-o", "--output-dir", metavar="STR", type=pl.Path, action="store", dest="outdir", help="The output directory to write output files to. [default: $PWD]", default=".", required=False)
    parser.add_argument("-S", "--samples-set-id", metavar="STR", type=str, action="store", dest="samples_set_id", help="String used as an identifier for the set of samples found in the dataset (found in the file provided to -d|--dataset). Special string 'auto' (case insensitive) will extract the name from the filename provided to -d|--dataset; the first capturing group from the following regex will be the resulting identifier: `^(?:.*[/_.])?([^/_.]+)\.[^./]+$`. [default: auto]", default="auto", required=False)
    parser.add_argument("-T", "--tolerance", metavar="FLOAT", type=float, action="store", dest="tol", help="Tolerance for stopping criteria. This value is passed directly to sklearn's LogisticRegression constructor. [default 1e-4]", default=1e-4, required=False)
    parser.add_argument("-t", "--threads", metavar="INT", type=int, action="store", dest="threads", help="The number of threads to use at once. [default: 1]", default=1, required=False)
    if len(sys.argv) == 1: sys.argv.append("-h")
    args = parser.parse_args()
    args.samples_set_id = processSamplesSetIdArg(args.dataset_fn, args.samples_set_id)
    args.chrom = processChromosomeArg(args.dataset_fn, args.chrom)
    return args

#---------------------- Classes ---------------------------------------------||

#---------------------- Functions -------------------------------------------||

# ------------- MAIN ----------------------------- ||
def _main():
    print("Not really ready to be run!", file=sys.stderr)
    sys.exit(1)

    # parse args
    args = parseArgs()

    # setup logging
    #log_fn = logdir / f"datasetReporting.{args.samples_set_id}.{args.chrom}.log"
    setupLogging(logfn=args.logfn)

    CHROM = args.chrom
    NUM_CPUS = args.threads
    C = args.C
    TOL = args.tol
    MAX_ITER = args.max_iter
    RANDOM_STATE = 39

    # create output directories
    outdir = args.outdir
    outdir.mkdir(mode=0o2775, parents=True, exist_ok=True)

    logger.info(f"Using up to {NUM_CPUS} CPUs")
    logger.info(f"Loading input data")
    logger.debug(f"Chromosome is {args.chrom}")
    logger.debug(f"Samples set ID is {args.samples_set_id}")

    # load in the model
    logger.info("Loading the model")
    start = timer()
    logreg = #TODO unpickle
    end = timer()
    logger.info(f"Elapsed time: {timedelta(seconds=end-start)}")

    # report the coefficients & intercept
    logger.info(f"The logistic regression used {logreg.n_iter_.item()} iterations (of {MAX_ITER} allowed)")
    logger.info(f"The following is the logistic regression intercept: {logreg.intercept_}\n")
    logger.info("The following are the logistic regression coefficients:\n" + '\n'.join(minifyLongListForReporting(list(map(str, logreg.coef_[0])))) + '\n')
    pd.DataFrame(zip(columns, logreg.coef_[0]), columns=['features', 'coef']).to_csv(outdir / f"{args.samples_set_id}.{CHROM}.model.coefficients.tsv", sep='\t', header=True, na_rep="NA")
    with open(outdir / f"{args.samples_set_id}.{CHROM}.model.intercept.txt", 'w') as intercept_fd: print(f"{logreg.intercept_[0]}", file=intercept_fd)
    with open(outdir / f"{args.samples_set_id}.{CHROM}.model.iters.txt", 'w') as iters_fd: print(f"{logreg.n_iter_[0]}", file=iters_fd)

    # plot the coefficients
    logger.info("Plotting the coefficients")
    start = timer()
    #TODO
    end = timer()
    logger.info(f"Elapsed time: {timedelta(seconds=end-start)}")

if __name__ == "__main__":
    _main()

