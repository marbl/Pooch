#! /usr/bin/env python3

#__author__ == "Brandon Pickett"

#---------------------- IMPORTS ---------------------------------------------||
import sys
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
import argparse
import pathlib as pl
import logging
#from packaging import version

from pofo_tools.utils import *

#---------------------- Misc Global Settings --------------------------------||
pd.options.mode.copy_on_write = True
logger = logging.getLogger(__name__)
#__all__ = ["blah_blah_blah"] # prevent unlisted functions from being imported via `from ... import *`
plt.set_loglevel("warning")
sns.set_style("white")

#---------------------- Command-line Args -----------------------------------||
def processLogFilename(logfn, datasetfn, logdir="logs"):
    if str(logfn).lower() == "auto":
        # stem removes one extension, presumably `.tsv`
        # sub removes `dataset` (possibly pre/suf-fixed w/ `.`) with just `.`
        logfn = pl.Path(logdir) / ( "diffDsReporting" + re.sub(r"\.?diffDs\.?", r".", datasetfn.stem) + ".log" )
        #logfn = pl.Path(logdir) / ( "diffDatasetReporting" + re.sub(r"\.?diffDataset\.?", r".", datasetfn.stem) + ".log" )
        #logfn = pl.Path(logdir) / f"diffDatasetReporting{re.sub(r'\.?diffDataset\.?', r'.', datasetfn.stem)}.log" # works only in Python 3.12+, otherwise no \ inside {} inside an f-string
    return logfn

def _parseArgs():
    parser = argparse.ArgumentParser(prog=sys.argv[0], description="Report some statistics and create some plots about a difference dataset. See the help message about -d|--dataset for a description of the dataset.")
    parser.add_argument("-c", "--chr", metavar="STR", type=str, action="store", dest="chrom", help="The chromosome to run. Technically, this could be any sequence identifier from a fasta file. It should match the sequence ID. Thus, if your fasta file has \"chrX\", provide \"chrX\", not \"X\". A special value of \"auto\" (case insensitive) may be specified to infer the chromosome from the dataset filename (provided to -d|--dataset); it will use the rightmost occurance of the following regex: `chr[0-9A-Za-z]+`. [default: auto]", default="auto", required=False)
    parser.add_argument("-d", "--dataset", metavar="FILE", type=pl.Path, action="store", dest="dataset_fn", help="The dataset TSV. It may optionally have label columns at the beginning. It must have a Sample column and LeftOpHapLabel and RightOpHapLabel columns. It should have N other columns, where N is the number of CpG sites in the dataset. It should have M+1 rows, where M is the number of samples and 1 is for the header row. Other than the optional label columns and the sample and Left/Right label columns that contain strings, all other values should be numeric. Represent NAs, if any, with \"NA\". The column order doesn't matter.", required=True)
    #parser.add_argument("-D", "--dss-dir", metavar="STR", type=pl.Path, action="store", dest="dss_dir", help="The input directory to read DSS files from. The DSS results are used for plotting and reporting purposes. [default: $PWD]", default=".", required=False)
    parser.add_argument("-l", "--log-file", metavar="FILE", type=pl.Path, action="store", dest="logfn", help="The output file to write logging output to. The special string 'auto' will select an output filename based on the basename of -d|--dataset (more specifically, the suffix (`.*`) will be removed and the string 'dataset' will be removed from within the filename) in a subdirectory `logs`. [default: auto]", default="auto", required=False)
    parser.add_argument("-o", "--output-dir", metavar="STR", type=pl.Path, action="store", dest="outdir", help="The output directory to write output files to. [default: $PWD]", default=".", required=False)
    parser.add_argument("-S", "--samples-set-id", metavar="STR", type=str, action="store", dest="samples_set_id", help="String used as an identifier for the set of samples found in the dataset (found in the file provided to -d|--dataset). Special string 'auto' (case insensitive) will extract the name from the filename provided to -d|--dataset; the first capturing group from the following regex will be the resulting identifier: `^(?:.*[/_.])?([^/_.]+)\.[^./]+$`. [default: auto]", default="auto", required=False)
    #parser.add_argument("-t", "--threads", metavar="INT", type=int, action="store", dest="threads", help="The number of threads to use at once. [default: 1]", default=1, required=False)
    if len(sys.argv) == 1: sys.argv.append("-h")
    args = parser.parse_args()
    args.samples_set_id = processSamplesSetIdArg(args.dataset_fn, args.samples_set_id)
    args.chrom = args.chrom if args.chrom.lower() == "all" else processChromosomeArg(args.dataset_fn, args.chrom)
    args.logfn = processLogFilename(args.logfn, args.dataset_fn)
    return args

#---------------------- Classes ---------------------------------------------||

#---------------------- Functions -------------------------------------------||

# ------------- MAIN ----------------------------- ||
if __name__ == "__main__":
    # parse args
    args = _parseArgs()

    # setup logging
    setupLogging(logfn=args.logfn)

    outdir = args.outdir
    #RANDOM_STATE = 39

    # create output directories
    outdir.mkdir(mode=0o2775, parents=True, exist_ok=True)

    #logger.info(f"Using up to {NUM_CPUS} CPUs")
    logger.info(f"Loading input data")
    logger.debug(f"Chromosome is {args.chrom}")
    logger.debug(f"Samples set ID is {args.samples_set_id}")

    # load in the DSS regions (DMRs)
    #regions = parseDssRegionsFile(args.dss_dir / f"{args.chrom}.dmrs.tsv", areastat_cutoff=6e4)

    # load in the DSS loci (DMLs)
    #loci = parseDssLociFile(args.dss_dir / f"{args.chrom}.dmls.tsv", dmrs_for_filtering=regions)

    # load in the dataset
    diff = parseDiffDatasetFile(args.dataset_fn)

    # extract the labels and samples for future use
    #left_label, right_label = diff["LeftOpHapLabel"].cat.categories.to_list() + diff["RightOpHapLabel"].cat.categories.to_list()
    left_label = diff["LeftOpHapLabel"].cat.categories.to_list()[0]
    right_label = diff["RightOpHapLabel"].cat.categories.to_list()[0]
    samples = sorted(diff["Sample"].cat.categories.to_list())

    logger.info(f"This diff dataset was created via '{left_label}' minus '{right_label}'")

    ################################
    #    Difference Scatterplot    #
    ################################

    # plot diff to view the distributions at each position
    logger.debug(f"diff scatterplot")
    #diff_for_plotting = diff.copy(deep=True) # we don't want the label/categorical columns
    diff_for_plotting = diff.filter(regex=r'^([0-9]+|Chromosome)$', axis="columns") # will keep only the data columns and, if present, Chromosome. (omitting the other label/categorical columns)
    #id_vars = ["Chromosome"] if "Chromosome" in diff_for_plotting.columns else []
    id_vars = []
    params = {"color": "black"}
    if "Chromosome" in diff_for_plotting.columns:
        id_vars.append("Chromosome")
        params = {"hue": "Chromosome"}
    diff_for_plotting = pd.melt(diff_for_plotting, id_vars=id_vars, var_name='x', value_name='y', ignore_index=True)
    diff_for_plotting.dropna(axis="index", how="any", subset='y', inplace=True, ignore_index=True)
    diff_for_plotting['x'] = pd.to_numeric(diff_for_plotting['x'])
    plt.figure()
    diff_scatter = sns.scatterplot(data=diff_for_plotting, x='x', y='y', s=0.1, alpha=0.2, edgecolor="none", **params)
    plt.title(f"diff Scatterplot")
    plt.xlabel(f"{args.chrom} Position")
    plt.ylabel(f"Difference in Methylation ({left_label}-{right_label})")
    plt.margins(x=0.005)
    plt.xlim(left=0)
    #plt.ylim(-1,1)
    plt.margins(y=0.005)
    plt.tick_params(axis="both", which="major", bottom=True, left=True)
    diff_scatter_fig = diff_scatter.get_figure()
    diff_scatter_fig.savefig(outdir / f"{args.samples_set_id}.{args.chrom}.diff.scatter.png")
    diff_scatter_fig.savefig(outdir / f"{args.samples_set_id}.{args.chrom}.diff.scatter.pdf")

    #######################
    #    Diff Variance    #
    #######################

    # let's plot the variance here for investigating options for our
    # VarianceThreshold (if we end up using it). ddof=1 is default and standard
    # for samples of data instead of the full population of data. axis=0 means
    # over the index instead of columns, which, unintuitively, gives me
    # column-level variances.
    #diff_var = diff.var(axis=0, skipna=True, ddof=1) # returns a Series with column names and scalar values for variance
    diff_var = diff.filter(regex=r'^[0-9]+$', axis="columns").var(axis=0, skipna=True, ddof=1) # returns a Series with column names and scalar values for variance
    logger.debug(f"Here's what the diff variance looks like (shape: {diff_var.shape}):\n{diff_var}")
    logger.debug(f"Summary information for that variance:\n{diff_var.describe()}")
    logger.debug(f"diff variances scatterplot")
    plt.figure()
    diff_var_hist = sns.histplot(data=diff_var, kde=True)
    plt.title(f"diff ({left_label}-{right_label}) Variances distribution")
    plt.xlabel(f"Variance")
    plt.ylabel(f"Frequency")
    plt.margins(x=0.005)
    plt.xlim(left=0)
    plt.ylim(bottom=0)
    plt.margins(y=0.005)
    plt.tick_params(axis="both", which="major", bottom=True, left=True)
    plt.axvline(x=diff_var.median(), color="red", linestyle="--", label=f"Median: {diff_var.median():.2f}")
    plt.legend()
    diff_var_hist_fig = diff_var_hist.get_figure()
    diff_var_hist_fig.savefig(outdir / f"{args.samples_set_id}.{args.chrom}.diff.variances.hist.png")
    diff_var_hist_fig.savefig(outdir / f"{args.samples_set_id}.{args.chrom}.diff.variances.hist.pdf")

