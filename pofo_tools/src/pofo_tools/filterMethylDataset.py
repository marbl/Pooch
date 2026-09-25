#! /usr/bin/env python3

#__author__ == "Brandon Pickett"

#---------------------- IMPORTS ---------------------------------------------||
import sys
import pandas as pd
#import numpy as np
import argparse
import pathlib as pl
import logging

from pofo_tools.utils import *

#---------------------- Misc Global Settings --------------------------------||
pd.options.mode.copy_on_write = True

#---------------------- Logging ---------------------------------------------||
logger = logging.getLogger(__name__)
#__all__ = [] # prevent unlisted functions from being imported via `from ... import *`

#---------------------- Command-line Args -----------------------------------||
def _parseArgs():
    '''
    Parse arguments with argparse provided to this script (__main__).

    Note that sys.argv must be available (i.e., having been imported, e.g., by `import sys`)

    Returns:
        argparse.Namespace: Namespace object from argparse.ArgumentParser.parse_args()

    Raises:
        argparse.ArgumentError: if arguments are somehow invalid, e.g., if
            options that should be mutually exclusive are used together.
    '''

    parser = argparse.ArgumentParser(prog=sys.argv[0], description="Filter a dataset according to certain criteria. See -d|--dataset for a description of the dataset.")
    parser.add_argument("-c", "--chr", metavar="STR", type=str, action="store", dest="chrom", help="The chromosome to run. Technically, this could be any sequence identifier from a fasta file. It should match the sequence ID. Thus, if your fasta file has \"chrX\", provide \"chrX\", not \"X\". A special value of \"auto\" (case insensitive) may be specified to infer the chromosome from the dataset filename (provided to -d|--dataset); it will use the rightmost occurance of the following regex: `chr[0-9A-Za-z]+`. [default: auto]", default="auto", required=False)
    parser.add_argument("-d", "--dataset", metavar="FILE", type=pl.Path, action="store", dest="dataset_fn", help="The dataset TSV. It should have >=N+2 columns, where N is the number of CpG sites in the dataset and the 2 extra are for the sample (e.g., HG01234) and label (e.g., mat/pat or hap1/hap2). It may have extra label columns. It should have 2M+1 rows, where M is the number of samples, 2 is the number of haplotypes, and 1 is for the header row. Other than the label columns that contain strings, all other values should be numeric. Represent NAs, if any, with \"NA\".", required=True)
    #parser.add_argument("-D", "--dss-dir", metavar="STR", type=pl.Path, action="store", dest="dss_dir", help="The input directory to read DSS files from. The DSS results are used for plotting and reporting purposes. [default: $PWD]", default=".", required=False)
    parser.add_argument("-l", "--log-file", metavar="FILE", type=pl.Path, action="store", dest="logfn", help="The output file to write logging output to. [default: no log file]", default=None, required=False)
    parser.add_argument("-o", "--output-file", metavar="FILE", type=pl.Path, action="store", dest="outfn", help="The output file to write the filtered dataset to. [default: same as file from -d|--dataset with .tsv replaced with .filtered.tsv]", default=None, required=False)
    parser.add_argument("-N", "--nonNA-prop", metavar="FLOAT", type=float, action="store", dest="nonNA_prop", help="The proportion of samples at a given CpG site for a particular haplotype that have valid values (i.e., not NAs). Anything below the provided threshold will be discarded. [default: 0 (i.e., no filtering)]", default=0, required=False)
    if len(sys.argv) == 1: sys.argv.append("-h")
    args = parser.parse_args()
    args.chrom = processChromosomeArg(args.dataset_fn, args.chrom)
    if not args.outfn: args.outfn = args.dataset_fn.with_suffix(".filtered.tsv")
    return args

#---------------------- Classes ---------------------------------------------||

#---------------------- Functions -------------------------------------------||
def _main():
    # parse args
    args = _parseArgs()

    # setup logging
    setupLogging(logfn=args.logfn)

    # report the command
    logger.info(f"Arguments provided to this python script: {sys.argv}")

    # create output directories
    args.outfn.parent.mkdir(mode=0o2775, parents=True, exist_ok=True)

    logger.info(f"Loading input data")
    logger.debug(f"Chromosome is {args.chrom}")

    # load in the dataset
    methyl = parseMethylDatasetFile(args.dataset_fn)

    # extract the labels and samples for future use
    cat_cols = [col_name for col_name in methyl.columns if re.fullmatch(r"[0-9]+", col_name) is None]
    hap1_label, hap2_label = sorted(methyl["Label"].cat.categories.to_list())
    samples = sorted(methyl["Sample"].cat.categories.to_list())

    # split the data by haplotype and make labels implicit
    methyl_hap1, methyl_hap2, labels_hap1, labels_hap2, S, other_cats = splitMethylByHaps(methyl, hap1_label, hap2_label)

    # calculate the difference between the haplotypes (hap1 - hap2)
    logger.info(f"Subtracting the values in each haplotype from eachother (one way for now) to find the difference in methylation level at each CpG")
    methyl_hap1_minus2 = methyl_hap1 - methyl_hap2 # note: an NA in either slot will return NA
    logger.debug(f"Here's what methyl_hap1_minus2 (i.e., {hap1_label}-{hap2_label}) looks like (shape: {methyl_hap1_minus2.shape}):\n" + headTailDataframeForReporting(methyl_hap1_minus2))

    #############
    # FILTERING #
    #############

    # FILTER 1 - Proportion of NAs
    NA_info = methyl.copy(deep=True)
    logger.debug(f"This is what NA_info looks like initially (same as methyl): {NA_info.shape}\n" + headTailDataframeForReporting(NA_info))

    NA_info_numHaplotypes = len(NA_info)
    NA_info_numSamples = len(NA_info) / 2

    NA_info_numNA = NA_info.drop(columns=cat_cols, inplace=False).isna().sum()
    NA_info_numNAhap1 = NA_info[NA_info["Label"] == hap1_label].drop(columns=cat_cols, inplace=False).isna().sum()
    NA_info_numNAhap2 = NA_info[NA_info["Label"] == hap2_label].drop(columns=cat_cols, inplace=False).isna().sum()

    NA_info = pd.DataFrame({"NumNA": NA_info_numNA, "NumNAhap1": NA_info_numNAhap1, "NumNAhap2": NA_info_numNAhap2})
    NA_info.reset_index(names="Pos", inplace=True)
    NA_info["NumSamples"] = NA_info_numSamples
    NA_info["NumNonNAhap1"] = NA_info["NumSamples"] - NA_info["NumNAhap1"]
    NA_info["NumNonNAhap2"] = NA_info["NumSamples"] - NA_info["NumNAhap2"]
    NA_info["PropNonNAhap1"] = NA_info["NumNonNAhap1"] / NA_info["NumSamples"]
    NA_info["PropNonNAhap2"] = NA_info["NumNonNAhap2"] / NA_info["NumSamples"]

    logger.debug(f"This is what NA_info looks like after calculating some things: {NA_info.shape}\n" + headTailDataframeForReporting(NA_info))

    NA_info_drop = NA_info  [
                                ( NA_info["PropNonNAhap1"]  < args.nonNA_prop )
                                |
                                ( NA_info["PropNonNAhap2"]  < args.nonNA_prop )
                            ]
    logger.debug(f"\nNA_info_drop items to remove from methyl (PropNonNAhap1 < {args.nonNA_prop} | PropNonNAhap2 < {args.nonNA_prop}). {NA_info_drop.shape}:\n" + headTailDataframeForReporting(NA_info_drop, show_idx=True))

    NA_info_keep = NA_info[~NA_info.index.isin(NA_info_drop.index)]
    logger.debug(f"\nNA_info_keep {NA_info_keep.shape}:\n" + headTailDataframeForReporting(NA_info_keep, show_idx=True))

    if len(NA_info_drop):
        methyl.drop(columns=NA_info_drop["Pos"].astype("str"), inplace=True)
        logger.debug(f"Here's what the cleaned methyl looks like after dropping columns with too many NAs (shape: {methyl.shape}):\n" + headTailDataframeForReporting(methyl, show_idx=True))
    else:
        logger.debug(f"No columns (i.e., genomic positions) needed to be dropped from methyl due to too many NAs and/or NA imbalance between haplotypes. shape: {methyl.shape}")

    # FILTER 2 - Remove anything w/ a std. dev. of 0. Note, we could choose a
    # different threshold, and, in that case, we'd want to implement this
    # differently.
    logger.info(f"Removing columns from methyl_hap1_minus2 with std. dev. of 0 (<2 unique non-NA values), if any.")
    cols_to_keep_mask = methyl_hap1_minus2.nunique(axis="index", dropna=True) > 1
    num_cols_to_drop = len(cols_to_keep_mask) - cols_to_keep_mask.sum()
    logger.debug(f"Removing {num_cols_to_drop} (of {len(cols_to_keep_mask)}) columns from methyl_hap1_minus2.")
    #cols_to_drop_mask = ~cols_to_keep_mask
    cols_to_keep_mask = cols_to_keep_mask.to_list()
    methyl_hap1_minus2 = methyl_hap1_minus2.loc[:, cols_to_keep_mask]
    logger.debug(f"Here's what the cleaned methyl_hap1_minus2 (i.e., {hap1_label}-{hap2_label}) looks like (shape: {methyl_hap1_minus2.shape}):\n" + headTailDataframeForReporting(methyl_hap1_minus2))
    num_cols_dropped = num_cols_to_drop
    cols_to_drop = [col for col in methyl if not (col in cat_cols or col in methyl_hap1_minus2.columns)]
    num_cols_to_drop = len(cols_to_drop)
    num_cols_to_keep = len(methyl.columns) - len(cols_to_drop)
    num_cols_already_dropped_filter1 = num_cols_dropped - num_cols_to_drop
    logger.debug(f"{num_cols_already_dropped_filter1}/{num_cols_dropped} of dropped cols in methyl_hap1_minu2 had already been dropped from methyl in the previous filter (i.e., too many NAs). Removing {num_cols_to_drop}/{len(methyl.columns)} cols from methyl (keeping {num_cols_to_keep}/{len(methyl.columns)}) based on columns dropped from methyl_hap1_minus2 due to having std. dev. 0 (<2 unique non-NA values).")
    methyl.drop(columns=cols_to_drop, inplace=True)
    logger.debug(f"Here's what the cleaned methyl looks like (shape: {methyl.shape}):\n" + headTailDataframeForReporting(methyl))

    ##########################
    # WRITE FILTERED DATASET #
    ##########################
    methyl.to_csv(args.outfn, mode='w', sep='\t', header=True, index=False, float_format="{:g}".format)

    # report completion
    logger.info(f"Filtering dataset completed.")

# ------------- MAIN ----------------------------- ||
if __name__ == "__main__":
    _main()

