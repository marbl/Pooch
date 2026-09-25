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
plt.set_loglevel("warning")
sns.set_style("white")

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

    parser = argparse.ArgumentParser(prog=sys.argv[0], description="Report some statistics and create some plots about a methylation dataset. See the help message about -d|--dataset for a description of the dataset.")
    parser.add_argument("-c", "--chr", metavar="STR", type=str, action="store", dest="chrom", help="The chromosome to run. Technically, this could be any sequence identifier from a fasta file. It should match the sequence ID. Thus, if your fasta file has \"chrX\", provide \"chrX\", not \"X\". A special value of \"auto\" (case insensitive) may be specified to infer the chromosome from the dataset filename (provided to -d|--dataset); it will use the rightmost occurance of the following regex: `chr[0-9A-Za-z]+`. [default: auto]", default="auto", required=False)
    parser.add_argument("-d", "--dataset", metavar="FILE", type=pl.Path, action="store", dest="dataset_fn", help="The dataset TSV. It should have N+2 columns, where N is the number of CpG sites in the dataset and the 2 extra are for the sample (e.g., HG01234) and label (e.g., mat/pat or hap1/hap2). It should have 2M+1 rows, where M is the number of samples, 2 is the number of haplotypes, and 1 is for the header row. Other than the sample and label columns that contain strings, all other values should be numeric. Represent NAs, if any, with \"NA\". The column order doesn't matter as long as there is a header with the non-position columns named as \"Label\" and \"Sample\" and the rest as integers representing the position of the CpG site in the reference sequence (e.g., 13859).", required=True)
    #parser.add_argument("-L", "--dss-dmls-file", metavar="FILE", type=pl.Path, action="store", dest="dss_dmls_fn", help="[default: no filtering]", default=None, required=False)
    parser.add_argument("-l", "--log-file", metavar="FILE", type=pl.Path, action="store", dest="logfn", help="The output file to write logging output to. [default: no log file]", default=None, required=False)
    parser.add_argument("-o", "--output-dir", metavar="STR", type=pl.Path, action="store", dest="outdir", help="The output directory to write output files to. [default: $PWD]", default=".", required=False)
    #parser.add_argument("-R", "--dss-dmrs-file", metavar="FILE", type=pl.Path, action="store", dest="dss_dmrs_fn", help="[default: no filtering]", default=None, required=False)
    parser.add_argument("-S", "--samples-set-id", metavar="STR", type=str, action="store", dest="samples_set_id", help="String used as an identifier for the set of samples found in the dataset (found in the file provided to -d|--dataset). Special string 'auto' (case insensitive) will extract the name from the filename provided to -d|--dataset; the first capturing group from the following regex will be the resulting identifier: `^(?:.*[/_.])?([^/_.]+)\.[^./]+$`. [default: auto]", default="auto", required=False)
    if len(sys.argv) == 1: sys.argv.append("-h")
    args = parser.parse_args()
    args.samples_set_id = processSamplesSetIdArg(args.dataset_fn, args.samples_set_id)
    args.chrom = processChromosomeArg(args.dataset_fn, args.chrom)
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
    args.outdir.mkdir(mode=0o2775, parents=True, exist_ok=True)

    logger.info(f"Loading input data")
    logger.debug(f"Chromosome is {args.chrom}")
    logger.debug(f"Samples set ID is {args.samples_set_id}")

    # load in the DSS regions (DMRs)
    #regions = parseDssRegionsFile(args.dss_dmrs_fn, chrom=args.chrom)

    # load in the DSS loci (DMLs)
    #loci = parseDssLociFile(args.dss_dmls_fn, dmrs_for_filtering=regions)

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
    methyl_hap1_minus2.to_csv(args.outdir / f"{args.samples_set_id}.{args.chrom}.methyl_hap1_minus2.txt", sep='\t', header=True, na_rep="NA")

    ######################
    #    NA Reporting    #
    ######################

    # output non-NA and ratio per position
    NA_reporting = methyl.copy(deep=True)
    logger.debug(f"This is what NA_reporting looks like initially (same as methyl): {NA_reporting.shape}\n" + headTailDataframeForReporting(NA_reporting))

    NA_reporting_numHaplotypes = len(NA_reporting)
    NA_reporting_numSamples = len(NA_reporting) / 2

    NA_reporting_numNA = NA_reporting.drop(columns=cat_cols, inplace=False).isna().sum()
    NA_reporting_numNAhap1 = NA_reporting[NA_reporting["Label"] == hap1_label].drop(columns=cat_cols, inplace=False).isna().sum()
    NA_reporting_numNAhap2 = NA_reporting[NA_reporting["Label"] == hap2_label].drop(columns=cat_cols, inplace=False).isna().sum()

    NA_reporting = pd.DataFrame({"NumNA": NA_reporting_numNA, "NumNAhap1": NA_reporting_numNAhap1, "NumNAhap2": NA_reporting_numNAhap2})
    NA_reporting.reset_index(names="Pos", inplace=True)
    NA_reporting["chrom"] = args.chrom
    NA_reporting["end"] = NA_reporting["Pos"].astype("uint64") + 1
    NA_reporting["strand"] = "."
    NA_reporting["NumSamples"] = NA_reporting_numSamples
    NA_reporting["NumHaplotypes"] = NA_reporting_numHaplotypes
    NA_reporting["NumNonNAhaplotypes"] = NA_reporting["NumHaplotypes"] - NA_reporting["NumNA"]
    NA_reporting["NumNonNAhap1"] = NA_reporting["NumSamples"] - NA_reporting["NumNAhap1"]
    NA_reporting["NumNonNAhap2"] = NA_reporting["NumSamples"] - NA_reporting["NumNAhap2"]
    NA_reporting["PropNonNAhap1"] = NA_reporting["NumNonNAhap1"] / NA_reporting["NumSamples"]
    NA_reporting["PropNonNAhap2"] = NA_reporting["NumNonNAhap2"] / NA_reporting["NumSamples"]
    NA_reporting["NonNAimbalance"] = 1 - ( pd.DataFrame([NA_reporting["NumNonNAhap1"], NA_reporting["NumNonNAhap2"]]).min() / pd.DataFrame([NA_reporting["NumNonNAhap1"], NA_reporting["NumNonNAhap2"]]).max() )
    NA_reporting.loc[NA_reporting["NonNAimbalance"] == np.inf, "NonNAimbalance"] = 1 # hack to fix the division by zero issue, 0/0 might as well be 1/1 for our purposes.
    NA_reporting["NonNAimbalanceDirection"] = "Equal"
    NA_reporting.loc[NA_reporting["NumNonNAhap1"] > NA_reporting["NumNonNAhap2"], "NonNAimbalanceDirection"] = hap1_label
    NA_reporting.loc[NA_reporting["NumNonNAhap1"] < NA_reporting["NumNonNAhap2"], "NonNAimbalanceDirection"] = hap2_label
    NA_reporting["NonNAimbalanceColor"] = "255,255,255" # black
    NA_reporting.loc[NA_reporting["NonNAimbalanceDirection"] == hap1_label, "NonNAimbalanceColor"] = "0,0,255" # blue
    NA_reporting.loc[NA_reporting["NonNAimbalanceDirection"] == hap2_label, "NonNAimbalanceColor"] = "255,0,0" # red
    logger.debug(f"\nNA_reporting {NA_reporting.shape}:\n" + headTailDataframeForReporting(NA_reporting, show_idx=True))
    logger.debug(f"\nNA_reporting.T {NA_reporting.T.shape}:\n" + headTailDataframeForReporting(NA_reporting.T, show_idx=True))

    na_reporting_fn = args.outdir / pl.Path(f"{args.samples_set_id}.{args.chrom}.naReporting.tsv")
    NA_reporting.to_csv(na_reporting_fn, mode='w', sep='\t', header=True, index=False, float_format="{:.2g}".format)
    nonNa_imbalance_fn = args.outdir / pl.Path(f"{args.samples_set_id}.{args.chrom}.nonNAimbalance.bed")
    pd.DataFrame(NA_reporting, columns=["chrom", "Pos", "end", "Pos", "NonNAimbalance", "strand", "Pos", "end", "NonNAimbalanceColor"]).to_csv(nonNa_imbalance_fn, mode='w', sep='\t', header=False, index=False, float_format="{:.2g}".format)

    # plot NA_reporting nonNAimbalance
    logger.debug(f"NA_reporting scatterplot")
    NA_reporting["Pos"] = pd.to_numeric(NA_reporting["Pos"])
    plt.figure()
    narprt_scatter = sns.scatterplot(data=NA_reporting, x="Pos", y="NonNAimbalance", s=1, hue="NonNAimbalanceDirection", alpha=0.5, edgecolor="none", palette={"Equal": "#000000", hap1_label: "#0000FF", hap2_label: "#FF0000"})
    plt.title(f"Non-NA Imbalance")
    plt.xlabel(f"{args.chrom} Position")
    plt.ylabel(f"Non-NA Imbalance (1 - min(N_h1,N_h2)/max(N_h1,N_h2))")
    plt.margins(x=0.005)
    plt.xlim(left=0)
    plt.ylim(bottom=0)
    plt.margins(y=0.005)
    plt.tick_params(axis="both", which="major", bottom=True, left=True)
    narprt_scatter_fig = narprt_scatter.get_figure()
    narprt_scatter_fig.savefig(args.outdir / f"{args.samples_set_id}.{args.chrom}.nonNAimbalance.scatter.png")
    narprt_scatter_fig.savefig(args.outdir / f"{args.samples_set_id}.{args.chrom}.nonNAimbalance.scatter.pdf")

    # plot reporting NA by position
    logger.debug(f"NA_reporting plotting NA by position")
    plt.figure()
    narprt_scatter = sns.scatterplot(data=NA_reporting, x="Pos", y="NumNA", s=1, alpha=1, edgecolor="none")
    plt.title(f"NAs by position")
    plt.xlabel(f"{args.chrom} Position")
    plt.ylabel(f"Number of NAs")
    plt.margins(x=0.005)
    plt.xlim(left=0)
    plt.ylim(bottom=0)
    plt.margins(y=0.005)
    plt.tick_params(axis="both", which="major", bottom=True, left=True)
    narprt_scatter_fig = narprt_scatter.get_figure()
    narprt_scatter_fig.savefig(args.outdir / f"{args.samples_set_id}.{args.chrom}.NAs.scatter.png")
    narprt_scatter_fig.savefig(args.outdir / f"{args.samples_set_id}.{args.chrom}.NAs.scatter.pdf")

    # plot reporting NA distribution
    logger.debug(f"NA_reporting plotting NA distribution summed haps ({hap1_label} + {hap2_label})")
    plt.figure()
    bin_edges = [0, 12, 24, 36, 48, 60, 72, 84, 96]
    narprt_hist = sns.histplot(data=NA_reporting_numNA, kde=False, log_scale=(False, False), bins=bin_edges)
    plt.title(f"Number of NAs distribution summed haps ({hap1_label} + {hap2_label})")
    plt.xlabel(f"Number of NAs at a CpG")
    plt.ylabel(f"Frequency")
    plt.margins(x=0.005)
    #plt.xlim(left=1)
    plt.xlim(left=0)
    plt.ylim(bottom=0)
    #plt.ylim(bottom=1)
    plt.margins(y=0.005)
    plt.tick_params(axis="both", which="major", bottom=True, left=True)
    narprt_hist.set_xticks(bin_edges)
    plt.axvline(x=NA_reporting_numNA.median(), color="red", linestyle="--", label=f"Median: {NA_reporting_numNA.median():.2f}")
    plt.legend()
    narprt_hist_fig = narprt_hist.get_figure()
    narprt_hist_fig.savefig(args.outdir / f"{args.samples_set_id}.{args.chrom}.NAs.summedHaps.hist.png")
    narprt_hist_fig.savefig(args.outdir / f"{args.samples_set_id}.{args.chrom}.NAs.summedHaps.hist.pdf")

    logger.debug(f"NA_reporting plotting NA distribution {hap2_label}")
    plt.figure()
    bin_edges = [0, 12, 24, 36, 48]
    narprt_hist = sns.histplot(data=NA_reporting_numNAhap2, kde=False, log_scale=(False, False), bins=bin_edges)
    plt.title(f"Number of NAs ({hap2_label}) distribution")
    plt.xlabel(f"Number of NAs at a CpG")
    plt.ylabel(f"Frequency")
    plt.margins(x=0.005)
    #plt.xlim(left=1)
    plt.xlim(left=0)
    plt.ylim(bottom=0)
    #plt.ylim(bottom=1)
    plt.margins(y=0.005)
    plt.tick_params(axis="both", which="major", bottom=True, left=True)
    narprt_hist.set_xticks(bin_edges)
    plt.axvline(x=NA_reporting_numNAhap2.median(), color="red", linestyle="--", label=f"Median: {NA_reporting_numNAhap2.median():.2f}")
    plt.legend()
    narprt_hist_fig = narprt_hist.get_figure()
    narprt_hist_fig.savefig(args.outdir / f"{args.samples_set_id}.{args.chrom}.NAs.{hap2_label}.hist.png")
    narprt_hist_fig.savefig(args.outdir / f"{args.samples_set_id}.{args.chrom}.NAs.{hap2_label}.hist.pdf")

    logger.debug(f"NA_reporting plotting NA distribution {hap1_label}")
    plt.figure()
    narprt_hist = sns.histplot(data=NA_reporting_numNAhap1, kde=False, log_scale=(False, False), bins=bin_edges)
    plt.title(f"Number of NAs ({hap1_label}) distribution")
    plt.xlabel(f"Number of NAs at a CpG")
    plt.ylabel(f"Frequency")
    plt.margins(x=0.005)
    #plt.xlim(left=1)
    plt.xlim(left=0)
    plt.ylim(bottom=0)
    #plt.ylim(bottom=1)
    plt.margins(y=0.005)
    plt.tick_params(axis="both", which="major", bottom=True, left=True)
    narprt_hist.set_xticks(bin_edges)
    plt.axvline(x=NA_reporting_numNAhap1.median(), color="red", linestyle="--", label=f"Median: {NA_reporting_numNAhap1.median():.2f}")
    plt.legend()
    narprt_hist_fig = narprt_hist.get_figure()
    narprt_hist_fig.savefig(args.outdir / f"{args.samples_set_id}.{args.chrom}.NAs.{hap1_label}.hist.png")
    narprt_hist_fig.savefig(args.outdir / f"{args.samples_set_id}.{args.chrom}.NAs.{hap1_label}.hist.pdf")

    logger.debug(f"NA_reporting plotting NA distribution both haps ({hap1_label} & {hap2_label})")
    both_haps_NAs = pd.concat([NA_reporting_numNAhap1, NA_reporting_numNAhap2], ignore_index=True)
    logger.debug(f"number of NAs summed both haps: {len(NA_reporting_numNA)}")
    logger.debug(f"number of NAs hap1: {len(NA_reporting_numNAhap1)}")
    logger.debug(f"number of NAs hap2: {len(NA_reporting_numNAhap2)}")
    logger.debug(f"number of NAs across both haps (concatenated): {len(both_haps_NAs)}")
    plt.figure()
    #bin_edges = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48]
    narprt_hist = sns.histplot(data=both_haps_NAs, kde=False, log_scale=(False, False), bins=bin_edges)
    plt.title(f"Number of NAs both haps ({hap1_label} & {hap2_label}) distribution")
    plt.xlabel(f"Number of NAs at a CpG")
    plt.ylabel(f"Frequency")
    plt.margins(x=0.005)
    #plt.xlim(left=1)
    plt.xlim(left=0)
    plt.ylim(bottom=0)
    #plt.ylim(bottom=1)
    plt.margins(y=0.005)
    plt.tick_params(axis="both", which="major", bottom=True, left=True)
    narprt_hist.set_xticks(bin_edges)
    plt.axvline(x=both_haps_NAs.median(), color="red", linestyle="--", label=f"Median: {both_haps_NAs.median():.2f}")
    plt.legend()
    narprt_hist_fig = narprt_hist.get_figure()
    narprt_hist_fig.savefig(args.outdir / f"{args.samples_set_id}.{args.chrom}.NAs.bothHaps.hist.png")
    narprt_hist_fig.savefig(args.outdir / f"{args.samples_set_id}.{args.chrom}.NAs.bothHaps.hist.pdf")

    #######################
    #    Out of Bounds    #
    #######################

    # Sanity check that all values fall between 0 and 1 in methyl_hap[12]
    logger.info(f"Sanity checking that methyl_hap1 and methyl_hap2 have all values between 0 and 1 (inclusive), other than NA, of course. Will plot all out-of-bounds values IF any such values exist.")
    methyl_hap1_outOfBounds = methyl_hap1.where((methyl_hap1 < 0) | (methyl_hap1 > 1), np.nan)
    if methyl_hap1_outOfBounds.notna().any().any():
        logger.warning(f"methyl_hap1 ({hap1_label}) contains 1+ values not between 0 and 1 (inclusive) which was NOT expected (not counting NAs).")
        methyl_hap1_outOfBounds = pd.melt(methyl_hap1_outOfBounds, var_name='x', value_name='y')
        methyl_hap1_outOfBounds['x'] = pd.to_numeric(methyl_hap1_outOfBounds['x'])
        plt.figure()
        hap1_scatter = sns.scatterplot(data=methyl_hap1_outOfBounds, x='x', y='y', s=0.1, color="black", alpha=0.2, edgecolor="none")
        plt.title(f"methyl_hap1 out-of-bounds Scatterplot")
        plt.xlabel(f"{args.chrom} Position")
        plt.ylabel(f"methyl_hap1 ({hap1_label}) Methylation")
        plt.margins(x=0.005)
        plt.xlim(left=0)
        #plt.ylim(0,1)
        plt.margins(y=0.005)
        plt.tick_params(axis="both", which="major", bottom=True, left=True)
        hap1_scatter_fig = hap1_scatter.get_figure()
        hap1_scatter_fig.savefig(args.outdir / f"{args.samples_set_id}.{args.chrom}.hap1-{hap1_label}-outOfBounds.scatter.png")
        hap1_scatter_fig.savefig(args.outdir / f"{args.samples_set_id}.{args.chrom}.hap1-{hap1_label}-outOfBounds.scatter.pdf")
    else:
        logger.info(f"methyl_hap1 ({hap1_label}) contains only values between 0 and 1 (inclusive) as expected, ignoring any NAs.")

    methyl_hap2_outOfBounds = methyl_hap2.where((methyl_hap2 < 0) | (methyl_hap2 > 1), np.nan)
    if methyl_hap2_outOfBounds.notna().any().any():
        logger.warning(f"methyl_hap2 ({hap2_label}) contains 1+ values not between 0 and 1 (inclusive) which was NOT expected (not counting NAs).")
        methyl_hap2_outOfBounds = pd.melt(methyl_hap2_outOfBounds, var_name='x', value_name='y')
        methyl_hap2_outOfBounds['x'] = pd.to_numeric(methyl_hap2_outOfBounds['x'])
        plt.figure()
        hap2_scatter = sns.scatterplot(data=methyl_hap2_outOfBounds, x='x', y='y', s=0.1, color="black", alpha=0.2, edgecolor="none")
        plt.title(f"methyl_hap2 out-of-bounds Scatterplot")
        plt.xlabel(f"{args.chrom} Position")
        plt.ylabel(f"methyl_hap2 ({hap2_label}) Methylation")
        plt.margins(x=0.005)
        plt.xlim(left=0)
        #plt.ylim(0,1)
        plt.margins(y=0.005)
        plt.tick_params(axis="both", which="major", bottom=True, left=True)
        hap2_scatter_fig = hap2_scatter.get_figure()
        hap2_scatter_fig.savefig(args.outdir / f"{args.samples_set_id}.{args.chrom}.hap2-{hap2_label}-outOfBounds.scatter.png")
        hap2_scatter_fig.savefig(args.outdir / f"{args.samples_set_id}.{args.chrom}.hap2-{hap2_label}-outOfBounds.scatter.pdf")
    else:
        logger.info(f"methyl_hap2 ({hap2_label}) contains only values between 0 and 1 (inclusive) as expected, ignoring any NAs.")
        
    ##################################
    #    Methylation Scatterplots    #
    ##################################

    # plot methyl_hap1 & methyl_hap2 to view the distributions at each position
    logger.debug(f"methyl_hap1 scatterplot")
    methyl_hap1_for_plotting = methyl_hap1.copy(deep=True)
    methyl_hap1_for_plotting = pd.melt(methyl_hap1_for_plotting, var_name='x', value_name='y')
    methyl_hap1_for_plotting['x'] = pd.to_numeric(methyl_hap1_for_plotting['x'])
    plt.figure()
    hap1_scatter = sns.scatterplot(data=methyl_hap1_for_plotting, x='x', y='y', s=0.1, color="black", alpha=0.2, edgecolor="none")
    plt.title(f"methyl_hap1 Scatterplot")
    plt.xlabel(f"{args.chrom} Position")
    plt.ylabel(f"methyl_hap1 ({hap1_label}) Methylation")
    plt.margins(x=0.005)
    plt.xlim(left=0)
    #plt.ylim(0,1)
    plt.margins(y=0.005)
    plt.tick_params(axis="both", which="major", bottom=True, left=True)
    hap1_scatter_fig = hap1_scatter.get_figure()
    hap1_scatter_fig.savefig(args.outdir / f"{args.samples_set_id}.{args.chrom}.hap1-{hap1_label}.scatter.png")
    hap1_scatter_fig.savefig(args.outdir / f"{args.samples_set_id}.{args.chrom}.hap1-{hap1_label}.scatter.pdf")

    logger.debug(f"methyl_hap2 scatterplot")
    methyl_hap2_for_plotting = methyl_hap2.copy(deep=True)
    methyl_hap2_for_plotting = pd.melt(methyl_hap2_for_plotting, var_name='x', value_name='y')
    methyl_hap2_for_plotting['x'] = pd.to_numeric(methyl_hap2_for_plotting['x'])
    plt.figure()
    hap2_scatter = sns.scatterplot(data=methyl_hap2_for_plotting, x='x', y='y', s=0.1, color="black", alpha=0.2, edgecolor="none")
    plt.title(f"methyl_hap2 Scatterplot")
    plt.xlabel(f"{args.chrom} Position")
    plt.ylabel(f"methyl_hap2 ({hap2_label}) Methylation")
    plt.margins(x=0.005)
    plt.xlim(left=0)
    #plt.ylim(0,1)
    plt.margins(y=0.005)
    plt.tick_params(axis="both", which="major", bottom=True, left=True)
    hap2_scatter_fig = hap2_scatter.get_figure()
    hap2_scatter_fig.savefig(args.outdir / f"{args.samples_set_id}.{args.chrom}.hap2-{hap2_label}.scatter.png")
    hap2_scatter_fig.savefig(args.outdir / f"{args.samples_set_id}.{args.chrom}.hap2-{hap2_label}.scatter.pdf")
    
    ################################
    #    Difference Scatterplot    #
    ################################

    # plot methyl_hap1_minus2 to view the distributions at each position
    logger.debug(f"methyl_hap1_minus2 scatterplot")
    methyl_hap1_minus2_for_plotting = methyl_hap1_minus2.copy(deep=True)
    if methyl_hap1_minus2_for_plotting.where((methyl_hap1_minus2_for_plotting < -1) | (methyl_hap1_minus2_for_plotting > 1), np.nan).notna().any().any():
        logger.critical(f"methyl_hap1_minus2_for_plotting contains values outside the range [-1,1] (ignoring NAs) which is _not_ expected.")
        #methyl_hap1_minus2_for_plotting.to_csv(args.outdir / f"{args.samples_set_id}.{args.chrom}.methyl_hap1_minus2_for_plotting.txt", sep='\t', header=True, na_rep="NA")
        sys.exit(1)
    methyl_hap1_minus2_for_plotting = pd.melt(methyl_hap1_minus2_for_plotting, var_name='x', value_name='y')
    methyl_hap1_minus2_for_plotting['x'] = pd.to_numeric(methyl_hap1_minus2_for_plotting['x'])
    plt.figure()
    diff_scatter = sns.scatterplot(data=methyl_hap1_minus2_for_plotting, x='x', y='y', s=0.1, color="black", alpha=0.2, edgecolor="none")
    plt.title(f"methyl_hap1_minus2 Scatterplot")
    plt.xlabel(f"{args.chrom} Position")
    plt.ylabel(f"Difference in Methylation ({hap1_label}-{hap2_label})")
    plt.margins(x=0.005)
    plt.xlim(left=0)
    #plt.ylim(-1,1)
    plt.margins(y=0.005)
    plt.tick_params(axis="both", which="major", bottom=True, left=True)
    diff_scatter_fig = diff_scatter.get_figure()
    diff_scatter_fig.savefig(args.outdir / f"{args.samples_set_id}.{args.chrom}.diff.scatter.png")
    diff_scatter_fig.savefig(args.outdir / f"{args.samples_set_id}.{args.chrom}.diff.scatter.pdf")

    ##########################
    #    Mean Scatterplot    #
    ##########################

    # plot X_mean to view the distributions at each position
    #logger.debug(f"X_mean scatterplot")
    #X_mean_for_plotting = X_train.loc[:, [col for col in columns if col.endswith(".mean")]]
    #if X_mean_for_plotting.where((X_mean_for_plotting < 0) | (X_mean_for_plotting > 1), np.nan).notna().any().any():
    #    logger.debug(f"X_mean_for_plotting contains values outside the range [0,1] (ignoring NAs) which is _not_ expected.")
    #    X_mean_for_plotting.to_csv(args.outdir / f"{args.samples_set_id}.{args.chrom}.X_mean_for_plotting.txt", sep='\t', header=True, na_rep="NA")
    #    #sys.exit()
    #X_mean_for_plotting.columns = X_mean_for_plotting.columns.str.rstrip(".mean")
    #X_mean_for_plotting = pd.melt(X_mean_for_plotting, var_name='x', value_name='y')
    #X_mean_for_plotting['x'] = pd.to_numeric(X_mean_for_plotting['x'])
    #plt.figure()
    #mean_scatter = sns.scatterplot(data=X_mean_for_plotting, x='x', y='y', s=0.1, color="black", alpha=0.2, edgecolor="none")
    #plt.title(f"X_mean Scatterplot")
    #plt.xlabel(f"{args.chrom} Position")
    #plt.ylabel(f"Mean Methylation between Haplotypes")
    #plt.margins(x=0.005)
    #plt.xlim(left=0)
    ##plt.ylim(0,1)
    #plt.margins(y=0.005)
    #plt.tick_params(axis="both", which="major", bottom=True, left=True)
    #mean_scatter_fig = mean_scatter.get_figure()
    #mean_scatter_fig.savefig(args.outdir / f"{args.samples_set_id}.{args.chrom}.mean.scatter.png")
    #mean_scatter_fig.savefig(args.outdir / f"{args.samples_set_id}.{args.chrom}.mean.scatter.pdf")

    ###################################
    #    Diff -x- Mean Scatterplot    #
    ###################################

    # do a diff -x- mean scatterplot
    #logger.debug(f"X_diff -x- X_mean Scatterplot")
    #X_mean_and_diff_for_plotting = pd.merge(X_mean_for_plotting, X_diff_for_plotting, on='x', suffixes=(".mean", ".diff"))
    #logger.debug(f"X_mean_and_diff_for_plotting:\n{X_mean_and_diff_for_plotting}")
    #plt.figure()
    #mean_diff_kde = sns.jointplot(data=X_mean_and_diff_for_plotting, x="y.mean", y="y.diff", kind="kde")
    ##mean_diff_scatter = sns.scatterplot(data=X_mean_and_diff_for_plotting, x="y.mean", y="y.diff", alpha=0.2, edgecolor="none")
    #plt.title(f"{args.chrom} X_mean -x- X_diff KDE")
    ##plt.title(f"{args.chrom} X_mean -x- X_diff Scatterplot")
    #plt.xlabel(f"Mean Methylation between Haplotypes")
    #plt.ylabel(f"Difference in Methylation between Haplotypes")
    #plt.margins(x=0.005)
    #plt.xlim(left=0)
    ##plt.ylim(0,1)
    #plt.margins(y=0.005)
    #plt.tick_params(axis="both", which="major", bottom=True, left=True)
    ##mean_diff_kde_fig = mean_diff_kde.get_figure() # <-- it's already a figure-level object instead of an Axes-level object
    ##mean_diff_kde_fig.savefig(args.outdir / f"{args.samples_set_id}.{args.chrom}.mean-x-diff.jointKde.png")
    ##mean_diff_kde_fig.savefig(args.outdir / f"{args.samples_set_id}.{args.chrom}.mean-x-diff.jointKde.pdf")
    #mean_diff_kde.savefig(args.outdir / f"{args.samples_set_id}.{args.chrom}.mean-x-diff.jointKde.png")
    #mean_diff_kde.savefig(args.outdir / f"{args.samples_set_id}.{args.chrom}.mean-x-diff.jointKde.pdf")


    #######################
    #    Diff Variance    #
    #######################

    # let's plot the variance here for investigating options for our
    # VarianceThreshold (if we end up using it). ddof=1 is default and standard
    # for samples of data instead of the full population of data. axis=0 means
    # over the index instead of columns, which, unintuitively, gives me
    # column-level variances.
    methyl_hap1_minus2_var = methyl_hap1_minus2.var(axis=0, skipna=True, ddof=1) # returns a Series with column names and scalar values for variance
    logger.debug(f"Here's what the methyl_hap1_minus2 variance looks like (shape: {methyl_hap1_minus2_var}):\n{methyl_hap1_minus2_var}")
    logger.debug(f"Summary information for that variance:\n{methyl_hap1_minus2_var.describe()}")
    logger.debug(f"methyl_hap1-minus2 variances scatterplot")
    plt.figure()
    methyl_hap1_minus2_var_hist = sns.histplot(data=methyl_hap1_minus2_var, kde=True)
    plt.title(f"methyl_hap1_minus2 Variances distribution")
    plt.xlabel(f"Variance")
    plt.ylabel(f"Frequency")
    plt.margins(x=0.005)
    plt.xlim(left=0)
    plt.ylim(bottom=0)
    plt.margins(y=0.005)
    plt.tick_params(axis="both", which="major", bottom=True, left=True)
    plt.axvline(x=methyl_hap1_minus2_var.median(), color="red", linestyle="--", label=f"Median: {methyl_hap1_minus2_var.median():.2f}")
    plt.legend()
    methyl_hap1_minus2_var_hist_fig = methyl_hap1_minus2_var_hist.get_figure()
    methyl_hap1_minus2_var_hist_fig.savefig(args.outdir / f"{args.samples_set_id}.{args.chrom}.methyl_hap1_minus2.variances.hist.png")
    methyl_hap1_minus2_var_hist_fig.savefig(args.outdir / f"{args.samples_set_id}.{args.chrom}.methyl_hap1_minus2.variances.hist.pdf")

    ## fill NAs in .diff columns with 0
    #logger.info(f"Filling NAs in methyl_hap1_minus2 with 0")
    #methyl_hap1_minus2.fillna(value=0, inplace=True)
    #logger.debug(f"Here's what the cleaned and NA->0 methyl_hap1_minus2 (i.e., {hap1_label}-{hap2_label}) looks like (shape: {methyl_hap1_minus2.shape}):\n{methyl_hap1_minus2}")

    ## rename the columns to add ".diff" or ".mean" since they'll otherwise have
    ## the same names (e.g., 1234 -> 1234.diff and 1234.mean)
    ##logger.info(f"Appending '.diff' and '.mean' to the columns of diff and mean")
    #logger.info(f"Appending '.diff' to the columns of diff")
    #methyl_hap1_minus2 = methyl_hap1_minus2.add_suffix(".diff")
    ##methyl_mean = methyl_mean.add_suffix(".mean")

    logger.info("Completed")

# ------------- MAIN ----------------------------- ||
if __name__ == "__main__":
    _main()

