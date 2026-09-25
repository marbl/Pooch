#! /usr/bin/env python3

#__author__ == "Brandon Pickett"

#---------------------- IMPORTS ---------------------------------------------||
import sys
import re
import pickle
import pandas as pd
#import numpy as np
from collections import defaultdict
import argparse
import pathlib as pl
import logging

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

    parser = argparse.ArgumentParser(prog=sys.argv[0], description="Create a \"diff\" dataset from a \"methyl\" dataset. The output diff dataset is a TSV file with 3+ categorical label columns followed by the differences (floats) between the two input haplotypes' methylation values at every position. The 3 required categorical columns are Sample, LeftOpHapLabel, and RightOpHapLabel (e.g., HG002, mat, and pat, signifying that the paternal haplotype methylation values were subtracted from the maternal haplotype methylation values for sample HG002). Other categorical columns could be added, e.g., Chromosome. Note that this program operates on only a single chromosome at a time, but other programs may handle multiple (e.g., mergeDiffDatasets). The column headers for the non-categorical columns are integers representing a position along the chromsome. Any NAs in the input methyl dataset on either haplotype will result in a 0 in the output diff dataset; this also means that there will be no NAs in the output (assuming none are present in any non-sample, non-haplotype-label categorical columns)..", epilog="Please note that, in theory, there may be NAs in a diff dataset when multiple chromosomes are represented in the same file because CpG sites will be at different position on different chromosomes. In such a case, it's not really 'missing' data as much as a convenience for storage by using a sparse matrix to avoid a larger matrix when positions collide between chromosomes. This fact should be considered when extracting data from such a diff dataset, but it is not a concern for this specific program because it operates on only a single chromosome at a time.")
    parser.add_argument("-C", "--emit-chr", action="store_true", dest="emit_chrom", help="Emit a 'Chromosome' column with the values matching the chromosome specified with -c|--chr. If such a column already exists in the input methyl dataset, it will be output regardless of whether this option is specified (it won't be duplicated if this option is specified). Adding this option when such a column is not already present in the input will ensure it is added to the output. [default: emit only if already in input methyl dataset]", required=False)
    parser.add_argument("-c", "--chr", metavar="STR", type=str, action="store", dest="chrom", help="The chromosome to run. Technically, this could be any sequence identifier from a fasta file. It should match the sequence ID. Thus, if your fasta file has \"chrX\", provide \"chrX\", not \"X\". A special value of \"auto\" (case insensitive) may be specified to infer the chromosome from the dataset filename (provided to -i|--input-methyl-dataset); it will use the rightmost occurance of the following regex: `chr[0-9A-Za-z]+`. [default: auto]", default="auto", required=False)
    parser.add_argument("-i", "--input-methyl-dataset", metavar="FILE", type=pl.Path, action="store", dest="methyl_dataset_fn", help="The input methyl dataset TSV. It should have >=(N+2) columns, where N is the number of CpG sites in the dataset and the 2 extra are for the sample (e.g., HG01234) and label (e.g., mat/pat or hap1/hap2). It may have extra categorical columns. It should have 2M+1 rows, where M is the number of samples, 2 is the number of haplotypes, and 1 is for the header row. Other than the sample and label columns (and any extra categorical columns) that contain strings, all other values should be numeric. Represent NAs, if any, with \"NA\". The column order doesn't matter as long as there is a header with the non-position columns named as \"Label\" and \"Sample\" (and whatever any extra categorical columns are called) and the rest as integers representing the position of the CpG site in the reference sequence (e.g., 13859).", required=True)
    parser.add_argument("-L", "--left-op-hap-label",  metavar="STR", type=str, action="store", dest="left_op_hap_label",  help="The haplotype label (e.g., hap1, hap2, mat, or pat) for the haplotype that will be the left operand in the subtraction to calculate methylation differences at every position, i.e., A in A - B. Reserved word 'auto' (case-insensitive) will infer the order from what was provided to -R|--right-op-hap-label, if provided, or, otherwise, from the first label found in the file provided to -i|--input-methyl-dataset. [default: auto]", default="auto", required=False)
    parser.add_argument("-l", "--log-file", metavar="FILE", type=pl.Path, action="store", dest="logfn", help="The output file to write logging output to. [default: no log file]", default=None, required=False)
    parser.add_argument("-R", "--right-op-hap-label", metavar="STR", type=str, action="store", dest="right_op_hap_label", help="See -L|--left-op-hap-label. This is the right operand instead of the left, i.e., B in A - B. [default: auto]", default="auto", required=False)
    parser.add_argument("-o", "--output-file", metavar="FILE", type=pl.Path, action="store", dest="outfn", help="The output diff dataset. [default: the same as -i|--input-methyl-dataset except with suffix .diffDataset.tsv]", default=None, required=False)
    if len(sys.argv) == 1: sys.argv.append("-h")
    args = parser.parse_args()
    args.chrom = processChromosomeArg(args.methyl_dataset_fn, args.chrom)
    if not args.outfn: args.outfn = args.methyl_dataset_fn.with_suffix(".diffDataset.tsv")
    return args

#---------------------- Classes ---------------------------------------------||

#---------------------- Functions -------------------------------------------||
def _determineOperandHapLabels(hap1_label, hap2_label, left_op_hap_label, right_op_hap_label):
    '''
    Determine which hap label is meant to be left and right

    The user will specify left and right operand via -L and -R. If just one is
    provided, the other can be inferred. If neither are provided, the order in
    the input to this function will be used (i.e., hap1_label then hap2_label),
    which, presumabely, is the order the labels were found in the input methyl
    dataset file. -L and/or -R will be considered "not provided" by the user if
    their values are 'auto' (case-insensitive). The purpose of this function is
    to (a) perform the inference, if needed, and (b) allow for case differences
    in the labels as found in the input methyl dataset file and as provided via
    -L and -R.

    Args:
        hap1_label (str): the label of the first haplotype
        hap2_label (str): the label of the second haplotype
        left_op_hap_label (str): the label of the haplotype that is meant to be
            the left operand in the subtraction used to create the difference.
            String 'auto' has special meaning (see function description).
        right_op_hap_label (str): the label of the haplotype that is meant to be
            the right operand in the subtraction used to create the difference
            String 'auto' has special meaning (see function description).

    Returns:
        str: the label that is meant to be the left operand
        str: the label that is meant to be the right operand

    Raises:
        argparse.ArgumentError: raised if hap1_label and hap2_label are not
            case-insensitive matches to left_op_hap_label and
            right_op_hap_label, not respectively.
    '''

    # lowercase the input labels for case-insensitive comparison
    l1 = hap1_label.lower()
    l2 = hap2_label.lower()
    ll = left_op_hap_label.lower()
    lr = right_op_hap_label.lower()

    # handle special case where both ll and lr are "auto"
    if ll == "auto" and lr == "auto": # both ll and lr are auto, use order provided by hap1_label and hap2_label
        return hap1_label, hap2_label

    # if either ll or lr are "auto" (but not both), infer the other from the
    # provided one and l1/l2 and return the labels in the appropriate order
    if ll == "auto": # and lr != "auto", i.e., infer lr label from ll and l1/l2
        if lr == l1:
            return hap2_label, hap1_label
        if lr == h2:
            return hap1_label, hap2_label
        #err_msg = f"the hap label for the left operand of the difference subtraction was meant to be inferred from the right operand hap label, but the right operand hap label ({right_operand_hap_label}) matched neither haplotype label: {hap1_label}, {hap2_label}"
        err_msg = f"the hap label for the left operand of the difference subtraction was meant to be inferred from the right operand hap label, but the right operand hap label ({left_op_hap_label}) matched neither haplotype label: {hap1_label}, {hap2_label}" # juhyun modified
        logger.critical(err_msg)
        raise argparse.ArgumentError(err_msg)

    if lr == "auto": # and ll != "auto", i.e., infer ll label from lr and l1/l2
        if ll == l1:
            return hap1_label, hap2_label
        if ll == h2:
            return hap2_label, hap1_label
        # err_msg = f"the hap label for the right operand of the difference subtraction was meant to be inferred from the left operand hap label, but the left operand hap label ({left_operand_hap_label}) matched neither haplotype label: {hap1_label}, {hap2_label}"
        err_msg = f"the hap label for the right operand of the difference subtraction was meant to be inferred from the left operand hap label, but the left operand hap label ({right_op_hap_label}) matched neither haplotype label: {hap1_label}, {hap2_label}" # juhyun modified
        logger.critical(err_msg)
        raise argparse.ArgumentError(err_msg)

    # neither ll nor lr were "auto", no inference required. Simply check for
    # matching and order
    if l1 == ll and l2 == lr:
        return hap1_label, hap2_label
    if l2 == ll and l1 == lr:
        return hap2_label, hap1_label
    # err_msg = f"One or both of the left and right operand hap labels ({left_operand_hap_label}, {right_operand_hap_label}) did not match either of the expected haplotype labels: {hap1_label}, {hap2_label}"
    err_msg = f"One or both of the left and right operand hap labels ({left_op_hap_label}, {right_op_hap_label}) did not match either of the expected haplotype labels: {hap1_label}, {hap2_label}" # juhyun modified
    print(err_msg)
    logger.critical(err_msg)
    # raise argparse.ArgumentError(err_msg)
    raise ValueError(err_msg) # juhyun modified

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
    methyl = parseMethylDatasetFile(args.methyl_dataset_fn)
    logger.debug(f"parseMethylDatasetFile is ready")
    # extract the samples labels for future use
    samples = sorted(methyl["Sample"].cat.categories.to_list())
    hap1_label, hap2_label = sorted(methyl["Label"].cat.categories.to_list()) # guaranteed to have only 2 labels after using parseMethylDatasetFile
    hap1_label, hap2_label = _determineOperandHapLabels(hap1_label, hap2_label, args.left_op_hap_label, args.right_op_hap_label)
    # NOTE: hap1_label and hap2_label can hereafter be considered
    # left_op_hap_label and right_op_hap_label, respectively, i.e., we'll be
    # doing hap1 - hap2 to get the difference. Technically, the order doesn't
    # matter, but we're now doing the user's preferred order, if they specified
    # a preference.

    # split the data by haplotype and make labels implicit
    methyl_hap1, methyl_hap2, labels_hap1, labels_hap2, S, other_cats = splitMethylByHaps(methyl, hap1_label, hap2_label)

    # add Chromosome to other_cats if outputting Chromosome is desired and it
    # is not already there
    if args.emit_chrom and not "Chromosome" in other_cats.columns:
        other_cats["Chromosome"] = args.chrom

    # calculate the difference between the haplotypes (hap1 - hap2)
    logger.info(f"Subtracting the values in each haplotype from eachother to find the difference in methylation level at each CpG")
    methyl_hap1_minus2 = methyl_hap1 - methyl_hap2 # note: an NA in either slot will return NA
    logger.debug(f"Here's what methyl_hap1_minus2 (i.e., {hap1_label}-{hap2_label}) looks like (shape: {methyl_hap1_minus2.shape}):\n" + headTailDataframeForReporting(methyl_hap1_minus2))

    # fill NAs with 0
    logger.info(f"Filling NAs in methyl_hap1_minus2 with 0")
    methyl_hap1_minus2.fillna(value=0, inplace=True)
    logger.debug(f"Here's what the NA->0 methyl_hap1_minus2 (i.e., {hap1_label}-{hap2_label}) looks like (shape: {methyl_hap1_minus2.shape}):\n" + headTailDataframeForReporting(methyl_hap1_minus2))

    # write the output
    logger.info(f"Writing diff dataset to {args.outfn}")
    writeXasDiffDataset(methyl_hap1_minus2, S, hap1_label, hap2_label, args.outfn, **other_cats)
    # 'X' in this function name refers to a pd.DataFrame of just the
    # differences, so called because it is what will eventually be used as the
    # input for training and prediction. For training, it would need to be
    # duplicated first to account for the hap2 - hap1 order showing up during
    # prediction time.

# ------------- MAIN ----------------------------- ||
if __name__ == "__main__":
    _main()

