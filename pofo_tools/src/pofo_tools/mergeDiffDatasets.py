#! /usr/bin/env python3

#__author__ == "Brandon Pickett"

#---------------------- IMPORTS ---------------------------------------------||
import sys
import re
import pandas as pd
import numpy as np
import argparse
import pathlib as pl
import logging

from pofo_tools.utils import *

#---------------------- Misc Global Settings --------------------------------||
pd.options.mode.copy_on_write = True

#---------------------- Logging ---------------------------------------------||
logger = logging.getLogger(__name__)
__all__ = ["mergeDiffDatasets"] # prevent unlisted functions from being imported via `from ... import *`

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

    parser = argparse.ArgumentParser(prog=sys.argv[0], description="Merge diff datasets based on chromosome (ignoring other label columns, so we're assuming this isn't an issue at this time). See the help message about -d|--dataset for a description of the dataset.")
    parser.add_argument("-c", "--chrs", "--chromosomes", metavar="STR", type=str, action="store", dest="chroms", nargs='+', help="The chromosome that is associated with each of the TSVs provided to -d|--datasets. The lists must be parallel. If omitted, an attempt will be made to autodetect the chromosome names from the filenames (but not the contents).", default=None, required=False)
    parser.add_argument("-d", "--datasets", metavar="FILE", type=pl.Path, action="store", dest="dataset_fns", nargs='+', help="The datasets TSVs. Each may optionally have label columns at the beginning. It must have a Sample column and LeftOpHapLabel and RightOpHapLabel columns. It should have at least N other columns, where N is the number of CpG sites in the dataset for that chromosome. Each chromosome may have a different number of columns for N. It should have M+1 rows, where M is the number of samples and 1 is for the header row. Other than the optional label columns and the sample and Left/Right label columns that contain strings (and any other label columns), all other values should be numeric. Represent NAs, if any, with \"NA\". The column order doesn't matter. You must provide 2+ files to this parameter.", required=True)
    parser.add_argument("-l", "--log-file", metavar="FILE", type=pl.Path, action="store", dest="logfn", help="The output file to write logging output to. [default: no log file]", default=None, required=False)
    parser.add_argument("-o", "--output-file", metavar="FILE", type=pl.Path, action="store", dest="outfn", help="The output filename to write merged file to. [default: $PWD]", default=".", required=False)
    if len(sys.argv) == 1: sys.argv.append("-h")
    args = parser.parse_args()
    if len(args.dataset_fns) < 2:
        raise argparse.ArgumentError(f"You must provide 2+ dataset filenames, you provided {len(args.dataset_fns)}.")
    if args.chroms and len(args.chroms) != len(args.dataset_fns):
        raise argparse.ArgumentError(f"You provided {len(args.dataset_fns)} dataset files, and the number of chromosomes ({len(args.chroms)}) did not match.")
    if not args.chroms:
        args.chroms = []
        for dataset_fn in args.dataset_fns:
            args.chroms.append(processChromosomeArg(dataset_fn, "auto")) # may also raise argparse.ArgumentError
    return args

#---------------------- Classes ---------------------------------------------||

#---------------------- Functions -------------------------------------------||
def mergeDiffDatasets(dataset_fns, chroms):
    '''
    Merge 2+ 'diff' datasets.

    Presumabely, each dataset partition is from the same general dataset, just
    from different chromosomes. This is the expectation; however, this is not
    strictly necessary. All that is technically required is for the two input
    lists to be parallel.

    Args:
        dataset_fns (list): paths (pl.Path) to dataset filenames. This list is
            parallel to the `chroms` list.
        chroms (list):  chromosome names (str). This list is parallel to the
            `dataset_fns` list.

    Returns:
        pd.DataFrame: the pandas DataFrame resulting from merging the DataFrames
            that were created from the individual files.
    '''

    #logger.debug(f"Chromosome to diff dataset filename mapping:\n{'\n'.join([f'{c}\t{d}' for c, d in sorted(zip(chroms, dataset_fns), key=lambda x: versionSortKey(x[0]))])}") # works only for python 3.12+
    logger.debug(f"Chromosome to diff dataset filename mapping:\n" + '\n'.join(f"{chrom_dataset_fn_pair[0]}\t" + str(chrom_dataset_fn_pair[1]) for chrom_dataset_fn_pair in sorted(zip(chroms, dataset_fns), key=lambda x: versionSortKey(x[0])))) # needed for python <3.12

    # do the merging
    logger.info("Looping through dataset files...")
    melted_dfs = []
    for dataset_fn, chrom in zip(dataset_fns, chroms):
        # parse the file
        logger.info("Parsing {dataset_fn}...")
        d = parseDiffDatasetFile(dataset_fn)

        # extract category column names
        cat_cols = [col_name for col_name in d.columns if re.fullmatch(r"[0-9]+", col_name) is None]

        # add chromosome if needed
        if not "Chromosome" in cat_cols:
            logger.info("Adding Chromosome ({chrom}) column to datatset from {dataset_fn}")
            d["Chromosome"] = chrom
            cat_cols.append("Chromosome")

        # melt
        logger.info("Melting dataset (from {dataset_fn})...")
        d = pd.melt (d, id_vars=cat_cols, var_name="Position", value_name="diffMethyl", ignore_index=True)

        # add the melted dataframe to the list of melted dataframes for later joining
        melted_dfs.append(d)
    
    # merge
    logger.info("Merging all melted datasets...")
    d = pd.concat(melted_dfs, copy=False, ignore_index=True)

    # pivot (widen/unmelt)
    logger.info("Widen (pivot) merged, melted datasets...")
    cat_cols = [col_name for col_name in d.columns if not col_name in ["Position", "diffMethyl"]]
    d = flattenMultiColumns(pd.pivot(d, index=cat_cols, columns=["Position"], values=["diffMethyl"]).reset_index())

    # return
    return d

def _main():
    # parse args
    args = _parseArgs()

    # setup logging for running this file directly
    setupLogging(logfn=args.logfn)

    # report the command
    logger.info(f"Arguments provided to this python script: {' '.join(sys.argv)}")

    # do the merging
    d = mergeDiffDatasets(args.dataset_fns, args.chroms)

    # create output directory
    args.outfn.parent.mkdir(mode=0o2775, parents=True, exist_ok=True)

    # write
    logger.info(f"Writing pivoted merged dataset to {args.outfn}...")
    d.to_csv(args.outfn, mode='w', sep='\t', header=True, index=False, float_format="{:g}".format, na_rep="NA")

    # report completion
    logger.info(f"Merging datasets completed.")

# ------------- MAIN ----------------------------- ||
if __name__ == "__main__":
    _main()

