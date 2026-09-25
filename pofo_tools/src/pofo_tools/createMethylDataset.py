#! /usr/bin/env python3

#__author__ == "Brandon Pickett"

#---------------------- IMPORTS ---------------------------------------------||
import sys
import re
import pickle
import pandas as pd
#import numpy as np
from subprocess import Popen, PIPE
from collections import defaultdict
from timeit import default_timer as timer
from datetime import timedelta
import argparse
import pathlib as pl
import logging
from packaging import version

from pofo_tools.utils import *

#---------------------- Misc Global Settings --------------------------------||
pd.options.mode.copy_on_write = True

#---------------------- Logging ---------------------------------------------||
logger = logging.getLogger(__name__)
__all__ = ["readOneInputBedMethylFile", "filterSignificantLoci", "filterRegions", "filterByModel", "parseSampleSexMappingFile", "removeSamplesBasedOnSex"] # prevent unlisted functions from being imported via `from ... import *`

#---------------------- Command-line Args -----------------------------------||
def _processInputPaths(input_paths):
    '''
    Create a sample -x- haplotype -x- bedmethyl file mapping
    
    Creates the mapping from the PATH(s) provided to -i|--input-fofn, which has
    1 or two parameters: (1) a TSV file with the mapping and (2) optionally a
    directory from which relative paths are assumed to start, defaulting to
    ${PWD}. The TSV file with the mapping (first item) has 3 columns: sample,
    haplotype, and bedmethyl file. No header is expected in the TSV file.

    Args:
        input_paths (list): a list of type str. Must be length 1 or 2. The first
            string is a path to a TSV file. The second string, if present, is a
            directory from which relative paths found in the TSV file are
            assumed to begin with. The directory provided as the second string
            could itself be relative (presumabely relative to the present
            directory), but the intention is that it would be an absolute path.
            If omitted, relative paths are assumed to be relative to ${PWD}.
            The TSV file should have 3 columns (in order): sample, haplotype,
            and bedmethyl file. No header is expected in the TSV file.

    Returns:
        pd.DataFrame: a DataFrame with 3 columns: "sample", "haplotype", and
            "bedmethyl". Only two haplotypes are expected. Every sample should
            have exactly two rows (one with each haplotype). The bedmethyl
            files should each be unique.

    Raises:
        arparse.ArgumentError: if too few or too many args are provided.
        DatasetCreationException: if the dataset expectations for the
            sample/haplotype pairings are not met.

    '''
    #if len(input_paths) == 1:
    #    input_paths.append('.')
    if len(input_paths) == 0 or len(input_paths) > 2:
        raise argparse.ArgumentError(f"-i|--input-fofn must have only one or two arguments provided, {len(input_paths)} provided.")

    fofn = pl.Path(input_paths[0])
    base_dir = None if len(input_paths) == 1 else pl.Path(input_paths[1])

    df = pd.read_csv(fofn, sep='\t', header=None, names=["sample", "haplotype", "bedmethyl"], dtype=defaultdict(lambda: "str", haplotype="category"))
    df["bedmethyl"] = df["bedmethyl"].map(pl.Path)

    if base_dir: # i.e., if not base_dir is None
        mask = ~df["bedmethyl"].apply(lambda p: p.is_absolute())
        df.loc[mask, "bedmethyl"] = df.loc[mask, "bedmethyl"].apply(lambda p: base_dir / p)

    # check that we have expected uniqueness and pairs
    if not df["bedmethyl"].is_unique:
        raise DatasetCreationException(f"Expected all ({len(df)}) files in the bedmethyl column of {fofn} to be unique, {df['bedmethyl'].nunique()} were unique.")
    if df["sample"].nunique() != len(df) / 2:
        raise DatasetCreationException(f"Expected each sample to be present exactly twice (i.e., {len(df)} rows / 2 = {len(df)/2} samples). {df['sample'].nunique()} unique sample names were present.")
    if df["haplotype"].nunique() != 2:
        raise DatasetCreationException(f"Expected exactly 2 haplotypes, got {df['haplotype'].nunique()} unique haplotypes.")
    hap_counts = df.groupby("sample")["haplotype"].nunique()
    if not (hap_counts == 2).all():
        raise DatasetCreationException(f"Expected each sample to be paired with each of the 2 haplotypes exactly once. {(hap_counts == 2).sum()}/{len(hap_counts)} samples met this expectation.")

    return df

def _processSamplesListArg(samples):
    '''
    Process an argument containing a list of samples

    Args:
        samples (list): a list of length >=0 with str entries. If only one
            entry, it is treated as a file and the sample identifiers are read
            from the file, one entry per line. If multiple entries are in the
            list, each is treated as a sample identifier directly. If there are
            no entries, the list is simply returned as an empty list.

    Returns:
        list: a list with str entries, each representing a sample identifier
    '''

    # it would probably be better if we made this a custom Action, but this was
    # faster to write during dev :shrug:
    if len(samples) != 1: # if 0, return the empty list. if >1, return the list of samples as-is.
        return samples
    samples_fn = pl.Path(samples[0])
    if not samples_fn.exists():
        #print(f"ERROR: {samples_fn} does not exist", file=sys.stderr)
        #sys.exit(1)
        raise FileNotFoundError(f"samples filename ({samples_fn}) provided to -s|--samples does not exist")
    with open(samples_fn, 'r') as ifd:
        return [line.rstrip('\n') for line in ifd]

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

    parser = argparse.ArgumentParser(prog=sys.argv[0], description="Create a 'methyl' dataset from bedmethyl files, possibly filtering by DSS's DMRs and/or DMLs (for training) or by the positions in a model (for prediction). A 'methyl' dataset is a precursor to the final matrix that will be trained on. This dataset type has methylation values between 0 and 1 at every position for every haplotype, thus it has a column labeling the haplotype, a column labeling the sample, optionally other label columns, and the main data columns are positions with the values being the methylation state (float between 0 and 1).")
    parser.add_argument("-c", "--chr", metavar="STR", type=str, action="store", dest="chrom", help="The chromosome to run. Technically, this could be any sequence identifier from a fasta file. It should match the sequence ID. Thus, if your fasta file has \"chrX\", provide \"chrX\", not \"X\".", required=True)
    parser.add_argument("-E", "--error-exit", action="store_true", dest="error_exit", help="When reading and filtering bedmethyl files, raise exception on subprocess shell error (non-zero exit code) or when filtering (e.g., via -R) leaves zero CpGs. Without this flag, the behavior is to proceed with dataset creation but without the sample that the error occurred on.", required=False)
    #parser.add_argument("-i", "--input-dir", metavar="DIR", type=pl.Path, action="store", dest="input_dir", help="The input directory to read bedmethyls files from. [default: $PWD]", default=".", required=False)
    parser.add_argument("-i", "--input-fofn", metavar="PATH", type=pl.Path, action="store", dest="input_paths", nargs='+', help="Provide one or two PATHS. Only the first is required. The first is a TSV file with 3 columns: (1) sample ID, (2) haplotype, and (3) path to BED file. The BED file path must be absolute or relative to the second PATH provided to this option. If omitted, the second PATH is assumed to be ${PWD}.", default=[], required=True)
    parser.add_argument("-l", "--log-file", metavar="FILE", type=pl.Path, action="store", dest="logfn", help="The output file to write logging output to. [default: no log file]", default=None, required=False)
    parser.add_argument("-o", "--output-file", metavar="FILE", type=pl.Path, action="store", dest="outfn", help="The output file to write the dataset to.", required=True)
    #parser.add_argument("-s", "--samples", metavar="STR", type=str, action="store", dest="samples", nargs='+', help="Either (a) the input file with a list of samples one per line or (b) a list of samples provided directly as subsequent arguments.", required=True)
    parser.add_argument("-s", "--samples", metavar="STR", type=str, action="store", dest="samples", nargs='+', help="A subset of samples to use instead of the full set (from the file provided as the first PATH argument to -i|--input-fofn). STR must be either (a) an input file with a list of samples one per line or (b) a list of samples provided directly as subsequent arguments. [default: all samples provided in -i|--input-fofn]", default=[], required=False)
    parser.add_argument("-X", "--emit-sex", action="store_true", dest="emit_sex", help="Add a 'Sex' column to the output. This is the sex of the sample, not the sex of the parent providing any given haplotype. Requires providing the mapping of sample ID to sex label via -x|--sex-mapping-file. For the purposes of adding the sex label to the output, the labels need not conform to any prescribed set or spelling. Though, that is not the case if -c|--chr is 'chrX' (see description of -x|--sex-mapping-file). In any case, consistency is recommended (e.g., don't use 'male' for one sample and 'Male' for another).", required=False)
    parser.add_argument("-x", "--sex-mapping-file", metavar="FILE", type=pl.Path, action="store", dest="samples_sex_map_fn", help="File containing the sex labels for the samples provided to -s|--samples. The file should be tab-separated with one record per line. Each line should have two columns (in order): (1) sample identifier and (2) sex label. If the provided path is a directory (i.e., ending in /), a file named 'samples-sex.tsv' will be searched for in that directory. This option is only relevant for two situations: (a) if -X|--emit-sex is specified and/or (b) if -c|--chr is 'chrX'. For (a), see the description for -X|--emit-sex for further explanation. For (b), the information is needed to remove male samples because there are not two copies of chrX in male samples and both haplotypes are needed to create a dataset. Thus, if -c|--chr is 'chrX', the program will fail if this option (-x|--sex-mapping-file) is not provided (or a file in the default location does not exist). For the purposes of removing male samples for chrX, the sex label must match 'Male' (case insensitive) for the sample to be considered male. Even with the case insensitive matching, mixing case usage between samples for the same sex label is not a good idea. [default: $PWD/]", default=None, required=False)
    filtering_group = parser.add_argument_group("bedmethyl filtering", "The following options control how the bedmethyl files are filtered. By default, no filtering is applied. If creating a dataset for testing (independent test set) or prediction purposes (as opposed to training a new model), you'll need to provide the model to -M. If training, using -R is recommended. -M and -L/-R are mutally exclusive.")
    filtering_group.add_argument("-L", "--dss-dmls-file", metavar="FILE", type=pl.Path, action="store", dest="dss_dmls_fn", help="If a file is provided, filter the bedmethyl data to include only CpGs that match the position of a DML (DMLs pre-determined using DSS). If used in conjunction with -R|--dss-dmrs-file, only DMLs within the DMRs will be used; otherwise, all DMLs (even those outside DMRs) will be included. This is intended to be used when creating the training/dev sets and does not make sense to use when making an independent test set or when predicting on new data. If a directory is provided instead (i.e., provided path ends in /), a file named <chromosome>.dmls.tsv (where <chromosome> is defined by -c|--chr) will be searched for in the provided directory. [default: no filtering]", default=None, required=False)
    filtering_group.add_argument("-R", "--dss-dmrs-file", metavar="FILE", type=pl.Path, action="store", dest="dss_dmrs_fn", help="If a file is provided, filter the bedmethyl data to include only CpGs within a DMR (DMRs pre-determined using DSS). If used in conjunction with -L|--dss-dmls-file, CpGs within the DMRs that are not significantly different on their own (i.e., are not DMLs as reported by DSS) will be excluded; otherwise, all CpGs within a DMR will be included. This is intended to be used when creating the training/dev sets and does not make sense to use when making an independent test set or when predicting on new data. If a directory is provided instead (i.e., provided path ends in /), a file named <chromosome>.dmrs.tsv (where <chromosome> is defined by -c|--chr) will be searched for in the provided directory. [default: no filtering]", default=None, required=False)
    filtering_group.add_argument("-A", "--areastat-cumsum", metavar="INT", type=sciInt, action="store", dest="areastat_cumsum", help="If filtering by DMRs via -R|--filter-dss-dmrs, restrict the cumulative sum of DSS's areaStat to INT (internally, the DMRs will be sorted in descending order by absolute value of the areaStat before the cumulative sum is calculated). Once the provided cumulative sum is reached, the remaining DMRs will be ignored. This option has no effect unless -R is also used. [default: no limit]", default=0, required=False)
    filtering_group.add_argument("-M", "--filter-model", metavar="FILE", type=pl.Path, action="store", dest="model_file", help="Model file (pickled) for filtering the bedmethyl data to include only CpGs that match the position of a CpG from a pre-trained model. This is needed for testing (independent test set) or prediction tasks because we want only the CpGs that were used in the model. The model file is expected to be pickle of a ModelPackage object; the transformation pipeline portion is unused here, just the feature names (i.e., positions) in the trained model are used. [default: no filtering]", default=None, required=False)
    if len(sys.argv) == 1: sys.argv.append("-h")
    args = parser.parse_args()
    if ( args.dss_dmls_fn or args.dss_dmrs_fn ) and args.model_file:
        raise argparse.ArgumentError("-L and/or -R are mutually exclusive with -M. If you use -M, it doesn't make sense to use -L and/or -R, and vise versa.")
    args.must_parse_samples_sex_map_file = args.chrom == "chrX" or args.emit_sex #juhyun hash
    if args.must_parse_samples_sex_map_file and not args.samples_sex_map_fn:
        raise argparse.ArgumentError("-x was not provided but is required when either or both of the following are true: (a) -c|--chr is 'chrX' and (b) the -X|--emit-sex flag is used.")
    if args.samples_sex_map_fn is None: args.samples_sex_map_fn = pl.Path("samples-sex.tsv")
    args.bedmethyls_df = _processInputPaths(args.input_paths)
    args.samples = _processSamplesListArg(args.samples)
    if args.samples: # i.e., the list isn't empty
        args.bedmethyls_df = args.bedmethyls_df[args.bedmethyls_df["sample"].isin(args.samples)] # subset the bedmethyls_df by samples
        if not args.bedmethyls_df["sample"].isin(args.samples).all():
            raise argparse.ArgumentError("One or more samples provided with -s|--samples were not present in -i|--input-fofn.")
    else: # i.e., the list is empty
        args.samples = args.bedmethyls_df["sample"].unique().tolist() # populate samples from bedmethyls_df
    return args

#---------------------- Classes ---------------------------------------------||
# None

#---------------------- Functions -------------------------------------------||
def readOneInputBedMethylFile(in_fn, chrom):
    '''
    Read one input bedmethyl file from ONT's modkit into a pandas DataFrame.

    The columns we want (1-based) are 1 (chrom; str), 2 (pos; int), 10
    (Nvalid_cov; int), and 12 (Nmod; int). Only the chromosome (or other
    sequence identifier) will be read. For speed, the entire file is not read
    with pd.read_csv as you might normally expect. Instead, a process is
    spawned to `grep` (or `zgrep`) the bedmethyl lines for the chromosome of
    interest. The appropriate columns are extracted at this time via `cut`.
    This limits the amount of data that must be processed by read_csv (and type
    checked, etc.) only to be dropped immediately after to get only the columns
    of interest. The resulting dataframe will have the 4 columns described. The
    chrom column will be of type str. The other columns will be of type uint64.

    Args:
        in_fn (str, pl.Path): the input filename of the bedmethyl file. May be
            gzipped (i.e., end in .gz).
        chrom (str): the chromosome (or other sequence identifier).

    Returns:
        pd.DataFrame: The Pandas DataFrame created from reading in the input bedmethyl file.

    Raises:
        DatasetCreationException: if the spawned process returns with non-zero
            exit code and/or the file is empty (or has no entries for the
            desired chromosome).
    '''

    #                          1          2            4         5     <-- column numbers in original file for `cut`
    #column_names = [      "chr",     "pos",          "N",       "X"]  # <-- column names desired for DSS (which doesn't matter here, just specifying for reference)
    #column_names = [    "chrom",   "start", "Nvalid_cov",    "Nmod"]  # <-- column names according to bedmethyl spec
    column_names  = [    "chrom",     "pos", "Nvalid_cov",    "Nmod"]  # <-- column names I want to use
    #                        str        int           int        int

    grep_cmd = "grep"
    if in_fn.name.endswith(".gz"): grep_cmd = "zgrep"
    cmd = f"set -o pipefail; {grep_cmd} -Ew ^{chrom} {in_fn} | cut -d $'\\t' -f 1,2,4,5"
    #cmd = f"set -o pipefail; {grep_cmd} -Ew ^{chrom} {in_fn} "
    

    data = None
    with Popen(cmd, shell=True, executable="bash", stdout=PIPE) as process:
        data = pd.read_csv  (   process.stdout,
                                sep='\t',
                                header=None,
                                names=column_names,
                                dtype=defaultdict(lambda: "uint64", chrom="str")
                            )
        process.wait()
        if process.returncode != 0:
            raise DatasetCreationException(f"Command `{cmd}` exited with non-zero exit code {process.returncode}.")
    
    #if len(data) == 0:
    #    raise DatasetCreationException(f"Dataset for {chrom} from {in_fn} has zero rows. Hmmm...") # should have returned non-zero exit code already.

    return data

def filterSignificantLoci(loci, sig_loci):
    '''
    Filter out any loci not considered significant by DSS.

    Technically, only the position is considered. We assume that both `loci`
    and `sig_loci` have only one (the same) chromosome represented. The chrom
    columns are still required due to implementation details, but it's never
    checked whether only one chromosome is present and whether they match.

    Args:
        loci (pd.DataFrame): The loci that will be filtered (i.e., the loci that
            will be kept or discarded based on the filtering DataFrame
            `sig_loci`). Required columns: chrom (str, pd.Categorical): the
            chromosome or other sequence ID, pos (Integer): the zero-based
            position of the loci.
        sig_loci (pd.DataFrame): the loci that will act as the filter, i.e., if
            loci in `loci` match the loci in this, they will be kept. Required
            columns: chrom (str, pd.Categorical): the chromosome or other
            sequence ID, pos (Integer): the zero-based position of the loci.

    Returns:
        pd.DataFrame: a copy of `loci` after filtering out an loci not in `sig_loci`

    Raises:
        DatasetCreationException: exception raised if no loci remain after filtering
    '''

    # merge on pos (assume chroms are equal and only one chrom exists)
    result = pd.merge(loci, sig_loci[["chrom", "pos"]], how="right", on="pos", suffixes=('', ".delete"))

    # next, drop columns we don't care about (i.e., everything from sig_loci
    cols_to_drop = [col for col in result.columns if col.endswith(".delete")] # should remove the chrom (well, "chrom.delete" now) and pos ("pos.delete") column originating from sig_loci
    result.drop(columns=cols_to_drop, inplace=True)

    if len(result) == 0:
        raise DatasetCreationException(f"No CpGs remain after filtering out insignificant loci.")

    return result

def filterRegions(cpgs, good_regions):
    '''
    Remove CpGs outside "good" regions (e.g., differentially methylated by DSS)

    Args:
        cpgs (pd.DataFrame): The CpGs that will be filtered (i.e., the CpGs that
            will be kept or discarded based on the filtering DataFrame
            `good_regions`). Required column: pos (Integer): the zero-based
            position of the CpGs. It is assumed, but not technically required
            for this function, that there is the following column: chrom (str,
            pd.Categorical), representing the chromosome (or other sequence
            ID). It is assumed that all positions in `pos` are on the same
            chromosome and that only one chromosome (the same one) will be
            represented in `good_regions`.
        good_regions (pd.DataFrame): the regions that will act as the filter, i.e., if
            positions in `cpgs` fall within the regions in this, they will be
            kept. Required columns: start (Integer): the zero-based start
            position of the good region, end (Integer): the exclusive end
            position of the good region. The coordinate pairs from start and
            end form a zero-based, half-open---[start, end)---coordinate
            system. It is assumed, but not technically required for this
            function, that there is the following column: chrom (str,
            pd.Categorical), representing the chromosome (or other sequence
            ID). It is assumed that all regions represented by `start` and
            `end` are on the same chromosome and that only one chromosome (the
            same one) will be represnted in `cpgs`.

    Returns:
        pd.DataFrame: a copy of `cpgs` after filtering out an CpGs not in `sig_cpgs`

    Raises:
        DatasetCreationException: exception raised if no CpGs remain after filtering
    '''

    result = pd.merge_asof  (
                                cpgs,
                                pd.DataFrame({"start": good_regions["start"], "end": good_regions["end"]}),
                                left_on="pos", right_on="start",
                                direction="backward", allow_exact_matches=True
                            )
    result = result[result["pos"] <= result["end"]]
    result.drop(columns=["start", "end"], inplace=True)
    result.index = pd.RangeIndex(len(result))

    if len(result) == 0:
        raise DatasetCreationException(f"No CpGs remain after filtering out any in undesired regions.")

    return result

def filterByModel(cpgs, model):
    '''
    Remove CpGs not found in the model

    Args:
        cpgs (pd.DataFrame): The CpGs that will be filtered (i.e., the CpGs that
            will be kept or discarded based on the positions in the filtering
            model. Required column: pos (Integer): the zero-based position of
            the CpGs. It is assumed, but not technically required for this
            function, that there is the following column: chrom (str,
            pd.Categorical), representing the chromosome (or other sequence
            ID). It is assumed that all positions in `pos` are on the same
            chromosome and that only one chromosome (the same one) will be
            represented in `good_regions`.
        model (LogisticRegression): the model that will act as a filter, i.e., if
            positions in `cpgs` match positions in the model, they will be
            kept. Note, the LogisticRegression object is a pre-trained
            sklearn.linear_model.LogisticRegression object. The positions are
            taken from feature names in `feature_names_in_`, which is expected
            to be an array of type object where each element is a string
            representing an integer, which in turn is a position along the
            chromosome that the model used as a feature. The coordinates are
            assumed to be in the same "space" (e.g., zero-based), but no check
            is made to that effect. It is assumed that all positions that are
            features in the model are on a single chromosome - the same
            chromosome for which there are positions in `cpgs`.

    Returns:
        pd.DataFrame: a copy of `cpgs` after filtering out an CpGs not in
            `model.feature_names_in_`.

    Raises:
        DatasetCreationException: exception raised if no CpGs remain after filtering
    '''
    # # model.feature_names_in_ will be an array of strings representing integers, e.g., 
    # # # array(['123650', '123660', '123684', ..., '193389285', '193389299', '193389307'], dtype=object)
    # pd.DataFrame({"pos": model.feature_names_in_.astype(np.uint64)})

    model_feature = pd.DataFrame({"pos": model.feature_names_in_.astype(np.uint64)}) # juhyun modified
    cpgs['pos'] = cpgs['pos'].astype(np.uint64) # juhyun modified
    # merge on pos (assume chroms are equal and only one chrom exists)
    result = pd.merge(cpgs, model_feature, how="right", on="pos", suffixes=('', ".delete")) # juhyun modified

    # next, drop columns we don't care about (i.e., everything from sig_loci
    cols_to_drop = [col for col in result.columns if col.endswith(".delete")] # should remove the pos (well, "pos.delete" now) column originating from the temporary DataFrame made from model.feature_names_in_
    result.drop(columns=cols_to_drop, inplace=True)

    if len(result) == 0:
        raise DatasetCreationException(f"No CpGs remain after filtering out insignificant loci.")

    return result

def parseSampleSexMappingFile(samples_sex_mapping_fn, samples=None):
    '''
    Parse the file with samples to sex mapping into a dictionary

    No checking is done on the validity of the contents and format of
    `samples_sex_mapping_fn`. If conflicting entries exist, it is undefined
    which entry will be in the resulting dict.

    Args:
        samples_sex_mapping_fn (str, pl.Path): file with sex labels. One sample
            per line with two columns: sample ID and sex.
        samples (Iterable): list (or set/frozenset or any Iterable container)
            containing strings representing sample IDs. Provided sample IDs
            will be retained in returned dict. All other sample IDs found in
            `samples_sex_mapping_fn` will be omitted. Helpful if many samples
            are listed in `samples_sex_mapping_fn` and only a small subset are
            actually needed. Will reduce RAM usage, but only by a negligible
            amount unless there is a VAST disparity in number of samples.

    Returns:
        dict: mapping of sample IDs (str) to sex (str)

    Raises:
        FileParsingError: if the file parsing procedure results in an empty dict
            for some reason (e.g., `samples` not present in
            `samples_sex_mapping_fn`).
    '''

    samples_x_sex = {}
    with open(samples_sex_mapping_fn, 'r') as samples_sex_mapping_fd:
        # no subsetting req'd, process entire file, immediately returning dict
        if not samples: # i.e., if samples is None
            return dict(tuple(line.rstrip('\n').split('\t')) for line in samples_sex_mapping_fd.readlines())

        # subsetting requested, process line by line (to reduce RAM usage, even
        # if only temporarily and even if the difference will be negligible
        # except in unlikely cases)
        for line in samples_sex_mapping_fd:
            sample, sex = line.rstrip('\n').split('\t')
            if sample in samples:
                samples_x_sex[sample] = sex
            
    if not samples_x_sex:
        subsetting_msg = "No subsetting was requested (i.e., samples==None)" if samples is None else f"{len(samples)} sample IDs were provided in the subsetting Iterable"
        raise FileParsingError(f"The sample-to-sex mapping dict was empty after processing {samples_sex_mapping_fn}. {subsetting_msg}.")

    return samples_x_sex

def removeSamplesBasedOnSex(samples, samples_sex=None, samples_sex_fn=None, sex_to_remove="Male"):
    '''
    Remove `sex_to_remove` samples from `samples`

    `samples_sex` and `samples_sex_fn` are both optional, but one MUST be
    provided. If both are provided, `samples_sex_fn` will be ignored and a
    warning will be emitted.

    Args:
        samples (list): sample IDs, all of type str
        samples_sex (dict, optional): mapping of key: sample ID (str), value:
            sex (str). All samples from `samples` must be present as keys.
        samples_sex_fn (str, pl.Path, optional): file with sex labels. One
            sample per line with two columns: sample ID and sex. For a given
            sample to be removed from `samples`, the sex column here must have
            its sex label match `sex_to_remove` (case insensitive). All samples
            in `samples` must be present in the file. Note: the case of the sex
            label will not be changed even though the comparison in this
            function will be case insensitive.
        sex_to_remove (str, optional): the label of the sex to be removed. The
            comparision will be case insensitive. default: Male.

    Returns:
        list: copy of `samples` with `sex_to_remove` samples omitted. Entries in
            list are of type str. Result may be empty (a warning will be
            issued).

    Raises:
        KeyError: if any given sample in `samples` does not have a matching
            entry in `samples_sex` or the `samples_sex_fn` file.
    '''

    logger.debug(f"Removing any '{sex_to_remove}' samples from the following list of samples:\n{samples}")

    # complain if neither `samples_sex` nor `samples_sex_fn` were provided
    if not ( samples_sex or samples_sex_fn ):
        err_msg = f"Neither `samples_sex` nor `samples_sex_fn` were provided. Expect a KeyError to be raised imminently."
        logger.critical(msg)
        #raise SomeException(msg)

    # warn the user that `samples_sex_fn` will not be used if it and `samples_sex` were provided
    if samples_sex and samples_sex_fn:
        logger.warning(f"`samples_sex` and `samples_sex_fn` were both provided. Sample-to-sex mapping from `samples_sex` will be utilized while the mapping in `samples_sex_fn` will be ignored.")

    # parse `samples_sex_fn` if it was provided and `samples_sex` is not provided
    if ( not samples_sex ) and samples_sex_fn:
        samples_sex = parseSampleSexMappingFile(samples_sex_fn, samples=frozenset(samples))
        # Note: samples_sex should exist as a variable outside this if block
        # because it is a named argument in the function definition.

    # return the list of samples that weren't removed. Note: samples_sex dict
    # has keys that are sample name strings (e.g., "HG00508") and values that
    # are sex strings (e.g., "Male" and "Female")

    #return [sample for sample in samples if samples_sex[sample].lower() != sex_to_remove.lower()] # <-- great, very pythonic option. Does not allow for reporting which were omitted, however.

    kept_samples = []
    dropped_samples = []
    sex_to_remove_lower = sex_to_remove.lower()
    for sample in samples:
        if samples_sex[sample].lower() != sex_to_remove_lower:
            kept_samples.append(sample)
            continue
        dropped_samples.append(sample)

    if dropped_samples:
        logger.info(f"The following samples are being dropped based on sex label '{sex_to_remove}':\n{sorted(dropped_samples)}")

    if kept_samples:
        logger.info(f"The following samples are retained after dropping samples based on sex label '{sex_to_remove}':\n{kept_samples}")
    else:
        logger.warning(f"No samples remain after dropping samples based on sex label '{sex_to_remove}'")

    return kept_samples

def _main():
    # parse args
    args = _parseArgs()

    # setup logging
    setupLogging(logfn=args.logfn)

    # report the command
    logger.info(f"Arguments provided to this python script: {sys.argv}")

    # get the samples' sexes
    samples_sexes = {}
    if args.must_parse_samples_sex_map_file:
        samples_sexes = parseSampleSexMappingFile(args.samples_sex_map_fn, samples=frozenset(args.samples))

    # get the list of samples from args, removing males if the chromosome is chrX
    logger.debug(f"The list of samples (either inferred or provided by the user directly or via a list in a file) is as follows:\n{args.samples}")
    samples = args.samples
    bedmethyls_df = args.bedmethyls_df
    # if args.chrom == "chrX": # juhyun modi 2 lines below as well
    #    samples = removeSamplesBasedOnSex(args.samples, samples_sexes, sex_to_remove="Male") # remove them from the samples list
    #    bedmethyls_df = bedmethyls_df[bedmethyls_df["sample"].isin(samples)] # remove them from the bedmethyls_df (by keeping only those that remained in samples after removal)

    # create output directories
    args.outfn.parent.mkdir(mode=0o2775, parents=True, exist_ok=True)

    # ------------------------------------------------------------------------------ #
    # load in the data or model that will be used for filtering the input bedmethyls #
    # ------------------------------------------------------------------------------ #

    # load the DSS DMRs if needed
    regions = None
    if args.dss_dmrs_fn:
        logger.info(f"Loading DSS's DMRs")
        start = timer()
        regions = parseDssRegionsFile(args.dss_dmrs_fn, areastat_cutoff=args.areastat_cumsum)
        end = timer()
        logger.info(f"Elapsed time: {timedelta(seconds=end-start)}")

        logger.info(f"Regions (DSS's DMRs) data from {args.dss_dmrs_fn} has been loaded and sorted by start position")
        logger.debug(f"This is what that regions data look like:\n{regions}")
        logger.debug(f"The dtypes for each column are as follows:\n{regions.dtypes}")

    # load in the DSS DMLs if needed
    loci = None
    if args.dss_dmls_fn:
        logger.info(f"Loading DSS's DMLs")
        start = timer()

        # remove DMLs not in DMRs if also filtering by DMRs
        if args.dss_dmrs_fn: logger.info(f"Removing any DMLs not in the (possibly filtered) DMRs")
        loci = parseDssLociFile(args.dss_dmls_fn, dmrs_for_filtering=regions)

        end = timer()
        logger.info(f"Elapsed time: {timedelta(seconds=end-start)}")

    # load in the model if needed
    model = None
    if args.model_file:
        logger.info(f"Loading the model from {args.model_file}")
        start = timer()
        with open(args.model_file, "rb") as fh:
            pipe = pickle.load(fh)
        # model = ModelPackage.unPickle(args.model_file).getModel()
        # model = pipe.named_steps["logisticregression"]
        model = pipe.named_steps['model']
        end = timer()
        logger.info(f"Elapsed time: {timedelta(seconds=end-start)}")

    # ---------------------------------------------------------------------- #
    # load in the bedmethyl files, filtering each one individually as needed #
    # ---------------------------------------------------------------------- #
    logger.info(f"Loading the the bedmethyl files for the specified samples")
    start = timer()
    #hap_cats = pd.CategoricalDtype(categories=["Maternal", "Paternal"], ordered=False)
    hap_cats = bedmethyls_df["haplotype"].dtype
    dataset = []
    samples_to_skip = set()
    for row in bedmethyls_df.itertuples():
        sample = row.sample
        hap = row.haplotype
        input_fn = row.bedmethyl
        

        logger.debug(f"Processing input CpGs from {sample} {hap}")
        
        if sample in samples_to_skip: continue

        try:
            data_1file = None

            if args.dss_dmls_fn:
                # filter by loci (which loci may already have been filtered by
                # region). If dss_dmrs_fn is also provided, then the loci
                # (DMLs) were already filtered to include only the loci within
                # those regions. If not, then this filter will use all loci
                # (DMLs) reported by DSS, not just the ones that are also
                # within DMRs.
                msg = f"Filtering input CpGs from {sample} {hap} using DSS's DMLs"
                if args.dss_dmrs_fn: msg += " (which DMLs were previously filtered by DSS's DMRs)"
                logger.debug(msg)
                data_1file = filterSignificantLoci(readOneInputBedMethylFile(input_fn, args.chrom), loci) # note: will have NAs if no match in input for sites in `loci`
                
            elif args.dss_dmrs_fn:
                # filter by region. If we've entered this block, then
                # dss_dmls_fn was not provided, which means we're filtering
                # only by region instead of by both region (DMRs) and loci
                # (DMLs) within those regions. Thus, this allows loci that DSS
                # didn't report as significant, even if they are in a region
                # that was considered significant.
                logger.debug(f"Filtering input CpGs from {sample} {hap} using DSS's (possibly filtered) DMRs")
                data_1file = filterRegions(readOneInputBedMethylFile(input_fn, args.chrom), regions)

            elif args.model_file:
                # use model to do filtering
                logger.debug(f"Filtering input CpGs from {sample} {hap} using provided model ({args.model_file})")
                data_1file = filterByModel(readOneInputBedMethylFile(input_fn, args.chrom), model)
                # data_1file['chrom'] = args.chrom.astype(str) # juhyun added
                # print(data_1file.shape)

            else:
                logger.debug(f"No filtering applied to input CpGs from {sample} {hap}")
                data_1file = readOneInputBedMethylFile(input_fn, args.chrom) # juhyun added

            # filtering done. Now add mod_frac, remove uneeded columns, add Sample and Label, and pivot
            
            
            logger.debug(f"Using the (possibly filtered) CpGs from {sample} {hap} which currently has {len(data_1file)} CpGs. Calculating mod_frac, removing uneeded columns, adding Sample and Label columns, and pivoting (wide).")
            data_1file["mod_frac"] = data_1file["Nmod"].astype("float64") / data_1file["Nvalid_cov"]
            data_1file.drop(columns=["chrom","Nmod","Nvalid_cov"], inplace=True)
            # print(data_1file.shape)
            # print(data_1file.head(4))
            data_1file.index = pd.RangeIndex(len(data_1file))
            data_1file["Sample"] = sample
            # data_1file["Label"]  = pd.Series([hap] * len(data_1file), dtype=hap_cats)
            data_1file["Label"]  = hap # juhyun modified
            cols_not_to_widen = ["Sample", "Label"]
            if args.emit_sex:
                data_1file["Sex"] = samples_sexes[sample]
                cols_not_to_widen.append("Sex")
            
            # print(data_1file.shape)
            # print(data_1file.head(4))
            # data_1file = flattenMultiColumns(pd.pivot(data_1file, index=cols_not_to_widen, columns=["pos"], values=["mod_frac"]).reset_index()) # note: I did test doing the pivot after all the files had been read. It was slower, but only slightly.
            data_1file = (data_1file.pivot_table(index=["Sample", "Label"],columns="pos",values="mod_frac",aggfunc="first", observed=False, dropna = False).reset_index()) # juhyun modified to escape the duplicated entries issue
            # print(data_1file.shape)
            logger.debug(f"This is what the data for sample {sample} hap {hap} looks like: {data_1file.shape}\n" + headTailDataframeForReporting(data_1file))
            # print(data_1file.iloc[:, :min(6, data_1file.shape[1])])
            dataset.append(data_1file) # should be fast, it _should_ just be adding a pointer to the list
            # print("after append", len(dataset))

        except DatasetCreationException as e:
            if args.error_exit:
                raise
            logger.warning(f"Caught DatasetCreationException when reading bedmethyl file for {sample} {hap}. Since the -E|--error-exit flag was not provided, this warning message is issued instead an exception being raised. The exception's message would have been \"{e}\". Now, this sample ({sample}) will be excluded from the resulting dataset (both this hap ({hap}) and the other hap).")
            samples_to_skip.add(sample)
            #break # break out of haplotype loop

    # -------------------------------------------- #
    # remove any samples that needed to be skipped #
    # -------------------------------------------- #
    if len(samples_to_skip):
        logger.warning("Omitting samples " + ", ".join(samples_to_skip))

        # remove samples that need to be skipped from samples list (of str)
        for i in range(len(samples) - 1, -1, -1):
            if samples[i] in samples_to_skip:
                #logger.debug(f"removing sample at position {i} in samples list: {samples[i]}.")
                samples.pop(i)
            #else:
            #    logger.debug(f"Keeping sample at position {i} in samples list: {samples[i]}.")

        # remove samples that need to be skipped from dataset list (of DataFrames)
        #logger.debug(f"dataset has length {len(dataset)}")
        for i in range(len(dataset) - 1, -1, -1):
            #logger.debug(f"inside the dataset removal for loop with index {i}")
            if dataset[i]["Sample"].isin(samples_to_skip).any():
                #logger.debug(f"removing dataframe at position {i} in dataset list. The dataframe to be removed looks like this:\n" + headTailDataframeForReporting(dataset[i]))
                dataset.pop(i)
            #else:
            #    logger.debug(f"Keeping dataframe at position {i} in dataset list. The dataframe to be kept looks like this:\n" + headTailDAtaframeForReporting(dataset[i]))

    # --------------------------------------- #
    # exit if there is nothing in our dataset #
    # --------------------------------------- #
    if len(dataset) == 0:
        err = "After attempting to process all bedmethyl files, the dataset list (a list of DataFrames) was empty."
        logger.error(err)
        raise DatasetCreationException(err)

    # ---------------------------------------------------- #
    # turn our list of dataframes into a single dataframe, #
    # functionally vertically concatenating them           #
    # ---------------------------------------------------- #

    # create a single dataframe via vertical concatenation
    # print(len(dataset))
    dataset = pd.concat(dataset, ignore_index=True, copy=False) # I don't expect copy=False to actually avoid copying, but it would be nice.
    # ideally, we sould avoid the copying associated with the concat. This is
    # the best option without more intensive solutions, e.g., not using
    # pd.read_csv and/or doing some other shenanigans (using awk to add the
    # sample and label columns to add those columns upon read in _and_ tracking
    # each column separately in Series or something before combining them
    # together with pd.DataFrame).

    # sort the columns to ensure the positions are in order (they should be
    # already, but let's enforce it)
    dataset.sort_index(axis="columns", ascending=True, inplace=True, key=lambda x: x.map(versionSortKey))

    # convert Sample column to type category instead of type str now that the
    # full set of available Sample IDs is present
    dataset["Sample"] = pd.Categorical(dataset["Sample"], ordered=False)

    # convert Sex column, if present, to type category instead of type str now
    # that the full set of available Sexes is present
    if args.emit_sex:
        dataset["Sex"] = pd.Categorical(dataset["Sex"], ordered=False)

    # it should now look like this (possibly with a Sex column after Label):
    #        0         1      2 ...    N
    #   Sample     Label  41902 ... NNNN
    #    HG002  Maternal    0.8 ... 0.35
    #    HG002  Paternal   0.62 ... 0.71
    #      ...       ...    ... ...  ...
    #  HG01234  Maternal   0.69 ... 0.84
    #  HG01234  Paternal    0.4 ... 0.53

    end = timer()
    logger.info(f"Elapsed time: {timedelta(seconds=end-start)}")

    # ------------------------------------------------------------ #
    # dataset creation complete! Write it to a file and we're done #
    # ------------------------------------------------------------ #

    # report on the final dataset
    logger.info(f"dataset (from the lifted bedmethyl files, not directly DSS's DMLs output) have been loaded and the column names (positions) have been sorted. Entries in the bedmethyl files were filtered, if requested, according to input parameters.")
    logger.debug(f"This is what that the dataset looks like: {dataset.shape}\n" + headTailDataframeForReporting(dataset))
    logger.debug(f"The dtypes for each column are as follows:\n{dataset.dtypes}")

    # write the final dataset
    logger.info(f"Writing the dataset to {args.outfn}")
    dataset.to_csv(args.outfn, mode='w', sep='\t', header=True, index=False, float_format="{:g}".format, na_rep="NA")

    # report completion
    logger.info(f"Dataset creation and writing to file completed.")

# ------------- MAIN ----------------------------- ||
if __name__ == "__main__":
    _main()

