#! /usr/bin/env python3

#__author__ == "Brandon Pickett"

#---------------------- IMPORTS ---------------------------------------------||
import sys
import re
import pandas as pd
import numpy as np
import pathlib as pl
import pickle
import hashlib
import logging
from packaging import version
import argparse
from collections import defaultdict

#---------------------- Misc Global Settings --------------------------------||
pd.options.mode.copy_on_write = True

#---------------------- Logging ---------------------------------------------||
logger = logging.getLogger(__name__)
#__all__ = ["blah_blah_blah"] # prevent anything other than what's in the list from being imported via `from ... import *`

def setupLogging(logfn=None, to_stderr=True, loglevel_to_file=logging.DEBUG):
    '''
    Setup logging for a script when called from __main__

    When running this file directly instead of as a module, configure the
    root logger. The reference to the root logger need not be returned
    because even _main can just rely on the global module-level logger which
    will propogate up to the root logger that will still exist even if we no
    longer have a direct reference to it after this function ends. This is a
    matter of scoping.
    References to `logger` outside this function will still refer to the
    module-level logger NOT the root logger, even though we used the variable
    `logger` in this function. Because we did NOT use the keyword `global` in
    this function, the variable `logger` used in this function is local to this
    function's scope.

    Args:
        logfn (pl.Path, optional): logfile to be written to. The parent dir need
            not exist first. default: do not log to a file.
        to_stderr (bool, optional): whether to log to stderr. level will be WARNING+. default: True.
        loglevel_to_file (int, optional): which logging level to use for the
            file. Only takes effect if a filename is provided to logfn. See
            allowable ints by looking up logging.DEBUG, logging.INFO, etc.
            default: 10 (i.e., logging.DEBUG).
    '''

    formatter = logging.Formatter("[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s")
    logger = logging.getLogger() # this is the root logger, not the module-level logger from logging.getLogger(__name__)
    logger.handlers.clear()

    logging_to_msg = 'Logging to '

    if to_stderr:
        stderr_handler = logging.StreamHandler(sys.stderr)
        stderr_handler.setLevel(logging.WARNING)
        stderr_handler.setFormatter(formatter)
        logger.addHandler(stderr_handler)
        logger.setLevel(logging.WARNING)
        logging_to_msg += f"stderr ({logging.getLevelName(logging.WARNING)})"
    
    if not logfn is None:
        logfn.parent.mkdir(mode=0o2775, parents=True, exist_ok=True) # create the directory of the logfn
        file_handler = logging.FileHandler(filename=logfn)
        file_handler.setLevel(loglevel_to_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.setLevel(loglevel_to_file)
        if to_stderr: logging_to_msg += " & "
        logging_to_msg += f"{logfn} ({logging.getLevelName(logging.DEBUG)})"
    
    if to_stderr or (not logfn is None): logger.info(logging_to_msg)


#---------------------- Command-line Args -----------------------------------||
def floatOrInt(value):
    if not re.match(r'^[0-9]+[Ee]?[0-9]+$', value) is None: # i.e., if it is clearly an integer (or a sciInt, e.g., 1e3 == 1000)
        return int(float(value)+0.1) # add 0.1 to handle float repr of int wierdness (e.g., 1e2 being represented as 99.9999999999)
    # else, if it's not an integer, we'll try to process it as a float
    try:
        return float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{value} is not a valid number")

def sciInt(value):
    try:
        f = float(value)
        if not f.is_integer():
            raise argparse.ArgumentTypeError(f"{value} is not an integer")
        return int(f)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{value} is not a valid number")

def processSamplesSetIdArg(dataset_fn, samples_set_id):
    if samples_set_id.lower() != "auto":
        return samples_set_id
    result = re.sub(r"^([^.]+).*$", r"\1", dataset_fn.name) # i.e., returns everything after the last / and before the first . after that
    if not result: # i.e., if result is "" or None
        raise argparse.ArgumentError(f"failed to find sample set ID from dataset filename ({dataset_fn})")
    return result

def processChromosomeArg(dataset_fn, chromosome):
    if chromosome.lower() != "auto":
        return chromosome
    result = re.sub(r"^.*(chr[0-9A-Za-z]+).*$", r"\1", str(dataset_fn)) # i.e., returns chr19 from v3-48.chr19.dataset.tsv.
    if not result: # i.e., if result is "" or None
        raise argparse.ArgumentError(f"failed to determine chromosome from dataset filename ({dataset_fn})")
    return result

#---------------------- Classes ---------------------------------------------||
class FileParsingError(Exception):
    '''
    Exception for when parsing a file fails.

    This is a simple extension of `Exception` with no additional attributes or
    functionality. It's just for aesthetics.
    '''
    pass

class DatasetCreationException(Exception):
    '''
    Exception for when dataset creation fails.

    This is a simple extension of `Exception` with no additional attributes or
    functionality. It's just for aesthetics.
    '''
    pass

class DatasetCompositionError(Exception):
    '''
    Exception for when the composition of a dataset does not match expectations.

    This is a simple extension of `Exception` with no additional attributes or
    functionality. It's just for aesthetics.
    '''
    pass

class ModelPackage():
    '''
    Simple container class for trained parts of a model/pipeline combo

    It enables pickling and unpickling to easily save and load.
    '''

    def __init__(self, model=None, xform_pipeline=None):
        self.model = model
        self.xform_pipeline = xform_pipeline

    def getModel(self):
        return self.model

    def getXformPipeline(self):
        return self.xform_pipeline

    def unpack(self):
        return self.model, self.xform_pipeline

    def pickle(self, f):
        '''
        Pickle this ModelPackage instance

        Write to f. Assume f is (1) an already-opened file descriptor (e.g.,
        sys.stdout.buffer or with open in mode 'wb') or (2) a filename as a str
        or pl.Path that will be opened before being written to.

        Args:
            f (str, pl.Path, sys.stdout.buffer, _io.BufferedWriter): the file
                (filename or writable file-like object) to write the pickle to.
        '''
        if isinstance(f, (str, pl.Path)):
            logger.info(f"Pickling ModelPackage into {f}")
            with open(fn, "wb") as pickle_fd: pickle.dump(self, pickle_fd, protocol=5)
            md5SumFileAndWriteMd5(fn)
            return
        logger.info(f"Pickling ModelPackage into {f.name}")
        pickle.dump(self, f, protocol=5)

    @classmethod
    def unPickle(cls, fn):
        obj = None
        with open(fn, "rb") as pickle_fd: obj = pickle.load(pickle_fd)
        if not isinstance(obj, cls): raise TypeError(f"Unpickled object is not a {cls.__name__}")
        return obj

#---------------------- Functions -------------------------------------------||
def flattenMultiColumns(df):
    def selectDeepestLevel(col):
        for index_level in range(len(col)-1, 0, -1):
            if col[index_level]:
                return col[index_level]
        return col[0]
    df.columns = df.columns.map(selectDeepestLevel)
    return df

def minifyLongListForReporting(l, n=3):
    if len(l) > n + n:
        return l[:n] + ["..."] + l[-n:]
    return l

def diffByElementNaAware(a, b):
    return a - b if not np.isnan([a, b]).any() else 0

def diffByColNaAware(col_a, col_b):
    return col_a.combine(col_b, diffByElementNaAware)

def tupleizeSplitIndexName_IntDotStr(name):
    fields = name.split('.')
    return (int(fields[0]), fields[1])

def PandasIndexSort_IntDotStr(idx):
    return pd.Index([tupleizeSplitIndexName_IntDotStr(name) for name in idx])

def md5SumFileAndWriteMd5(fn):
    digest = None
    with open(fn, "rb") as fd:
        try:
            digest = hashlib.file_digest(fd, "md5")
        except AttributeError:
            digest = hashlib.md5()
            for chunk in iter(lambda: fd.read(4096), b""):
                digest.update(chunk)
    digest = digest.hexdigest()
    with open(f"{fn}.md5", "w") as md5_fd:
        print(f"{digest}  {fn}", file=md5_fd)

def headTailDataframeForReporting(d, max_rows=10, max_cols=10, f="{:g}".format, show_idx=False):
    return d.to_string(index=show_idx, justify="right", header=True, float_format=f, max_cols=max_cols, max_rows=max_rows, show_dimensions=True)

def versionSortKey(value):
    if re.match(r"^[0-9.]+$", str(value)) is None:
        return version.parse("0.0.0+" + re.sub(r"[._-]+", ".", re.sub(r"^\.+", "", re.sub(r"([A-z]+|[0-9]+)", r".\1", str(value)))))
    return version.parse(str(value))

def reCategorySeriesOrderedByVersionSort(ser):
    '''
    Order a pandas Categorical Series by version ordering

    Args:
        ser (pd.Series): any pandas Series with dtype pd.Categorical.

    Returns:
        pd.Series: a copy of the input series with new categories that have been
            ordered. Categories determined by all unique values of `ser`.
            Ordering determined by a version sort of the unique values.
    '''
    return pd.Categorical   (
                                ser,
                                categories=ser.drop_duplicates  (
                                                                    keep="first",
                                                                    inplace=False,
                                                                    ignore_index=True
                                                                ).sort_values   (
                                                                                    ascending=True,
                                                                                    inplace=False,
                                                                                    ignore_index=True,
                                                                                    key=lambda s: s.apply(versionSortKey)
                                                                                ),
                                ordered=True
                            )

def parseAkbari2023KnownIdmrsFixedFile(idmrs_path, chrom=''):
    '''
    Read the iDMRs file into a DataFrame

    The iDMRs file is the table of known iDMRs as reported by Akbari et. al.
    2023 (DOI: 10.1016/j.xgen.2022.100233). The filename was originally
    `Imprinted_DMR_List_V1.tsv`, as downloaded from the GitHub repo in the
    paper: https://github.com/vahidAK/PatMat. It is assumed that the version
    loaded into this function has been semi-manually cleaned up to fix the
    various inconsistencies in the original file and had extra information
    added. We also assume it has been lifted from GRCh38, if needed, to the
    reference coordinate system for the DSS analysis (e.g., CHM13v2.0). The
    file is expected to have the following columns (without a header row):
    "Chromosome", "Start", "End", "Haplotype", "Identifier", "Loci", and
    "Sources". These names will be used as the column names in the returned
    pandas DataFrame.

    Args:
        idmrs_path (pl.Path, str): path to the iDMRs file.
        chrom (str, optional): chromosome (or other sequence ID) to subset the
            input to before returning.

    Returns:
        pd.DataFrame: the parsed file as a pandas DataFrame
    '''

    colnames = [ "Chromosome", "Start", "End", "Haplotype", "Identifier", "Loci", "Sources" ]

    # load in the data
    idmrs = pd.read_csv(    idmrs_path,
                            sep='\t', header=None, names=colnames,
                            dtype=  {   "Chromosome": "str", "Start": "uint64", "End": "uint64",
                                        "Haplotype": "category", "Identifier": "str", "Loci": "str",
                                        "Sources": "str"
                                    }
                        )

    #idmrs.sort_values(by="Start", ascending=True, inplace=True, ignore_index=True)
    #idmrs.sort_values(by="Chromosome", ascending=True, kind="stable", inplace=True, ignore_index=True, key=lambda s: s.apply(versionSortKey))

    # order the chromosome categorical
    idmrs["Chromosome"] = reCategorySeriesOrderedByVersionSort(idmrs["Chromosome"])

    # drop any chromosome but the requested one (if one was requested)
    if chrom: # i.e., if chrom != ''
        logger.info(f"dropping any rows where 'Chromosome' isn't {chrom}")
        idmrs = idmrs.loc[idmrs["Chromosome"] == chrom]

    # report
    logger.debug(f"This is what that iDMRs data look like:\n{idmrs}")
    logger.debug(f"The dtypes for each column are as follows:\n{idmrs.dtypes}")
    
    return idmrs


def parseDssRegionsFile(regions_path, chrom='', areastat_cutoff=0):
    '''
    Read the DMRs TSV from one chromosome, output by DSS, into a dataframe

    Args:
        regions_path (pl.Path): path to the the TSV file. If a directory is
            given, and the directory exists, a file <chrom>.dmrs.tsv will be
            used (must provide `chrom`). If no `chrom` is provided, the path
            must be to a TSV file.
        chrom (str, optional): chromosome or other sequence ID. Used when
            `regions_path` is a directory to construct the filename. Also used
            to subset to a specific chromosome, which is important if
            `regions_path` is a merged file with multiple chromosomes in it.
            caveat emptor: if `chrom` is a sequence identifier that does not
            encapsulate a full chromosome (e.g., contig-1234) and the provided
            `regions_path` is for a single chromosome spread across multiple
            sequences (e.g., contig-1234 & contig-5678), the result will be to
            subset to only the sequences matching the identifier provided to
            `chrom` (contig-1234 in this example). Generally, the assumption is
            that `chrom` is a sequence representing the full chromosome.
        areastat_cutoff (int, optional): if provided (or if 0), the contents of
            `regions_path` will be (temporarily) sorted in descending order by
            the absolute value of the areastat column, and when the cumulative
            sum of the areastats exceeds `areastat_cutoff`, all subsequent rows
            are dropped. As implemented, only makes sense to use this when
            operating on only one chromosome, so please ensure that the
            `regions_path` contains only DMRs from a single chromosome or that
            `chrom` is provided. default: no filtering.

    Returns:
        pd.DataFrame: the data from `regions_path`, possibly filtered by
            cumulative areastat according to `areastat_cutoff` and/or `chrom`.

    Raises:
        FileParsingError: when no `chrom` is provided but `regions_path` is a
            directory.
    '''

    # read in the file
    #    0         1         2      3    4                  5                 6                  7                 8
    #  chr     start       end length  nCG         meanMethy1        meanMethy2         diff.Methy          areaStat
    #chr19  56602126  56635238  33113  632  0.874928215980582  0.64312644120783  0.231801774772751  20884.2914317026
    if regions_path.is_dir():
        if not chrom: # i.e., if chrom == ''
            raise FileParsingError(f"No chromosome provided when path to DSS regions (i.e., DMRs) file was a directory ({regions_path}). Either provide the path to the file or provide the desired chromosome (e.g., chr19) and ensure <chrom>.dmrs.tsv exists in the provided directory.")
        regions_path = regions_path / f"{chrom}.dmrs.tsv"

    regions = pd.read_csv(regions_path, sep='\t', header=0, names=["chrom", "start", "end", "length", "nCG", "meanMethyl1", "meanMethyl2", "diffMeanMethyl", "areaStat"], dtype=defaultdict(lambda: "float64", chrom="str", start="uint64", end="uint64", length="uint64", nCG="uint64"))

    if chrom: # i..e, if chrom != ''
        regions = regions[regions["chrom"] == chrom] # keep only entries for the chromosome of interest

    # filter out regions/DMRs based on areaStat (if desired)
    if areastat_cutoff > 0:
        regions = regions.sort_values(by="areaStat", ascending=False, inplace=False)
        regions = regions[regions["areaStat"].cumsum() <= areastat_cutoff] # remove rows after cumsum of descending sorted areastat reaches areastat_cutoff

    # sort by start position
    regions.sort_values(by="start", ascending=True, inplace=True, ignore_index=True)

    # if we have multiple chromosomes present, also sort by chromosome
    # (ensuring stability from the previous sort on start position)
    if not chrom: # i.e., if chrom == '':
        regions.sort_values(by="chrom", ascending=True, kind="stable", inplace=True, ignore_index=True, key=lambda s: s.apply(versionSortKey))

    # turn the chrom column into a category (ordered)
    regions["chrom"] = reCategorySeriesOrderedByVersionSort(regions["chrom"])
#    regions["chrom"] = pd.Categorical   (
#                                            regions["chrom"], 
#                                            categories=regions["chrom"].drop_duplicates (
#                                                                                            keep="first",
#                                                                                            inplace=False,
#                                                                                            ignore_index=True
#                                                                                        ).sort_values   (
#                                                                                                            ascending=True,
#                                                                                                            inplace=False,
#                                                                                                            ignore_index=True,
#                                                                                                            key=lambda s: s.apply(versionSortKey)
#                                                                                                        ), 
#                                            ordered=True
#                                        )
#
    # log
    log_str = f"Regions (DSS's DMRs) data from {regions_path} has been loaded and sorted by chromosome and start position."
    if chrom: # i.e., if chrom != ''
        log_str += f" Only DMRs in chromosome '{chrom}' were kept."
    if areastat_cutoff > 0:
        log_str += f" If any, extra DMRs after a cumulative sum of the descending sorted absolute values of the areaStat reaches {areastat_cutoff} were dropped."
    logger.info(log_str)
    logger.debug(f"This is what that regions data look like:\n{regions}")
    logger.debug(f"The dtypes for each column are as follows:\n{regions.dtypes}")

    return regions

def parseDssLociFile(loci_path, chrom='', dmrs_for_filtering=None):
    # if dmrs_for_filtering is not None, it is assumed to be a dataframe
    # (possibly subsequently filtered) as that resulting from parseDssRegionsFile
    # chrom (e.g., chr19) need be provided only if loci_path is a directory. We
    # assume that there is only one chromosome's worth of loci and regions,
    # i.e., all values of `chr` must match, though no such check is made.
    
    # read in the file
    #    0         1                  2                 3                 4                  5                 6                   7                  8     9   10
    #  chr       pos                mu1               mu2              diff            diff.se              stat                phi1               phi2  pval  fdr
    #chr19  56616945  0.907268961790591  0.15767947243355  0.74958948935704  0.005559728688588  134.824832531066  0.0821380116175496  0.339297811432783     0    0
    if loci_path.is_dir():
        if not chrom: # i.e., if chrom == ''
            raise FileParsingError(f"No chromosome provided when path to DSS loci (i.e., DMLs) file was a directory ({loci_path}). Either provide the path to the file or provide the desired chromosome (e.g., chr19) and ensure <chrom>.dmls.tsv exists in the provided directory.")
        loci_path = loci_path / f"{chrom}.dmls.tsv"
    loci = pd.read_csv(loci_path, sep='\t', header=0, names=["chrom", "pos", "mean1", "mean2", "diff", "diffSE", "stat", "phi1", "phi2", "pval", "fdr"], dtype=defaultdict(lambda: "float64", chrom="str", pos="uint64"))

    # sort by position
    loci.sort_values(by="pos", ascending=True, inplace=True, ignore_index=True)

    # remove DMLs not in DMRs (if DMRs are provided)
    if not dmrs_for_filtering is None:
        loci = pd.merge_asof(
                                loci.loc[loci["pos"] >= dmrs_for_filtering["start"][0]],
                                pd.DataFrame({"start": dmrs_for_filtering["start"], "end": dmrs_for_filtering["end"]}),
                                left_on="pos", right_on="start",
                                direction="backward", allow_exact_matches=True
                            )
        loci = loci[loci["pos"] <= loci["end"]]
        loci.drop(columns=["start", "end"], inplace=True)
        loci.index = pd.RangeIndex(len(loci))
    

    # log
    log_str = f"Loci (DSS's DMLs) data from {loci_path} has been loaded and sorted by position."
    if not dmrs_for_filtering is None:
        log_str += " Loci not in DMRs were omitted."
    logger.info(log_str)
    logger.debug(f"This is what that loci data look like:\n{loci}")
    logger.debug(f"The dtypes for each column are as follows:\n{loci.dtypes}")
    
    return loci
    
def parseMethylDatasetFile(dataset_fn):

    # load in the dataset
    #        0         1      2 ...    N
    #   Sample     Label  41902 ... NNNN
    #    HG002  Maternal    0.8 ... 0.35
    #    HG002  Paternal   0.62 ... 0.71
    #      ...       ...    ... ...  ...
    #  HG01234  Maternal   0.69 ... 0.84
    #  HG01234  Paternal    0.4 ... 0.53
    #d = pd.read_csv(f"{dataset_fn}", sep='\t', header=0, dtype=defaultdict(lambda: "float64", Sample="category", Label="category")) # <-- OK if all non-position, categorical column names are known

    # We assume that _all_ non-position columns are of type category and all
    # position columns are of type float64
    cat_cols = [col_name for col_name in pd.read_csv(f"{dataset_fn}", sep='\t', header=0, nrows=0).columns if re.fullmatch(r"[0-9]+", col_name) is None]
    d = pd.read_csv(f"{dataset_fn}", sep='\t', header=0, dtype=defaultdict(lambda: "float64", **{cat_col: "category" for cat_col in cat_cols}))

    # log
    logger.info(f"Differentially methylated CpGs (from the lifted bedmethyl files) have been loaded from {dataset_fn}")
    logger.debug(f"This is what that the data looks like: {d.shape}\n" + headTailDataframeForReporting(d))
    logger.debug(f"The dtypes for each column are as follows:\n{d.dtypes}")
    
    # sanity checks
    ## only 2 labels
    hap_labels = d["Label"].cat.categories.to_list()
    if len(hap_labels) != 2:
        err_msg = f"Expected 2 levels in the 'Label' column, got {len(hap_labels)}: {hap_labels}"
        logger.critical(err_msg)
        raise DatasetCompositionError(err_msg)
    hap1_label, hap2_label = hap_labels
    logger.debug(f"hap_labels: {hap_labels}")
    logger.debug(f"hap1_label: {hap1_label}")
    logger.debug(f"hap2_label: {hap2_label}")

    ## same number of rows per label
    label_counts = d["Label"].value_counts(dropna=False)
    hap1_nrows = label_counts[hap1_label]
    hap2_nrows = label_counts[hap2_label]
    if hap1_nrows != hap2_nrows:
        err_msg = f"Expected the same number of rows for each haplotype, got {hap1_label}={hap1_nrows}, {hap2_label}={hap2_nrows}"
        logger.critical(err_msg)
        raise DatasetCompositionError(err_msg)

    ## number of rows for each hap equals the number of samples (by now, we
    ## know the two haps have the same number of rows as eachother)
    samples = sorted(d["Sample"].cat.categories.to_list())
    if len(samples) != hap1_nrows:
        err_msg = f"Expected each hap to have the number of rows equal to the number of samples (i.e., {len(samples)}), each hap had {hap1_nrows}"
        logger.critical(err_msg)
        raise DatasetCompositionError(err_msg)

    ## same samples in each hap. Note: if this test passes, then it's also true
    ## the samples in each haplotype are the same set across both haplotypes;
    ## thus, obviating the need to check for something like this:
    ##     samples == sorted(hap1_samples)
    hap1_samples = d.loc[d["Label"] == hap1_label, "Sample"].reset_index(drop=True)
    hap2_samples = d.loc[d["Label"] == hap2_label, "Sample"].reset_index(drop=True)
    #logger.debug(f"hap1_samples\n: {hap1_samples}")
    #logger.debug(f"hap2_samples\n: {hap2_samples}")
    if hap1_samples.ne(hap2_samples).any():
        idx = hap1_samples.eq(hap2_samples).to_list().index(False)
        elem1 = hap1_samples[idx]
        elem2 = hap2_samples[idx]
        err_msg = f"Expected each hap to have the same samples in the same order, but they were not. The first difference occurs at (zero-based indexing) element {i}: {elem1} != {elem2}"
        logger.critical(err_msg)
        raise DatasetCompositionError(err_msg)
    
    return d

def splitMethylByHaps(methyl_dataset_df, hap1_label, hap2_label):
    # methyl_dataset_df is the dataframe resulting from reading in a dataset as
    # via parseMethylDatasetFile. Caveat emptor: it is expected to pass the
    # checks/assumptions from the parseMethylDatasetFile function, even if it
    # wasn't read in via that function or has since been filtered or modified in
    # some way.
    # hap1_label & hap2_label are strings labelling the haplotypes. Expected
    # values are mat & pat or hap1 & hap2 or similar.
    # Final return values:
    #### methyl_hap[12]: same as input methyl_dataset_df except (a) sorted by
    ####                 Sample name (it may already have been, but now we're
    ####                 certain it will be), (b) with the Sample and Label
    ####                 columns removed (and any other categorical columns),
    ####                 and (c) split into two dataframes by haplotype. Each
    ####                 will be the same length as S.
    #### labels_hap[12]: 1 column ("Label") dataframes (one dataframe per each of
    ####                 the two haplotype labels). In each dataframe, every row is
    ####                 the same; i.e., labels_hap1 would contain all "mat" or
    ####                 "hap1" or similar labels and labels_hap2 would contain all
    ####                 "pat" or "hap2" or similar labels. Each will be the same
    ####                 length as S.
    ####              S: 1 column ("Sample") dataframe with sample names in sorted
    ####                 order.
    ####     other_cats: 0+ column dataframe containing non-sample,
    ####                 non-haplotype-label categorical columns, if any.

    # separate mat from pat (or hap1 from hap2), but without knowing about the
    # naming of the Label other than that we expect there to be two categories.
    # Then sort by Sample.
    logger.info(f"Separating the dataset into two groups by haplotype (i.e., {hap1_label} & {hap2_label}) using the 'Label' column")
    methyl_groupedby_labels = methyl_dataset_df.groupby("Label") # does not copy everything, just a view-like reference object
    methyl_hap1 = methyl_groupedby_labels.get_group(hap1_label)
    methyl_hap2 = methyl_groupedby_labels.get_group(hap2_label)
    methyl_hap1.sort_values(by="Sample", inplace=True, ignore_index=True)
    methyl_hap2.sort_values(by="Sample", inplace=True, ignore_index=True)
    logger.debug(f"Initial shapes ({hap1_label}, {hap2_label}): {methyl_hap1.shape}, {methyl_hap2.shape}")

    # separate out the data from the labels (sample and haplotype). Note: the
    # labels are now implicit because we've grouped them into separate
    # dataframes. We're still saving the labels (haplotype assignment) because
    # we'll need it for checking if we predict correctly, but once we do only
    # classfiication using the trained model, we won't know the answer and will
    # discard the uniformative Label column (presumabely, something like
    # hap1/hap2).
    cat_cols    = frozenset([col_name for col_name in methyl_dataset_df.columns if re.fullmatch(r"[0-9]+", col_name) is None])
    methyl_cols = [col_name for col_name in methyl_dataset_df.columns if not col_name in cat_cols]
    other_cats  = methyl_hap1.drop(columns=methyl_cols, inplace=False).drop(columns=["Sample", "Label"], inplace=False) # other_cats_hap1 would equal other_cats__hap2, so we'll just use one and call it other_cats
    S           = methyl_hap1.loc[:, methyl_hap1.columns == "Sample"] # S_hap1 would equal S_hap2, so we'll just use one and call it S
    labels_hap1 = methyl_hap1.loc[:, methyl_hap1.columns == "Label"]
    labels_hap2 = methyl_hap2.loc[:, methyl_hap2.columns == "Label"]
    cat_cols    = list(cat_cols)
    methyl_hap1.drop(columns=cat_cols, inplace=True)
    methyl_hap2.drop(columns=cat_cols, inplace=True)

    logger.debug(f"Result of splitting by haplotypes shapes...")
    #logger.debug(f"methyl.shape ({hap1_label}, {hap2_label}): {methyl_hap1.shape}, {methyl_hap2.shape}")
    #logger.debug(f"labels.shape ({hap1_label}, {hap2_label}): {labels_hap1.shape}, {labels_hap2.shape}")
    #logger.debug(f"S.shape: {S.shape}")
    logger.debug(f"This is what the hap1 methylation data dataframe \"methyl_hap1\" looks like: {methyl_hap1.shape}\n" + headTailDataframeForReporting(methyl_hap1))
    logger.debug(f"This is what the hap2 methylation data dataframe \"methyl_hap2\" looks like: {methyl_hap2.shape}\n" + headTailDataframeForReporting(methyl_hap2))
    logger.debug(f"This is what the hap1 labels dataframe \"labels_hap1\" looks like: {labels_hap1.shape}\n" + headTailDataframeForReporting(labels_hap1))
    logger.debug(f"This is what the hap2 labels dataframe \"labels_hap2\" looks like: {labels_hap2.shape}\n" + headTailDataframeForReporting(labels_hap2))
    logger.debug(f"This is what the samples dataframe \"S\" looks like: {S.shape}\n" + headTailDataframeForReporting(S))
    logger.debug(f"This is what the non-sample, non-hap-labels category columns dataframe \"other_cats\" looks like: {other_cats.shape}\n" + headTailDataframeForReporting(other_cats))

    return methyl_hap1, methyl_hap2, labels_hap1, labels_hap2, S, other_cats
    
def parseDiffDatasetFile(dataset_fn):

    # load in the dataset
    #        0               1                2      3 ...    N
    #   Sample  LeftOpHapLabel  RightOpHapLabel  41902 ... NNNN
    #    HG002        Maternal         Paternal    0.6 ... 0.35
    #      ...             ...              ...    ... ...  ...
    #  HG01234        Maternal         Paternal    0.2 ... 0.53

    #d = pd.read_csv(f"{dataset_fn}", sep='\t', header=0, dtype=defaultdict(lambda: "float64", Sample="category", LeftOpHapLabel="category", RightOpHapLabel="category")) # <-- OK if all non-position, categorical column names are known

    # We assume that _all_ non-position columns are of type category and all
    # position columns are of type float64
    cat_cols = [col_name for col_name in pd.read_csv(f"{dataset_fn}", sep='\t', header=0, nrows=0).columns if re.fullmatch(r"[0-9]+", col_name) is None]
    d = pd.read_csv(f"{dataset_fn}", sep='\t', header=0, dtype=defaultdict(lambda: "float64", **{cat_col: "category" for cat_col in cat_cols}))

    # log
    logger.info(f"Differences in methylation at CpGs between two haplotypes have been loaded from {dataset_fn}")
    logger.debug(f"This is what that the data looks like: {d.shape}\n" + headTailDataframeForReporting(d))
    logger.debug(f"The dtypes for each column are as follows:\n{d.dtypes}")

    # extract an arbitrary (we'll go with the first) left label and right label
    # and use them hereafter as left and right, even though there may be a mix
    # in the case of a duplicated+inverted dataset
    left_label  = d["LeftOpHapLabel" ].iat[0]
    right_label = d["RightOpHapLabel"].iat[0]
    logger.debug(f"arbitrary (first) left_label: {left_label}")
    logger.debug(f"arbitrary (first) right_label: {right_label}")
    
    # sanity checks
    ## only 2 labels
    hap_labels = list(frozenset(d["LeftOpHapLabel"].cat.categories.to_list() + d["RightOpHapLabel"].cat.categories.to_list()))
    if len(hap_labels) != 2:
        err_msg = f"Expected 2 levels in union of the 'LeftHapOpLabel' and 'RightHapOpLabel' columns, got {len(hap_labels)}: {hap_labels}"
        logger.critical(err_msg)
        raise DatasetCompositionError(err_msg)
    left_labels  = d["LeftOpHapLabel" ].cat.categories.to_list()
    right_labels = d["RightOpHapLabel"].cat.categories.to_list()
    logger.debug(f"hap_labels: {hap_labels}")
    logger.debug(f"left_label(s): {left_label}")
    logger.debug(f"right_label(s): {right_label}")

    ## expecting only one label per column in LeftOpHapLabel and RightOpHapLabel
    ## OR two labels per column if the number of leftOp and rightOp even out
    ## and are opposite to eachother at each entry and the other categoricals are
    ## identical otherwise between the two sets, presumabely sans one categorical
    ## that demarcates the difference.
    if len(left_labels) == 2 and len(right_labels) == 2:
        # We already know that the union of the two sets has length two. The
        # question now is whether both have length 2 or if we're dealing with a
        # different situation. We need either (a) both to have length 2 or (b)
        # both to have length 1. This if block deals with case 'a', the else
        # with case 'b'. If both are length 2, we need to check that the counts
        # match up to exactly half each.
        leftOp_label_counts  = d["LeftOpHapLabel"].value_counts()
        rightOp_label_counts = d["RightOpHapLabel"].value_counts()
        if not ( leftOp_label_counts[left_label] == len(d) / 2 and rightOp_label_counts[left_label] == len(d) / 2 ):
            # we can infer if both are true that the same would be true
            # swapping in `right_label` as the key. This saves some checks. The
            # negation is the error state.
            err_msg = f"Expected 2 levels with equal number of instances in the 'LeftOpHapLabel' and 'RightOpHapLabel' columns"
            logger.critical(err_msg)
            raise DatasetCompositionError(err_msg)
        # there are certainly more checks we could do, but this will likely
        # catch most issues

    else:
        # this else block is dealing with situation 'b' described above
        if len(left_labels) != 1:
            err_msg = f"Expected 1 level in the 'LeftOpHapLabel' column, got {len(left_labels)}: {left_labels}"
            logger.critical(err_msg)
            raise DatasetCompositionError(err_msg)
        if len(right_labels) != 1:
            err_msg = f"Expected 1 level in the 'RightOpHapLabel' column, got {len(right_labels)}: {right_labels}"
            logger.critical(err_msg)
            raise DatasetCompositionError(err_msg)

    ## same number of entries per label in the union of Left and Right Operand
    ## Hap Labels (roughly akin to the check for number of rows per haplotype
    ## in the methylation dataset). This check is good for the case 'b' above.
    ## It also complements the check for case 'a' above.
    label_counts = pd.Series(pd.api.types.union_categoricals([d["LeftOpHapLabel"].array, d["RightOpHapLabel"].array])).value_counts()
    left_nentries = label_counts[left_label]
    right_nentries = label_counts[right_label]
    if left_nentries != right_nentries:
        err_msg = f"Expected the same number of entries for each haplotype in the union of the LeftOpHapLabel and RightOpHapLabel, got {left_label}={left_nentries}, {right_label}={right_nentries}"
        logger.critical(err_msg)
        raise DatasetCompositionError(err_msg)

    ### there are no duplicate samples # <-- only works if this input file is from a single chromosome.
    #samples = sorted(d["Sample"].cat.categories.to_list())
    #if len(samples) != len(d["Sample"]):
    #    err_msg = f"Expected no duplicated samples (i.e., one row per sample), got {len(samples)} unique sample names across {len(d['Sample'])} rows"
    #    logger.critical(err_msg)
    #    raise DatasetCompositionError(err_msg)

    ## there are no duplicate samples relative to other categoricals. A more
    ## sophisticated test would be to more careful check based on different
    ## combinations, but we'll keep it a bit simpler to allow for arbitrary label
    ## columns other than the few required as we've currently defined it.
    samples = sorted(d["Sample"].cat.categories.to_list())
    exclude_cat_col_names = ["Sample"]
    if "Chromosome" in d.columns and len(d["Chromosome"].cat.categories.to_list()) == 1: exclude_cat_col_names.append("Chromosome")
    d["NonSampleCatCols"] = d[ [col_name for col_name in cat_cols if not col_name in exclude_cat_col_names ] ].astype(str).agg('_'.join, axis=1)
    uniq_nonSample_cat_cols = sorted(list(set(d["NonSampleCatCols"])), key=lambda x: versionSortKey(x))
    for uniq_nonSample_cat_col in uniq_nonSample_cat_cols:
        num_occurences_this_cat_col_combo = (d["NonSampleCatCols"] == uniq_nonSample_cat_col).sum()
        num_samples_this_cat_col_combo = d.loc[d["NonSampleCatCols"] == uniq_nonSample_cat_col, "Sample"].drop_duplicates(keep="first", inplace=False, ignore_index=True).count()
        if num_samples_this_cat_col_combo != num_occurences_this_cat_col_combo:
            err_msg = f"Expected no duplicated samples (i.e., one row per sample) per unique combination of non-Sample categorical columns. This unique combination of non-Sample categorical columns (i.e., {uniq_nonSample_cat_col}) occurred {num_occurences_this_cat_col_combo} times. Of those, there were {num_samples_this_cat_col_combo} unique samples represented. We expect these numbers to be identical."
            logger.critical(err_msg)
            raise DatasetCompositionError(err_msg)
        #if num_samples_this_cat_col_combo != len(samples):
        #    warn_msg = f"In most cases (but not all), the number of unique samples ({len(samples)}) for this dataset ({dataset_fn}) should be the number of unique samples present for any given combination of non-Sample categorical columns. This combo, {uniq_nonSample_cat_col}, had {num_samples_this_cat_col_combo}. Please investigate if anything looks amiss."
        #    if "chrX" in uniq_nonSample_cat_col:
        #        warn_msg += f"Note that this combo has the string \"chrX\" in it, so the difference may simply be the absence of the male samples. If the numbers look good to you, great! Otherwise, you should investigate."
        #    logger.warning(warn_msg)
    d.drop(columns=["NonSampleCatCols"], inplace=True)

    ### number of entries for each hap equals the number of samples (by now, we # <-- only works if this input file has no other categoricals beyond sample and left/right op hap label
    ### know the two haps have the same number of entries as eachother and there
    ### are no duplicate samples)
    #if len(samples) != left_nentries:
    #    err_msg = f"Expected each hap pair to have the number of entries equal to the number of samples (i.e., {len(samples)}), each hap had {left_nentries}"
    #    logger.critical(err_msg)
    #    raise DatasetCompositionError(err_msg)

    return d

def writeXnumpyAsDiffDataset(X, S, left_op_hap_label, right_op_hap_label, X_col_names, dataset_fn, **other_label_columns):
    # X is assumed to be a numpy 2d-array. Each value is a float64 in the range
    #     [0,1] representing a difference in methylation between two haplotypes.
    #     It is the result of an operation like this: X = X_hap1 - X_hap2.
    # S is a pd.Series of type Category or Object, i.e., a bunch of strings
    #     representing sample names. It should be the same length as there are
    #     rows in X.
    # left_op_hap_label is the haplotype label (e.g., hap1 or hap2 or mat or
    #     pat) for the "hap1" referred to in `X_hap1` in the explanation above
    #     for the X argument.
    # right_op_hap_label is the same as left_op_hap_label except for the "hap2"
    #     in `X_hap2` in the explanation above for the X argument.
    # X_col_names is a list of positions (integers) in the reference genome,
    #     one entry per column in X. They need to be of type str, so if the
    #     type is of type int instead, it will be converted to type str. All
    #     entries are expected to share a type. Heterogenous types or non-str,
    #     non-int types results in undefined behavior.
    # dataset_path is a pl.Path of the output filename for the dataset to be written to
    # **other_label_columns are any additional columns that the function caller
    #     wishes to have written. The key is the column name, the value is a
    #     pd.Series or single value that can be broadcast into one.

    if all(isinstance(col_name, (int, np.integer)) for col_name in X_col_names):
        # i.e., if all elements of X_col_names are of type int or np.integer
        X_col_names = [str(col_name) for col_name in X_col_names] # convert to str
    
    if not ( all(isinstance(col_name, str) for col_name in X_col_names) and re.search(r'[^0-9]', ''.join(X_col_names)) is None ):
        # i.e., if not (1 and 2), where
        #       1: all elements of X_col_names are of type str and
        #       2: every string contains only characters 0-9
        # i.e., if any col_name is not a string or not containing only chars 0-9 if it is a string
        err_msg = f"X_col_names provided to writeXasDiffDataset was not all strings that look like integers (or that can be coerced to that), please investigate."
        logger.critical(err_msg)
        raise DatasetCompositionError(err_msg)

    # init correctly shaped and typed dataframe without values
    columns = ["Sample", "LeftOpHapLabel", "RightOpHapLabel"]
    if not other_label_columns: # i.e., if len(other_label_columns) == 0
        columns.extend(other_label_columns.keys())
    columns.extend(X_col_names)
    d = pd.DataFrame(
                        index=range(len(S)),
                        columns=columns,
                        dtype=defaultdict(lambda: "float64", Sample="str", LeftOpHapLabel="str", RightOpHapLabel="str", **{other_label_column: "str" for other_label_column in other_label_columns.keys()})
                    )

    # add the values for the Sample and Left/RightOpHapLabel columns
    d["Sample"] = S
    d["LeftOpHapLabel"] = left_op_hap_label
    d["RightOpHapLabel"] = right_op_hap_label

    # add any extra label columns
    for col_name, col_value in other_label_columns.items():
        d[col_name] = col_value

    # add the values for the main data columns
    for i, col_name in enumerate(X_col_names):
        d[col_name] = X[:, i]

    # set a values sort order
    values_sort_by = []
    values_sort_ascending = []
    if "Dataset" in other_label_columns.keys():
        values_sort_by.append("Dataset")
        values_sort_ascending.append(False)
    #values_sort_by.extend(["Sample", "LeftOpHapLabel"])
    values_sort_by.extend(["LeftOpHapLabel", "Sample"])
    values_sort_ascending.extend([True, True])
    if "Chromosome" in other_label_columns.keys():
        values_sort_by.append("Chromosome")
        values_sort_ascending.append(True)

    # sort
    d.sort_values(by=values_sort_by, ascending=values_sort_ascending, kind="stable", inplace=True, ignore_index=True, key=lambda s: s.apply(versionSortKey))

    # write to file
    d.to_csv(dataset_fn, mode='w', sep='\t', header=True, index=False, float_format="{:g}".format, na_rep="NA")

def writeXasDiffDataset(X, S, left_op_hap_label, right_op_hap_label, dataset_fn, **other_label_columns):
    # X is assumed to be a pd.DataFrame with all values of type float64 in the
    #     range [0,1] representing a difference in methylation between two
    #     haplotypes. It is the result of an operation like this:
    #     X = X_hap1 - X_hap2. Column names are expected to be present. They
    #     are strings that look like integers (i.e., only chars 0-9).
    # S is a pd.Series of type Category or Object, i.e., a bunch of strings
    #     representing sample names. It should be the same length as there are
    #     rows in X.
    # left_op_hap_label is the haplotype label (e.g., hap1 or hap2 or mat or
    #     pat) for the "hap1" referred to in `X_hap1` in the explanation above
    #     for the X argument.
    # right_op_hap_label is the same as left_op_hap_label except for the "hap2"
    #     in `X_hap2` in the explanation above for the X argument.
    # dataset_path is a pl.Path of the output filename for the dataset to be written to
    # **other_label_columns are any additional columns that the function caller
    #     wishes to have written. The key is the column name, the value is a
    #     pd.Series or single value that can be broadcast into one.

    # get the positional columns
    d = X.copy(deep=False)

    # add the known label columns
    d["Sample"] = S
    d["LeftOpHapLabel"] = left_op_hap_label
    d["RightOpHapLabel"] = right_op_hap_label

    # add any extra label columns
    for col_name, col_value in other_label_columns.items():
        d[col_name] = col_value

    # reorder to move all positional columns to end
    X_cols = X.columns.tolist()
    new_cols = [col for col in d.columns.tolist() if not col in X_cols]
    new_cols.extend(X_cols)
    d = d[new_cols]

    # set a values sort order
    values_sort_by = []
    values_sort_ascending = []
    if "Dataset" in other_label_columns.keys():
        values_sort_by.append("Dataset")
        values_sort_ascending.append(False)
    #values_sort_by.extend(["Sample", "LeftOpHapLabel"])
    values_sort_by.extend(["LeftOpHapLabel", "Sample"])
    values_sort_ascending.extend([True, True])
    if "Chromosome" in other_label_columns.keys():
        values_sort_by.append("Chromosome")
        values_sort_ascending.append(True)

    # sort
    d.sort_values(by=values_sort_by, ascending=values_sort_ascending, kind="stable", inplace=True, ignore_index=True, key=lambda s: s.apply(versionSortKey))

    # write to file
    d.to_csv(dataset_fn, mode='w', sep='\t', header=True, index=False, float_format="{:g}".format, na_rep="NA")

def enforceOneChromosomeInDataFrame(df, chromosome=None, chromosome_column_name="Chromosome", warn_on_exclude=False, error_on_exclude=False):
    '''
    Ensure pd.DataFrame `df` has rows for only one chromosome

    Check if `df` contains only one value in the `chromosome_column_name` column
    (i.e., every row is the same in that column). By default, the column name
    for that column in the pandas DataFrame is 'Chromosome'. If it is something
    else, it must be specified. If `chromosome_column_name` is not present in
    the `df`, an error will be raised. When `chromosome` is None, an error will
    also be raised if multiple chromosome IDs are found. When multiple IDs are
    found and `chromosome` is not None, all rows not matching that chromosome
    will be excluded. Upon such exclusion, an error will be raised if
    `error_on_exclude` is True. In leiu of that, a warning will be raised if
    `warn_on_exclude` is True and `error_on_exclude` is False.

    Args:
        df (pd.DataFrame): any pd.DataFrame with a categorical or str column
            named after `chromosome_column_name`.
        chromosome (str, optional): Specify which chromosome must be present
            when checking that only one chromosome is present in the dataframe.
            If multiple are present, only those matching `chromosome` will be
            retained. If None, check only that one unique identifier is present
            (whatever it may be) in df's `chromosome_column_name`. Default:
            None
        chromosome_column_name (str, optional): the column name for the relevant
            column in the dataframe. Default: "Chromosome"
        warn_on_exclude (bool, optional): if rows must be excluded to retain
            only `chromosome`, log a warning. Default: no warning.
        error_on_exclude (bool, optional): same was `warn_on_exclude`, except an
            error is raised instead of a logging a warning. When both are True,
            this takes precedence over the warning. Default: no error reported
            or raised

    Returns:
        pd.DataFrame: Copy of `df`, possibly filtered.

    Raises:
        DatasetCompositionError: when assumptions about the file structure are
            broken.

    '''
    # check that the col we need exists
    if not chromosome_column_name in df.columns:
        msg = f"{chromosome_column_name} not found in df.columns"
        logger.error(msg)
        raise DatasetCompositionError(msg)

    # check whether there is only one chromosome in said column
    only1 = df[chromosome_column_name].nunique() == 1

    # handle things if chromosome is specified
    if chromosome:
        present = chromosome in df[chromosome_column_name]
        if only1 and present:
            return df
        if only1: # i.e., elif only1 and not present
            msg = f"{chromosome} not found in df['{chromosome_column_name}']"
            logger.error(msg)
            raise DatasetCompositionError(msg)
        if present: # i.e., elif present and not only1
            if error_on_exclude:
                msg = f"Multiple chromosomes found in df['{chromosome_column_name}'], only {chromosome} expected/allowed"
                logger.error(msg)
                raise DatasetCompositionError(msg)
            if warn_on_exclude:
                msg = f"Multiple chromosomes found in df['{chromosome_column_name}'], retaining only rows matching {chromosome}"
                logger.warning(msg)
            return df[df[chromosome_column_name] == chromosome]
        # i.e., else or elif not present and not only1
        msg = f"Multiple chromosomes found in df['{chromosome_column_name}']. {chromosome} was expected/required but was not one of the ones found."
        logger.error(msg)
        raise DatasetCompositionError(msg)
    # i.e., else or elif not chromosome
    if only1: return df
    # i.e., else or elif not only1 (and also not chromosome)
    msg = f"Multiple chromosomes found in df['{chromosome_column_name}'], expected only 1 (no specific one required, just that there weren't multiple)."
    logger.error(msg)
    raise DatasetCompositionError(msg)

# ------------- MAIN ----------------------------- ||
if __name__ == "__main__":
    raise ImportError(f"This file ({sys.argv[0]}) is not meant to be run directly, only imported.")

