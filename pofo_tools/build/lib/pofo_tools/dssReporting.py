#! /usr/bin/env python3

#__author__ == "Brandon Pickett"

#---------------------- IMPORTS ---------------------------------------------||
import sys
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib as mpl
import seaborn as sns
import argparse
import pathlib as pl
import logging
from scipy.stats import pearsonr

from pofo_tools.utils import *

#---------------------- Misc Global Settings --------------------------------||
pd.options.mode.copy_on_write = True
plt.set_loglevel("warning")
#sns.set_style("white")
empty_circle_marker=mpl.markers.MarkerStyle("o", fillstyle="none")
filled_circle_marker=mpl.markers.MarkerStyle("o", fillstyle="full")
filled_ul_circle_marker=mpl.markers.MarkerStyle("o", fillstyle="left").rotated(deg=315)
filled_br_circle_marker=mpl.markers.MarkerStyle("o", fillstyle="right").rotated(deg=315)

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

    parser = argparse.ArgumentParser(prog=sys.argv[0], description="Report basic info and create some plots about a DSS's DMRs.")
    parser.add_argument("-i", "--idmrs", metavar="FILE", type=pl.Path, action="store", dest="idmrs_fn", help="The TSV file with iDMRs.", required=True)
    #parser.add_argument("-L", "--dss-dmls-file", metavar="FILE", type=pl.Path, action="store", dest="dss_dmls_fn", help="[default: NA]", default=None, required=False)
    parser.add_argument("-l", "--log-file", metavar="FILE", type=pl.Path, action="store", dest="logfn", help="The output file to write logging output to. [default: no log file]", default=None, required=False)
    parser.add_argument("-o", "--output-dir", metavar="FILE", type=pl.Path, action="store", dest="outdir", help="The output directory to write output files to. [default: $PWD]", default=".", required=False)
    parser.add_argument("-R", "--dss-dmrs-files", metavar="FILE", type=pl.Path, action="store", dest="dss_dmrs_fns", nargs='+', help="The TSV file(s) with DMRs output by DSS. You can provide one file (e.g., all.dmrs.tsv) or multiple (e.g., chr{1..22}.dmrs.tsv chrX.dmrs.tsv). The expectation is that all DMRs from all chromosomes are present.", required=True)
    #parser.add_argument("-S", "--samples-set-id", metavar="STR", type=str, action="store", dest="samples_set_id", help="String used as an identifier for the set of samples found in the dataset (found in the file provided to -d|--dataset). Special string 'auto' (case insensitive) will extract the name from the filename provided to -d|--dataset; the first capturing group from the following regex will be the resulting identifier: `^(?:.*[/_.])?([^/_.]+)\.[^./]+$`. [default: auto]", default="auto", required=False)
    if len(sys.argv) == 1: sys.argv.append("-h")
    args = parser.parse_args()
    #args.samples_set_id = processSamplesSetIdArg(args.dataset_fn, args.samples_set_id)
    return args

#---------------------- Classes ---------------------------------------------||

class ExpLogFormatter(mpl.ticker.LogFormatterMathtext):
    def __call__(self, x, pos=None):
        return super().__call__(10**x, pos) # Treat x as log10(value), so convert back to value = 10**x before formatting

#---------------------- Functions -------------------------------------------||
def _overlap(s1, e1, s2, e2): # really more like overlapping or encompassing
    #return (s1 <= e2 and s1 >= s2) or (e1 <= e2 and e1 >= s2) or (s1 <= s2 and e1 >= e2) or (s2 <= s1 and e2 >= e1)
    return not ( (s1 < s2 and e1 < s2) or (s2 < s1 and e2 < s1) )

def _corrfunc(x, y, **kwargs):
    (r, p) = pearsonr(x, y)
    ax = plt.gca()
    ax.annotate(f"r = {r:.2f} ", xy=(.1, .9), xycoords=ax.transAxes, color="red")
    ax.annotate(f"p = {p:.3g}",  xy=(.5, .9), xycoords=ax.transAxes, color="red")

def _pairplottify(d, output_fn, logpfx=""):
    if pl.Path(output_fn).exists():
        logger.info(f"{logpfx}Skipping. {output_fn} already exists.")
        return
    # plot pairplots
    alpha_level = 0.6 if logpfx else 0.1
    logger.info(f"{logpfx}Plotting pairplots...(be patient)...")
    p = sns.pairplot(d, plot_kws=dict(marker=empty_circle_marker, size=0.1, linewidth=0.1, color="black", alpha=alpha_level), diag_kws=dict(fill=False, linewidth=0.1, color="black"))
    p.map(_corrfunc)
    logger.info(f"{logpfx}Saving to {output_fn}...")
    plt.savefig(output_fn, dpi=600, format="pdf")
    for dpi in [300, 600, 1200, 2400]:
        plt.savefig(output_fn.with_suffix(f".{dpi}ppi.png"), dpi=dpi, format="png")
    plt.clf()
    plt.cla()
    plt.close()

def _lenNumCGsCorrPlottify(d, output_fn, output_log_fn, logpfx='', titlesuffix=''):
    if pl.Path(output_fn).exists() and pl.Path(output_log_fn).exists():
        logger.info(f"{logpfx}Skipping. {output_fn} and {output_log_fn} already exists.")
        return
    logger.info(f"{logpfx}Plotting...")
    plt.scatter(d["Length"], d["Num_CGs"], s=0.1, c="black", marker=empty_circle_marker)
    w = np.linalg.lstsq( np.hstack( ( np.array(d["Length"].values, copy=True).reshape((len(d["Length"]), 1)),
                                      np.ones(
                                                ( len(d["Length"]), 1 )
                                             )
                                    )
                                  ),
                         d["Num_CGs"],
                         rcond=None)[0]
    x = np.linspace(*plt.gca().get_xlim()).T
    plt.plot(x, w[0]*x + w[1], ls='-', lw=1, c="red")
    plt.title(f"Length -x- Number of DMLs for each DMR{titlesuffix}")
    plt.xlabel("DMR Length")
    plt.ylabel("Number of DMLs in DMR")
    logger.info(f"{logpfx}Saving to {output_fn}...")
    plt.savefig(output_fn, dpi=600, format="pdf")
    for dpi in [300, 600, 1200, 2400]:
        plt.savefig(output_fn.with_suffix(f".{dpi}ppi.png"), dpi=dpi, format="png")
    plt.clf()
    plt.cla()
    plt.close()

    #plt.xscale("log")
    #plt.yscale("log")
    d["logLength"] = np.log10(d["Length"])
    d["logNum_CGs"] = np.log10(d["Num_CGs"])
    plt.scatter(d["logLength"], d["logNum_CGs"], s=0.1, c="black", marker=empty_circle_marker)
    #plt.scatter(d["Length"], d["Num_CGs"], s=0.1, c="black", marker=empty_circle_marker)
    w = np.linalg.lstsq( np.hstack( ( np.array(d["logLength"].values, copy=True).reshape((len(d["logLength"]), 1)),
                                      np.ones(
                                                ( len(d["logLength"]), 1 )
                                             )
                                    )
                                  ),
                         d["logNum_CGs"],
                         rcond=None)[0]
    x = np.linspace(*plt.gca().get_xlim()).T
    plt.plot(x, w[0]*x + w[1], ls='-', lw=1, c="red")
    plt.title(f"Length -x- Number of DMLs for each DMR{titlesuffix}")
    plt.xlabel("DMR Length (log)")
    plt.ylabel("Number of DMLs in DMR (log)")
    #ax = plt.gca()
    ##ax.set_xscale("log")
    ##ax.set_yscale("log")
    #ax.xaxis.set_major_locator(mpl.ticker.LogLocator(base=10.0, subs=(1.0,)))
    #ax.xaxis.set_major_formatter(ExpLogFormatter(base=10.0))
    ##ax.xaxis.set_major_formatter(mpl.ticker.LogFormatterMathtext(base=10.0))
    ##ax.xaxis.set_minor_locator(mpl.ticker.LogLocator(base=10.0, subs="auto"))
    #ax.xaxis.set_minor_locator(mpl.ticker.LogLocator(base=10.0, subs=np.arange(2,10)))
    #ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())

    #ax.yaxis.set_major_locator(mpl.ticker.LogLocator(base=10.0, subs=(1.0,)))
    #ax.yaxis.set_major_formatter(ExpLogFormatter(base=10.0))
    ##ax.yaxis.set_major_formatter(mpl.ticker.LogFormatterMathtext(base=10.0))
    ##ax.yaxis.set_minor_locator(mpl.ticker.LogLocator(base=10.0, subs="auto"))
    #ax.yaxis.set_minor_locator(mpl.ticker.LogLocator(base=10.0, subs=np.arange(2,10)))
    #ax.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    ##plt.xscale("log")
    ##plt.yscale("log")
    logger.info(f"{logpfx}Saving to {output_log_fn}...")
    plt.savefig(output_log_fn, dpi=600, format="pdf")
    for dpi in [300, 600, 1200, 2400]:
        plt.savefig(output_log_fn.with_suffix(f".{dpi}ppi.png"), dpi=dpi, format="png")
    plt.clf()
    plt.cla()
    plt.close()

    d.drop(columns=["logLength", "logNum_CGs"], inplace=True)

def _scatterMeanMethylPlottify(d, output_fn, logpfx='', titlesuffix=''):
    if pl.Path(output_fn).exists():
        logger.info(f"{logpfx}Skipping. {output_fn} already exists.")
        return

    #cyan = "#0d60bd"
    #brick = "#c83126"

    logger.info(f"{logpfx}Plotting...")
    plt.plot((0,1), (0,1), ls='-', lw=0.75, c="black", alpha=0.5) # original figure color used grey
    '''
    Dropped this code to just plot a black identity line instead of a grey
    identity line and a black line of best fit (linear). They were very close
    anyway, and slight deviation from identity is obvious.
    w = np.linalg.lstsq( np.hstack( ( np.array(d["Mean_Methyl_Group1_Mat"].values, copy=True).reshape((len(d["Mean_Methyl_Group1_Mat"]), 1)),
                                      np.ones(
                                                ( len(d["Mean_Methyl_Group1_Mat"]), 1 )
                                             )
                                    )
                                  ),
                         d["Mean_Methyl_Group2_Pat"],
                         rcond=None)[0]
    x = np.linspace(*plt.gca().get_xlim()).T
    plt.plot(x, w[0]*x + w[1], ls='-', lw=0.75, c="Black", alpha=0.9)
    '''
    #plt.scatter(d["Mean_Methyl_Group1_Mat"], d["Mean_Methyl_Group2_Pat"], s=0.1, c="black", marker=empty_circle_marker)
    #for name, group in d.groupby("Known_iDMRs_Haplotypes"):
    #    plt.scatter(group["Mean_Methyl_Group1_Mat"], group["Mean_Methyl_Group2_Pat"], s=0.1, c="black", marker=empty_circle_marker, label=name)
    mat_pat_colors = {"Paternal": "#0d60bd", "Maternal": "#c83126", "Both": "white", "unknown": "grey"}
    mat_pat_size = {"Paternal": 1, "Maternal": 1, "Both": 6, "unknown": 0.25}
    mat_pat_alpha = {"Paternal": 1, "Maternal": 1, "Both": 1, "unknown": 0.2}
    #mat_pat_markers = {"Paternal": filled_circle_marker, "Maternal": filled_circle_marker, "Both": filled_circle_marker, "unknown": empty_circle_marker}
    mat_pat_markers = {"Paternal": filled_circle_marker, "Maternal": filled_circle_marker, "Both": empty_circle_marker, "unknown": empty_circle_marker}
    for name in ("unknown", "Maternal", "Paternal", "Both"):
        grp = d.loc[d["Known_iDMRs_Haplotypes"] == name]
        logger.debug(f"{logpfx}Plotting {len(grp)} {name} points...")
        if name == "Both":
            filled_ul_circle_marker
            plt.scatter(grp["Mean_Methyl_Group1_Mat"], grp["Mean_Methyl_Group2_Pat"], s=mat_pat_size[name], c=mat_pat_colors["Maternal"], linewidth=0, marker=filled_br_circle_marker, alpha=mat_pat_alpha[name])
            plt.scatter(grp["Mean_Methyl_Group1_Mat"], grp["Mean_Methyl_Group2_Pat"], s=mat_pat_size[name], c=mat_pat_colors["Paternal"], linewidth=0, marker=filled_ul_circle_marker, alpha=mat_pat_alpha[name])
            continue
        #plt.scatter(grp["Mean_Methyl_Group1_Mat"], grp["Mean_Methyl_Group2_Pat"], s=0.1, c=mat_pat_colors[name], marker=empty_circle_marker, label=name)
        plt.scatter(grp["Mean_Methyl_Group1_Mat"], grp["Mean_Methyl_Group2_Pat"], s=mat_pat_size[name], c=mat_pat_colors[name], marker=mat_pat_markers[name], alpha=mat_pat_alpha[name], label=name)
    plt.title(f"Mean Methylation for each DMR{titlesuffix}")
    plt.xlabel("Maternal Mean Methylation")
    plt.ylabel("Paternal Mean Methylation")
    plt.legend()
    logger.info(f"{logpfx}Saving to {output_fn}...")
    plt.savefig(output_fn, dpi=600, format="pdf")
    for dpi in [300, 600, 1200, 2400]:
        plt.savefig(output_fn.with_suffix(f".{dpi}ppi.png"), dpi=dpi, format="png")
    plt.clf()
    plt.cla()
    plt.close()
    #del w, x

def _scatterMeanMethylPlottifySubplots(data, output_fn, chroms, logpfx=''):
    if pl.Path(output_fn).exists():
        logger.info(f"{logpfx}Skipping. {output_fn} already exists.")
        return

    logger.info(f"Plotting...")
    num_cols = 5
    num_rows = int((len(chroms) / float(num_cols))+0.5)
    fig, axs = plt.subplots(num_rows, num_cols, sharex=True, sharey=True, layout="constrained", subplot_kw={"adjustable" : "box", "aspect" : "equal"}, figsize=(12,12))

    # restrict axis numbers etc to only the outer plots
    for ax in axs.flat:
        ax.label_outer()
    #xticks = [0, 0.25, 0.5, 0.75, 1]
    #yticks = [0, 0.25, 0.5, 0.75, 1]

    for r in range(0, num_rows, 1):
        for c in range(0, num_cols, 1):
            chrom_idx = r * num_cols + c
            if chrom_idx >= len(chroms):
                axs[r, c].axis('off')
                #axs[r-1, c].set_xticklabels(axs[r, 0].get_xticklabels(which="both"))
                continue
            chrom = chroms[chrom_idx]
            d = data.loc[data["Chromosome"] == chrom]
            axs[r, c].plot((0,1), (0,1), ls='-', lw=0.75, c="grey", alpha=0.5)
            w = np.linalg.lstsq( np.hstack( ( np.array(d["Mean_Methyl_Group1_Mat"].values, copy=True).reshape((len(d["Mean_Methyl_Group1_Mat"]), 1)),
                                              np.ones(
                                                        ( len(d["Mean_Methyl_Group1_Mat"]), 1 )
                                                     )
                                            )
                                          ),
                                 d["Mean_Methyl_Group2_Pat"],
                                 rcond=None)[0]
            x = np.linspace(*axs[r, c].get_xlim()).T
            axs[r, c].plot(x, w[0]*x + w[1], ls='-', lw=0.75, c="Black", alpha=0.9)
            mat_pat_colors = {"Paternal": "#0d60bd", "Maternal": "#c83126", "Both": "white", "unknown": "grey"}
            mat_pat_size = {"Paternal": 1, "Maternal": 1, "Both": 6, "unknown": 0.25}
            mat_pat_alpha = {"Paternal": 1, "Maternal": 1, "Both": 1, "unknown": 0.2}
            #mat_pat_markers = {"Paternal": filled_circle_marker, "Maternal": filled_circle_marker, "Both": filled_circle_marker, "unknown": empty_circle_marker}
            mat_pat_markers = {"Paternal": filled_circle_marker, "Maternal": filled_circle_marker, "Both": empty_circle_marker, "unknown": empty_circle_marker}
            for name in ("unknown", "Maternal", "Paternal", "Both"):
                grp = d.loc[d["Known_iDMRs_Haplotypes"] == name]
                logger.debug(f"\t{chrom}: Plotting {len(grp)} {name} points...")
                if name == "Both":
                    filled_ul_circle_marker
                    axs[r, c].scatter(grp["Mean_Methyl_Group1_Mat"], grp["Mean_Methyl_Group2_Pat"], s=mat_pat_size[name], c=mat_pat_colors["Maternal"], linewidth=0, marker=filled_br_circle_marker, alpha=mat_pat_alpha[name])
                    axs[r, c].scatter(grp["Mean_Methyl_Group1_Mat"], grp["Mean_Methyl_Group2_Pat"], s=mat_pat_size[name], c=mat_pat_colors["Paternal"], linewidth=0, marker=filled_ul_circle_marker, alpha=mat_pat_alpha[name])
                    continue
                #axs[r, c].scatter(grp["Mean_Methyl_Group1_Mat"], grp["Mean_Methyl_Group2_Pat"], s=0.1, c=mat_pat_colors[name], marker=empty_circle_marker, label=name)
                #axs[r, c].scatter(grp["Mean_Methyl_Group1_Mat"], grp["Mean_Methyl_Group2_Pat"], s=mat_pat_size[name], c=mat_pat_colors[name], marker=mat_pat_markers[name], alpha=mat_pat_alpha[name], label=name)
                axs[r, c].scatter(grp["Mean_Methyl_Group1_Mat"], grp["Mean_Methyl_Group2_Pat"], s=mat_pat_size[name], c=mat_pat_colors[name], marker=mat_pat_markers[name], alpha=mat_pat_alpha[name])
            axs[r, c].set_title(chrom)
            axs[r, c].set_xlim([0, 1])
            axs[r, c].set_ylim([0, 1])
            #axs[r, c].set(adjustable='box-forced', aspect='equal')

    fig.suptitle(f"Mean Methylation for each DMR by Chromosome", fontsize="xx-large", fontweight="black")
    fig.supxlabel("Maternal Mean Methylation", fontsize="large", fontweight="bold")
    fig.supylabel("Paternal Mean Methylation", fontsize="large", fontweight="bold")

    #fig.legend()
    logger.info(f"Saving to {output_fn}...")
    plt.savefig(output_fn, dpi=2400, format="pdf")
    for dpi in [300, 600, 1200, 2400]:
        plt.savefig(output_fn.with_suffix(f".{dpi}ppi.png"), dpi=dpi, format="png")
    plt.clf()
    plt.cla()
    plt.close()
    #del w, x

def _lengthDistPlottify(d, output_fn, logpfx='', titlesuffix=''):
    # d = the data
    # output_fn is the output filename
    # logpfx is what to prefix my makeshift logging messages with
    # titlesuffix is something add to the end of the title, e.g., " chr7"
    if pl.Path(output_fn).exists():
        logger.info(f"{logpfx}Skipping. {output_fn} already exists.")
        return

    logger.info(f"{logpfx}Plotting...")
    #sns.displot(data=d, x="Length", kind="hist", color="grey", kde=True, kde_kws={"color": "black"})
    #sns.histplot(data=d, x="Length", color="grey", kde=True, line_kws={"color": "black"})
    #ax = sns.histplot(data=d, x="Length", color="grey")
    #sns.kdeplot(data=d, x="Length", color="black", linewidth=2, ax=ax)
    ax = sns.histplot(data=d, x="Length", color="grey", kde=True)
    ax.lines[0].set_color((0, 0, 0, 0.7))
    ax.lines[0].set_linewidth(0.5)
    plt.title(f"Distribution of DSS's DMR Lengths{titlesuffix}")
    plt.xlabel("DMR Length")
    logger.info(f"{logpfx}Saving to {output_fn}...")
    plt.savefig(output_fn, dpi=600, format="pdf")
    for dpi in [300, 600, 1200, 2400]:
        plt.savefig(output_fn.with_suffix(f".{dpi}ppi.png"), dpi=dpi, format="png")
    plt.clf()
    plt.cla()
    plt.close()

def _lengthDistPlottifySubplots(data, output_fn, chroms, logpfx=''):
    if pl.Path(output_fn).exists():
        logger.info(f"{logpfx}Skipping. {output_fn} already exists.")
        return

    logger.info(f"Plotting...")
    num_cols = 5
    num_rows = int((len(chroms) / float(num_cols))+0.5)
    #fig, axs = plt.subplots(num_rows, num_cols, sharex=True, sharey=True, layout="constrained", subplot_kw={"adjustable" : "box", "aspect" : "equal"}, figsize=(12,12))
    fig, axs = plt.subplots(num_rows, num_cols, sharex=False, sharey=False, layout="constrained", subplot_kw={"adjustable" : "datalim", "aspect" : "auto"}, figsize=(12,12))

    # restrict axis numbers etc to only the outer plots
    #for ax in axs.flat:
    #    ax.label_outer()
    ##xticks = [0, 0.25, 0.5, 0.75, 1]
    ##yticks = [0, 0.25, 0.5, 0.75, 1]

    for r in range(0, num_rows, 1):
        for c in range(0, num_cols, 1):
            chrom_idx = r * num_cols + c
            if chrom_idx >= len(chroms):
                axs[r, c].axis('off')
                #axs[r-1, c].set_xticklabels(axs[r, 0].get_xticklabels(which="both"))
                continue
            chrom = chroms[chrom_idx]
            d = data.loc[data["Chromosome"] == chrom]
            #sns.displot(d, x="Length", kind="hist", kde=True, ax=axs[r, c], color="grey", kde_kws={"color": "black"})
            #sns.histplot(data=d, x="Length", color="grey", kde=True, line_kws={"color": "black"}, ax=axs[r, c])
            #sns.histplot(data=d, x="Length", color="grey", kde=True, kde_kws={"color": "black"}, ax=axs[r, c])
            sns.histplot(data=d, x="Length", color="grey", kde=True, ax=axs[r, c])
            axs[r, c].lines[0].set_color((0, 0, 0, 0.7))
            axs[r, c].lines[0].set_linewidth(0.5)
            axs[r, c].set_title(chrom)
            max_count = max(patch.get_height() for patch in axs[r, c].patches)
            axs[r, c].set_ylim([0, max_count * 1.05])
            #axs[r, c].set(adjustable='datalim', aspect='equal')


    fig.suptitle(f"Distribution of DSS's DMR Lengths by Chromosome", fontsize="xx-large", fontweight="black")
    fig.supxlabel("DMR Length", fontsize="large", fontweight="bold")
    fig.supylabel("Count", fontsize="large", fontweight="bold")

    #fig.legend()
    logger.info(f"Saving to {output_fn}...")
    plt.savefig(output_fn, dpi=2400, format="pdf")
    for dpi in [300, 600, 1200, 2400]:
        plt.savefig(output_fn.with_suffix(f".{dpi}ppi.png"), dpi=dpi, format="png")
    plt.clf()
    plt.cla()
    plt.close()

def _main():
    # parse args
    args = _parseArgs()

    # setup logging
    setupLogging(logfn=args.logfn)

    # report the command
    logger.info(f"Arguments provided to this python script: {sys.argv}")

    # ------------------------ #
    # input file parsing, etc. #
    # ------------------------ #

    # create output directories
    args.outdir.mkdir(mode=0o2775, parents=True, exist_ok=True)

    # load in the known iDMRs (Akbari et. al. 2023)
    logger.info("Loading iDMRs...")
    idmrs = parseAkbari2023KnownIdmrsFixedFile(args.idmrs_fn)
    logger.debug(f"iDMRS ({idmrs.shape}):\n" + headTailDataframeForReporting(idmrs))

    # load in the DSS regions (DMRs) file(s), concatenating them
    dmrs = []
    if len(args.dss_dmrs_fns) == 1:
        logger.info(f"Loading DSS DMRs from a single file ({args.dss_dmrs_fns[0]})...")
        dmrs = parseDssRegionsFile(args.dss_dmrs_fns[0])
        # note: should be sorted
    else:
        logger.info("Loading DSS DMRs from multiple files...")
        for dss_dmrs_fn in args.dss_dmrs_fns:
            logger.info(f"Loading DSS DMRs from {dss_dmrs_fn}...")
            dmrs.append(parseDssRegionsFile(dss_dmrs_fn))
        logger.info("Merging DSS DMRs from multiple files, fixing chromosome categories and sorting by chromosome and start position...")
        dmrs = pd.concat(dmrs, ignore_index=True, copy=False) # I don't expect copy=False to actually avoid copying, but it would be nice.
        dmrs["chrom"] = reCategorySeriesOrderedByVersionSort(dmrs["chrom"]) # update categories since a single file likely didn't contain all chromosomes
        dmrs.sort_values(by="start", ascending=True, inplace=True, ignore_index=True) # sort by start position
        dmrs.sort_values(by="chrom", ascending=True, kind="stable", inplace=True, ignore_index=True) # sort (stably) by chromosome. Note: `key=lambda s: s.apply(versionSortKey)` not needed since chrom is ordered a few lines above.

    # reset the colnames for dmrs to play nice with the way everything has been
    # written in this script:
    #                          0        1      2         3          4                         5                         6                              7                8     <-- 0-based position in list
    #                        chr    start    end    length        nCG                meanMethy1                meanMethy2                     diff.Methy         areaStat     <-- original names from DSS
    #                      chrom    start    end    length        nCG               meanMethyl1               meanMethyl2                 diffMeanMethyl         areaStat     <-- names set in parseDssRegionsFile
    dmrs.columns = [ "Chromosome", "Start", "End", "Length", "Num_CGs", "Mean_Methyl_Group1_Mat", "Mean_Methyl_Group2_Pat", "Difference_Mean_Methylation", "DSS_Area_Stat" ] #<-- desired names for this script

    # extract chromosome list from dmrs
    chroms = idmrs["Chromosome"].drop_duplicates(keep="first", inplace=False, ignore_index=True).to_list() # no need to sort because dmrs should already be sorted
    logger.debug(f"iDMRs chroms: {chroms}")
    logger.debug(f"iDMRs chroms ordered?: {idmrs['Chromosome'].cat.ordered}")
    logger.debug(f"iDMRs chroms: {idmrs['Chromosome'].cat.categories}")
    logger.debug(f"iDMRs chroms: {idmrs['Chromosome'].describe()}")
    chroms = dmrs["Chromosome"].drop_duplicates(keep="first", inplace=False, ignore_index=True).to_list() # no need to sort because dmrs should already be sorted
    logger.debug(f"DMRs chroms: {chroms}")
    logger.debug(f"DMRs chroms ordered?: {dmrs['Chromosome'].cat.ordered}")
    logger.debug(f"DMRs chroms: {dmrs['Chromosome'].cat.categories}")
    logger.debug(f"DMRs chroms: {dmrs['Chromosome'].describe()}")

    # load in the DSS loci (DMLs)
    #loci = parseDssLociFile(args.dss_dmls_fn)

    # ----------------------------------------------------------------------------- #
    # find overlap w/ akbari 2023 and DSS DMRs. Add Mat/Pat labels to DSS DMRs data #
    # ----------------------------------------------------------------------------- #
    logger.info(f"Finding overlap with DSS's DMRs and the known iDMRs...")
    dmrs["Known_iDMRs_Haplotypes"] = pd.Categorical(["unknown"] * len(dmrs), categories=["Maternal", "Paternal", "Both", "unknown"], ordered=False)
    for i in range(0,len(dmrs),1):
        #logger.debug(f"DMRS ({dmrs.shape}):\n" + headTailDataframeForReporting(dmrs))
        #logger.debug(f"iDMRS ({idmrs.shape}):\n" + headTailDataframeForReporting(idmrs))
        #logger.debug(f"{dmrs.at[i, 'Chromosome']} in {idmrs['Chromosome'].cat.categories} :" + str(dmrs.at[i, "Chromosome"] in idmrs["Chromosome"].cat.categories))
        if dmrs.at[i, "Chromosome"] in idmrs["Chromosome"].cat.categories:
            #logger.debug(f"equal1? {idmrs['Chromosome'] == dmrs.at[i, 'Chromosome']}")
            idmrs_chr_subset = idmrs.loc[idmrs["Chromosome"] == dmrs.at[i, "Chromosome"]]
            idmrs_chr_subset.reset_index(drop=True, inplace=True)
            #logger.debug(f"iDMRS subset ({idmrs_chr_subset.shape}):\n" + headTailDataframeForReporting(idmrs_chr_subset))
            for j in range(0,len(idmrs_chr_subset),1):
                if _overlap(dmrs.at[i, "Start"], dmrs.at[i, "End"], idmrs_chr_subset.at[j, "Start"], idmrs_chr_subset.at[j, "End"]):
                    #if dmrs.at[i, "Chromosome"] == "chr20" and dmrs.at[i, "Start"] == 60593948:
                    #    logger.debug(f"idmrs_chr_subset {dmrs.at[i, 'Chromosome']}:\n{idmrs_chr_subset.describe()}\n" + headTailDataframeForReporting(idmrs_chr_subset))
                    hap = idmrs_chr_subset.at[j, "Haplotype"]
                    if dmrs.at[i, "Known_iDMRs_Haplotypes"] != "unknown" and dmrs.at[i, "Known_iDMRs_Haplotypes"] != hap:
                        dmrs.at[i, "Known_iDMRs_Haplotypes"] = "Both"
                        continue
                    dmrs.at[i, "Known_iDMRs_Haplotypes"] = hap

    logger.info(f"iDMR status (unknown vs Mat/Pat/Both) added to DSS's DMRs:\n" + headTailDataframeForReporting(dmrs))
    logger.info(f"Here's the breakdown by iDMR status group:")
    for name, grp in dmrs.groupby("Known_iDMRs_Haplotypes"):
        logger.info(f"{name} (len={len(grp)}): {grp.describe()}")

    # -------- #
    # plotting #
    # -------- #

    # plot meanMethyls against eachother
    meanmethdir = args.outdir / "scatter_meanMethyl"
    logger.info(f"Plotting scatterplot of mean methylation from each hap against eachother to {meanmethdir}...")
    meanmethdir.mkdir(mode=0o2775, parents=True, exist_ok=True)
    _scatterMeanMethylPlottify(dmrs, meanmethdir / pl.Path("scatter_meanMethyl.pdf"), logpfx='', titlesuffix='')
    for chrom in chroms:
        _scatterMeanMethylPlottify(dmrs.loc[dmrs["Chromosome"] == chrom], meanmethdir / pl.Path(f"scatter_meanMethyl.{chrom}.pdf"), logpfx=f"\t{chrom}: ", titlesuffix=f" ({chrom})")
    _scatterMeanMethylPlottifySubplots(dmrs, meanmethdir / pl.Path("scatter_meanMethyl.all.pdf"), chroms)

    # plot Length / Num_CGs correlation
    lenNCGdir = args.outdir / "scatter_len-x-dmls"
    logger.info(f"Plotting scatterplot of DMR length against the number of CG sites in the DMR to {lenNCGdir}...")
    lenNCGdir.mkdir(mode=0o2775, parents=True, exist_ok=True)
    _lenNumCGsCorrPlottify(dmrs, lenNCGdir / pl.Path("scatter_len-x-dmls.pdf"), lenNCGdir / pl.Path("scatter_len-x-dmls_log.pdf"), logpfx='', titlesuffix='')
    for chrom in chroms:
        break
        _lenNumCGsCorrPlottify(dmrs.loc[dmrs["Chromosome"] == chrom], lenNCGdir / pl.Path(f"scatter_len-x-dmls.{chrom}.pdf"), lenNCGdir / pl.Path(f"scatter_len-x-dmls_log.{chrom}.pdf"), logpfx=f"\t{chrom}: ", titlesuffix=f" ({chrom})")

    # plot pairplots
    pairplotdir = args.outdir / "pairplot_dmrs"
    logger.info(f"Plotting pairplots of all variables in the DMRs dataframe to {pairplotdir}...")
    pairplotdir.mkdir(mode=0o2775, parents=True, exist_ok=True)
    _pairplottify(dmrs, pairplotdir / pl.Path("pairplot_dmrs.pdf"), logpfx="")
    for chrom in chroms:
        _pairplottify(dmrs.loc[dmrs["Chromosome"] == chrom], pairplotdir / pl.Path(f"pairplot_dmrs.{chrom}.pdf"), logpfx=f"\t{chrom}: ")

    # plot length length distribution
    lengthdistdir = args.outdir / "length_dists"
    logger.info(f"Plotting the distribution of the length of the DMRs to {lengthdistdir}...")
    lengthdistdir.mkdir(mode=0o2775, parents=True, exist_ok=True)
    _lengthDistPlottify(dmrs, lengthdistdir / "length_dist.pdf", logpfx='', titlesuffix='')
    for chrom in chroms:
        _lengthDistPlottify(dmrs.loc[dmrs["Chromosome"] == chrom], lengthdistdir / f"length_dist.{chrom}.pdf", logpfx=f"\t{chrom}: ", titlesuffix=f" ({chrom})")
    _lengthDistPlottifySubplots(dmrs, lengthdistdir / "length_dist.all.pdf", chroms)

# ------------- MAIN ----------------------------- ||
if __name__ == "__main__":
    _main()

