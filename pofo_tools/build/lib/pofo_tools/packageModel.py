#! /usr/bin/env python3

#__author__ == "Brandon Pickett"

#---------------------- IMPORTS ---------------------------------------------||
import sys
import pickle
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

    parser = argparse.ArgumentParser(prog=sys.argv[0], description="Combine a pickled transformation Pipeline and a pickled trained model (e.g., LogisticRegression) into a ModelPackage and pickle the package.")
    parser.add_argument("-l", "--log-file", metavar="FILE", type=pl.Path, action="store", dest="logfn", help="The output file to write logging output to. [default: no log file]", default=None, required=False)
    parser.add_argument("-m", "--input-model-pickle", metavar="FILE", type=pl.Path, action="store", dest="model_fn", help="The pre-trained model in a pickle.", required=True)
    parser.add_argument("-o", "--output-pickle", metavar="FILE", type=argparse.FileType('wb'), action="store", dest="outfd", help="The output filename to pickle the packaged model and pipeline to. [default: stdout]", default=sys.stdout.buffer, required=False)
    parser.add_argument("-p", "--input-xform-pipeline-pickle", metavar="FILE", type=pl.Path, action="store", dest="xform_pipeline_fn", help="The 'fitted' transformation Pipeline in a pickle.", required=True)
    if len(sys.argv) == 1: sys.argv.append("-h")
    args = parser.parse_args()
    args.outfn = None
    if not args.outfd is sys.stdout.buffer:
        args.outfn = pl.Path(args.outfd.name)
    return args

#---------------------- Classes ---------------------------------------------||

#---------------------- Functions -------------------------------------------||
def _unPickleObject(pickle_fn):
    with open(pickle_fn, "rb") as pickle_fd:
        return pickle.load(pickle_fd)

def _unPickleModel(model_fn):
    return _unPickleObject(model_fn)

def _unPicklePipeline(pipeline_fn):
    return _unPickleObject(pipeline_fn)

def _packagePipelineAndModel(model, pipeline):
    return ModelPackage(model=model, xform_pipeline=pipeline)

def _main():
    # parse args
    args = _parseArgs()

    # setup logging
    setupLogging(logfn=args.logfn)

    # create output directories (if not stdout)
    if args.outfn:
        args.outfn.parent.mkdir(mode=0o2775, parents=True, exist_ok=True)

    # load input model and pipeline
    logger.info(f"Loading model")
    model = _unPickleModel(args.model_fn)

    logger.info(f"Loading pipeline")
    pipeline = _unPicklePipeline(args.xform_pipeline_fn)

    # package model and pipeline
    logger.info(f"packaging the model and pipeline into a ModelPackage")
    pkg = _packagePipelineAndModel(model, pipeline)

    # pickle the package
    pkg.pickle(args.outfd)

    # close the output file
    if not args.outfd is sys.stdout.buffer:
        args.outfd.close()

    # report completion
    logger.info("Completed.")

# ------------- MAIN ----------------------------- ||
if __name__ == "__main__":
    _main()

