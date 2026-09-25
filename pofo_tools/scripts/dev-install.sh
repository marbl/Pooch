#! /usr/bin/env bash

set -euo pipefail

# load modules
module purge
module load python/3.10

# add updated scikit-learn
export PYTHONPATH=/home/pickettbd/.local/lib/python3.10/site-packages:${PYTHONPATH:-}

# confirm we're in the right location
if [ "$(readlink "`dirname "${BASH_SOURCE[0]}"`")" != "`readlink .`" ]
then
	printf 'ERROR: This script is expected to be run from the project root (presumably %s).\n' "`dirname "${BASH_SOURCE[0]}"`" 1>&2
	exit 1
fi

# install package with pip in editable mode
pip install -e .

