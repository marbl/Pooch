#!/bin/bash
if command -v module >/dev/null 2>&1; then
    module load minimap2
    module load nextflow
    module load blat
    module load java
    module load crossmap
    module load graphviz
    module load ucsc
    module load python/3.9
fi

SCRIPT_DIR=$(dirname "$(realpath "$0")")
printf "Adding %s/envs/bin to PATH\n" "$SCRIPT_DIR"
export PATH="$SCRIPT_DIR/../envs/bin/:$PATH"

# check if all command tools are available
# rb : rustybam
for tool in nextflow minimap2 blat crossmap python maf-convert rb chaintools_bio; do
    if ! command -v $tool >/dev/null 2>&1; then
        echo "Error: $tool is not installed or not in PATH."
        exit 1
    fi
done
echo "All required tools are available!"


# check if all arguments are provided
if [ "$#" -ne 5 ]; then
    echo "Usage: $0 <source_fasta> <target_fasta> <output_dir> <ref_name> <sample_name>"
    exit 1
fi

SOURCE_IN=$1
TARGET_IN=$2
OUTDIR_IN=$3

export SOURCEFASTA=$(realpath "$SOURCE_IN") # chm13
export TARGETFASTA=$(realpath "$TARGET_IN") # personalized reference
export OUTDIR=$(realpath "$OUTDIR_IN")
export REFNAME=$4
export SAMPLE=$5

if [ ! -f "$SOURCEFASTA" ]; then
    echo "Error: source fasta not found: $SOURCEFASTA"
    exit 1
fi

if [ ! -f "$TARGETFASTA" ]; then
    echo "Error: target fasta not found: $TARGETFASTA"
    exit 1
fi

mkdir -p "$OUTDIR"

# nf-LO PREPROC:tgt2bit and splittgt are more reliable with an uncompressed FASTA.
if [[ "$TARGETFASTA" == *.gz ]]; then
    TARGET_UNCOMPRESSED="$OUTDIR/$(basename "${TARGETFASTA%.gz}")"
    if [ ! -f "$TARGET_UNCOMPRESSED" ]; then
        echo "Decompressing target fasta to: $TARGET_UNCOMPRESSED"
        gunzip -c "$TARGETFASTA" > "$TARGET_UNCOMPRESSED"
    else
        echo "Using existing decompressed target fasta: $TARGET_UNCOMPRESSED"
    fi
    TARGETFASTA="$TARGET_UNCOMPRESSED"
fi

export PREFIX=${REFNAME}_to_${SAMPLE}
# export PREFIX=`echo $OUTDIR | sed 's:.*/::'`

# for debugging
echo "SOURCEFASTA: $SOURCEFASTA"
echo "TARGETFASTA: $TARGETFASTA"
echo "OUTDIR: $OUTDIR"
echo "REFNAME: $REFNAME"
echo "SAMPLE: $SAMPLE"
echo "PREFIX: $PREFIX"

# nextflow run nf-LO/main.nf --source $SOURCEFASTA --target $TARGETFASTA --outdir $OUTDIR -profile local --aligner minimap2 --max_cpus 2 --max_memory 64g -resume &&
nextflow run evotools/nf-LO --source "$SOURCEFASTA" --target "$TARGETFASTA" --outdir "$OUTDIR" -profile local --aligner minimap2 --max_cpus 2 --max_memory 250g -resume &&
if [ ! -f "$OUTDIR/chainnet/liftover.chain" ]; then
    echo "Error: nf-LO finished but chain file is missing: $OUTDIR/chainnet/liftover.chain"
    echo "Hint: inspect Nextflow logs in $OUTDIR/.nextflow.log and $OUTDIR/work/*/.command.err"
    exit 1
fi
python chaintools_bio split -c $OUTDIR/chainnet/liftover.chain -o $OUTDIR/$PREFIX-split.chain &&
python chaintools_bio to_paf -c $OUTDIR/$PREFIX-split.chain -t $SOURCEFASTA -q $TARGETFASTA -o $OUTDIR/$PREFIX-split.paf  &&
#awk '{short1=$1; short2=$6; gsub("_1", "", short1); gsub("_1", "", short2); if(short1==short2) {print}}' $PREFIX-split.paf > $PREFIX-samechr-split.paf  &&
#cat $PREFIX-samechr-split.paf | rb break-paf --max-size 10000 | rb trim-paf -r | rb invert | rb trim-paf -r | rb invert > $PREFIX.paf  &&
cat $OUTDIR/$PREFIX-split.paf | rb break-paf --max-size 10000 | rb trim-paf -r | rb invert | rb trim-paf -r | rb invert > $OUTDIR/$PREFIX.paf  &&
paf2chain -i $OUTDIR/$PREFIX.paf > $OUTDIR/$PREFIX.chain  &&
python chaintools_bio invert -c $OUTDIR/$PREFIX.chain -o $OUTDIR/$PREFIX.inverted.chain &&
touch $OUTDIR/step02.chain.done