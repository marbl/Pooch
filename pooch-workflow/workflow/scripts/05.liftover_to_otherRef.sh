#! /bin/bash

# Liftover methylation calls to another reference genome using modkit
# Usage: ./03.liftover_to_otherRef.sh <bam.file> <output.dir> <sample.name> <core>
if command -v module >/dev/null 2>&1; then
    module load crossmap
    module load bedtools
fi

########################## INPUTS ##########################
PERSONAL_BED="$1" #filt methylation bed file based on personalized diploid coordinates
MAINDIR="$2" # main working direcotory
SAMPLE="$3" # name of the sample
HAPNUM="$4" # number of haplotype
CHAIN_HAP="$5" # chain file for haplotype
########################## DEFAULTS ##########################
REF_NAME=CHM13
###################################################################
mkdir -p "$MAINDIR/methylation"
mkdir -p "$MAINDIR/logs"

tools=(crossmap bedtools)
for tool in "${tools[@]}"; do
    if ! command -v "$tool" >/dev/null 2>&1; then
        echo "Error: $tool is not installed or not in PATH."
        exit 1
    fi
done

# make new chain file for liftover CpG methylation calls to the reference genome
for STRAND in "+" "-"
do
NEW_CHAIN="$MAINDIR/methylation/chm13v2_to_${SAMPLE}_${HAPNUM}.inverted.filtered.${STRAND}.chain"
FIRST_LIFT_BED=$MAINDIR/methylation/${SAMPLE}_${HAPNUM}.meth.CHM13.${STRAND}.1.bed
SECOND_LIFT_BED=$MAINDIR/methylation/${SAMPLE}_${HAPNUM}.meth.CHM13.${STRAND}.2.bed
FINAL_LIFT_BED=$MAINDIR/methylation/${SAMPLE}_${HAPNUM}.meth.CHM13.bed

if [[ ! -s "$NEW_CHAIN" ]]; then
    echo "Filtering chain file for strand $STRAND"
    if [[ "$STRAND" == "+" ]]; then
        s="pos"
    else
        s="neg"
    fi

    awk -v strand="$STRAND" '
    $1=="chain" && $10==strand {print; keep=1; next}
    $1!="chain" && keep {print; next}
    {keep=0}
    ' $CHAIN_HAP > \
    "$NEW_CHAIN"
else
    echo "Filtered chain file for strand $STRAND already exists"
fi

# Liftover methylation calls to the reference genome using CrossMap
crossmap bed $NEW_CHAIN $PERSONAL_BED $FIRST_LIFT_BED

# Adjust the coordinates based on the strand information
if [[ $STRAND == "-" ]]; then
echo -n "Adjusting coordinates for strand $STRAND"
awk '{print $1,$2-1,$3-1,$4,$5}' OFS='\t' $FIRST_LIFT_BED > $SECOND_LIFT_BED &&
rm $FIRST_LIFT_BED
else
mv $FIRST_LIFT_BED $SECOND_LIFT_BED
fi
done

# merge the methylation results and remove potential duplications
cat $MAINDIR/methylation/${SAMPLE}_${HAPNUM}.meth.CHM13.+.2.bed $MAINDIR/methylation/${SAMPLE}_${HAPNUM}.meth.CHM13.-.2.bed | bedtools sort -i > $FINAL_LIFT_BED &&
sed -i '/^chr/!s/^/chr/' $FINAL_LIFT_BED &&
rm $MAINDIR/methylation/${SAMPLE}_${HAPNUM}.meth.CHM13.+.2.bed $MAINDIR/methylation/${SAMPLE}_${HAPNUM}.meth.CHM13.-.2.bed