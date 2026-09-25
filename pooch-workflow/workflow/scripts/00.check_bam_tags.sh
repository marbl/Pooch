#!/bin/bash
# load module and load samtools if available
if command -v module >/dev/null 2>&1; then
    module load samtools
fi

# check if samtools is available
if ! command -v samtools >/dev/null 2>&1; then
    echo "Error: samtools is not installed or not in PATH."
    exit 1
fi

echo -n "Checking BAM file for MM tag..."
bam=$1
outdir=$2
sample=$3

met=$(samtools view "$bam" | head -10000 | grep --extended-regexp "MM:Z:C\+m\?|ML:B:C" | wc -l)
if [ "$met" -gt 0 ]; then
    echo "MM tag found in BAM file."
    touch "${outdir}/alignments/${sample}.step0_methylation_tag_check.done"
else
    echo "MM tag not found in BAM file. Please ensure the BAM file contains methylation tags (MM and ML)."
    exit 1
fi