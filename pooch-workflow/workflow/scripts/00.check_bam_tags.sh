#!/bin/bash
# load module and load samtools if available
# check if samtools module is available
if command -v module >/dev/null 2>&1; then
    for module_name in samtools; do
        if ! module load "$module_name"; then
            printf 'Skipping unavailable module: %s\n' "$module_name" >&2
        fi
    done
fi

echo -n "Checking BAM file for MM tag..."
bam=$1
outdir=$2
sample=$3

# MM or Mm
met=$(samtools view "$bam" | head -10000 | grep --extended-regexp "Mm:Z:C|MM:Z:C\+m\?|ML:B:C" | wc -l)
if [ "$met" -gt 0 ]; then
    echo "MM tag found in BAM file."
    touch "${outdir}/alignments/${sample}.step0_methylation_tag_check.done"
else
    echo "MM tag not found in BAM file. Please ensure the BAM file contains methylation tags (MM and ML)."
    exit 1
fi