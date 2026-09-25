#! /bin/bash

# Call methylation from PacBio HiFi reads using modkit
# Usage: ./01.call_methylation.sh <bam.file> <output.dir> <sample.name> <core>

BAM="$1"
mainDir="$2"
sample="$3"
CORE="$4"

mkdir -p "$mainDir/methylation"
mkdir -p "$mainDir/logs"

# Methylation calling using aligned_bam_to_cpg_scores
if [ ! -f $BAM.bai ]; then
    echo -e "Indexing BAM file $BAM"
    samtools index $BAM
fi

# linked file doesnt work. use physical path to BAM file.
# if there are supplementary alignments, this script will fail. need to filter out supplementary alignments first.
if [ ! -f "$mainDir/methylation/${sample}.pri.meth.combined.bed" ]; then
    echo -e "Calling methylation for $sample using aligned_bam_to_cpg_scores"
    cmd="aligned_bam_to_cpg_scores \
                --bam $BAM \
                --output-prefix $mainDir/methylation/${sample}.pri.meth \
                --threads $CORE"

# --model $CPG_PILEUP_MODEL/pileup_calling_model.v1.tflite \
    echo $cmd
    eval $cmd
fi

cat  "$mainDir/methylation/${sample}.pri.meth.combined.bed"  | grep -v "^#" | cut -f 1,2,3,6,7 > "$mainDir/methylation/${sample}.pri.meth.filt.bed"