#! /bin/bash

if command -v module >/dev/null 2>&1; then
    module load samtools
fi

PERSONALIZED_GENOME=$1
OUTDIR=$2
HAP1=$3
HAP2=$4
SAMPLE=$5

if [ ! -f $PERSONALIZED_GENOME.fai ]; then
    samtools faidx $PERSONALIZED_GENOME
fi

HAP1_LIST=$(grep "$HAP1" $PERSONALIZED_GENOME.fai | cut -f1 | tr '\n' ' ' | sed 's/,$//')
if [ -z "$HAP1_LIST" ]; then
    echo "Error: No sequences found for haplotype 1 ($HAP1) in the personalized genome index."
    exit 1
fi
HAP2_LIST=$(grep "$HAP2" $PERSONALIZED_GENOME.fai | cut -f1 | tr '\n' ' ' | sed 's/,$//')
if [ -z "$HAP2_LIST" ]; then
    echo "Error: No sequences found for haplotype 2 ($HAP2) in the personalized genome index."
    exit 1
fi

samtools faidx $PERSONALIZED_GENOME $HAP1_LIST > ${OUTDIR}/${SAMPLE}.hap1.fa
samtools faidx $PERSONALIZED_GENOME $HAP2_LIST > ${OUTDIR}/${SAMPLE}.hap2.fa

samtools faidx ${OUTDIR}/${SAMPLE}.hap1.fa
samtools faidx ${OUTDIR}/${SAMPLE}.hap2.fa