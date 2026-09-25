#! /bin/bash
set -o pipefail
# Align PacBio HiFi reads to the reference genome using Winnowmap2
# Usage: ./00.align_winnowmap.sh <reference.fasta> <output.dir> <sample.name> <reads.fastq> <platform>

if command -v module >/dev/null 2>&1; then
    module load minimap2
    module load samtools
    module load meryl
fi

ref="$1"
mainDir="$2"
sample="$3"
fastqs="$4" # comma-separated list of fastq files

if [ "$5" = "ONT" ]; then
    platform="map-ont"
elif [ "$5" = "PACBIO" ]; then
    platform="map-pb"
fi
core=$6

fastq_list=$(echo $fastqs | tr ',' ' ')
ref_index=$(dirname "$ref")
ref_basename=$(basename "${ref%%.*}")

mkdir -p "${mainDir}/alignments"

sort_tmp_root="${TMPDIR:-/tmp}"
sort_tmp_dir="${sort_tmp_root%/}/pooch_samtools_sort"
mkdir -p "$sort_tmp_dir"
if [ ! -w "$sort_tmp_dir" ]; then
    echo "Error: sort tmp directory is not writable: $sort_tmp_dir"
    exit 1
fi
sort_tmp_prefix="$sort_tmp_dir/${sample}.$$"

# Check index
if [ ! -f "${mainDir}/alignments/${sample}.idx.done" ]; then
    echo -e "Index not found. Creating index for $ref"
    # minimap2 -x map-ont -d "$mainDir/alignments/$ref_basename.mmi" "$ref"
    touch "${mainDir}/alignments/${sample}.idx.done"
fi

# Mapping
if [ ! -f "$mainDir/alignments/$sample.mapping.done" ]; then
    echo -e "Mapping $sample reads to $ref using Minimap2"
    minimap2 -ax $platform -t $core --MD --cs -Y -y "$ref" $fastq_list | \
    samtools sort -@ ${core} -O bam -o "$mainDir/alignments/$sample.bam" -T "$sort_tmp_prefix" &&
    samtools index -@ ${core} "$mainDir/alignments/$sample.bam" &&
    touch "$mainDir/alignments/$sample.mapping.done"
fi

# filter primary alignments
if [ ! -f "$mainDir/alignments/$sample.pri.bam.done" ]; then
    echo -e "Filtering primary alignments for $sample"
    samtools view -@ ${core}  -F 0x104 -hb $mainDir/alignments/$sample.bam > $mainDir/alignments/$sample.pri.bam &&
    samtools index -@ ${core} $mainDir/alignments/$sample.pri.bam &&
    rm $mainDir/alignments/$sample.bam $mainDir/alignments/$sample.bam.bai && 
    touch $mainDir/alignments/$sample.pri.bam.done
fi