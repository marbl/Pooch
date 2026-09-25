#! /bin/bash
set -euo pipefail
# Align PacBio HiFi reads to the reference genome using Winnowmap2
# Usage: ./00.align_winnowmap.sh <reference.fasta> <output.dir> <sample.name> <reads.fastq> <platform>

if command -v module >/dev/null 2>&1; then
    module load winnowmap
    module load samtools
    module load meryl
fi

# check if the tools are available
if ! command -v winnowmap >/dev/null 2>&1; then
    echo "Error: winnowmap is not installed or not in PATH."
    exit 1
fi
if ! command -v samtools >/dev/null 2>&1; then
    echo "Error: samtools is not installed or not in PATH."
    exit 1
fi
if ! command -v meryl >/dev/null 2>&1; then
    echo "Error: meryl is not installed or not in PATH."
    exit 1
fi

############################################# ARGUMENTS #############################################
ref="$1"
mainDir="$2"
sample="$3"
fastqs="$4" # comma-separated list of fastq files
platform_input="$5"
core="$6"

echo -e "Reference: $ref\nOutput Directory: $mainDir\nSample Name: $sample\nReads: $fastqs\nPlatform: $platform_input\nThreads: $core"
############################################# ARGUMENTS #############################################

# check inputs 
if [ -z "$ref" ] || [ -z "$mainDir" ] || [ -z "$sample" ] || [ -z "$fastqs" ] || [ -z "$platform_input" ] || [ -z "$core" ]; then
    echo "Usage: $0 <reference.fasta> <output.dir> <sample.name> <reads.fastq> <platform> <threads> [ref_index]"
    exit 1
fi
if [ ! -f "$ref" ]; then
    echo "Error: Reference genome file '$ref' not found."
    exit 1
fi

if [ "$platform_input" = "ONT" ]; then
    platform="map-ont"
elif [ "$platform_input" = "PACBIO" ]; then
    platform="map-pb"
else
    echo "Error: Unsupported platform '$platform_input'. Supported platforms are ONT and PACBIO."
    exit 1
fi

IFS=',' read -r -a fastq_arr <<< "$fastqs"

mkdir -p "${mainDir}/alignments"

bam_path="$mainDir/alignments/$sample.bam"
pri_bam_path="$mainDir/alignments/$sample.pri.bam"
pri_bai_path="$mainDir/alignments/$sample.pri.bam.bai"

sort_tmp_root="${TMPDIR:-/tmp}"
sort_tmp_dir="${sort_tmp_root%/}/pooch_samtools_sort"
mkdir -p "$sort_tmp_dir"
if [ ! -w "$sort_tmp_dir" ]; then
    echo "Error: sort tmp directory is not writable: $sort_tmp_dir"
    exit 1
fi
sort_tmp_prefix="$sort_tmp_dir/${sample}.$$"

# Check index
if [ ! -f  "${mainDir}/alignments/repetitive_k15_${sample}.txt" ]; then
    echo -e "Index not found. Creating index for $ref"

    meryl count k=15 output "${mainDir}/alignments/${sample}_merylDB" "$ref" &&
    meryl print greater-than distinct=0.9998 "${mainDir}/alignments/${sample}_merylDB" > "${mainDir}/alignments/repetitive_k15_${sample}.txt"
else
    echo -e "Index already exists. Skipping index creation."
fi

# filter primary alignments
if [ -s "$pri_bam_path" ] && [ -s "$pri_bai_path" ]; then
    echo -e "Primary BAM already exists. Skipping mapping/filtering."
else
    # If pri BAM is missing, force a fresh mapping when source BAM is unavailable.
    if [ ! -s "$bam_path" ]; then
        echo -e "Mapping $sample reads to $ref using Winnowmap2"
        winnowmap -W "${mainDir}/alignments/repetitive_k15_${sample}.txt" -ax "$platform" -y -Y --MD -I12g -t "$core" "$ref" "${fastq_arr[@]}" | \
        samtools sort -@ "$core" -O bam -o "$bam_path" -T "$sort_tmp_prefix"
    else
        echo -e "BAM already exists. Skipping mapping."
    fi

    if [ -s "$bam_path" ] && [ ! -f "$bam_path.bai" ]; then
        echo -e "Indexing alignments for $sample"
        samtools index -@ ${core} "$bam_path"
    fi

    if [ -s "$bam_path" ] && [ ! -f "$pri_bam_path" ]; then
        echo -e "Filtering primary alignments for $sample"
        samtools view -@ ${core} -F 0x104 -hb "$bam_path" > "$pri_bam_path" && 
        samtools index -@ ${core} "$pri_bam_path" && 
        rm -f "$bam_path" "$bam_path.bai"
    else
        echo -e "Primary BAM already exists. Skipping filtering."
    fi
fi