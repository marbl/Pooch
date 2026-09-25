#!/bin/bash


echo -n "Checking fastq headers for MM tag..."
fastq_list=$1
outdir=$2
sample=$3


IFS=',' read -r -a fastq_list_sep <<< "$fastq_list"

count_MM=0
count_no_MM=0
for fastq in "${fastq_list_sep[@]}"
do
header_check=$(less $fastq | head -1| grep -w "MM")
# echo $fastq
if [[ -z "$header_check" ]]; then
    echo "Error: $fastq does not contain MM tag in the header. Please check the fastq file."
    count_no_MM=$((count_no_MM + 1))
else
    count_MM=$((count_MM + 1))
fi
done

if [[ $count_no_MM -eq 0 ]]; then
    echo -e "All fastqs have MM tag in the header."
    touch "${outdir}/alignments/${sample}.step0_methylation_tag_check.done"
else
    echo -e "$count_MM fastqs have MM tag in the header."
    echo -e "$count_no_MM fastqs don't have MM tag in the header. Please check the fastq files."
fi