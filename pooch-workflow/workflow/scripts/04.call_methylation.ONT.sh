#! /bin/bash

# Call methylation from PacBio HiFi reads using modkit
# Usage: ./01.call_methylation.sh <bam.file> <output.dir> <sample.name> <core>
if command -v module >/dev/null 2>&1; then
    module load modkit
    # ml modkit/0.4.1
fi

# check if modkit is available
if ! command -v modkit >/dev/null 2>&1; then
    echo "Error: modkit is not installed or not in PATH."
    exit 1
fi

BAM="$1"
mainDir="$2"
sample="$3"
CORE="$4"
Dip_genome="$5"

# Make methylation directory
mkdir -p "$mainDir/methylation"


# Run Modkit pileup command
echo -e "Methylation calling using Modkit\n"

# mokit v0.6.2
cmd1="modkit pileup -t $CORE \
$BAM $mainDir/methylation/${sample}.pri.meth.bed \
--log-filepath $mainDir/logs/${sample}_modkit.log" 

# modkit v0.6.1 test 
cmd2="modkit pileup -t $CORE --cpg --reference $Dip_genome \
$BAM $mainDir/methylation/${sample}.pri.meth.bed \
--modified-bases 5mC \
--log-filepath $mainDir/methylation/${sample}_modkit.log" 

cmd3="modkit pileup -t $CORE \
--cpg --reference $Dip_genome \
$BAM $mainDir/methylation/${sample}.pri.meth.bed \
--modified-bases 5mC --combine-strands \
--log-filepath $mainDir/methylation/${sample}_modkit.log" 

# modkit v0.4.1
cmd4="modkit pileup -t $CORE \
--preset traditional \
--ref $Dip_genome $BAM $mainDir/methylation/${sample}.pri.meth.bed \
--log-filepath $mainDir/methylation/${sample}_modkit.log"

if [ -f "$mainDir/methylation/${sample}.pri.meth.bed" ]; then
    echo "Methylation file already exists. Skipping Modkit pileup."
else
    echo "Running Modkit pileup command..."
    echo "CMD : $cmd3" &&
    eval $cmd3 
fi

# Filter columns that we need for downstream analysis (chr, start, end, coverage, and read modified)
grep -w "m" "$mainDir/methylation/${sample}.pri.meth.bed" |\
cut -f 1,2,3,12,10 > "$mainDir/methylation/${sample}.pri.meth.filt.bed"

# Clean up intermediate files
# rm "$mainDir/methylation/${sample}.pri.meth.bed"