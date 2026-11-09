#!/bin/bash
# Parent of Origin Prediction using Pooch

# Reduce known pandas warning noise from external pofo_tools utilities.
warning_filters="ignore:The default of observed=False is deprecated and will be changed to True in a future version of pandas:FutureWarning,ignore:DataFrame is highly fragmented:Warning"
if [ -n "${PYTHONWARNINGS:-}" ]; then
	export PYTHONWARNINGS="${PYTHONWARNINGS},${warning_filters}"
else
	export PYTHONWARNINGS="$warning_filters"
fi

pooch_tool_dir=$(realpath $1)
# pooch_tool_dir points at the pofo_tools package dir; its parent (src/) must be
# on PYTHONPATH so "from pofo_tools.utils import *" resolves when running the
# scripts directly instead of via an installed package.
export PYTHONPATH="$(dirname "$pooch_tool_dir")${PYTHONPATH:+:$PYTHONPATH}"
mainDir=$(realpath $2)
# Expand glob pattern first, then resolve path
model_glob=( $3 )
model=$(realpath "${model_glob[0]}")
sample=$4
REF_NAME=$5
chr=$6
# Snakefile passes hap1_bed, hap2_bed, then the sex mapping file, in that order
hap1_met=$7
hap2_met=$8
mapFile=$9

mkdir -p "$mainDir/prediction"
fofn=$mainDir/prediction/$sample.fofn
sex=$(grep -w "$sample" "$mapFile" | awk '{print $2}')

if [ "$sex" == "Male" ] && [ "$chr" == "chrX" ]; then
	echo "$sample is Male, skipping chromosome X for parent of origin prediction."
	touch $mainDir/prediction/$chr.predict.out
	exit 0
fi

if [ "$sex" == "Female" ] && [ "$chr" == "chrX" ]; then
	echo -e "$sample is Female and chromosome is $chr"
	echo -e "Calculate XIST activity" 
	this_script="$(realpath "$(dirname "$0")")"
	xist_bed=$this_script/../../resources/xist_promoter.bed
	hap1_xist_methylation_proportion=$(bedtools intersect -a $xist_bed -b $hap1_met -wa -wb | bedtools merge -i - -c 7,8 -o sum,sum | \
	awk -v hap="hap1" -v sample=$sample '{print $0,sample,hap}' OFS='\t' | awk '{print $5/$4}')
	hap2_xist_methylation_proportion=$(bedtools intersect -a $xist_bed -b $hap2_met -wa -wb | bedtools merge -i - -c 7,8 -o sum,sum | \
	awk -v hap="hap2" -v sample=$sample '{print $0,sample,hap}' OFS='\t' | awk '{print $5/$4}')
	diff=$(awk -v h1="$hap1" -v h2="$hap2" 'BEGIN {print h1 - h2}')
	echo -e "Sample\tXIST_diff" > $mainDir/prediction/xist_level.tsv
	echo -e "$sample\t$diff" >> $mainDir/prediction/xist_level.tsv
	predict_script=$pooch_tool_dir/predict_parent_origin_xist.py
	predict_param="--xist $mainDir/prediction/xist_level.tsv"
	createMethylDataset_script=$pooch_tool_dir/createMethylDataset_xist.py
else
	predict_script=$pooch_tool_dir/predict.py
	createMethylDataset_script=$pooch_tool_dir/createMethylDataset.py
fi


if [ ! -s $fofn ]; then 
# echo -e "$fofn is not -not empty or does not exist. Please check the file path and ensure it is correct." 
# exit 1
echo "$fofn is empty, so generating one"
touch $fofn
echo -e "$sample\tHap1\t$hap1_met" >> $fofn
echo -e "$sample\tHap2\t$hap2_met" >> $fofn
fi

# for bed_file in $(cut -f 3 $fofn ); do
#    sed -i '/^chr/!s/^/chr/' $bed_file
# done
echo "Creating methylation dataset for $sample on chromosome $chr"
## Step1 : Run Pooch : createMethylDataset
cmd="python $createMethylDataset_script \
	--chr $chr \
	--input-fofn $fofn \
	--output-file $mainDir/prediction/$chr.createMethylDataset.out \
	--log-file $mainDir/prediction/$chr.createMethylDataset.log \
	--sex-mapping-file $mapFile \
	--filter-model $model"
eval $cmd &&

## Step2 : Run Pooch : createDiffDatasetFromMethylDataset
if [[ -f $mainDir/prediction/$chr.createMethylDataset.out ]]; then
echo "Creating differential methylation dataset for $sample on chromosome $chr"
cmd="python $pooch_tool_dir/createDiffDatasetFromMethylDataset.py \
	--input-methyl-dataset $mainDir/prediction/$chr.createMethylDataset.out \
	-L Hap1 -R Hap2 \
	-c $chr \
	-l $mainDir/prediction/$chr.createDiffDatasetFromMethylDataset.log \
	--output-file $mainDir/prediction/$chr.createDiffDatasetFromMethylDataset.out"
eval $cmd
else
	echo "Error: $mainDir/prediction/$chr.createMethylDataset.out not found. Differential methylation dataset creation failed for $sample on chromosome $chr."
	exit 1
fi	

## Step3 : Run Pooch : predict
if [[ -f $mainDir/prediction/$chr.createDiffDatasetFromMethylDataset.out ]]; then
echo "Predicting parent of origin for $sample on chromosome $chr"
cmd="python $predict_script \
	-i $mainDir/prediction/$chr.createDiffDatasetFromMethylDataset.out \
	-M $model \
	-o $mainDir/prediction/$chr.predict.out \
	-l $mainDir/prediction/$chr.predict.log $predict_param"
eval $cmd
else 
	echo "Error: $mainDir/prediction/$chr.createDiffDatasetFromMethylDataset.out not found. Prediction failed for $sample on chromosome $chr."
	exit 1
fi