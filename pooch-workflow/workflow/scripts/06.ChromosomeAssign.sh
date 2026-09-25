#! /bin/bash

# Assign chromosomes to parental haplotypes using alignment paf file
# Usage: ./06.ChromosomeAssign.sh <paf.file>

mainDir=$1
sample=$2
REF_NAME=$3
min_align_length=$4 # 10000
chain_hap1=$5
chain_hap2=$6

cat $chain_hap1 $chain_hap2 | \
grep "^chain" | \
awk -v min_len=$min_align_length '$7-$6 > min_len { print $8,$3,$5,$7-$6 }' OFS='\t' | \
    awk -F'\t' '{
        key = $1 FS $2 FS $3
        sum[key] += $4
    }
    END {
        for (k in sum) print k, sum[k]
    }' OFS='\t' | sort -k1,1 -k2,2 -k3,3 > $mainDir/chain/${REF_NAME}_to_${sample}.chromAssign.all.txt &&

awk -F'\t' '
{
    if (!($1 in max) || $4 > max[$1]) {
        max[$1] = $4
        row[$1] = $0
    }
}
END {
    for (k in row) print row[k]
}' $mainDir/chain/${REF_NAME}_to_${sample}.chromAssign.all.txt | sort -k1 > \
${mainDir}/chain/${REF_NAME}_to_${sample}.chromAssign.filtered.txt