#!/bin/bash

mainDir=$(realpath $1)
sample=$2

grep -v "PredictionCorrect" $mainDir/prediction/chr*.predict.out \
