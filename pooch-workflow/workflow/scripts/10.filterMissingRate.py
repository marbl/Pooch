import pandas as pd
import seaborn as sns
import glob as glob
import sys

# get arguments from outside
chr=sys.argv[1]

default_missing_rate = 5.0  # default missing rate if not provided
missing_rate = float(sys.argv[2]) if len(sys.argv) > 2 else default_missing_rate  # percentage of missing values allowed in each column

print(f"chr : {chr}")
print(f"missing_rate : {missing_rate}")

# calculate variance for each column
file = f"{chr}.createMethylDataset.out"

df_train_curr = pd.read_csv(file, sep = '\t', header = 0 )

print(f"before filtering : {df_train_curr.shape}")

# Keep metadata columns unchanged; filter only feature columns by NA rate
meta_cols = df_train_curr.columns[:3]
feature_cols = df_train_curr.columns[3:]

na_counts_column = df_train_curr[feature_cols].isna().sum(axis=0)
max_na = df_train_curr.shape[0] * missing_rate / 100

print(f"number of feature columns above threshold ({max_na:.2f} NAs): {(na_counts_column > max_na).sum()}")
keep_feature_cols = na_counts_column[na_counts_column <= max_na].index
df_train = pd.concat([df_train_curr[meta_cols], df_train_curr[keep_feature_cols]], axis=1)

print(f"after filtering : {df_train.shape}")

df_train.to_csv(f"{chr}.missing_rate_{missing_rate}.createMethylDataset.out", sep='\t', header = True, index = False)