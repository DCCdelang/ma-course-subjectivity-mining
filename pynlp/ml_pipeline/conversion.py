import pandas as pd
import numpy as np

# file = "data/test/trainData"

# df = pd.read_csv(file+".csv",delimiter="\t")
# print(df.head())

# df.loc[df.Label=="hate","Label"] = "OFF"
# df.loc[df.Label=="noHate","Label"] = "NOT"

# df.to_csv(file+"_converted.csv", sep='\t', index=False)

file = "data/gibert_vua_format/trainData"

df = pd.read_csv(file+".csv",delimiter="\t")
print(df.head())

df.loc[df.Label=="hate","Label"] = "OFF"
df.loc[df.Label=="noHate","Label"] = "NOT"

df.to_csv(file+"_converted.csv", sep='\t', index=False)