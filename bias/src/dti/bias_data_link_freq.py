import numpy as np 
import pandas as pd
import os

import random

# from rdkit import Chem
# from rdkit.Chem import AllChem
# from rdkit.Chem import rdMolDescriptors

dataset_names = ["human", "celegans"]

def calc(x):
    if x <= 0.01:
        return 0.01
    elif x >= 0.99:
        return 0.99
    else:
        return x

def observed(x):
    coin = random.random()
    if x > coin:
        return 1
    else:
        return 0

def calc_weight(x, counts):
    weight = 1 / counts[x]
    weight = max( weight, 0.1)
    weight = min( weight, 0.9)
    return weight

def counting(x):
    return count_compound[x]


for dataset_name in dataset_names:
    filename = "../../data/maked/default/" + dataset_name + ".csv"
    for random_state in range(100):
        df = pd.read_csv("../../data/maked/default/" + dataset_name + ".csv", header=0)
        df = df.sample(frac=1, random_state=random_state)
        N = len(df)
        train_len = int(N * 0.1)
        df_test = df[:train_len]
        df = df[train_len:]
        count_compound = df["compound"].value_counts().to_dict()
        df["molecular_weight"] = df["compound"].map(lambda x: count_compound[x])
        df["molecular_weight"] = df["molecular_weight"].map(lambda x: 20 if x > 20 else x)
        df["weight_norm"] = (df["molecular_weight"] - df["molecular_weight"].min()) / (df["molecular_weight"].max() - df["molecular_weight"].min())
        df["prob"] = df["weight_norm"].map(calc)
        df["observed"] = df["prob"].map(observed)
        df_observed = df[df["observed"] == 1]
        count_compound2 = df_observed["compound"].value_counts().to_dict()
        df_observed["weight"] = df_observed["compound"].apply(calc_weight, counts=count_compound2)

        os.makedirs("../../data/maked/bias/" + str(random_state) + "/", exist_ok=True)

        df_observed[["compound", "protein", "label", "weight"]].to_csv("../../data/maked/bias/" + str(random_state) + "/train_" + dataset_name + "_link_freq.csv", index=None)
        df_test.to_csv("../../data/maked/bias/" + str(random_state) + "/test_" + dataset_name + "_link_freq.csv", index=None)
