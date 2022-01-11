import numpy as np 
import pandas as pd
import os

import random

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdMolDescriptors

dataset_names = ["human", "celegans"]

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(x))

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

for dataset_name in dataset_names:
    filename = "../../data/maked/default/" + dataset_name + ".csv"
    for random_state in range(10):
        df = pd.read_csv("../../data/maked/default/" + dataset_name + ".csv", header=0)
        df = df.sample(frac=1, random_state=random_state)
        N = len(df)
        train_len = int(N * 0.1)
        df_test = df[:train_len]
        df = df[train_len:]
        df_compound = pd.DataFrame(df["compound"].unique(), columns=["compound"])
        df_compound["weight"] = df_compound["compound"].map(lambda sm: rdMolDescriptors._CalcMolWt(Chem.MolFromSmiles(sm)))
        df_compound["weight_norm"] = (df_compound["weight"] - df_compound["weight"].mean()) / df_compound["weight"].std()
        df_compound["prob"] = df_compound["weight_norm"].map(sigmoid)
        df_compound["observed"] = df_compound["prob"].map(observed)
        df = pd.merge(df, df_compound[["compound", "observed"]], how="left", on="compound")
        df_observed = df[df["observed"] == 1]
        count_compound = df_observed["compound"].value_counts().to_dict()
        df_observed["weight"] = df_observed["compound"].apply(calc_weight, counts=count_compound)

        # os.mkdir("../../data/maked/bias/" + str(random_state) + "/")

        df_observed[["compound", "protein", "label", "weight"]].to_csv("../../data/maked/bias/" + str(random_state) + "/train_" + dataset_name + ".csv", index=None)
        df_test.to_csv("../../data/maked/bias/" + str(random_state) + "/test_" + dataset_name + ".csv", index=None)
