import pathlib
import pandas as pd
from collections import defaultdict
import numpy as np


def process_splits(path):

    csv_files = list(path.rglob("*test.csv"))

    folds = defaultdict(list)

    for csv_file in csv_files:

        participants = pd.read_csv(csv_file)['id'].unique().tolist()
        folds[csv_file.stem] = participants

    return folds


base_path = pathlib.Path("/homes/bacharya/rPPG-Toolbox/")

splits = base_path  / "splits"

metrics_paths = splits.rglob("*.json")

folds = process_splits(splits)

#calcualte the metrics for each fold

for metrics_pat in metrics_paths:
    mae = []

    metrics = pd.read_json(metrics_pat)
    print("----")
    print(metrics_pat.stem)

    for fold in folds.keys():

        participants = folds[fold]

        filtered_df = metrics[metrics.index.str.split('_').str[0].isin(participants)]

        diff = abs(filtered_df['GT_HR'] - filtered_df['Pred_HR'] )
        mae.append(diff.mean())

    MAE = sum(mae) / len(mae)
    STD = np.std(mae)/ np.sqrt(len(mae))

    print(MAE, STD)