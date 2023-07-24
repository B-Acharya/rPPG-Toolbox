import pathlib
from pprint import pprint
import pandas as pd
import numpy as np

def MAE(dataframe):
    mae = np.mean(np.abs(dataframe["HR_GT"] - dataframe["HR_Pred"]))
    return mae
def process_csv(path):
    method = path.stem
    result = {'LowHR_Bright': {}, 'LowHR_Dark': {}, 'HighHR_Dark': {}, 'HighHR_Bright': {}}
    with open(path, "r") as f:
        first = True
        for line in f.readlines():
            if first:
                first= False
                continue
            index, HR_GT, HR_pred = line.strip().split(",")
            if index[-1] == "0":
                result['LowHR_Bright'][index[:-1]] = {"HR_GT": float(HR_GT), "HR_Pred": float(HR_pred)}
            elif index[-1] == "1":
                result['LowHR_Dark'][index[:-1]] = {"HR_GT": float(HR_GT), "HR_Pred": float(HR_pred)}
            if index[-1] == "2":
                result['HighHR_Dark'][index[:-1]] = {"HR_GT":float(HR_GT), "HR_Pred": float(HR_pred)}
            if index[-1] == "3":
                result['HighHR_Bright'][index[:-1]] = {"HR_GT": float(HR_GT), "HR_Pred": float(HR_pred)}

    print(f"--------{method}--------")
    for key in result.keys():
        dataframe = pd.DataFrame.from_dict(result[key]).T
        mae = MAE(dataframe)
        print(f"--{key}--")
        print(mae)

if __name__ == "__main__":
    # path = pathlib.Path("./CMBP_SizeW72_SizeH72_ClipLength180_DataTypeRaw_LabelTypeRaw_Large_boxTrue_Large_size1.5_Dyamic_DetFalse_det_len180_unsupervised/No_facecrop/")
    path = pathlib.Path("./CMBP_SizeW72_SizeH72_ClipLength180_DataTypeRaw_LabelTypeRaw_Large_boxTrue_Large_size1.5_Dyamic_DetFalse_det_len180_unsupervised/")
    paths = path.glob("*.csv")
    for p in paths:
        process_csv(p)


