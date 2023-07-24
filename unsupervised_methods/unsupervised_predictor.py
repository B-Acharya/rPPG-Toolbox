"""Unsupervised learning methods including POS, GREEN, CHROME, ICA, LGI and PBV."""

import logging
import os
from collections import OrderedDict

import numpy as np
import pandas as pd
import torch
from evaluation.post_process import *
from unsupervised_methods.methods.CHROME_DEHAAN import *
from unsupervised_methods.methods.GREEN import *
from unsupervised_methods.methods.ICA_POH import *
from unsupervised_methods.methods.LGI import *
from unsupervised_methods.methods.PBV import *
from unsupervised_methods.methods.POS_WANG import *
from tqdm import tqdm
import mlflow

def calcualte_mae_per_setting(dataframe):
    mae = np.mean(np.abs(dataframe["HR_GT"] - dataframe["HR_Pred"]))
    standard_error = np.std(np.abs(dataframe["HR_GT"] - dataframe["HR_Pred"])) / np.sqrt(len(dataframe))
    return mae, standard_error


def unsupervised_predict(config, data_loader, method_name):
    """ Model evaluation on the testing dataset."""
    if data_loader["unsupervised"] is None:
        raise ValueError("No data for unsupervised method predicting")
    print("===Unsupervised Method ( " + method_name + " ) Predicting ===")
    predict_hr_peak_all = []
    gt_hr_peak_all = []
    predict_hr_fft_all = []
    gt_hr_fft_all = []
    predictions_dict = dict()
    SNR_all = []
    sbar = tqdm(data_loader["unsupervised"], ncols=80)
    for _, test_batch in enumerate(sbar):
        batch_size = test_batch[0].shape[0]
        for idx in range(batch_size):
            data_input, labels_input = test_batch[0][idx].cpu().numpy(), test_batch[1][idx].cpu().numpy()
            filename = test_batch[2][idx]
            index = test_batch[2]
            MAE_per_scenarios = dict()
            if method_name == "POS":
                BVP = POS_WANG(data_input, config.UNSUPERVISED.DATA.FS)
            elif method_name == "CHROM":
                BVP = CHROME_DEHAAN(data_input, config.UNSUPERVISED.DATA.FS)
            elif method_name == "ICA":
                BVP = ICA_POH(data_input, config.UNSUPERVISED.DATA.FS)
            elif method_name == "GREEN":
                BVP = GREEN(data_input)
            elif method_name == "LGI":
                BVP = LGI(data_input)
            elif method_name == "PBV":
                BVP = PBV(data_input)
            else:
                raise ValueError("unsupervised method name wrong!")

            video_frame_size = test_batch[0].shape[1]
            if config.INFERENCE.EVALUATION_WINDOW.USE_SMALLER_WINDOW:
                window_frame_size = config.INFERENCE.EVALUATION_WINDOW.WINDOW_SIZE * config.UNSUPERVISED.DATA.FS
                if window_frame_size > video_frame_size:
                    window_frame_size = video_frame_size
            else:
                window_frame_size = video_frame_size

            for i in range(0, len(BVP), window_frame_size):
                BVP_window = BVP[i:i+window_frame_size]
                label_window = labels_input[i:i+window_frame_size]

                if len(BVP_window) < 9:
                    print(f"Window frame size of {len(BVP_window)} is smaller than minimum pad length of 9. Window ignored!")
                    continue

                if config.INFERENCE.EVALUATION_METHOD == "peak detection":
                    gt_hr, pre_hr, SNR = calculate_metric_per_video(BVP_window, label_window, diff_flag=False,
                                                                    fs=config.UNSUPERVISED.DATA.FS, hr_method='Peak')
                    gt_hr_peak_all.append(gt_hr)
                    predict_hr_peak_all.append(pre_hr)
                    SNR_all.append(SNR)
                elif config.INFERENCE.EVALUATION_METHOD == "FFT":
                    gt_fft_hr, pre_fft_hr, SNR = calculate_metric_per_video(BVP_window, label_window, diff_flag=False,
                                                                    fs=config.UNSUPERVISED.DATA.FS, hr_method='FFT')
                    gt_hr_fft_all.append(gt_fft_hr)
                    predict_hr_fft_all.append(pre_fft_hr)
                    SNR_all.append(SNR)
                else:
                    raise ValueError("Inference evaluation method name wrong!")
            if config.INFERENCE.EVALUATION_METHOD == "peak detection":
                gt_hr, pre_hr = calculate_metric_per_video(BVP, labels_input, diff_flag=False,
                                                                fs=config.UNSUPERVISED.DATA.FS, hr_method='Peak')
                predict_hr_peak_all.append(pre_hr)
                gt_hr_peak_all.append(gt_hr)
            if config.INFERENCE.EVALUATION_METHOD == "FFT":
                gt_fft_hr, pre_fft_hr, SNR_predicted = calculate_metric_per_video(BVP, labels_input, diff_flag=False,
                                                                   fs=config.UNSUPERVISED.DATA.FS, hr_method='FFT')
                predict_hr_fft_all.append(pre_fft_hr)
                gt_hr_fft_all.append(gt_fft_hr)

                predictions_dict[filename] = {"GT_HR":gt_fft_hr, "Pred_HR":pre_fft_hr}

            MAE = np.mean(np.abs(gt_fft_hr - pre_fft_hr))
            print("FFT MAE (FFT Label): {0} ".format(MAE))
            mlflow.log_metric(filename, MAE)

    print("Used Unsupervised Method: " + method_name)
    dataframe = pd.DataFrame.from_dict(predictions_dict).T
    dataframe.to_csv(f"{config.UNSUPERVISED.DATA.CACHED_PATH}/{method_name}.csv")
    if config.INFERENCE.EVALUATION_METHOD == "peak detection":
        predict_hr_peak_all = np.array(predict_hr_peak_all)
        gt_hr_peak_all = np.array(gt_hr_peak_all)
        SNR_all = np.array(SNR_all)
        num_test_samples = len(predict_hr_peak_all)
        for metric in config.UNSUPERVISED.METRICS:
            if metric == "MAE":
                MAE_PEAK = np.mean(np.abs(predict_hr_peak_all - gt_hr_peak_all))
                standard_error = np.std(np.abs(predict_hr_peak_all - gt_hr_peak_all)) / np.sqrt(num_test_samples)
                print("Peak MAE (Peak Label): {0} +/- {1}".format(MAE_PEAK, standard_error))
            elif metric == "RMSE":
                RMSE_PEAK = np.sqrt(np.mean(np.square(predict_hr_peak_all - gt_hr_peak_all)))
                standard_error = np.std(np.square(predict_hr_peak_all - gt_hr_peak_all)) / np.sqrt(num_test_samples)
                print("PEAK RMSE (Peak Label): {0} +/- {1}".format(RMSE_PEAK, standard_error))
            elif metric == "MAPE":
                MAPE_PEAK = np.mean(np.abs((predict_hr_peak_all - gt_hr_peak_all) / gt_hr_peak_all)) * 100
                standard_error = np.std(np.abs((predict_hr_peak_all - gt_hr_peak_all) / gt_hr_peak_all)) / np.sqrt(num_test_samples) * 100
                print("PEAK MAPE (Peak Label): {0} +/- {1}".format(MAPE_PEAK, standard_error))
            elif metric == "Pearson":
                Pearson_PEAK = np.corrcoef(predict_hr_peak_all, gt_hr_peak_all)
                correlation_coefficient = Pearson_PEAK[0][1]
                standard_error = np.sqrt((1 - correlation_coefficient**2) / (num_test_samples - 2))
                print("PEAK Pearson (Peak Label): {0} +/- {1}".format(correlation_coefficient, standard_error))
            elif metric == "SNR":
                SNR_FFT = np.mean(SNR_all)
                standard_error = np.std(SNR_all) / np.sqrt(num_test_samples)
                print("FFT SNR (FFT Label): {0} +/- {1}".format(SNR_FFT, standard_error))
            else:
                raise ValueError("Wrong Test Metric Type")
    elif config.INFERENCE.EVALUATION_METHOD == "FFT":
        predict_hr_fft_all = np.array(predict_hr_fft_all)
        gt_hr_fft_all = np.array(gt_hr_fft_all)
        SNR_all = np.array(SNR_all)
        num_test_samples = len(predict_hr_fft_all)
        for metric in config.UNSUPERVISED.METRICS:
            if metric == "MAE":
                MAE_FFT = np.mean(np.abs(predict_hr_fft_all - gt_hr_fft_all))
                standard_error = np.std(np.abs(predict_hr_fft_all - gt_hr_fft_all)) / np.sqrt(num_test_samples)
                print("FFT MAE (FFT Label): {0} +/- {1}".format(MAE_FFT, standard_error))
                mlflow.log_metric("MAE", MAE_FFT)
                mlflow.log_metric("MAE_error", standard_error)
            elif metric == "RMSE":
                RMSE_FFT = np.sqrt(np.mean(np.square(predict_hr_fft_all - gt_hr_fft_all)))
                standard_error = np.std(np.square(predict_hr_fft_all - gt_hr_fft_all)) / np.sqrt(num_test_samples)
                print("FFT RMSE (FFT Label): {0} +/- {1}".format(RMSE_FFT, standard_error))
                mlflow.log_metric("RMSE", RMSE_FFT)
                mlflow.log_metric("RMSE_error", standard_error)
            elif metric == "MAPE":
                MAPE_FFT = np.mean(np.abs((predict_hr_fft_all - gt_hr_fft_all) / gt_hr_fft_all)) * 100
                standard_error = np.std(np.abs((predict_hr_fft_all - gt_hr_fft_all) / gt_hr_fft_all)) / np.sqrt(num_test_samples) * 100
                print("FFT MAPE (FFT Label): {0} +/- {1}".format(MAPE_FFT, standard_error))
                mlflow.log_metric("MAPE", MAPE_FFT)
                mlflow.log_metric("MAPE_error", standard_error)
            elif metric == "Pearson":
                Pearson_FFT = np.corrcoef(predict_hr_fft_all, gt_hr_fft_all)
                correlation_coefficient = Pearson_FFT[0][1]
                standard_error = np.sqrt((1 - correlation_coefficient**2) / (num_test_samples - 2))
                print("FFT Pearson (FFT Label): {0} +/- {1}".format(correlation_coefficient, standard_error))
            elif metric == "SNR":
                SNR_PEAK = np.mean(SNR_all)
                standard_error = np.std(SNR_all) / np.sqrt(num_test_samples)
                print("FFT SNR (FFT Label): {0} +/- {1}".format(SNR_PEAK, standard_error))
            else:
                raise ValueError("Wrong Test Metric Type")
    else:
        raise ValueError("Inference evaluation method name wrong!")

    if config.UNSUPERVISED.DATA.DATASET == "CMBP":
        result = {'LowHR_Bright': {}, 'LowHR_Dark': {}, 'HighHR_Dark': {}, 'HighHR_Bright': {}}
        for index in predictions_dict.keys():
            HR_GT, HR_pred = predictions_dict[index]["GT_HR"],predictions_dict[index]["Pred_HR"]

            if index[-1] == "0":
                result['LowHR_Bright'][index[:-1]] = {"HR_GT": float(HR_GT), "HR_Pred": float(HR_pred)}
            elif index[-1] == "1":
                result['LowHR_Dark'][index[:-1]] = {"HR_GT": float(HR_GT), "HR_Pred": float(HR_pred)}
            if index[-1] == "2":
                result['HighHR_Dark'][index[:-1]] = {"HR_GT": float(HR_GT), "HR_Pred": float(HR_pred)}
            if index[-1] == "3":
                result['HighHR_Bright'][index[:-1]] = {"HR_GT": float(HR_GT), "HR_Pred": float(HR_pred)}

        for key in result.keys():
            dataframe = pd.DataFrame.from_dict(result[key]).T
            mae, standard_error = calcualte_mae_per_setting(dataframe)
            print(f"--{key}--")
            print(mae, standard_error)
            mlflow.log_metric(key, mae)
            mlflow.log_metric(key + "_std_error", standard_error)
