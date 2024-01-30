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
from evaluation.BlandAltmanPy import BlandAltman
from tqdm import tqdm
import pickle


def metrics_calculations(ground_truth, predictions, SNR, config, logger):

    predictions = np.array(predictions)
    ground_truth = np.array(ground_truth)
    SNR = np.array(SNR)
    num_test_samples = len(predictions)
    method = config.INFERENCE.EVALUATION_METHOD
    filename_id = config.TRAIN.MODEL_FILE_NAME
    for metric in config.UNSUPERVISED.METRICS:
        if metric == "MAE":
            MAE = np.mean(np.abs(predictions - ground_truth))
            standard_error = np.std(np.abs(predictions - ground_truth)) / np.sqrt(num_test_samples)
            print("{2} MAE : {0} +/- {1}".format(MAE, standard_error, method))
            logger.log_metrics({"MAE": MAE})
        elif metric == "RMSE":
            RMSE = np.sqrt(np.mean(np.square(predictions - ground_truth)))
            standard_error = np.std(np.square(predictions - ground_truth)) / np.sqrt(num_test_samples)
            print("{2} RMSE : {0} +/- {1}".format(RMSE, standard_error, method))
            logger.log_metrics({"RMSE": RMSE})
        elif metric == "MAPE":
            MAPE = np.mean(np.abs((predictions - ground_truth) / ground_truth)) * 100
            standard_error = np.std(np.abs((predictions - ground_truth) / ground_truth)) / np.sqrt(
                num_test_samples) * 100
            print("{2} MAPE : {0} +/- {1}".format(MAPE, standard_error, method))
            logger.log_metrics({"MAPE": MAPE})
        elif metric == "Pearson":
            Pearson = np.corrcoef(predictions, ground_truth)
            correlation_coefficient = Pearson[0][1]
            standard_error = np.sqrt((1 - correlation_coefficient ** 2) / (num_test_samples - 2))
            print("{2} Pearson : {0} +/- {1}".format(correlation_coefficient, standard_error, method))
            logger.log_metrics({"Pearson": Pearson})
        elif metric == "SNR":
            SNR = np.mean(SNR)
            standard_error = np.std(SNR) / np.sqrt(num_test_samples)
            print("{2} SNR : {0} +/- {1}".format(SNR, standard_error, method))
            logger.log_metrics({"SNR": SNR})
        else:
            raise ValueError("Wrong Test Metric Type")
    compare = BlandAltman(ground_truth, predictions, config, logger=logger, averaged=True)
    compare.scatter_plot(
        x_label='GT PPG HR [bpm]',
        y_label='rPPG HR [bpm]',
        show_legend=True, figure_size=(5, 5),
        the_title=f'{filename_id}_peak_BlandAltman_ScatterPlot',
        file_name=f'{filename_id}_peak_BlandAltman_ScatterPlot.pdf')
    compare.difference_plot(
        x_label='Difference between rPPG HR and GT PPG HR [bpm]',
        y_label='Average of rPPG HR and GT PPG HR [bpm]',
        show_legend=True, figure_size=(5, 5),
        the_title=f'{filename_id}_peak_BlandAltman_DifferencePlot',
        file_name=f'{filename_id}_peak_BlandAltman_DifferencePlot.pdf')

def save_test_outputs( predictions, labels, config, method_name):
    if config.TOOLBOX_MODE == 'train_and_test' or config.TOOLBOX_MODE == 'only_test':
        output_dir = config.TEST.OUTPUT_SAVE_DIR
    else:
        output_dir = config.UNSUPERVISED.DATA.CACHED_PATH

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # Filename ID to be used in any output files that get saved
    if config.TOOLBOX_MODE == 'train_and_test':
        filename_id = config.MODEL
    elif config.TOOLBOX_MODE == 'only_test':
        model_file_root = config.INFERENCE.MODEL_PATH.split("/")[-1].split(".pth")[0]
        filename_id = model_file_root + "_" + config.TEST.DATA.DATASET
    elif config.TOOLBOX_MODE == 'unsupervised_method':
        filename_id = method_name

    else:
        raise ValueError('Metrics.py evaluation only supports train_and_test and only_test!')
    output_path = os.path.join(output_dir, filename_id + '_outputs.pickle')

    data = dict()
    data['predictions'] = predictions
    data['labels'] = labels
    data['label_type'] = config.TEST.DATA.PREPROCESS.LABEL_TYPE
    data['fs'] = config.TEST.DATA.FS

    with open(output_path, 'wb') as handle:  # save out frame dict pickle file
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print('Saving outputs to:', output_path)

def calcualte_mae_per_setting(dataframe):
    mae = np.mean(np.abs(dataframe["HR_GT"] - dataframe["HR_Pred"]))
    standard_error = np.std(np.abs(dataframe["HR_GT"] - dataframe["HR_Pred"])) / np.sqrt(len(dataframe))
    return mae, standard_error


def unsupervised_predict(config, data_loader, method_name, logger, log=True, save_outputs=True):
    """ Model evaluation on the testing dataset."""
    if data_loader["unsupervised"] is None:
        raise ValueError("No data for unsupervised method predicting")
    print("===Unsupervised Method ( " + method_name + " ) Predicting ===")
    gt_hr_all = []
    predict_hr_all = []
    predictions_dict = dict()
    predictions = dict()
    labels = dict()
    SNR_all = []

    if config.UNSUPERVISED.DATA.DATASET == "DST":
        print("Using DST")
        DST = True
    else:
        DST = False

    sbar = tqdm(data_loader["unsupervised"], ncols=80)
    for _, test_batch in enumerate(sbar):
        batch_size = test_batch[0].shape[0]
        for idx in range(batch_size):
            if DST:
                data_input, gt_hr = test_batch[0][idx].cpu().numpy(), test_batch[1][idx].cpu().numpy()
            else:
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
            elif method_name == "dummy":
                if DST:
                    pass
                else:
                    BVP = labels_input
            else:
                raise ValueError("unsupervised method name wrong!")

            if save_outputs:
                predictions[filename] = BVP
                if DST:
                    labels[filename] = gt_hr
                else:
                    labels[filename] = labels_input

            video_frame_size = test_batch[0].shape[1]
            if config.INFERENCE.EVALUATION_WINDOW.USE_SMALLER_WINDOW:
                window_frame_size = config.INFERENCE.EVALUATION_WINDOW.WINDOW_SIZE * config.UNSUPERVISED.DATA.FS
                if window_frame_size > video_frame_size:
                    window_frame_size = video_frame_size
            else:
                window_frame_size = video_frame_size

            for i in range(0, len(BVP), window_frame_size):
                BVP_window = BVP[i:i+window_frame_size]
                if DST:
                    print("DST")
                    pass
                else:
                    label_window = labels_input[i:i+window_frame_size]

                if len(BVP_window) < 9:
                    print(f"Window frame size of {len(BVP_window)} is smaller than minimum pad length of 9. Window ignored!")
                    continue

                if config.INFERENCE.EVALUATION_METHOD == "peak detection":
                    if DST:
                        pre_hr = calculate_HR(BVP_window, fs=config.UNSUPERVISED.DATA.FS, diff_flag=False, hr_method='Peak')
                    else:
                        gt_hr, pre_hr, SNR = calculate_metric_per_video(BVP_window, label_window, diff_flag=False,
                                                                    fs=config.UNSUPERVISED.DATA.FS, hr_method='Peak')
                elif config.INFERENCE.EVALUATION_METHOD == "FFT":
                    if DST:
                        pre_hr = calculate_HR(BVP_window, fs=config.UNSUPERVISED.DATA.FS, diff_flag=False, hr_method='FFT')
                    else:
                        gt_hr, pre_hr, SNR = calculate_metric_per_video(BVP_window, label_window, diff_flag=False,
                                                                            fs=config.UNSUPERVISED.DATA.FS,
                                                                            hr_method='FFT')
                elif config.INFERENCE.EVALUATION_METHOD == "Welch":
                    if DST:
                        pre_hr = calculate_HR(BVP_window, fs=config.UNSUPERVISED.DATA.FS, diff_flag=False, hr_method='Welch')
                    else:
                        gt_hr, pre_hr, SNR = calculate_metric_per_video(BVP_window, label_window, diff_flag=False,
                                                                            fs=config.UNSUPERVISED.DATA.FS,
                                                                            hr_method='Welch')
                else:
                    raise ValueError("Inference evaluation method name wrong!")

                gt_hr_all.append(gt_hr)
                predict_hr_all.append(pre_hr)

                if not DST:
                    SNR_all.append(SNR)

                #creating a dict for further processing
                predictions_dict[filename] = {"GT_HR": gt_hr, "Pred_HR": pre_hr}

                #MAE for each segment
                MAE = np.mean(np.abs(gt_hr - pre_hr))

                if config.INFERENCE.EVALUATION_WINDOW.USE_SMALLER_WINDOW:
                    print("FFT MAE for segment {1}: {0} ".format(MAE, filename))
                    if log:
                        logger.log_metrics({filename+"_smaller_window_"+str(i): MAE})
                else:
                    print("FFT MAE and GT for segment {1}: {0} {2}".format(MAE, filename, gt_hr))
                    if log:
                        logger.log_metrics({filename: MAE})

    if method_name == "dummy":
        print("Running mean predicition")
        mean_hr = np.mean(gt_hr_all)
        for key in predictions_dict.keys():
            predictions_dict[key]["Pred_HR"] = mean_hr
        for i in range(len(predict_hr_all)):
            predict_hr_all[i] = mean_hr
    if save_outputs:
        save_test_outputs(predictions, labels, config, method_name)

    print("Used Unsupervised Method: " + method_name)
    dataframe = pd.DataFrame.from_dict(predictions_dict).T
    dataframe.to_csv(f"{config.UNSUPERVISED.DATA.CACHED_PATH}/{method_name}_{config.INFERENCE.EVALUATION_METHOD}.csv")
    logger.experiment.log_dataframe_profile(dataframe, "whole-data")

    metrics_calculations(gt_hr_all, predict_hr_all, SNR_all, config, logger)

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

        df = pd.DataFrame.from_dict(dict(sorted(predictions_dict.items())))
        logger.experiment.log_dataframe_profile(df, "whole-data")

        for key in result.keys():
            dataframe = pd.DataFrame.from_dict(result[key]).T
            logger.experiment.log_dataframe_profile(dataframe, key)
            print(dataframe)
            mae, standard_error = calcualte_mae_per_setting(dataframe)
            print(f"--{key}--")
            print(mae, standard_error)
            logger.log_metrics({key: mae})
            logger.log_metrics({key + "_std_error": standard_error})
