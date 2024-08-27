"""Unsupervised learning methods including POS, GREEN, CHROME, ICA, LGI and PBV."""

import logging
import os
from collections import OrderedDict

import matplotlib.pyplot as plt
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
from evaluation.metrics import metrics_calculations
from tqdm import tqdm
import random
import pickle
from unsupervised_methods.utils import ecg_processing



def save_test_outputs( predictions, labels, config, method_name):
    if config.TOOLBOX_MODE == 'train_and_test' or config.TOOLBOX_MODE == 'only_test':
        output_dir = config.TEST.OUTPUT_SAVE_DIR
    else:
        output_dir = config.UNSUPERVISED.OUT_SAVE_DIR

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
        DST = True
    else:
        DST = False

    sbar = tqdm(data_loader["unsupervised"], ncols=80)
    for _, test_batch in enumerate(sbar):
        batch_size = test_batch[0].shape[0]
        for idx in range(batch_size):
            data_input, labels_input = test_batch[0][idx].cpu().numpy(), test_batch[1][idx].cpu().numpy()
            filename = test_batch[2][idx]
            if len(labels_input) == 0:
                print("--------------")
                print("--------------")
                print("--------------")
                print("--------------")
                print("--------------")
                print("ECG data missing, please check", filename)
                print("--------------")
                print("--------------")
                print("--------------")
                print("--------------")
                print("--------------")
                continue
            if data_input.shape[0] == 0:
                print("--------------")
                print("--------------")
                print("--------------")
                print("--------------")
                print("Video data missing, please check", filename)
                print("--------------")
                print("--------------")
                print("--------------")
                print("--------------")
                continue
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
            elif method_name == "RED":
                BVP = RED(data_input)
            elif method_name == "BLUE":
                BVP = BLUE(data_input)
            elif method_name == "LGI":
                BVP = LGI(data_input)
            elif method_name == "PBV":
                BVP = PBV(data_input)
            elif method_name == "dummy" or method_name == "random":
                BVP = labels_input
            else:
                raise ValueError("unsupervised method name wrong!")

            if save_outputs:
                predictions[filename] = BVP
                print(BVP.shape)
                labels[filename] = labels_input

            if DST:
                #TODO: should be changed and moved to dataloader, bit hacky
                label_frame_size = len(labels_input)
                if label_frame_size>25000 and label_frame_size<30000:
                    sampling_rate = 300
                else:
                    sampling_rate = 1000

            video_frame_size = test_batch[0].shape[1]
            if config.INFERENCE.EVALUATION_WINDOW.USE_SMALLER_WINDOW:
                window_frame_size = config.INFERENCE.EVALUATION_WINDOW.WINDOW_SIZE * config.UNSUPERVISED.DATA.FS
                if window_frame_size > video_frame_size:
                    window_frame_size = video_frame_size
                if DST:
                    label_frame_size = config.INFERENCE.EVALUATION_WINDOW.WINDOW_SIZE * sampling_rate
                    if method_name == "dummy" or method_name == "random":
                        window_frame_size = label_frame_size
            else:
                # if method_name == "dummy" or method_name == "random":
                #     window_frame_size = label_frame_size
                # else:
                window_frame_size = video_frame_size




            for i in range(0, len(BVP), window_frame_size):
                BVP_window = BVP[i:i+window_frame_size]

                if DST:
                    #conversion of number of samples to match the same window (BVP -> ECG )
                    if method_name == "dummy" or method_name == "random":
                        label_window = labels_input[i:i + label_frame_size]
                    else:
                        j = (i//config.UNSUPERVISED.DATA.FS) * sampling_rate
                        label_window = labels_input[j:j + label_frame_size]
                else:
                    label_window = labels_input[i:i + window_frame_size]

                if len(BVP_window) < 9:
                    print(f"Window frame size of {len(BVP_window)} is smaller than minimum pad length of 9. Window ignored!")
                    continue

                if DST and len(BVP_window)< window_frame_size//2:
                    print(f"Window smaller than the smaller window size. Window ignored")
                    continue

                if config.INFERENCE.EVALUATION_METHOD == "peak detection":
                    if DST:
                        pre_hr = calculate_HR(BVP_window, fs=config.UNSUPERVISED.DATA.FS, diff_flag=False, hr_method='Peak')
                        gt_hr = ecg_processing(label_window, sampling_rate=sampling_rate)
                    else:
                        gt_hr, pre_hr, SNR = calculate_metric_per_video(BVP_window, label_window, diff_flag=False,
                                                                    fs=config.UNSUPERVISED.DATA.FS, hr_method='Peak')
                elif config.INFERENCE.EVALUATION_METHOD == "FFT":
                    if DST:
                        pre_hr = calculate_HR(BVP_window, fs=config.UNSUPERVISED.DATA.FS, diff_flag=False, hr_method='FFT')
                        gt_hr = ecg_processing(label_window, sampling_rate=sampling_rate)
                    else:
                        gt_hr, pre_hr, SNR = calculate_metric_per_video(BVP_window, label_window, diff_flag=False,
                                                                            fs=config.UNSUPERVISED.DATA.FS,
                                                                            hr_method='FFT')
                elif config.INFERENCE.EVALUATION_METHOD == "Welch":
                    if DST:
                        if method_name == "dummy" or method_name == "random":
                            pre_hr = ecg_processing(label_window, sampling_rate=sampling_rate)
                        else:
                            pre_hr = calculate_HR(BVP_window, fs=config.UNSUPERVISED.DATA.FS, diff_flag=False, hr_method='Welch')
                        gt_hr = ecg_processing(label_window, sampling_rate=sampling_rate)
                    else:
                        gt_hr, pre_hr, SNR = calculate_metric_per_video(BVP_window, label_window, diff_flag=False,
                                                                            fs=config.UNSUPERVISED.DATA.FS,
                                                                            hr_method='Welch')
                else:
                    raise ValueError("Inference evaluation method name wrong!")

                if np.isnan(gt_hr):
                    print("Nan encountered in GT estimate")
                    continue

                gt_hr_all.append(gt_hr)
                predict_hr_all.append(pre_hr)
                if not DST:
                    SNR_all.append(SNR)

                #creating a dict for further processing
                if config.INFERENCE.EVALUATION_WINDOW.USE_SMALLER_WINDOW:
                    if method_name == "dummy" or method_name == "random":
                        predictions_dict[filename + str(i//sampling_rate)] = {"GT_HR": gt_hr, "Pred_HR": pre_hr}
                    else:
                        predictions_dict[filename + str(i // config.UNSUPERVISED.DATA.FS)] = {"GT_HR": gt_hr, "Pred_HR": pre_hr}
                else:
                    predictions_dict[filename ] = {"GT_HR": gt_hr, "Pred_HR": pre_hr}

                #MAE for each segment
                MAE = np.mean(np.abs(gt_hr - pre_hr))

                if config.INFERENCE.EVALUATION_WINDOW.USE_SMALLER_WINDOW:
                    print("FFT MAE for segment {1}: {0} ".format(MAE, filename))
                    if log:
                        logger.log_metrics({filename+"_smaller_window_"+str(i): MAE})
                else:
                    print("FFT MAE for segment {1}: {0} ".format(MAE, filename))
                    if log:
                        logger.log_metrics({filename: MAE})

    if method_name == "dummy":
        print("Running mean predicition")
        mean_hr = np.mean(gt_hr_all)
        for key in predictions_dict.keys():
            predictions_dict[key]["Pred_HR"] = mean_hr
        for i in range(len(predict_hr_all)):
            predict_hr_all[i] = mean_hr
    if method_name == "random":
        print("running random predicitons")
        min, max = np.int(np.min(gt_hr_all)),np.int(np.max(gt_hr_all))
        print(min, max)
        print(len(predict_hr_all), len(predictions_dict.keys()))
        for i, key in enumerate(predictions_dict.keys()):
            sample = random.sample(range(min, max), 1)[0]
            predictions_dict[key]["Pred_HR"] = sample
            predict_hr_all[i] = sample

    if save_outputs:
        save_test_outputs(predictions, labels, config, method_name)

    print("Used Unsupervised Method: " + method_name)

    dataframe = pd.DataFrame.from_dict(predictions_dict).T
    dataframe.to_csv(f"{config.UNSUPERVISED.OUT_SAVE_DIR}/{method_name}_{config.INFERENCE.EVALUATION_METHOD}.csv")
    logger.experiment.log_dataframe_profile(dataframe, "whole-data")
    dataframe['GT_HR'].plot.hist()
    logger.experiment.log_figure(figure=plt, figure_name="GT-Histogram")
    plt.close()

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
