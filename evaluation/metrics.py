import numpy as np
import pandas as pd
import torch
from evaluation.post_process import *
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle
import os
from evaluation.BlandAltmanPy import BlandAltman
from unsupervised_methods.utils import ecg_processing


def metrics_calculations(ground_truth, predictions, SNR, config, logger ):

    ground_truth = np.array(ground_truth)
    predictions = np.array(predictions)
    SNR = np.array(SNR)
    num_test_samples = len(predictions)
    method = config.INFERENCE.EVALUATION_METHOD
    filename_id = config.TRAIN.MODEL_FILE_NAME

    if config.TOOLBOX_MODE == "unsupervised_method":
        metrics = config.UNSUPERVISED.METRICS
    else:
        metrics = config.TEST.METRICS

    for metric in metrics:
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
    if config.TOOLBOX_MODE == 'train_and_test' or config.TOOLBOX_MODE == 'only_test' or config.TOOLBOX_MODE == "train_and_test_enrich":
        output_dir = config.TEST.OUT_SAVE_DIR
    else:
        output_dir = config.UNSUPERVISED.DATA.CACHED_PATH

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # Filename ID to be used in any output files that get saved
    if config.TOOLBOX_MODE == 'train_and_test' or config.TOOLBOX_MODE == "LOO" or config.TOOLBOX_MODE == "ENRICH" or config.TOOLBOX_MODE == "train_and_test_enrich":
        filename_id = config.TRAIN.MODEL_FILE_NAME
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
    return output_dir

def calcualte_mae_per_setting(dataframe):
    mae = np.mean(np.abs(dataframe["HR_GT"] - dataframe["HR_Pred"]))
    return mae

def read_label(dataset):
    """Read manually corrected labels."""
    df = pd.read_csv("label/{0}_Comparison.csv".format(dataset))
    out_dict = df.to_dict(orient='index')
    out_dict = {str(value['VideoID']): value for key, value in out_dict.items()}
    return out_dict

def log_HR_HR_plot(logger, predictions):

    HR_gt = []
    HR_pred = []
    print("keys=====")
    print(predictions.keys())
    for key in predictions.keys():
        HR_gt.append(predictions[key]['GT_HR'])
        HR_pred.append(predictions[key]['Pred_HR'])

    # min_HR = min(HR_pred)
    max_HR = max(HR_pred)
    plt.scatter(HR_pred, HR_gt)
    plt.plot([0, max_HR], [0, max_HR], "r--")
    plt.xlabel("HR from Video")
    plt.ylabel("HR ground Truth")
    logger.experiment.log_figure(figure=plt, figure_name="GT HR vs Pred HR ")
    plt.close()

def plot_bland(logger, predictions):
    HR_gt = []
    HR_pred = []
    print("keys=====")
    print(predictions.keys())
    for key in predictions.keys():
        HR_gt.append(predictions[key]['GT_HR'])
        HR_pred.append(predictions[key]['Pred_HR'])

    hr = np.concatenate((np.array(HR_gt).reshape(-1, 1), np.array(HR_pred).reshape(-1, 1)), axis=1)
    averages_NR = np.mean(hr, axis=1)
    diff_NR = np.diff(hr, axis=1)
    diff_NR_mean, diff_NR_std = np.mean(diff_NR), np.std(diff_NR)
    upper_limit_NR = diff_NR_mean + 1.96 * diff_NR_std
    lower_limit_NR = diff_NR_mean - 1.96 * diff_NR_std

    x_value = np.max(averages_NR)

    plt.scatter(averages_NR, diff_NR)
    plt.hlines(upper_limit_NR, min(averages_NR), max(averages_NR), colors="red", linestyle="dashed", label="+1.96SD")
    plt.hlines(lower_limit_NR, min(averages_NR), max(averages_NR), colors="red", linestyle="dashed", label="-1.96SD")
    plt.hlines(diff_NR_mean, min(averages_NR), max(averages_NR), colors="Blue", linestyle="solid", label="Mean")
    plt.text(x_value, upper_limit_NR + 1, "+1.96SD")
    plt.text(x_value, upper_limit_NR - 1, f"{upper_limit_NR:.2f}")
    plt.text(x_value, lower_limit_NR + 1, "+1.96SD")
    plt.text(x_value, lower_limit_NR - 1, f"{lower_limit_NR:.2f}")
    plt.text(x_value, diff_NR_mean + 1, "Mean")
    plt.text(x_value, diff_NR_mean - 1, f"{diff_NR_mean:.2f}")
    plt.xlabel("Average of the estimated HR and Ground truth")
    plt.ylabel("Difference between estimated HR and ground truth HR")
    logger.experiment.log_figure(figure=plt, figure_name="Bland-Altman-Plot")
    plt.close()

def read_hr_label(feed_dict, index):
    """Read manually corrected UBFC labels."""
    # For UBFC only
    if index[:7] == 'subject':
        index = index[7:]
    video_dict = feed_dict[index]
    if video_dict['Preferred'] == 'Peak Detection':
        hr = video_dict['Peak Detection']
    elif video_dict['Preferred'] == 'FFT':
        hr = video_dict['FFT']
    else:
        hr = video_dict['Peak Detection']
    return index, hr


def _reform_data_from_dict(data, flatten=True):
    """Helper func for calculate metrics: reformat predictions and labels from dicts. """
    sort_data = sorted(data.items(), key=lambda x: x[0])
    sort_data = [i[1] for i in sort_data]
    sort_data = torch.cat(sort_data, dim=0)

    if flatten:
        sort_data = np.reshape(sort_data.cpu(), (-1))
    else:
        sort_data = np.array(sort_data.cpu())

    return sort_data

def calculate_metrics(predictions, labels, config, logger, mean_HR = 70, save_outputs=True):
    """Calculate rPPG Metrics (MAE, RMSE, MAPE, Pearson Coef.)."""
    pred_hr_all = list()
    gt_hr_all = list()
    predict_hr__all = list()
    gt_hr_peak_all = list()
    SNR_all = list()
    predictions_dict = dict()

    if config.TEST.DATA.DATASET == "DST":
        DST = True
    else:
        DST = False

    for index in tqdm(predictions.keys(), ncols=80):
        prediction = _reform_data_from_dict(predictions[index])
        label = _reform_data_from_dict(labels[index])

        if len(label) == 0:
            print("--------------")
            print("--------------")
            print("--------------")
            print("--------------")
            print("--------------")
            print("ECG data missing, please check")
            print("--------------")
            print("--------------")
            print("--------------")
            print("--------------")
            print("--------------")
            continue
        if DST:
            # TODO: should be changed and moved to dataloader, bit hacky
            label_frame_size = len(label)
            if label_frame_size > 25000 and label_frame_size < 30000:
                sampling_rate = 300
            else:
                sampling_rate = 1000

        video_frame_size = prediction.shape[0]
        print("Video frame size", video_frame_size, index)
        if config.INFERENCE.EVALUATION_WINDOW.USE_SMALLER_WINDOW:
            window_frame_size = config.INFERENCE.EVALUATION_WINDOW.WINDOW_SIZE * config.TEST.DATA.FS
            if window_frame_size > video_frame_size:
                window_frame_size = video_frame_size
            if DST:
                label_frame_size = config.INFERENCE.EVALUATION_WINDOW.WINDOW_SIZE * sampling_rate
        else:
            window_frame_size = video_frame_size

        for i in range(0, len(prediction), window_frame_size):
            pred_window = prediction[i:i+window_frame_size]
            if DST:
                j = (i // config.TEST.DATA.FS) * sampling_rate
                label_window = label[j:j + label_frame_size]
            else:
                label_window = label[i:i+window_frame_size]

            if len(pred_window) < 9:
                print(f"Window frame size of {len(pred_window)} is smaller than minimum pad length of 9. Window ignored!")
                continue

            if config.TEST.DATA.PREPROCESS.LABEL_TYPE == "Standardized" or \
                    config.TEST.DATA.PREPROCESS.LABEL_TYPE == "Raw":
                diff_flag_test = False
            elif config.TEST.DATA.PREPROCESS.LABEL_TYPE == "DiffNormalized":
                diff_flag_test = True
            else:
                raise ValueError("Unsupported label type in testing!")
            
            if config.INFERENCE.EVALUATION_METHOD == "peak detection":
                if DST:
                    pre_hr = calculate_HR(pred_window, fs=config.TEST.DATA.FS, diff_flag=diff_flag_test, hr_method='Peak')
                    gt_hr = ecg_processing(label_window, sampling_rate=sampling_rate)
                else:
                    gt_hr, pre_hr, SNR = calculate_metric_per_video(
                        pred_window, label_window, diff_flag=diff_flag_test, fs=config.TEST.DATA.FS, hr_method='Peak')
            elif config.INFERENCE.EVALUATION_METHOD == "FFT":
                if DST:
                    pre_hr = calculate_HR(pred_window, fs=config.TEST.DATA.FS, diff_flag=diff_flag_test, hr_method='FFT')
                    gt_hr = ecg_processing(label_window, sampling_rate=sampling_rate)
                else:
                    gt_hr, pre_hr, SNR = calculate_metric_per_video(
                        pred_window, label_window, diff_flag=diff_flag_test, fs=config.TEST.DATA.FS, hr_method='FFT')
            elif config.INFERENCE.EVALUATION_METHOD == "Welch":
                if DST:
                    pre_hr = calculate_HR(pred_window, fs=config.TEST.DATA.FS, diff_flag=diff_flag_test,
                                          hr_method='Welch')
                    gt_hr = ecg_processing(label_window, sampling_rate=sampling_rate)
                else:
                    gt_hr, pre_hr, SNR = calculate_metric_per_video(
                        pred_window, label_window, high_pass=3.0, diff_flag=diff_flag_test, fs=config.TEST.DATA.FS, hr_method='Welch')
            else:
                raise ValueError("Inference evaluation method name wrong!")

            gt_hr_all.append(gt_hr)
            pred_hr_all.append(pre_hr)
            if not DST:
                SNR_all.append(SNR)

            predictions_dict[str(i)+"_"+index] = {"GT_HR":gt_hr, "Pred_HR":pre_hr}
    print(gt_hr_all)
    print(pred_hr_all)
    print(predictions_dict)

    if config.MODEL.NAME == "Mean":
        print("Running mean predicition")
        for key in predictions_dict.keys():
            predictions_dict[key]["Pred_HR"] = mean_HR
        for i in range(len(pred_hr_all)):
            pred_hr_all[i] = mean_HR

    log_HR_HR_plot(logger, predictions_dict)
    plot_bland(logger, predictions_dict)
    filename_id = config.TRAIN.MODEL_FILE_NAME


    dataframe = pd.DataFrame.from_dict(predictions_dict).T
    if save_outputs:
        out_dir = save_test_outputs(predictions, labels, config, filename_id)
        dataframe.to_csv(f"{out_dir}/{config.MODEL.NAME}_{config.TRAIN.MODEL_FILE_NAME}_{config.INFERENCE.EVALUATION_METHOD}.csv")

    logger.experiment.log_dataframe_profile(dataframe, "whole-data")

    metrics_calculations(gt_hr_all, pred_hr_all, SNR_all, config, logger)

    if config.TEST.DATA.DATASET == "CMBP":
        result = {'LowHR_Bright': {}, 'LowHR_Dark': {}, 'HighHR_Dark': {}, 'HighHR_Bright': {}}
        for index in predictions_dict.keys():
            print(index)
            HR_GT, HR_pred = predictions_dict[index]["GT_HR"], predictions_dict[index]["Pred_HR"]

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
            mae = calcualte_mae_per_setting(dataframe)
            print(f"--{key}--")
            print(mae)
            logger.log_metrics({key: mae})