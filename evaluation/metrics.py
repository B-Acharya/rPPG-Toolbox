import numpy as np
import pandas as pd
import torch
from evaluation.post_process import *
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle
import os
from evaluation.BlandAltmanPy import BlandAltman

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


def _reform_data_from_dict(data):
    """Helper func for calculate metrics: reformat predictions and labels from dicts. """
    sort_data = sorted(data.items(), key=lambda x: x[0])
    sort_data = [i[1] for i in sort_data]
    sort_data = torch.cat(sort_data, dim=0)
    return np.reshape(sort_data.cpu(), (-1))


def calculate_metrics(predictions, labels, config, logger, save_outputs=True):
    """Calculate rPPG Metrics (MAE, RMSE, MAPE, Pearson Coef.)."""
    predict_hr_fft_all = list()
    gt_hr_fft_all = list()
    predict_hr_peak_all = list()
    gt_hr_peak_all = list()
    SNR_all = list()
    gt_hr_fft_dict = dict()
    pred_hr_fft_dict = dict()
    predictions_dict = dict()
    for index in tqdm(predictions.keys(), ncols=80):
        prediction = _reform_data_from_dict(predictions[index])
        label = _reform_data_from_dict(labels[index])

        video_frame_size = prediction.shape[0]
        print("Video frame size", video_frame_size, index)
        if config.INFERENCE.EVALUATION_WINDOW.USE_SMALLER_WINDOW:
            window_frame_size = config.INFERENCE.EVALUATION_WINDOW.WINDOW_SIZE * config.TEST.DATA.FS
            if window_frame_size > video_frame_size:
                window_frame_size = video_frame_size
        else:
            window_frame_size = video_frame_size

        for i in range(0, len(prediction), window_frame_size):
            pred_window = prediction[i:i+window_frame_size]
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
                gt_hr_peak, pred_hr_peak, SNR = calculate_metric_per_video(
                    prediction, label, diff_flag=diff_flag_test, fs=config.TEST.DATA.FS, hr_method='Peak')
                gt_hr_peak_all.append(gt_hr_peak)
                predict_hr_peak_all.append(pred_hr_peak)
                SNR_all.append(SNR)
                predictions_dict[index] = {"GT_HR": gt_hr_peak, "Pred_HR": pred_hr_peak}
            elif config.INFERENCE.EVALUATION_METHOD == "FFT":
                gt_hr_fft, pred_hr_fft, SNR = calculate_metric_per_video(
                    prediction, label, diff_flag=diff_flag_test, fs=config.TEST.DATA.FS, hr_method='FFT')
                gt_hr_fft_all.append(gt_hr_fft)
                predict_hr_fft_all.append(pred_hr_fft)
                SNR_all.append(SNR)
                predictions_dict[index] = {"GT_HR": gt_hr_fft, "Pred_HR": pred_hr_fft}
            elif config.INFERENCE.EVALUATION_METHOD == "Welch":
                gt_hr_fft, pred_hr_fft, SNR = calculate_metric_per_video(
                    prediction, label, high_pass=3.0, diff_flag=diff_flag_test, fs=config.TEST.DATA.FS, hr_method='FFT')
                gt_hr_fft_all.append(gt_hr_fft)
                predict_hr_fft_all.append(pred_hr_fft)
                SNR_all.append(SNR)
                predictions_dict[index] = {"GT_HR": gt_hr_fft, "Pred_HR": pred_hr_fft}
            elif config.INFERENCE.EVALUATION_METHOD == "Welch_fft":
                gt_hr_fft, pred_hr_fft, SNR = calculate_metric_per_video(
                    prediction, label, high_pass=3.0, diff_flag=diff_flag_test, fs=config.TEST.DATA.FS, hr_method='Welch')
                gt_hr_fft_all.append(gt_hr_fft)
                predict_hr_fft_all.append(pred_hr_fft)
                SNR_all.append(SNR)
                predictions_dict[index] = {"GT_HR": gt_hr_fft, "Pred_HR": pred_hr_fft}
            else:
                raise ValueError("Inference evaluation method name wrong!")

            for metric in config.TEST.METRICS:
                if metric == "MAE":
                    MAE = np.mean(np.abs(gt_hr_fft - pred_hr_fft))
                    print("FFT MAE (FFT Label): {0} ".format(MAE))
                    logger.log_metrics({index:MAE} )
    log_HR_HR_plot(logger, predictions_dict)
    plot_bland(logger, predictions_dict)
    filename_id = config.TRAIN.MODEL_FILE_NAME

    if save_outputs:
        save_test_outputs(predictions, labels, config, method_name)

    if config.INFERENCE.EVALUATION_METHOD == "FFT" or config.INFERENCE.EVALUATION_METHOD == "Welch" or config.INFERENCE.EVALUATION_METHOD == "Welch_fft":
        gt_hr_fft_all = np.array(gt_hr_fft_all)
        predict_hr_fft_all = np.array(predict_hr_fft_all)
        SNR_all = np.array(SNR_all)
        num_test_samples = len(predict_hr_fft_all)
        for metric in config.TEST.METRICS:
            if metric == "MAE":
                MAE_FFT = np.mean(np.abs(predict_hr_fft_all - gt_hr_fft_all))
                standard_error = np.std(np.abs(predict_hr_fft_all - gt_hr_fft_all)) / np.sqrt(num_test_samples)
                print("FFT MAE (FFT Label): {0} +/- {1}".format(MAE_FFT, standard_error))
                logger.log_metrics({"FFT MAE": MAE_FFT, "FFT MAE std":standard_error})
            elif metric == "RMSE":
                RMSE_FFT = np.sqrt(np.mean(np.square(predict_hr_fft_all - gt_hr_fft_all)))
                standard_error = np.std(np.square(predict_hr_fft_all - gt_hr_fft_all)) / np.sqrt(num_test_samples)
                print("FFT RMSE (FFT Label): {0} +/- {1}".format(RMSE_FFT, standard_error))
                logger.log_metrics({"FFT RMSE": RMSE_FFT, "FFT RMSE std":standard_error})
            elif metric == "MAPE":
                MAPE_FFT = np.mean(np.abs((predict_hr_fft_all - gt_hr_fft_all) / gt_hr_fft_all)) * 100
                standard_error = np.std(np.abs((predict_hr_fft_all - gt_hr_fft_all) / gt_hr_fft_all)) / np.sqrt(num_test_samples) * 100
                print("FFT MAPE (FFT Label): {0} +/- {1}".format(MAPE_FFT, standard_error))
                logger.log_metrics({"FFT MAPE": MAPE_FFT, "FFT MAPE std":standard_error})
            elif metric == "Pearson":
                Pearson_FFT = np.corrcoef(predict_hr_fft_all, gt_hr_fft_all)
                correlation_coefficient = Pearson_FFT[0][1]
                standard_error = np.sqrt((1 - correlation_coefficient**2) / (num_test_samples - 2))
                print("FFT Pearson (FFT Label): {0} +/- {1}".format(correlation_coefficient, standard_error))
                logger.log_metrics({"FFT Pearson ": correlation_coefficient, "FFT Pearson std": standard_error})

            elif metric == "SNR":
                SNR_FFT = np.mean(SNR_all)
                standard_error = np.std(SNR_all) / np.sqrt(num_test_samples)
                print("FFT SNR (FFT Label): {0} +/- {1}".format(SNR_FFT, standard_error))
                logger.log_metrics({"FFT SNR_FFT": SNR_FFT, "FFT SNR std": standard_error})
            else:
                raise ValueError("Wrong Test Metric Type")

        compare = BlandAltman(gt_hr_fft_all, predict_hr_fft_all, config, logger=logger, averaged=True)
        try:
            compare.scatter_plot(
                x_label='GT PPG HR [bpm]',
                y_label='rPPG HR [bpm]',
                show_legend=True, figure_size=(5, 5),
                the_title=f'{filename_id}_FFT_BlandAltman_ScatterPlot',
                file_name=f'{filename_id}_FFT_BlandAltman_ScatterPlot.pdf')
        except:
            print("Plot not available")
        try:
            compare.difference_plot(
                x_label='Difference between rPPG HR and GT PPG HR [bpm]',
                y_label='Average of rPPG HR and GT PPG HR [bpm]',
                show_legend=True, figure_size=(5, 5),
                the_title=f'{filename_id}_FFT_BlandAltman_DifferencePlot',
                file_name=f'{filename_id}_FFT_BlandAltman_DifferencePlot.pdf')
        except:
            print("Difference plot not available")

    elif config.INFERENCE.EVALUATION_METHOD == "peak detection":
        gt_hr_peak_all = np.array(gt_hr_peak_all)
        predict_hr_peak_all = np.array(predict_hr_peak_all)
        SNR_all = np.array(SNR_all)
        num_test_samples = len(predict_hr_peak_all)
        for metric in config.TEST.METRICS:
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
                SNR_PEAK = np.mean(SNR_all)
                standard_error = np.std(SNR_all) / np.sqrt(num_test_samples)
                print("FFT SNR (FFT Label): {0} +/- {1}".format(SNR_PEAK, standard_error))
            else:
                raise ValueError("Wrong Test Metric Type")
    else:
        raise ValueError("Inference evaluation method name wrong!")

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