import math

import cv2
import numpy as np
import torch
from scipy import io as scio
from scipy import linalg
from scipy import signal
from scipy import sparse
from skimage.util import img_as_float
from sklearn.metrics import mean_squared_error
from scipy.signal import resample
import heartpy as hp
import matplotlib.pyplot as plt

def ecg_processing(ecg, sampling_rate, plot_hp=False ):
    """
    This function returns the bpm and rmssd of the ecg data in the provided time frame.

    :param sensor_df: pd.DataFrame, with (at least) columns "ECG" and "time_seconds"
    :param start_time: float, starting time
    :param end_time: float, ending time
    :param sampling_rate: int, sampling rate
    :return: bpm, rmssd
    """
    if torch.is_tensor(ecg):
        ecg = ecg.numpy()
    # upsample if necessary
    if sampling_rate == 300:
        ecg = resample(ecg, len(ecg) * 4)
        sampling_rate = sampling_rate * 4
    # also convert signal from 1000Hz to numpy array, so both have same type for future calculations

    try:
        wd, m = hp.process(hp.scale_data(ecg), sampling_rate)
    except ValueError:
        print(ecg, sampling_rate)
    if plot_hp:
        hp.plotter(wd, m)
        plt.show()
    rmssd = m["rmssd"]
    bpm = m["bpm"]
    if bpm > 200 or bpm < 40:
        print(bpm)
        print("BPM out of range")
    return bpm


def detrend(input_signal, lambda_value):
    signal_length = input_signal.shape[0]
    # observation matrix
    H = np.identity(signal_length)
    ones = np.ones(signal_length)
    minus_twos = -2 * np.ones(signal_length)
    diags_data = np.array([ones, minus_twos, ones])
    diags_index = np.array([0, 1, 2])
    D = sparse.spdiags(diags_data, diags_index,
                (signal_length - 2), signal_length).toarray()
    filtered_signal = np.dot(
        (H - np.linalg.inv(H + (lambda_value ** 2) * np.dot(D.T, D))), input_signal)
    return filtered_signal


def process_video(frames):
    RGB = []
    for frame in frames:
        summation = np.sum(np.sum(frame, axis=0), axis=0)
        RGB.append(summation / (frame.shape[0] * frame.shape[1]))
    RGB = np.asarray(RGB)
    RGB = RGB.transpose(1, 0).reshape(1, 3, -1)
    return np.asarray(RGB)
