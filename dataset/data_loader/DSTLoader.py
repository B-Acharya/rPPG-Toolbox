"""The dataloader for MBP-PPG datasets.

TODO:Update the informaiton link and citations for MBP-PPG
Details for the MBP-PPG Dataset see .
"""
import glob
import os
import pathlib
import re
from multiprocessing import Pool, Process, Value, Array, Manager
import h5py
import random
import pickle
import cv2
import numpy as np
from dataset.data_loader.BaseLoader import BaseLoader
from tqdm import tqdm


class DSTLoader(BaseLoader):
    """The data loader for the DST videos in the DST versus TSST study. All necessary information are stored in a
       pickle file in /data/dst_tsst_22_bi_multi_nt_lab/processed/sensor_files.pkl. It contains a tuple with the following structure:
       ---
       Pickle File Structure
       ---
        sensor_files[paradigm + '_' + token] = (timer1_marker,
                                                csv_marker_0,
                                                csv_marker_1,
                                                back_from_study,
                                                sensor_txt, # Path to ECG txt file
                                                sensor_df, #  DataFrame of sensor signals
                                                sampling_rate,
                                                dst_files = {"dst_meta": dst_meta,
                                                             "dst_speech_video": dst_speech_video,
                                                             "dst_math_video": dst_math_video,
                                                             "dst_math_task_results": dst_math_task_results,
                                                             "dst_speech_task_audio": dst_speech_task_audio},
                                                speech_task_start_time_sensor,
                                                speech_task_end_time_sensor,
                                                math_task_start_time_sensor,
                                                math_task_end_time_sensor)
       ---
       Data structure
       -----------------
             raw/mainstudy
             |   |TOKEN1[DSTTSST]/
             |       |-- TOKEN1_tsst_video.MOV
             |       |-- TOKEN1_tsst_sensors_converted_*.txt
             |       |-- TOKEN1_tsst_sensors_*.txt
             |       |-- TOKEN1_tsst_sensors_*.h5
             |       |-- TOKEN1_tsst_sensors_EventsAnnotation_*.txt
             |       |-- TOKEN1_dst_sensors_converted_*.txt
             |       |-- TOKEN1_dst_sensors_*.txt
             |       |-- TOKEN1_dst_sensors_*.h5
             |       |-- TOKEN1_dst_sensors_EventsAnnotation_*.txt
             |       |-- study-result_[DST-ID]/
             |          |-- comp-result_[DST-ID]/
             |              |-- [DST-ID]_speechTask_*.webm
             |              |-- [DST-ID]_mathTask_*.webm
             |              |-- [DST-ID]_introduction_*.webm
             |              |-- [DST-ID]_mathTask.json
             |              |-- [DST-ID]_speechTask.json
             |              |-- [DST-ID]_metaParticipantData.json
             |...
             |   |TOKENn[DSTTSST]/
             |...
        -----------------
    """


    def __init__(self, name, data_path, config_data, model):
        """Initializes a DST dataloader.
            Args:
                data_path(str): path of a folder which stores raw video and bvp data.
                e.g. data_path should be "RawData" for below dataset structure:
                name(string): name of the dataloader.
                config_data(CfgNode): data settings(ref:config.py).
        """
        super().__init__(name, data_path, config_data, model)

    def get_raw_data(self, data_path):
        """Returns data directories under the path(For CMBP dataset)."""
        print("Files loading from pickle in: ", data_path, "...")
        with open(data_path, 'rb') as f:
            sensor_files = pickle.load(f)
        print("Files loaded from Pickle.")
        filtered_tuple = {key: value for key, value in sensor_files.items() if 'dst' in key}
        print(len(filtered_tuple), "DST Files in Pickle-File")
        dirs = list()
        for token, (timer1_marker, csv_marker_0, csv_marker_1, back_from_study,
                    sensor_txt, sensor_df, sampling_rate, dst_files,
                    speech_task_start_time_sensor, speech_task_end_time_sensor, math_task_start_time_sensor,
                    math_task_end_time_sensor) in tqdm.tqdm(filtered_tuple.items()):
            subject = str(token[-8:])
            for dst_key in ["dst_speech_video", "dst_math_video"]:
                data_dir = dst_files[dst_key]
                task = subject + "_" + dst_key[4:-6]
                if "speech" in task:
                    start_time = speech_task_start_time_sensor
                    end_time = speech_task_end_time_sensor
                else:
                    start_time = math_task_start_time_sensor
                    end_time = math_task_end_time_sensor
                ecg_df = sensor_df[(sensor_df["time_seconds"] >= start_time) & (sensor_df["time_seconds"] <= end_time)]
                dirs.append({"task":task, "subject":subject, "video_path": str(data_dir), "ecg_df": ecg_df})

        return dirs

    def split_raw_data(self, data_dirs, begin, end):
        """Returns a subset of data dirs, split with begin and end values."""
        # return the full directory
        if begin == 0 and end == 1:
            return data_dirs

        # get info about the dataset: subject list and num vids per subject
        data_info = dict()
        for data in data_dirs:
            subject = data['subject']
            data_dir = data['video_path']
            task = data['task']
            ecg_df = data["ecg_df"]
            # creates a dictionary of data_dirs indexed by subject number
            if subject not in data_info:  # if subject not in the data info dictionary
                data_info[subject] = []  # make an empty list for that subject
            # append a tuple of the filename, subject num, trial num, and chunk num
            data_info[subject].append({"task": task, "video_path": data_dir, "subject": subject, "ecg_df": ecg_df})

        subj_list = list(data_info.keys())  # all subjects by number ID (1-27)
        subj_list = sorted(subj_list)
        print("Before Shuffle:", subj_list)
        if self.shuffle:
            random.Random(4).shuffle(subj_list)
            print("After Shuffle:", subj_list)
        else:
            print("No Shuffle")
        num_subjs = len(subj_list)  # number of unique subjects

        # get split of data set (depending on start / end)
        subj_range = list(range(0, num_subjs))
        if begin != 0 or end != 1:
            subj_range = list(range(int(begin * num_subjs), int(end * num_subjs)))

        # compile file list
        data_dirs_new = []
        for i in subj_range:
            subj_num = subj_list[i]
            subj_files = data_info[subj_num]
            data_dirs_new += subj_files  # add file information to file_list (tuple of fname, subj ID, trial num,
            # chunk num)

        return data_dirs_new

    def preprocess_dataset_subprocess(self, data_dirs, config_preprocess, i, file_list_dict):
        """ invoked by preprocess_dataset for multi_process."""
        saved_filename = data_dirs[i]['task']
        video_path = data_dirs[i]['video_path']
        ecg_df = data_dirs[i]['ecg_df']
        print("os path exists", os.path.exists(video_path))
        frames = self.read_video(video_path)
        ecgs = self.read_wave(ecg_df)
        print("frame shape", frames.shape)
        print("bvps shape", ecgs.shape)
        target_length = frames.shape[0]
        ecgs = BaseLoader.resample_ppg(ecgs, target_length)

        frames_clips, bvps_clips = self.preprocess(frames, ecgs, config_preprocess)
        print("saving", saved_filename)
        input_name_list, label_name_list = self.save_multi_process(frames_clips, bvps_clips, saved_filename)
        file_list_dict[i] = input_name_list

    @staticmethod
    def read_video(video_file):
        """Reads a video file, returns frames(T, H, W, 3) """
        print("enter read")
        VidObj = cv2.VideoCapture(video_file)
        VidObj.set(cv2.CAP_PROP_POS_MSEC, 0)
        success, frame = VidObj.read()
        frames = list()
        while success:
            frame = cv2.cvtColor(np.array(frame), cv2.COLOR_BGR2RGB)
            frame = np.asarray(frame)
            frames.append(frame)
            success, frame = VidObj.read()
        print("exit read")
        return np.asarray(frames)

    @staticmethod
    def read_wave(ecg_df):
        """Reads the ecg_df and returns the ECG signal."""
        return ecg_df["ECG"]