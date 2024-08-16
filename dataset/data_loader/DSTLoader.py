"""The dataloader for MBP-PPG datasets.

TODO:Update the informaiton link and citations for MBP-PPG
Details for the MBP-PPG Dataset see .
"""
import glob
import os
import pathlib
import re
from multiprocessing import Pool, Process, Value, Array, Manager
import random
import json
import cv2
from dataset.data_loader.BaseLoader import BaseLoader
from tqdm import tqdm
from ast import literal_eval
import pandas as pd
import numpy as np
import subprocess
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
                                                sampling_rate,
                                                dst_files = {"dst_meta": dst_meta,
                                                             "dst_speech_video": dst_speech_video,
                                                             "dst_math_video": dst_math_video,
                                                             "dst_math_task_results": dst_math_task_results,
                                                             "dst_speech_task_audio": dst_speech_task_audio},
                                                speech_task_start_time_sensor,
                                                speech_task_end_time_sensor,
                                                math_task_start_time_sensor,

                                                math_task_end_time_sensor,
                                                bpm_rmssd_dict = {"speech_bpm": - float,
                                                                  "speech_rmssd": - float,
                                                                  "math_bpm": - float,
                                                                  "math_rmssd - float})
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
        """Returns data directories under the path (for DST dataset)."""
        print("Files loading from json in: ", data_path, "...")
        with open(data_path, 'rb') as f:
            sensor_files = json.load(f)
        print("Files loaded from json.")
        filtered_tuple = {key: value for key, value in sensor_files.items() if 'dst' in key}
        print(len(filtered_tuple), "DSTs with each speech and math in Json-File")
        dirs = list()
        for token, (timer1_marker, csv_marker_0, csv_marker_1, back_from_study,
                    sensor_txt, sampling_rate, dst_files,
                    speech_task_start_time_sensor, speech_task_end_time_sensor, math_task_start_time_sensor,
                    math_task_end_time_sensor, bpm_rmssd_dict) in tqdm(filtered_tuple.items()):
            subject = str(token[-8:])
            for dst_key in ["dst_speech_video", "dst_math_video"]:
                data_dir = dst_files[dst_key][0]
                task = subject + "_" + dst_key[4:-6]
                if "speech" in task:
                    start_time = speech_task_start_time_sensor
                    end_time = speech_task_end_time_sensor
                    hr = bpm_rmssd_dict['speech_bpm']
                else:
                    start_time = math_task_start_time_sensor
                    end_time = math_task_end_time_sensor
                    hr = bpm_rmssd_dict['math_bpm']
                dirs.append({"task":task, "subject":subject, "video_path": str(data_dir), "sensor_txt": sensor_txt, "start_time": start_time, "end_time": end_time, "hr": hr})

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
            sensor_txt = data['sensor_txt']
            start_time = data['start_time']
            end_time = data['end_time']
            hr = data['hr']

            # creates a dictionary of data_dirs indexed by subject number
            if subject not in data_info:  # if subject not in the data info dictionary
                data_info[subject] = []  # make an empty list for that subject
            # append a tuple of the filename, subject num, trial num, and chunk num
            data_info[subject].append({"task":task, "subject":subject, "video_path": data_dir, "sensor_txt": sensor_txt, "start_time": start_time, "end_time": end_time, 'hr': hr })

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
            data_dirs_new += subj_files  #add file information to file_list (tuple of fname, subj ID, trial num,
            # chunk num)

        return data_dirs_new

    def preprocess_dataset_subprocess(self, data_dirs, config_preprocess, i, file_list_dict):
        """ invoked by preprocess_dataset for multi_process."""
        saved_filename = data_dirs[i]['task']
        video_path = data_dirs[i]['video_path']
        print("video_path: ", video_path)
        # Extract the path components
        path_components = video_path.split(os.sep)
        # Get the first two levels of directories as part of the initial directory
        initial_directory = os.path.join(*path_components[:3])
        # Define the path for the converted videos
        mp4_directory =  "/" + os.path.join(initial_directory, 'processed', 'converted_videos')
        original_filename = os.path.splitext(path_components[-1])[0]
        video_path_converted = os.path.join(mp4_directory, f'{original_filename}.mp4')
        print("video_path_converted", video_path_converted)
        sensor_txt = data_dirs[i]['sensor_txt']
        start_time = data_dirs[i]['start_time']
        end_time = data_dirs[i]['end_time']
        hr = data_dirs[i]['hr']
        sensor_df = self.sensor_txt_to_df(sensor_txt)
        ecg_df = sensor_df[(sensor_df["time_seconds"] >= start_time) & (sensor_df["time_seconds"] <= end_time)]
        print("os path exists converted", os.path.exists(video_path_converted))
        print("os path exists", os.path.exists(video_path))
        frames = self.read_video(video_path_converted)
        ecgs = self.read_wave(ecg_df)

        print("frame shape", frames.shape)
        print("bvps shape", ecgs.shape)
        target_length = frames.shape[0]

        # ecgs = BaseLoader.resample_ppg(ecgs, target_length)
        frames_clips, bvps_clips = self.preprocess(frames, ecgs, config_preprocess)

        #overide ecg to HR
        input_name_list, label_name_list = self.save_multi_process(frames_clips, ecgs, saved_filename)
        file_list_dict[i] = input_name_list

    @staticmethod
    def read_video(video_file):
        """Reads a video file, returns frames(T, H, W, 3)."""
        print("enter read")
        print(video_file)
        try:
            cap = cv2.VideoCapture(video_file)
        except:
            print("cap not working.")
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        cap.release()
        print("exit read")
        return np.asarray(frames)

    @staticmethod
    def read_wave(ecg_df):
        """Reads the ecg_df and returns the ECG signal."""
        return ecg_df["ECG"]

    @staticmethod
    def sensor_txt_to_df(filepath):
        """
        This function generates a sensor DataFrame from the sensor txt file, with columns "t" for the timestep (time * sampling rate) (careful as float) and sensor_names based on the sensors in the text file. For our study those will always be "ECG", "RESPIRATION" and "BVP". Also added column "time_seconds".

        :param filepath: path to the sensor.txt file
        :return: the sensor pd.DataFrame
        """
        data = []
        # Read the file and parse it
        with open(filepath, 'r') as file:
            lines = file.readlines()
            read_data = False
            metadata = lines[1]
            metadata_aux = metadata[2:-1]
            header_dict = literal_eval(metadata_aux)
            for device in header_dict:
                column_names = ["t", "DI"] + (header_dict[device]["sensor"])
                sampling_rate = header_dict[device]["sampling rate"]
            for line in lines[3:]:
                row_data = line.strip().split("\t")
                # print(row_data)
                data.append(row_data)
            sensor_df = pd.DataFrame(data, columns=column_names)
            sensor_df = sensor_df.astype(float)
            sensor_df = sensor_df.drop(columns="DI")
            sensor_df["t"] = (sensor_df["t"]).astype("int")
            sensor_df["time_seconds"] = sensor_df["t"] / sampling_rate
            return sensor_df