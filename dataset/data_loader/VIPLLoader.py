"""The dataloader for MBP-PPG datasets.

TODO:Update the informaiton link and citations for MBP-PPG
Details for the MBP-PPG Dataset see .
"""
import glob
import os
import pathlib
import re
from multiprocessing import Pool, Process, Value, Array, Manager
#import h5py

import cv2
import numpy as np
from dataset.data_loader.BaseLoader import BaseLoader
from tqdm import tqdm
import random


class VIPLLoader(BaseLoader):
    """The data loader for the MBP-PPG dataset.
       Data structure
       -----------------
             RawData/
             |   |-- subject1/
             |       |-- v1/
             |          |-- source1/
             |              |-- gt_HR.csv
             |              |-- gt_SpO2.csv
             |              |-- time.txt
             |              |-- video.avi
             |              |-- wave.csv
             |          |-- source2/
             |              |-- gt_HR.csv
             |              |-- gt_SpO2.csv
             |              |-- video.avi
             |              |-- wave.csv
             |          |-- source3/
             |              |-- gt_HR.csv
             |              |-- gt_SpO2.csv
             |              |-- time.txt
             |              |-- video.avi
             |              |-- wave.csv
             |          |-- source4/
             |              |-- gt_HR.csv
             |              |-- gt_SpO2.csv
             |              |-- time.txt
             |              |-- video.avi
             |              |-- wave.csv
             |       |-- v2/
             |          ...
             |       |-- v3/
             |          ...
             |       |-- v5/
             |          ...
             |       |-- v6/
             |          ...
             |       |-- v7/
             |          ...
             |       |-- v8/
             |          ...
             |       |-- v9/
             |          ...
             |   |-- subjectn/
             |       |-- v1/
             |...
        -----------------
    """


    def __init__(self, name, data_path,config_data, model, sources=None, scenarios=None ):
        """Initializes an UBFC dataloader.
            Args:
                data_path(str): path of a folder which stores raw video and bvp data.
                e.g. data_path should be "RawData" for below dataset structure:
                name(string): name of the dataloader.
                config_data(CfgNode): data settings(ref:config.py).
                source(list): list of sources to consider
        """
        if sources == None:
            self.sources = ["source2", "source1", "source3"]
        else:
            self.sources = sources

        if scenarios == None:
            self.scenarios = ['v1', 'v2','v3','v4','v5','v6','v7', 'v8', 'v9']
        else:
            self.scenarios = scenarios

        super().__init__(name, data_path, config_data, model)

    def get_raw_data(self, data_path):
        """Returns data directories under the path(For UBFC dataset)."""
        data_path = pathlib.Path(data_path)

        data_dirs = list()
        for dir in data_path.iterdir():
            if dir.is_dir():
                for scenario in self.scenarios:
                    for source in self.sources:
                        dataPath = dir/ scenario /source
                        if dataPath.is_dir():
                            data_dirs.append(dataPath)

        if not data_dirs:
            raise ValueError(self.dataset_name + " data paths empty!")

        print(len(data_dirs))

        dirs = list()
        for data_dir in data_dirs:
            subject = data_dir.parent.parent.stem.strip("p")
            index = subject + data_dir.parent.stem + data_dir.stem
            dirs.append({"index":index, "subject":subject, "path":str(data_dir)})
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
            data_dir = data['path']
            index = data['index']
            # creates a dictionary of data_dirs indexed by subject number
            if subject not in data_info:  # if subject not in the data info dictionary
                data_info[subject] = []  # make an emplty list for that subject
            # append a tuple of the filename, subject num, trial num, and chunk num
            data_info[subject].append({"index": index, "path": data_dir, "subject": subject})

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
        filename = os.path.split(data_dirs[i]['path'])[-1]
        saved_filename = data_dirs[i]['index']

        video_path = str(data_dirs[i]['path'])


        frames = self.read_video(
            os.path.join(video_path, "video.avi"))
        bvps = self.read_wave(
            os.path.join(video_path, "wave.csv"))

        target_length = frames.shape[0]
        bvps = BaseLoader.resample_ppg(bvps, target_length)

        frames_clips, bvps_clips = self.preprocess(frames, bvps, config_preprocess)
        input_name_list, label_name_list = self.save_multi_process(frames_clips, bvps_clips, saved_filename)
        file_list_dict[i] = input_name_list

    @staticmethod
    def read_video(video_file):
        """Reads a video file, returns frames(T, H, W, 3) """
        VidObj = cv2.VideoCapture(video_file)
        VidObj.set(cv2.CAP_PROP_POS_MSEC, 0)
        success, frame = VidObj.read()
        frames = list()
        while success:
            frame = cv2.cvtColor(np.array(frame), cv2.COLOR_BGR2RGB)
            frame = np.asarray(frame)
            frames.append(frame)
            success, frame = VidObj.read()
        return np.asarray(frames)

    @staticmethod
    def read_wave(bvp_file):
        """Reads a bvp signal file."""
        bvp_wave = []
        first = True
        with open(bvp_file) as f:
            for line in f.readlines():
                if first:
                    first=False
                    continue
                bvp_wave.append(np.float64(line.strip()))
        return np.array(bvp_wave)