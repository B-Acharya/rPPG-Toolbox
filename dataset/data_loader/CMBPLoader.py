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

import cv2
import numpy as np
from dataset.data_loader.BaseLoader import BaseLoader
from tqdm import tqdm


class CMBPLoader(BaseLoader):
    """The data loader for the MBP-PPG dataset.
       Data structure
       -----------------
             RawData/
             |   |-- subject1/
             |       |-- 1/
             |          |-- *.MOV
             |          |-- data.hdf5
             |       |-- 2/
             |          |-- *.MOV
             |          |-- data.hdf5
             |       |-- 3/
             |          |-- *.MOV
             |          |-- data.hdf5
             |       |-- 4/
             |          |-- *.MOV
             |          |-- data.hdf5
             |   |-- subject2/
             |       |-- 1/
             |          |-- *.MOV
             |          |-- data.hdf5
             |       |-- 2/
             |          |-- *.MOV
             |          |-- data.hdf5
             |       |-- 3/
             |          |-- *.MOV
             |          |-- data.hdf5
             |       |-- 4/
             |          |-- *.MOV
             |          |-- data.hdf5
             |...
             |   |-- subjectn/
             |       |-- 1/
             |          |-- *.MOV
             |          |-- data.hdf5
             |...
        -----------------
    """


    def __init__(self, name, data_path, config_data, model):
        """Initializes an UBFC dataloader.
            Args:
                data_path(str): path of a folder which stores raw video and bvp data.
                e.g. data_path should be "RawData" for below dataset structure:
                name(string): name of the dataloader.
                config_data(CfgNode): data settings(ref:config.py).
        """
        super().__init__(name, data_path, config_data, model)

    def get_raw_data(self, data_path):
        """Returns data directories under the path(For UBFC dataset)."""
        data_path = pathlib.Path(data_path)
        data_dirs = [dir for dir in data_path.iterdir() if dir.is_dir()]
        _temp = list()
        for data_dir in data_dirs:
            for dir in data_dir.iterdir():
                if dir.is_dir():
                    _temp.append(dir)
        data_dirs = sorted(_temp)
        if not data_dirs:
            raise ValueError(self.dataset_name + " data paths empty!")
        # if len(data_dirs) != 80:
        #     raise ValueError("Some files are missing")
        print(len(data_dirs))
        dirs = list()
        for data_dir in data_dirs:
            # subject = data_dir.parent.stem.strip("p")
            # index = subject + data_dir.stem.strip("p")
            subject = data_dir.parent.stem
            index = subject + "_" + data_dir.stem

            dirs.append({"index":index, "subject":subject, "path":str(data_dir)})
        return dirs

    def split_raw_data(self, data_dirs, begin, end):
        """Returns a subset of data dirs, split with begin and end values."""
        if begin == 0 and end == 1:  # return the full directory if begin == 0 and end == 1
            return data_dirs

        file_num = len(data_dirs)
        choose_range = range(int(begin * file_num), int(end * file_num))
        data_dirs_new = []

        for i in choose_range:
            data_dirs_new.append(data_dirs[i])

        return data_dirs_new

    def preprocess_dataset_subprocess(self, data_dirs, config_preprocess, i, file_list_dict):
        """ invoked by preprocess_dataset for multi_process."""
        filename = os.path.split(data_dirs[i]['path'])[-1]
        saved_filename = data_dirs[i]['index']

        video_path = data_dirs[i]['path']
        setting = saved_filename[-1]
        #different setting have different file names
        if setting == '0':
           video_filename = "LowHR-Bright.MOV"
        elif setting == '1':
            video_filename = "LowHR-Dark.MOV"
        elif setting == '2':
            video_filename = "HighHR-Dark.MOV"
        elif setting == '3':
            video_filename = "HighHR-Bright.MOV"

        print("os path exists", os.path.exists(os.path.join(video_path, video_filename)))
        print("data path exists", os.path.exists(os.path.join(video_path, "data.hdf5")))
        frames = self.read_video(
            os.path.join(video_path, video_filename))
        bvps = self.read_wave(
            os.path.join(video_path, "data.hdf5"))

        print("frame shape",frames.shape)
        print("bvps shape", bvps.shape)

        target_length = frames.shape[0]
        bvps = BaseLoader.resample_ppg(bvps, target_length)

        frames_clips, bvps_clips = self.preprocess(frames, bvps, config_preprocess)
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
    def read_wave(bvp_file):
        """Reads a bvp signal file."""
        with h5py.File(bvp_file, 'r') as f:
            bvp = np.array(f['bvp'])
        return bvp