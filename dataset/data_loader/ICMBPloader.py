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

import cv2
import numpy as np
from dataset.data_loader.BaseLoader import BaseLoader
from tqdm import tqdm


from pathlib import Path
import pprint
import polars as pl
import matplotlib.pyplot as plt
def read_ppg(file):
    pass

def read_images():
    pass

class shimmer_data():

    def __init__(self, path, image_paths):

        self.path = path
        self.image_paths = image_paths

        if "A188" in str(path):
            self.device = "A188"
            self.column_names = ['Shimmer_A188_TimestampSync_Unix_CAL', 'Shimmer_A188_PPG_A13_CAL']
            # self.column_names = ['Shimmer_A188_Timestamp_Unix_CAL', 'Shimmer_A188_PPG_A13_CAL']

        elif "A543" in str(path):
            self.device = "A543"
            self.column_names = ['Shimmer_A543_TimestampSync_Unix_CAL', 'Shimmer_A543_PPG_A13_CAL']
            # self.column_names = ['Shimmer_A543_Timestamp_Unix_CAL', 'Shimmer_A543_PPG_A13_CAL']
        self._read_as_df()
        self._create_image_df()

        print(self.df_filetered)
        print(self.image_df)

        self.combined = self.image_df.join_asof(self.df_filetered, on=self.column_names[0], strategy='backward')

        print(self.combined)


    def _read_as_df(self):
        try:
            self.df = pl.read_csv(self.path, separator='\t')
        except:
            print("skipping first row")
            self.df = pl.read_csv(self.path, separator='\t', skip_rows=1)

        self.df_filetered = self.df.select(self.column_names)
        self.df_filetered = self.df_filetered.slice(1)

        self.df_filetered = self.df_filetered.with_columns(
            pl.col(self.column_names[0])
            .cast(pl.Float64)
        )

        self.df_filetered = self.df_filetered.with_columns(
            pl.col(self.column_names[0])*1000
        )

        self.df_filetered = self.df_filetered.with_columns(
            pl.col(self.column_names[0]).cast(pl.Int64)
        )

        self.df_filetered = self.df_filetered.with_columns(
            pl.col(self.column_names[0]).cast(pl.Datetime(time_unit='us'))
        )

        self.df_filetered = self.df_filetered.with_columns(
            pl.col(self.column_names[1]).cast(pl.Float64)
        )




    def _create_image_df(self):

        timestamps = [path.stem.strip('.tiff') for path in self.image_paths]
        # print(timestamps)

        data = [
            pl.Series(self.column_names[0], timestamps).cast(pl.Int64),
            pl.Series("paths", self.image_paths)
        ]
        self.image_df = pl.DataFrame(data)


        self.image_df = self.image_df.with_columns(
            pl.col(self.column_names[0]).cast(pl.Datetime(time_unit='ns')).cast(pl.Datetime(time_unit='us'))
        )

    def get_combined(self):
        return self.combined[self.column_names[1]]

class ICMBPLoader(BaseLoader):
    """The data loader for the MBP-PPG dataset.
       Data structure
       -----------------
             RawData/
             |   |-- subject1/
             |       |-- HighHR/
             |          |-- Bright/
             |              |-- *.tiff
             |              |-- sensor
             |                  |--
             |          |-- Dark/
             |              |-- *.tiff
             |              |-- sensor
             |                  |--
             |       |-- LowHR/
             |          |-- Bright/
             |              |-- *.tiff
             |              |-- sensor
             |                  |--
             |          |-- Dark/
             |              |-- *.tiff
             |              |-- sensor
             |                  |--
             |   |-- subject2/
             |       |-- HighHR/
             |          |-- Bright/
             |              |-- *.tiff
             |              |-- sensor
             |                  |--
             |          |-- Dark/
             |              |-- *.tiff
             |              |-- sensor
             |                  |--
             |       |-- LowHR/
             |          |-- Bright/
             |              |-- *.tiff
             |              |-- sensor
             |                  |--
             |          |-- Dark/
             |              |-- *.tiff
             |              |-- sensor
             |                  |--
             |   |-- subjectn/
             |       |-- HighHR/
             |          |-- Bright/
             |              |-- *.tiff
             |              |-- sensor
             |                  |--
             |          |-- Dark/
             |              |-- *.tiff
             |              |-- sensor
             |                  |--
             |       |-- LowHR/
             |          |-- Bright/
             |              |-- *.tiff
             |              |-- sensor
             |                  |--
             |          |-- Dark/
             |              |-- *.tiff
             |              |-- sensor
             |                  |--
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
        """Returns data directories under the path(For CMBP dataset)."""
        data_path = pathlib.Path(data_path)
        data_dirs = [folder for folder in data_path.iterdir() if folder.is_dir()]

        if not data_dirs:
            raise ValueError(self.dataset_name + " data paths empty!")

        print("Number of participants", len(data_dirs))

        # HR settings
        hr_settings = ['HighHR', 'LowHR']
        brightness_settings = ['Bright', 'Dark']

        dirs = []

        for data_dir in data_dirs:

            for hr_setting in hr_settings:

                for brightness in brightness_settings:

                    path = data_dir / hr_setting / brightness
                    subject = data_dir.stem
                    index = subject + "_" + hr_setting + brightness
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

        video_path = data_dirs[i]['path']

        print("os path exists", os.path.exists(os.path.join(video_path, video_path)))

        frames = self.read_video(
            os.path.join(video_path))
        bvps = self.read_wave(
            os.path.join(video_path))

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

        frames = list()
        all_png = sorted(glob.glob(video_file + '*.tiff'))
        for png_path in all_png:
            img = cv2.imread(png_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            frames.append(img)

        return np.asarray(frames)

    @staticmethod
    def read_wave(bvp_file):
        """Reads a bvp signal file."""

        with h5py.File(bvp_file, 'r') as f:
            bvp = np.array(f['bvp'])
        return bvp