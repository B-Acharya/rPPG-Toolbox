"""The dataloader for COHFACE datasets.

Details for the COHFACE Dataset see https://www.idiap.ch/en/dataset/cohface
If you use this dataset, please cite the following publication:
Guillaume Heusch, André Anjos, Sébastien Marcel, “A reproducible study on remote heart rate measurement”, arXiv, 2016.
http://publications.idiap.ch/index.php/publications/show/3688
"""

import glob
import os
import re

import cv2
import h5py
import numpy as np
from syncpos.utils.alignsignals import AlignSignals

from dataset.data_loader.BaseLoader import BaseLoader


class COHFACELoader(BaseLoader):
    """The data loader for the COHFACE dataset."""

    def __init__(
            self,
            name,
            data_path,
            config_data,
            model,
            device,
            align=None,
            sensor_type=None,  # Added to match the same path to all the datasets
            pseudo_label_type=None,
            transform=None,):
        """Initializes an COHFACE dataloader.
            Args:
                data_path(str): path of a folder which stores raw video and bvp data.
                e.g. data_path should be "RawData" for below dataset structure:
                -----------------
                     RawData/
                     |   |-- 1/
                     |      |-- 0/
                     |          |-- data.avi
                     |          |-- data.hdf5
                     |      |...
                     |      |-- 3/
                     |          |-- data.avi
                     |          |-- data.hdf5
                     |...
                     |   |-- n/
                     |      |-- 0/
                     |          |-- data.avi
                     |          |-- data.hdf5
                     |      |...
                     |      |-- 3/
                     |          |-- data.avi
                     |          |-- data.hdf5
                -----------------
                name(str): name of the dataloader.
                config_data(CfgNode): data settings(ref:config.py).
        """
        if align is not None:
            self.align_signals = AlignSignals(align, config_data.FS)
        else:
            self.align_signals = None

        if name == "train":
            self.split_path = config_data.SPLIT_PATH
        elif name == "valid":
            self.split_path = config_data.SPLIT_PATH
        elif name == "test":
            self.split_path = config_data.SPLIT_PATH
        elif name == "unsupervised":
            self.split_path = config_data.SPLIT_PATH

        if self.split_path == None:
            self.use_predefined_splits = False
        else:
            self.use_predefined_splits = True

        super().__init__(name, data_path, config_data, model)

    def _read_split_path(self):
        data_paths = []
        with open(self.split_path, "r") as f:
            for line in f.readlines():
                data_paths.append(self.raw_data_path + line.strip())
        return data_paths

    def get_raw_data(self, data_path):
        """Returns data directories under the path(For COHFACE dataset)."""
        data_dirs = glob.glob(data_path + os.sep + "*")
        if not data_dirs:
            raise ValueError(self.dataset_name + " data paths empty!")
        dirs = list()
        if self.use_predefined_splits:
            data_dirs = self._read_split_path()
            for data_dir in data_dirs:
                subject = data_dir.split("/")[-3]
                i = data_dir.split("/")[-2]
                dirs.append(
                    {
                        "index": int("{0}0{1}".format(subject, i)),
                        "path": os.path.join(data_dir),
                    }
                )

        else:
            data_dirs = glob.glob(data_path + os.sep + "*")
            print("data_dirs:", data_dirs)
            for data_dir in data_dirs:
                for i in range(4):
                    subject = os.path.split(data_dir)[-1]
                    if subject.isnumeric():
                        dirs.append(
                            {
                                "index": int("{0}0{1}".format(subject, i)),
                                "path": os.path.join(data_dir, str(i)),
                            }
                        )
        if not data_dirs:
            raise ValueError(self.dataset_name + " data paths empty!")

        return dirs

    def preprocess_dataset(self, data_dirs, config_preprocess):
        """Preprocesses the raw data."""
        filename = os.path.split(data_dirs[i]["path"])[-1]
        saved_filename = data_dirs[i]["index"]
        print("saved filename", saved_filename)
        print(data_dirs[i])
        frames = self.read_video(os.path.join(data_dirs[i]["path"], "data.avi"))
        bvps = self.read_wave(os.path.join(data_dirs[i]["path"], "data.hdf5"))
        print(frames.shape)
        print(data_dirs[i]["path"])
        target_length = frames.shape[0]
        bvps = BaseLoader.resample_ppg(bvps, target_length)
        frames_clips, bvps_clips, bvps_pseudo_clips = self.preprocess(frames, bvps, config_preprocess)
        input_name_list, label_name_list, _ = self.save_multi_process(
            frames_clips, bvps_clips, bvps_pseudo_clips, saved_filename
        )
        print("frames_clips shape", frames_clips.shape)
        print("bvps_clips shape", bvps_clips.shape)
        print("bvps_pseudo_clips shape", bvps_pseudo_clips.shape)

        # raise ValueError("stop")

        file_list_dict[i] = input_name_list

    @staticmethod
    def read_video(video_file):
        """Reads a video file, returns frames(T,H,W,3) """
        VidObj = cv2.VideoCapture(video_file)
        VidObj.set(cv2.CAP_PROP_POS_MSEC, 0)
        success, frame = VidObj.read()
        frames = list()
        while (success):
            frame = cv2.cvtColor(np.array(frame), cv2.COLOR_BGR2RGB)
            frame = np.asarray(frame)
            frame[np.isnan(frame)] = 0  # TODO: maybe change into avg
            frames.append(frame)
            success, frame = VidObj.read()

        return np.asarray(frames)

    @staticmethod
    def read_wave(bvp_file):
        """Reads a bvp signal file."""
        f = h5py.File(bvp_file, 'r')
        pulse = f["pulse"][:]
        return pulse
