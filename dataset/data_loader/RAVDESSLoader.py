import glob
import os
import random

import cv2
import h5py
import numpy as np
from syncpos.utils.alignsignals import AlignSignals
from numpy.typing import NDArray

from rPPG_Toolbox.dataset.data_loader.BaseLoader import BaseLoader

class RAVDESSLoader(BaseLoader):
    """The RAVDESS data loader"""

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
        """Initializes a RAVDESS dataloader.
        Args:
            data_path(str): path of a folder which stores raw video.
            e.g. data_path should be "RawData" for below dataset structure:
            -----------------
                 RawData/
                 |   |-- Actor_01/
                 |      |-- 01-02-01-01-01-01-01/
                 |         |-- 01-02-01-01-01-01-01.mp4
                 |      |-- 01-02-01-01-01-02-01/
                 |         |-- 01-02-01-01-01-01-02.mp4
                 |   |-- Actor_02/
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

        self.pseudo_label_type = pseudo_label_type

        super().__init__(name, data_path, config_data, model)

    def get_raw_data(self, data_path):
        dirs = list()
        data_dirs = glob.glob(data_path + os.sep + "*")

        for data_dir in data_dirs:
            for sub_dir in glob.glob(data_dir + os.sep + "*"):
                subject = int(os.path.split(data_dir)[-1].split("_")[-1])
                video_path = os.path.join(data_dir, sub_dir)
                dirs.append(
                    {
                        "index": subject,
                        "path": video_path,
                    }
                )
        if not data_dirs:
            raise ValueError(self.dataset_name + " data paths empty!")
        return dirs

    def split_raw_data(self, data_dirs, begin, end):
        """Returns a subset of data dirs, split with begin and end values,
        and ensures no overlapping subjects between splits"""
        # return the full directory
        if begin == 0 and end == 1:
            return data_dirs

        # get info about the dataset: subject list and num vids per subject
        data_info = dict()
        for data in data_dirs:
            subject = data["subject"]
            data_dir = data["path"]
            index = data["index"]
            # creates a dictionary of data_dirs indexed by subject number
            if subject not in data_info:  # if subject not in the data info dictionary
                data_info[subject] = []  # make an emplty list for that subject
            # append a tuple of the filename, subject num, trial num, and chunk num
            data_info[subject].append(
                {"index": index, "path": data_dir, "subject": subject}
            )

        subj_list = list(data_info.keys())  # all subjects by number ID
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
        """Invoked by preprocess_dataset for multi_process."""
        filename = os.path.split(data_dirs[i]["path"])[-1]
        saved_filename = data_dirs[i]["index"]
        video_path = data_dirs[i]["path"]
        print("Processing video: ", video_path)
        frames = self.read_video(os.path.join(video_path, "data_faces.hdf5"))
        bvps = self.generate_pos_uf(frames, fs=self.fs)

        frames_clips, bvps_clips, bvps_pseudo_clips = self.preprocess(
            frames, bvps, config_preprocess
        )

        if self.pseudo_label_type == "POS_UF":
            print("Using unfiltered POS to generate pseudo_labels")
            bvps_pseudo_clips = bvps

            # preprocessing required for the pseudo labels
            chunk_length = config_preprocess.CHUNK_LENGTH
            clip_num = frames.shape[0] // chunk_length

            bvps_pseudo_clips = self._preprocess_for_alignment(
                bvps_pseudo_clips, config_preprocess, clip_num, chunk_length
            )
        else:
            pass

        input_name_list, label_name_list, _ = (
            self.save_multi_process(
                frames_clips, bvps_clips, bvps_pseudo_clips, saved_filename
            )
        )
        file_list_dict[i] = input_name_list

    @staticmethod
    def _preprocess_for_alignment(
        bvps, config_preprocess, clip_num: int, chunk_length: int
    ) -> NDArray:
        if config_preprocess.LABEL_TYPE == "Raw":
            pass
        elif config_preprocess.LABEL_TYPE == "DiffNormalized":
            bvps = BaseLoader.diff_normalize_label(bvps)
        elif config_preprocess.LABEL_TYPE == "Standardized":
            bvps = BaseLoader.standardized_label(bvps)
        else:
            raise ValueError("Unsupported label type!")

        if config_preprocess.DO_CHUNK:  # chunk data into snippets
            bvps_clips = [
                bvps[i * chunk_length : (i + 1) * chunk_length] for i in range(clip_num)
            ]
            bvps_clips = np.array(bvps_clips)
        else:
            bvps_clips = np.array([bvps])

        return bvps_clips

    @staticmethod
    def read_video(video_file):
        """Reads face crops from HDF5, returns (T, H, W, 3)."""
        with h5py.File(video_file, "r") as f:
            if "faces" not in f:
                raise KeyError(
                    f"'faces' key not found in {video_file}. Available: {list(f.keys())}"
                )
            return np.array(f["faces"])

    @staticmethod
    def check_video_length(video_file):
        """Checks if video is at least 100 frames long."""
        VidObj = cv2.VideoCapture(video_file)
        frame_count = int(VidObj.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count < 100:
            return False
        return True