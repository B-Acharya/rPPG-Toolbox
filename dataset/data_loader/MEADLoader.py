import glob
import os
import random

import cv2
import h5py
import numpy as np
from syncpos.utils.alignsignals import AlignSignals
from numpy.typing import NDArray

from rPPG_Toolbox.dataset.data_loader.BaseLoader import BaseLoader

class MEADLoader(BaseLoader):
    """The MEAD data loader"""

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
        """Initializes a MEAD dataloader.
        Args:
            data_path(str): path of a folder which stores raw video.
            e.g. data_path should be "RawData" for below dataset structure:
            -----------------
                 RawData/
                 |   |-- M003/
                 |      |-- perspective/ (e.g. top, left)
                 |         |-- emotion/ (e.g. angry, surprised)
                 |              |-- level/ (e.g. level_1, level_2)
                 |                  |-- 001.mp4
                 |                  |-- 002.mp4
                 |   |-- M004/
                 |      |-- perspective/ (e.g. top, left)
                 |         |-- emotion/ (e.g. angry, surprised)
                 |              |-- level/ (e.g. level_1, level_2)
                 |                  |-- 001.mp4
                 |                  |-- 002.mp4
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

        for subject in data_dirs:
            for perspective in glob.glob(subject + os.sep + "video" + os.sep + "*"):
                for emotion in glob.glob(perspective + os.sep + "*"):
                    for level in glob.glob(emotion + os.sep + "*"):
                        sublevel = glob.glob(level + os.sep + "*")
                        for vid in sublevel:
                            subject_index = subject.split(os.sep)[-1]
                            dirs.append(
                                {
                                    "index": subject_index,
                                    "path": vid,
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

        frames = self.read_video(video_path)
        bvps = self.generate_pos_uf(frames, fs=self.fs)

        frames_clips, bvps_clips, bvps_pseudo_clips = self.preprocess(
            frames, bvps, config_preprocess
        )

        if self.pseudo_label_type == "POS_UF":
            print("Using unfiltered POS to generate pseudo_labels")
            bvps_pseudo_clips = self.generate_pos_uf(frames, fs=self.fs)

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
        """Reads a video file, returns frames(T,H,W,3)"""
        print("start of read_video")
        VidObj = cv2.VideoCapture(video_file)
        VidObj.set(cv2.CAP_PROP_POS_MSEC, 0)
        success, frame = VidObj.read()
        frames = list()
        while success:
            frame = cv2.cvtColor(np.array(frame), cv2.COLOR_BGR2RGB)
            frame = np.asarray(frame)
            if np.isnan(frame).any():
                frame[np.isnan(frame)] = 0  # TODO: maybe change into avg
            frames.append(frame)
            success, frame = VidObj.read()
        print("end of read_video")
        return np.asarray(frames)
