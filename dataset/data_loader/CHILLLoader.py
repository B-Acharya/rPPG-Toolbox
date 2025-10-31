"""The dataloader for MBP-PPG datasets.

TODO:Update the informaiton link and citations for MBP-PPG
Details for the MBP-PPG Dataset see .
"""

import os
import pathlib
import h5py
import random
from numpy.typing import NDArray

import numpy as np
from rPPG_Toolbox.dataset.data_loader.BaseLoader import BaseLoader
from syncpos.utils.alignsignals import AlignSignals


class CHILLLoader(BaseLoader):
    """The data loader for the MBP-PPG dataset.
    Data structure
    -----------------
          RawData/
          |   |-- subject1/
          |       |-- 1/
          |          |-- faces_yolo_data.hdf5
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

    def __init__(
        self, name, data_path, config_data, model, device, align=None, transform=None
    ):
        """Initializes an UBFC dataloader.
        Args:
            data_path(str): path of a folder which stores raw video and bvp data.
            e.g. data_path should be "RawData" for below dataset structure:
            name(string): name of the dataloader.
            config_data(CfgNode): data settings(ref:config.py).
        """

        if align is not None:
            self.align_signals = AlignSignals(align, config_data.FS)
        else:
            self.align_signals = None

        super().__init__(name, data_path, config_data, model, device, transform)

    def get_raw_data(self, data_path):
        """Returns data directories under the path(For CMBP dataset)."""
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

            dirs.append({"index": index, "subject": subject, "path": str(data_dir)})
        return dirs

    def split_raw_data(self, data_dirs, begin, end):
        """Returns a subset of data dirs, split with begin and end values."""
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

    def preprocess_dataset_subprocess(
        self, data_dirs, config_preprocess, i, file_list_dict
    ):
        """invoked by preprocess_dataset for multi_process."""
        filename = os.path.split(data_dirs[i]["path"])[-1]
        saved_filename = data_dirs[i]["index"]

        video_path = data_dirs[i]["path"]
        setting = saved_filename[-1]
        # different setting have different file names
        video_filename = "faces_yolo_data.hdf5"
        print(
            "os path exists", os.path.exists(os.path.join(video_path, video_filename))
        )

        frames = self.read_video(os.path.join(video_path, video_filename))

        bvps = self.read_wave(os.path.join(video_path, "data.hdf5"))

        target_length = frames.shape[0]
        bvps = BaseLoader.resample_ppg(bvps, target_length)

        if self.align_signals is not None:
            # generate the psuedo_labels
            # These are hilbert envelopes , TODO: Maybe use the genreal algo to extract the signal
            bvp_psuedo = self.generate_pos_psuedo_labels(frames, fs=self.fs)

            aligned_bvps, _, video_start_idx, video_end_idx = self.align_signals(
                bvps, bvp_psuedo
            )

            print(f"start-> {video_start_idx}, end-> {video_end_idx}")

            # create the synced video frames
            frames = frames[video_start_idx:video_end_idx]

            # aligned signals are preprocessed
            frames_clips, bvps_aligned_clips, bvps_psuedo_clips = self.preprocess(
                frames, bvps, config_preprocess
            )

            # need similar preprocessing as the syncronized signal
            chunk_length = config_preprocess.CHUNK_LENGTH
            clip_num = frames.shape[0] // chunk_length
            bvps_clips = self._preprocess_for_alignment(
                bvps, config_preprocess, clip_num, chunk_length
            )

            #
            input_name_list, label_name_list, label_psuedo_name_list = (
                self.save_multi_process(
                    frames_clips, bvps_clips, bvps_aligned_clips, saved_filename
                )
            )

        else:
            # the data is preprocessed if align signals is none
            frames_clips, bvps_clips, bvps_psuedo_clips = self.preprocess(
                frames, bvps, config_preprocess
            )

            input_name_list, label_name_list, label_psuedo_name_list = (
                self.save_multi_process(
                    frames_clips, bvps_clips, bvps_psuedo_clips, saved_filename
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
        """Reads a video file, returns frames(T, H, W, 3)"""
        with h5py.File(video_file, "r") as hdf5_file:
            if "faces" in hdf5_file:
                faces = np.array(hdf5_file["faces"])
            else:
                raise KeyError(
                    f"'faces' dataset not found. Available keys: {list(hdf5_file.keys())}"
                )
        return faces

    @staticmethod
    def read_wave(bvp_file):
        """Reads a bvp signal file."""
        with h5py.File(bvp_file, "r") as f:
            bvp = np.array(f["bvp"])
        return bvp
