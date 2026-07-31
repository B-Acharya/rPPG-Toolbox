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
            transform=None,
            hydra_config=None):
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
                video_dir = sub_dir.split(os.sep)[-1]
                video_path = os.path.join(data_dir, sub_dir)
                dirs.append(
                    {
                        "index": f"{subject}_{video_dir}",
                        "subject": subject,
                        "path": video_path,
                    }
                )
        if not data_dirs:
            raise ValueError(self.dataset_name + " data paths empty!")
        return dirs

    def split_raw_data(self, data_dirs, begin, end):
        """Returns a subset of data dirs, split with begin/end, no subject overlap."""
        if begin == 0 and end == 1:
            return data_dirs

        data_info = {}
        for data in data_dirs:
            subject = data["subject"]
            if subject not in data_info:
                data_info[subject] = []
            data_info[subject].append(data)

        subj_list = sorted(data_info.keys())
        print("Before Shuffle:", subj_list)
        if self.shuffle:
            random.Random(4).shuffle(subj_list)
            print("After Shuffle:", subj_list)
        else:
            print("No Shuffle")

        num_subjs = len(subj_list)
        subj_range = list(range(int(begin * num_subjs), int(end * num_subjs)))

        data_dirs_new = []
        for i in subj_range:
            data_dirs_new += data_info[subj_list[i]]
        return data_dirs_new

    def preprocess_dataset_subprocess(self, data_dirs, config_preprocess, i, file_list_dict):
        """Invoked by preprocess_dataset for multi_process."""
        filename = os.path.split(data_dirs[i]["path"])[-1]
        saved_filename = data_dirs[i]["index"]
        video_path = data_dirs[i]["path"]
        print("Processing video: ", video_path)
        frames = self.read_video(os.path.join(video_path, "data_faces.hdf5"))

        if self.pseudo_label_type == "POS_UF":
            print("Using unfiltered POS to generate pseudo_labels")
            bvps = self.generate_pos_uf(frames, fs=self.fs)
        elif self.pseudo_label_type == "CHROM":
            print("Using CHROM to generate pseudo_labels")
            bvps = self.generate_chrom_pseudo_labels(frames, fs=self.fs)
        else:
            raise NotImplementedError("The pseudo labels type has to be set to POS_UF or CHROM")

        min_len = min(frames.shape[0], bvps.shape[0])
        frames = frames[:min_len]
        bvps = bvps[:min_len]

        assert frames.shape[0] == bvps.shape[0]

        chunk_length = config_preprocess.CHUNK_LENGTH

        usable_len = (min_len // chunk_length) * chunk_length

        frames = frames[:usable_len]
        bvps = bvps[:usable_len]

        frames_clips, bvps_clips, _ = self.preprocess(frames, bvps, config_preprocess)

        bvps_pseudo_clips = bvps_clips

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