import glob
import os
import random

import cv2
import h5py
import numpy as np
from syncpos.utils.alignsignals import AlignSignals
from numpy.typing import NDArray
from pathlib import Path
import subprocess
import json
import time
import shutil

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
        self.config_data = config_data

        super().__init__(name, data_path, config_data, model)

    def get_raw_data(self, data_path):
        dirs = list()
        data_dirs = glob.glob(data_path + os.sep + "*")
        db_file_path = "/homes/cgerz/datasets/mead_frame_counts.json"
        db_file = self.load_db(db_file_path)

        dst_root = Path("/data/rppg_20_mead_video_nt_lab/processed/tmp")
        src_root = Path("/data/rppg_20_mead_video_nt_lab/processed/cropped_face")

        for subject in data_dirs:
            for perspective in glob.glob(subject + os.sep + "video" + os.sep + "*"):
                perspective_name = perspective.split(os.sep)[-1]
                if perspective_name != "front":
                    print(f"Skipping perspective {perspective_name} for subject {subject}")
                    continue
                for emotion in glob.glob(perspective + os.sep + "*"):
                    emotion_name = emotion.split(os.sep)[-1]
                    # if emotion_name != "happy":
                    #     print(f"Skipping emotion {emotion_name} for subject {subject} and perspective {perspective_name}")
                    #     continue
                    for level in glob.glob(emotion + os.sep + "*"):
                        level_name = level.split(os.sep)[-1]
                        if level_name != "level_1":
                            print(f"Skipping level {level_name} for subject {subject}")
                            continue
                        sublevel = glob.glob(level + os.sep + "*")
                        level_name = level.split(os.sep)[-1]
                        for vid in sublevel:
                            start = time.time()
                            subject_index = subject.split(os.sep)[-1]
                            vid_name = vid.split(os.sep)[-1]
                            vid_content = os.listdir(vid)
                            if "data_faces.hdf5" not in vid_content:
                                print(f"Skipping video {vid} because data_faces.hdf5 not found")
                                continue

                            data_faces_path = os.path.join(vid, "data_faces.hdf5")
                            relative_path = Path(data_faces_path).relative_to(src_root)
                            dst = dst_root / relative_path

                            parts = Path(data_faces_path).parts
                            root = Path(*parts[:3])
                            emotion_path = Path(*parts[5:-2])
                            clip_id = parts[-2]

                            new_path = (
                                    root
                                    / "raw"
                                    / emotion_path
                                    / f"{clip_id}.mp4"
                            )

                            if data_faces_path not in db_file:
                                duration = self.read_video(data_faces_path).shape[0]
                                print(f"Video {new_path} has duration {duration} frames")
                                print(f"Add video {new_path} with duration {duration} to db_file")
                                db_file[data_faces_path] = duration
                            else:
                                duration = db_file[data_faces_path]

                            if duration < 100:
                                print(f"Skipping video {new_path} because duration {duration} is less than CHUNK_LENGTH {self.config_data.PREPROCESS.CHUNK_LENGTH}")
                                dst.parent.mkdir(parents=True, exist_ok=True)
                                shutil.move(data_faces_path, dst)
                                continue
                            dirs.append(
                                {
                                    "index": f"{subject_index}_{perspective_name}_{emotion_name}_{level_name}_{vid_name}",
                                    "subject": subject,
                                    "path": vid,
                                }
                            )
                            end = time.time()

        if not data_dirs:
            raise ValueError(self.dataset_name + " data paths empty!")

        self.save_db(db_file, db_file_path)
        print("Number of files: ", len(dirs))
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
        bvps = self.generate_pos_uf(frames, fs=self.fs)

        if self.pseudo_label_type is not None:

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
    def load_db(db_file):
        if Path(db_file).exists():
            with open(db_file, "r") as f:
                return json.load(f)
        return {}

    @staticmethod
    def save_db(db, db_file):
        print("Saving database...")
        with open(db_file, "w") as f:
            json.dump(db, f)