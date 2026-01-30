"""The dataloader for MBP-PPG datasets.

TODO:Update the informaiton link and citations for MBP-PPG
Details for the MBP-PPG Dataset see .
"""

import os
from pathlib import Path
import pandas as pd
import pathlib
from re import U
import h5py
import random
from numpy.typing import NDArray
from typing import List

import numpy as np
import cv2
from rPPG_Toolbox.dataset.data_loader.BaseLoader import BaseLoader
from syncpos.utils.alignsignals import AlignSignals
from typing import Literal, Union

DeviceType = Union[Literal["A188", "A543"], None]


class CHILLINDLoader(BaseLoader):
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
        self,
        name,
        data_path,
        config_data,
        model,
        device,
        sensor_type: DeviceType = None,
        psuedo_label_type=None,
        align=None,
        transform=None,
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

        # Hardcoded data path
        self.sync_data_path = Path(
            "/data/rppg_24_bit_video_nt_lab/processed/synchronized_data"
        )

        self.raw_data_path: Path = Path("data/rppg_24_bit_video_nt_lab/raw/")

        self.sensor_type: DeviceType = sensor_type

        self.path_to_dataframes = list(
            self.sync_data_path.rglob(f"{self.sensor_type}_synchronized.csv")
        )
        self.hr_conditions = ["HighHR", "LowHR"]
        self.illu_conditions = ["Bright", "Dark"]

        self.psuedo_label_type = psuedo_label_type

        super().__init__(name, data_path, config_data, model, device, transform)

    def get_raw_data(self, data_path):
        """Returns data directories under the path for dataset)."""

        data_dirs = []

        participants_paths = [
            Path(participant)
            for participant in os.scandir(self.sync_data_path)
            if participant.is_dir()
        ]

        for participant_path in participants_paths:

            participant = participant_path.stem

            for hr_condition in self.hr_conditions:
                for illu_condition in self.illu_conditions:
                    csv_file = (
                        participant_path
                        / hr_condition
                        / illu_condition
                        / f"{self.sensor_type}_synchronized.csv"
                    )
                    unique_id = f"{participant}_{hr_condition}_{illu_condition}"
                    data_dirs.append(
                        {
                            "index": hash(unique_id),  # Unique numeric identifier
                            "path": str(Path(csv_file).parent),
                            "subject": participant,
                            "csv_path": csv_file,
                        }
                    )

        return data_dirs

    def split_raw_data(self, data_dirs, begin, end):
        """Returns a subset of data dirs, split with begin and end values."""

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

    @staticmethod
    def update_paths_to_face_cropped(
        paths: List[str],
        participant_id: str,
        hr_condition: Literal["HighHR", "LowHR"],
        illu_condition: Literal["Bright", "Dark"],
    ) -> List[Path]:
        """Updates the path to the facecropped paths from the directory"""

        updated_paths: List[Path] = list()

        base_path: Path = (
            Path("/data/rppg_24_bit_video_nt_lab/processed/chill_ind_face_crop")
            / participant_id
        )

        for path in paths:
            image_timestamp = Path(path).stem + "_face.tiff"
            updated_paths.append(
                base_path / hr_condition / illu_condition / image_timestamp
            )

        return updated_paths

    def preprocess_dataset_subprocess(
        self, data_dirs, config_preprocess, i, file_list_dict
    ):
        """invoked by preprocess_dataset for multi_process."""

        entry = data_dirs[i]
        csv_path = entry["csv_path"]

        filename = os.path.split(data_dirs[i]["path"])[-1]
        saved_filename = data_dirs[i]["index"]
        csv_path = data_dirs[i]["csv_path"]

        # Read and sort TIFF sequence
        sync_data = pd.read_csv(csv_path)
        print(sync_data.columns)
        image_paths = sync_data["paths"]

        illu_condition = csv_path.parent.name
        hr_condition = csv_path.parent.parent.name

        # updates the path files to cropped face
        # TOOD: Adding a flag to processed raw images

        # image_paths = self.update_paths_to_face_cropped(
        #    image_paths,
        #    participant_id=entry["subject"],
        #    hr_condition=hr_condition,
        #    illu_condition=illu_condition,
        # )

        # Read Frames
        if "None" in config_preprocess.DATA_AUG:
            # Utilize dataset-specific function to read video
            frames = self.read_video(image_paths)
        elif "Motion" in config_preprocess.DATA_AUG:
            # Utilize general function to read video in .npy format
            frames = self.read_video(image_paths)
        else:
            raise ValueError(
                f"Unsupported DATA_AUG specified for {self.dataset_name} dataset! Received {config_preprocess.DATA_AUG}."
            )

        # Process signals
        if config_preprocess.USE_PSEUDO_PPG_LABEL:
            bvps = self.generate_pos_psuedo_labels(frames, self.config_data.FS)
        else:
            bvps = self.read_wave(sync_data, self.sensor_type)

        target_length = frames.shape[0]

        # TOOD: Check if resampling is needed for CHILL-IND dataset
        bvps = BaseLoader.resample_ppg(bvps, target_length)

        if self.align_signals is not None:

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

            input_name_list, label_name_list, label_psuedo_name_list = (
                self.save_multi_process(
                    frames_clips, bvps_clips, bvps_aligned_clips, saved_filename
                )
            )

        else:

            frames_clips, bvps_clips, bvps_psuedo_clips = self.preprocess(
                frames, bvps, config_preprocess
            )

            if self.psuedo_label_type == "POS_UF":
                print("Using unfiltered POS to generate psuedo_labels")
                bvps_psuedo_clips = self.generate_pos_uf(frames, fs=self.fs)

                # need similar preprocessing as the pseudo signal

                chunk_length = config_preprocess.CHUNK_LENGTH
                clip_num = frames.shape[0] // chunk_length

                bvps_psuedo_clips = self._preprocess_for_alignment(
                    bvps_psuedo_clips, config_preprocess, clip_num, chunk_length
                )
            else:
                pass

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
    def read_video(paths):
        """Reads a video file, returns frames(T, H, W, 3)"""
        frames = list()
        for tiff_path in paths:
            if Path(tiff_path).exists():
                img = cv2.imread(tiff_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                frames.append(img)
            else:
                raise FileNotFoundError(f"{tiff_path}: Check the file path bruh")
        return np.asarray(frames)

    @staticmethod
    def read_wave(bvp_file, sensor_type):
        """Reads a bvp signal file."""
        if sensor_type == "A188":
            bvps = bvp_file["Shimmer_A188"].values.astype(np.float32)
        elif sensor_type == "A543":
            bvps = bvp_file["Shimmer_A543"].values.astype(np.float32)
        else:
            raise NotImplementedError(f"Wrong {sensor_type}")

        return bvps
