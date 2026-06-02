"""The dataloader for COHFACE datasets.

Details for the COHFACE Dataset see https://www.idiap.ch/en/dataset/cohface
If you use this dataset, please cite the following publication:
Guillaume Heusch, André Anjos, Sébastien Marcel, "A reproducible study on remote heart rate measurement", arXiv, 2016.
http://publications.idiap.ch/index.php/publications/show/3688

Data paths (mirrors COHFACEProcessor class constants):
  Face crops : OUTPUT_ROOT / {participant} / {session}
               → HDF5 file named by session int (no extension), key "faces"
               → e.g. /data/rppg_14_bit_video_nt_lab/cropped_face/1/0
  PPG signal : BVP_DATA_ROOT / {participant} / {session} / data.hdf5
               → key "pulse"
               → e.g. /data/rppg_14_cohface_video_nt_lab/raw/1/0/data.hdf5

data_path in the YAML must point to the face-crop root (OUTPUT_ROOT).
"""

import os
import pathlib
import random
from pathlib import Path
from numpy.typing import NDArray
import cv2

import h5py
import numpy as np
from rPPG_Toolbox.dataset.data_loader.BaseLoader import BaseLoader
from syncpos.utils.alignsignals import AlignSignals


class COHFACELoader(BaseLoader):
    """The data loader for the COHFACE dataset."""

    # Mirrors COHFACEProcessor class constants so both sides stay consistent.
    BVP_DATA_ROOT: Path = Path("/data/rppg_14_cohface_video_nt_lab/raw")

    def __init__(
        self,
        name,
        data_path,
        config_data,
        model,
        device,
        align=None,
        sensor_type=None,
        pseudo_label_type=None,
        transform=None,
    ):
        if align is not None:
            self.align_signals = AlignSignals(align, config_data.FS)
        else:
            self.align_signals = None

        self.pseudo_label_type = pseudo_label_type
        super().__init__(name, data_path, config_data, model, device, transform)

    def get_raw_data(self, data_path):
        """Returns data directories under the path (for COHFACE dataset).

        COHFACE structure: {data_path}/{subject_int}/{session_int}/
        Subjects and sessions are integers; sessions are always 0-3.
        Both levels are sorted numerically to avoid "10" < "2" lexicographic ordering.
        """
        dirs = []
        data_path = pathlib.Path(data_path)
        for subject_dir in sorted(
            (p for p in data_path.iterdir() if p.is_dir()),
        ):
            subject = subject_dir.name
            if not subject.isnumeric():
                continue
            subject = int(subject)
            i = 0
            for session_dir in sorted(
                (p for p in subject_dir.iterdir() if p.is_dir()),
                key=lambda p: int(p.name),
            ):
                if i >= 4:
                    continue
                dirs.append(
                    {
                        "index": f"{subject}_{session_dir.name}",
                        "subject": subject,
                        "path": str(session_dir),
                    }
                )
                i = i+1
        if not dirs:
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

    def split_raw_data_loo(self, data_dirs, participant_ids):
        """Returns entries whose subject is in participant_ids (leave-one-out support)."""
        return [d for d in data_dirs if d["subject"] in participant_ids]

    def preprocess_dataset_subprocess(
        self, data_dirs, config_preprocess, i, file_list_dict
    ):
        """Invoked by preprocess_dataset for multi-process preprocessing."""
        saved_filename = data_dirs[i]["index"]
        session_path = data_dirs[i]["path"]

        face_file = os.path.join(session_path, "data.avi")
        bvp_file = os.path.join(session_path, "data.hdf5")

        frames = self.read_video(face_file)
        bvps = self.read_wave(bvp_file)

        target_length = frames.shape[0]
        bvps = BaseLoader.resample_ppg(bvps, target_length)

        if self.align_signals is not None:
            bvp_pseudo = self.generate_pos_pseudo_labels(frames, fs=self.fs)
            aligned_bvps, _, video_start_idx, video_end_idx = self.align_signals(
                bvps, bvp_pseudo
            )
            print(f"start-> {video_start_idx}, end-> {video_end_idx}")
            frames = frames[video_start_idx:video_end_idx]

            frames_clips, bvps_aligned_clips, bvps_pseudo_clips = self.preprocess(
                frames, bvps, config_preprocess
            )
            chunk_length = config_preprocess.CHUNK_LENGTH
            clip_num = frames.shape[0] // chunk_length
            bvps_clips = self._preprocess_for_alignment(
                bvps, config_preprocess, clip_num, chunk_length
            )
            input_name_list, label_name_list, label_pseudo_name_list = (
                self.save_multi_process(
                    frames_clips, bvps_clips, bvps_aligned_clips, saved_filename
                )
            )
        else:
            frames_clips, bvps_clips, bvps_pseudo_clips = self.preprocess(
                frames, bvps, config_preprocess
            )

            if self.pseudo_label_type == "POS_UF":
                print("Using unfiltered POS to generate pseudo_labels")
                bvps_pseudo_clips = self.generate_pos_uf(frames, fs=self.fs)
                chunk_length = config_preprocess.CHUNK_LENGTH
                clip_num = frames.shape[0] // chunk_length
                bvps_pseudo_clips = self._preprocess_for_alignment(
                    bvps_pseudo_clips, config_preprocess, clip_num, chunk_length
                )

            input_name_list, label_name_list, label_pseudo_name_list = (
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

        if config_preprocess.DO_CHUNK:
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

    @staticmethod
    def read_wave(bvp_file):
        """Reads ground-truth PPG from HDF5."""
        with h5py.File(bvp_file, "r") as f:
            return np.array(f["pulse"])
