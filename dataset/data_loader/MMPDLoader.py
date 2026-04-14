"""The dataloader for MMPD datasets."""

import os
import glob
import random
import numpy as np
from numpy.typing import NDArray

from rPPG_Toolbox.dataset.data_loader.BaseLoader import BaseLoader
import pandas as pd
import scipy.io as sio
from warnings import simplefilter
from syncpos.utils.alignsignals import AlignSignals

simplefilter(action="ignore", category=FutureWarning)


class MMPDLoader(BaseLoader):
    """The data loader for the MMPD dataset."""

    def __init__(
        self,
        name,
        data_path,
        config_data,
        model,
        device,
        align=None,
        sensor_type=None,  # Added to match the same interface as all loaders
        pseudo_label_type=None,
        transform=None,
    ):
        """Initializes an MMPD dataloader.
        Args:
            data_path(str): path of a folder which stores raw video and bvp data.
            e.g. data_path should be "mat_dataset" for below dataset structure:
            -----------------
                 mat_dataset/
                 |   |-- subject1/
                 |       |-- p1_0.mat
                 |       |-- p1_1.mat
                 |       |...
                 |   |-- subject2/
                 |       |-- p2_0.mat
                 |       |-- p2_1.mat
                 |       |...
                 |...
                 |   |-- subjectn/
                 |       |-- pn_0.mat
                 |       |-- pn_1.mat
                 |       |...
            -----------------
            name(string): name of the dataloader.
            config_data(CfgNode): data settings(ref:config.py).
        """
        self.info = config_data.INFO

        if align is not None:
            self.align_signals = AlignSignals(align, config_data.FS)
        else:
            self.align_signals = None

        self.pseudo_label_type = pseudo_label_type

        super().__init__(name, data_path, config_data, model, device, transform)

    def get_raw_data(self, raw_data_path):
        """Returns data directories under the path(For MMPD dataset)."""

        data_dirs = glob.glob(raw_data_path + os.sep + "subject*")
        if not data_dirs:
            raise ValueError(self.dataset_name + " data paths empty!")
        dirs = list()
        data_dirs = [data_dir for data_dir in data_dirs if os.path.isdir(data_dir)]
        for data_dir in data_dirs:
            subject = data_dir.split(os.sep)[-1]
            mat_dirs = os.listdir(data_dir)
            for mat_dir in mat_dirs:
                index = mat_dir.split("_")[-1].split(".")[0]
                dirs.append(
                    {
                        "index": index,
                        "path": data_dir + os.sep + mat_dir,
                        "subject": subject,
                    }
                )
        return dirs

    def split_raw_data(self, data_dirs, begin, end):
        """Returns a subset of data dirs, split with begin and end values,
        and ensures no overlapping subjects between splits"""

        # return the full directory
        if begin == 0 and end == 1:
            return data_dirs

        data_info = dict()
        for data in data_dirs:
            index = data["index"]
            data_dir = data["path"]
            subject = data["subject"]
            # creates a dictionary of data_dirs indexed by subject number
            if subject not in data_info:
                data_info[subject] = list()
            data_info[subject].append(data)

        subj_list = list(data_info.keys())
        subj_list = sorted(subj_list)
        print("Before Shuffle:", subj_list)
        if self.shuffle:
            random.Random(4).shuffle(subj_list)
            print("After Shuffle:", subj_list)
        else:
            print("No Shuffle")
        num_subjs = len(subj_list)

        # get split of data set (depending on start / end)
        subj_range = list(range(num_subjs))
        if begin != 0 or end != 1:
            subj_range = list(range(int(begin * num_subjs), int(end * num_subjs)))
        print("used subject ids for split:", [subj_list[i] for i in subj_range])

        # compile file list
        data_dirs_new = list()
        for i in subj_range:
            subj_num = subj_list[i]
            data_dirs_new += data_info[subj_num]

        return data_dirs_new

    def split_raw_data_loo(self, data_dirs, participant_ids):
        """Returns subset of data dirs for leave-one-out splits."""
        data_dirs_new = list()
        for data_dir in data_dirs:
            if data_dir["subject"] in participant_ids:
                data_dirs_new.append(data_dir)
        return data_dirs_new

    def preprocess_dataset_subprocess(
        self, data_dirs, config_preprocess, i, file_list_dict
    ):
        """Invoked by preprocess_dataset for multi_process."""
        (
            frames,
            bvps,
            light,
            motion,
            exercise,
            skin_color,
            gender,
            glasser,
            hair_cover,
            makeup,
        ) = self.read_mat(data_dirs[i]["path"])

        saved_filename = "subject" + str(data_dirs[i]["subject"])
        saved_filename += f"_L{light}_MO{motion}_E{exercise}_S{skin_color}_GE{gender}_GL{glasser}_H{hair_cover}_MA{makeup}"

        frames = (np.round(frames * 255)).astype(np.uint8)
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

    def read_mat(self, mat_file):
        try:
            mat = sio.loadmat(mat_file)
        except Exception:
            for _ in range(20):
                print(mat_file)
        frames = np.array(mat["video"])
        bvps = np.array(mat["GT_ppg"]).T.reshape(-1)

        light = mat["light"]
        motion = mat["motion"]
        exercise = mat["exercise"]
        skin_color = mat["skin_color"]
        gender = mat["gender"]
        glasser = mat["glasser"]
        hair_cover = mat["hair_cover"]
        makeup = mat["makeup"]
        information = [
            light,
            motion,
            exercise,
            skin_color,
            gender,
            glasser,
            hair_cover,
            makeup,
        ]

        light, motion, exercise, skin_color, gender, glasser, hair_cover, makeup = (
            self.get_information(information)
        )

        return (
            frames,
            bvps,
            light,
            motion,
            exercise,
            skin_color,
            gender,
            glasser,
            hair_cover,
            makeup,
        )

    def load_preprocessed_data(self):
        """Loads the preprocessed data listed in the file list, filtered by INFO criteria.

        The MMPD dataset encodes metadata in the filename (e.g., subject1_L2_MO1_E2_S4_GE1_GL2_H2_MA2).
        This override filters recordings by the INFO fields specified in config.
        """
        file_list_path = self.file_list_path
        file_list_df = pd.read_csv(file_list_path)
        inputs_temp = file_list_df["input_files"].tolist()
        inputs = []
        for each_input in inputs_temp:
            info = each_input.split(os.sep)[-1].split("_")
            light = int(info[1][-1])
            motion = int(info[2][-1])
            exercise = int(info[3][-1])
            skin_color = int(info[4][-1])
            gender = int(info[5][-1])
            glasser = int(info[6][-1])
            hair_cover = int(info[7][-1])
            makeup = int(info[8][-1])
            if (
                (light in self.info.LIGHT)
                and (motion in self.info.MOTION)
                and (exercise in self.info.EXERCISE)
                and (skin_color in self.info.SKIN_COLOR)
                and (gender in self.info.GENDER)
                and (glasser in self.info.GLASSER)
                and (hair_cover in self.info.HAIR_COVER)
                and (makeup in self.info.MAKEUP)
            ):
                inputs.append(each_input)
        if not inputs:
            raise ValueError(self.dataset_name + " dataset loading data error!")
        inputs = sorted(inputs)
        labels = [input_file.replace("input", "label") for input_file in inputs]
        labels_pseudo = [
            input_file.replace("input", "label_pseudo") for input_file in inputs
        ]
        self.inputs = inputs
        self.labels = labels
        self.labels_pseudo = labels_pseudo
        self.preprocessed_data_len = len(inputs)

    @staticmethod
    def get_information(information):
        light = ""
        if information[0] == "LED-low":
            light = 1
        elif information[0] == "LED-high":
            light = 2
        elif information[0] == "Incandescent":
            light = 3
        elif information[0] == "Nature":
            light = 4
        else:
            raise ValueError(
                "Error with MMPD or Mini-MMPD dataset labels! "
                "The following lighting label is not supported: {0}".format(
                    information[0]
                )
            )

        motion = ""
        if (
            information[1] == "Stationary"
            or information[1] == "Stationary (after exercise)"
        ):
            motion = 1
        elif information[1] == "Rotation":
            motion = 2
        elif information[1] == "Talking":
            motion = 3
        # 'Watching Videos' is an erroneous label from older versions of the MMPD dataset,
        #  it should be handled as 'Walking'.
        elif information[1] == "Walking" or information[1] == "Watching Videos":
            motion = 4
        else:
            raise ValueError(
                "Error with MMPD or Mini-MMPD dataset labels! "
                "The following motion label is not supported: {0}".format(
                    information[1]
                )
            )

        exercise = ""
        if information[2] == "True":
            exercise = 1
        elif information[2] == "False":
            exercise = 2
        else:
            raise ValueError(
                "Error with MMPD or Mini-MMPD dataset labels! "
                "The following exercise label is not supported: {0}".format(
                    information[2]
                )
            )

        skin_color = information[3][0][0]

        if skin_color != 3 and skin_color != 4 and skin_color != 5 and skin_color != 6:
            raise ValueError(
                "Error with MMPD or Mini-MMPD dataset labels! "
                "The following skin_color label is not supported: {0}".format(
                    information[3][0][0]
                )
            )

        gender = ""
        if information[4] == "male":
            gender = 1
        elif information[4] == "female":
            gender = 2
        else:
            raise ValueError(
                "Error with MMPD or Mini-MMPD dataset labels! "
                "The following gender label is not supported: {0}".format(
                    information[4]
                )
            )

        glasser = ""
        if information[5] == "True":
            glasser = 1
        elif information[5] == "False":
            glasser = 2
        else:
            raise ValueError(
                "Error with MMPD or Mini-MMPD dataset labels! "
                "The following glasser label is not supported: {0}".format(
                    information[5]
                )
            )

        hair_cover = ""
        if information[6] == "True":
            hair_cover = 1
        elif information[6] == "False":
            hair_cover = 2
        else:
            raise ValueError(
                "Error with MMPD or Mini-MMPD dataset labels! "
                "The following hair_cover label is not supported: {0}".format(
                    information[6]
                )
            )

        makeup = ""
        if information[7] == "True":
            makeup = 1
        elif information[7] == "False":
            makeup = 2
        else:
            raise ValueError(
                "Error with MMPD or Mini-MMPD dataset labels! "
                "The following makeup label is not supported: {0}".format(
                    information[7]
                )
            )

        return light, motion, exercise, skin_color, gender, glasser, hair_cover, makeup
