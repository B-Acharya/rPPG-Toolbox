import torch
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
import glob
import os
import re
from collections import defaultdict
import random
import pandas as pd
import time




class OptimisedMMPDDataset(Dataset):

    def __init__(self, csv_file_path, frame_depth= 90, sampling_mode = 'sliding',
                 stride = None, overlap_ratio = 0.5):
        self.csv_file_path = csv_file_path
        self.frame_depth = frame_depth
        self.sampling_mode = sampling_mode
        self.overlap_ratio = overlap_ratio


        if sampling_mode == "non_overlapping":
            self.stride = frame_depth
        elif sampling_mode == "sliding"and stride is None:
            self.stride = int(frame_depth * (1-overlap_ratio))
        else:
            self.stride = stride or frame_depth

        # Initialize data structures
        self.file_info = []
        self.subject_to_indices = defaultdict(list)
        self._load_from_csv()

        # Get unique subjects
        self.subjects = list(self.subject_to_indices.keys())

    def _load_from_csv(self):
        df = pd.read_csv(self.csv_file_path)

        input_files = None

        if 'input_files' in df.columns:
            input_files = df['input_files'].tolist()

        if input_files is None:
            raise ValueError(f"Could not find files columns in CSV file.")

        label_files = []

        for input_file in input_files:
            label_file = input_file.replace("_input", "_label")
            label_files.append(label_file)

        self._process_file_list(input_files, label_files)

    def _process_file_list(self, input_files, label_files):

        total_files = len(input_files)
        processed = 0
        skipped = 0
        total_windows = 0

        for input_file, label_file in zip(input_files, label_files):

            subject_id = self._extract_subject_id(input_file)

            try:
                frames = np.load(input_file, mmap_mode='r')
                chunk_length = frames.shape[0]

                if chunk_length >=  self.frame_depth:
                    num_windows = int((chunk_length - self.frame_depth) // self.stride + 1)

                    for window_idx in range(num_windows):
                        start_frame = window_idx * self.stride
                        if start_frame + self.frame_depth <= chunk_length:
                            idx = len(self.file_info)
                            self.file_info.append((input_file, label_file,start_frame, subject_id))
                            if subject_id:
                                self.subject_to_indices[subject_id].append(idx)
                            total_windows += 1

                processed += 1

                if processed % 100 == 0:
                    print(f"Processed {processed} / {total_files} files")

            except Exception as e:
                print(f"Error processing {input_file}: {e}")
                skipped += 1
                continue

    def _extract_subject_id(self, filepath):
        file_name = os.path.basename(filepath)
        subject_id = file_name.split("_")[0]
        return subject_id if subject_id.startswith("subject") else None

    def __len__(self):
        return len(self.file_info)

    def get_subjects(self):
        return self.subjects

    def __getitem__(self, idx):

        input_file, label_file, start_frame, subject_id = self.file_info[idx]

        try:
            frames = np.load(input_file, mmap_mode='r')
            bvp = np.load(label_file, mmap_mode='r')

            if frames.ndim == 4:
                chunk_length = frames.shape[0]

                if self.sampling_mode == 'random' and chunk_length > self.frame_depth:
                    start_frame = np.random.randint(0, chunk_length - self.frame_depth + 1)

                end_frame = start_frame + self.frame_depth

                if end_frame <= chunk_length:
                    frames = frames[start_frame:end_frame]
                    bvp = bvp[start_frame:end_frame]
                else:
                    frames = frames[start_frame:]
                    bvp = bvp[start_frame:]
                    pad_length = self.frame_depth - len(frames)
                    frames = np.pad(frames, ((0, pad_length), (0,0),(0,0),(0,0)), mode='edge')
                    bvp = np.pad(bvp,(0,pad_length),mode='edge')

                frames = np.moveaxis(frames, -1, 0)

                frames_tensor = torch.from_numpy(frames).float()
                bvp_tensor = torch.from_numpy(bvp).float()

            return frames_tensor, bvp_tensor

        except Exception as e:
            print(f"Error processing {input_file}: {e}")
            frames = torch.zeros((3,self.frame_depth, 72,72))
            bvp = torch.zeros(self.frame_depth)
            return frames, bvp


def create_dataloaders_from_csv(csv_file_path, frame_depth= 90, batch_size = 32,
                                num_workers = 4, train_ratio = 0.8, seed = 42):

    train_dataset = OptimisedMMPDDataset(csv_file_path = csv_file_path, frame_depth = frame_depth, sampling_mode = 'sliding',
                                         stride = frame_depth//2)

    test_dataset = OptimisedMMPDDataset(csv_file_path = csv_file_path, frame_depth = frame_depth, sampling_mode = 'non_overlapping')

    subjects = sorted(train_dataset.get_subjects())
    rng = random.Random(seed)
    rng.shuffle(subjects)
    n_train = int(len(subjects) * train_ratio)
    train_subjects = set(subjects[:n_train])
    test_subjects = set(subjects[n_train:])

    train_indices = []
    for subj in train_subjects:
        train_indices.extend(train_dataset.subject_to_indices.get(subj, []))

    test_indices = []
    for subj in test_subjects:
        test_indices.extend(test_dataset.subject_to_indices.get(subj, []))

    # Create subsets
    train_subset = Subset(train_dataset, train_indices)
    test_subset = Subset(test_dataset, test_indices)

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                               pin_memory=True,
                              drop_last=True, persistent_workers=num_workers>0,
                              prefetch_factor=2 if num_workers>0 else None)

    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                              pin_memory=True,
                              drop_last=True, persistent_workers=num_workers > 0,
                              prefetch_factor=2 if num_workers > 0 else None)

    return train_loader, test_loader

if __name__ == '__main__':

    csv_path = "/data/rppg_23_mmpd_video_nt_lab/processed/DataFileLists/MMPD_SizeW72_SizeH72_ClipLength180_DataTypeRaw_DataAugNone_LabelTypeRaw_Crop_faceTrue_BackendY5F_Large_boxFalse_Large_size1.5_Dyamic_DetTrue_det_len1_Median_face_boxFalse_PSEUDO_LABELFalse_0.0_1.0.csv"
    start_time = time.time()
    dataset = OptimisedMMPDDataset(csv_file_path = csv_path, sampling_mode = 'sliding')
    dataset_creation_time = time.time() - start_time
    print(dataset_creation_time)
    print(len(dataset))

    if len(dataset) > 0:
        frames, bvp = dataset[0]
        print(frames.shape, bvp.shape)























