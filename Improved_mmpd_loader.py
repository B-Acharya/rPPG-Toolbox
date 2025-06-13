import torch
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
import scipy.io as sio
from scipy import signal
import glob
import os
import cv2 as cv
from collections import defaultdict
import random


# This is the dataset class that loads the raw MMPD data directly from the .mat files with subject based splitting as it  is with the original MMPDdataloader. I didn't do the resising here but it will only take a couple of line changes.
class ImprovedMMPDDataset(Dataset):

    def __init__(self, data_path, frame_depth=64):

        self.frame_depth = frame_depth
        self.data_path = data_path
        self.mat_files = []
        # check which mat file belongs to which subject
        self.file_to_subject = {}

        # Finding all subject directories
        all_subject_items = glob.glob(os.path.join(self.data_path, 'subject*'))
        subject_dirs = [item for item in all_subject_items if os.path.isdir(item)]
        subject_dirs = sorted(subject_dirs)

        print(len(subject_dirs))
        # need all the files for a specific subject
        self.subject_to_files = defaultdict(list)

        for subject_dir in subject_dirs:
            subject_name = os.path.basename(subject_dir)
            subject_files = glob.glob(os.path.join(subject_dir, '*.mat'))

            if subject_files:
                for file_path in subject_files:
                    self.mat_files.append(file_path)
                    self.file_to_subject[file_path] = subject_name
                    self.subject_to_files[subject_name].append(file_path)

        self.subjects = list(self.subject_to_files.keys())

        print(f"Found {len(self.subjects)} subjects with data")
        print(f"Total {len(self.mat_files)} .mat files")

    def __len__(self):
        return len(self.mat_files)

    def get_subjects(self):
        ## Return the list of all subjects
        return self.subjects

    def get_subject_for_idx(self, idx):
        ##  subject ID for a given sample index
        return self.file_to_subject[self.mat_files[idx]]

    def preprocess_frames(self, frames):

        frames = frames.astype(np.float32)

        if frames.max() <= 1.0:
            frames = frames * 255.0

        enhanced_frames = np.zeros_like(frames)

        for i in range(frames.shape[0]):

            frame_uint8 = frames[i].astype(np.uint8)

            lab = cv.cvtColor(frame_uint8, cv.COLOR_RGB2Lab)
            l,a,b = cv.split(lab)

            clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            l = clahe.apply(l.astype(np.uint8))

            #merge and convert back

            enhanced = cv.merge([l,a,b])

            enhanced_frames[i] = cv.cvtColor(enhanced, cv.COLOR_LAB2RGB)

        enhanced_frames = enhanced_frames/255.0

        return enhanced_frames

    def preprocess_bvp(self, bvp):
        bvp_detrended = signal.detrend(bvp)

        fs = 30
        nyquist_freq = fs / 2
        low = 0.7 / nyquist_freq
        high = 4.0 / nyquist_freq

        try:
            b, a = signal.butter(4, [low, high], btype='band')
            bvp_filter = signal.filtfilt(b, a, bvp_detrended)
        except:
            bvp_filter = bvp_detrended

        mean_val = np.mean(bvp_filter)
        std_val = np.std(bvp_filter)
        # Fix the normalization
        if std_val > 0:
            bvp_normalized = (bvp_filter - mean_val) / (std_val + 1e-8)
        else:
            bvp_normalized = bvp_filter - mean_val

        return bvp_normalized

    def __getitem__(self, idx):
        try:
            mat_data = sio.loadmat(self.mat_files[idx])

            # (T, H, W, C)
            frames = np.array(mat_data['video'])
            # (T,)
            bvp = np.array(mat_data['GT_ppg']).T.reshape(-1)

            if frames.shape[0] < self.frame_depth:
                repetitions = int(np.ceil(self.frame_depth / frames.shape[0]))
                repeated_frames = np.tile(frames, (repetitions, 1, 1, 1))
                frames = repeated_frames[:self.frame_depth]

                repeated_bvp = np.tile(bvp, repetitions)
                bvp = repeated_bvp[:self.frame_depth]
            else:
                frames = frames[:self.frame_depth]
                bvp = bvp[:self.frame_depth]

            # Apply preprocessing (CLAHE enhancement)
            frames = self.preprocess_frames(frames)

            # Normalize BVP
            bvp = self.preprocess_bvp(bvp)

            # Transpose for PyTorch (C, T, H, W)
            frames = frames.transpose(3, 0, 1, 2)

            frames = torch.FloatTensor(frames)
            bvp = torch.FloatTensor(bvp)

            return frames, bvp

        except Exception as e:
            print(f"Error loading file {self.mat_files[idx]}: {str(e)}")
            frames = torch.zeros((3, self.frame_depth, 80, 60))
            bvp = torch.zeros(self.frame_depth)
            return frames, bvp


def create_subject_split_indices(dataset, train_ratio=0.8, random_seed=42):

    all_subjects = dataset.get_subjects()
    n_subjects = len(all_subjects)
    n_train_subjects = int(n_subjects * train_ratio)

    random.seed(random_seed)
    subjects_shuffled = all_subjects.copy()
    random.shuffle(subjects_shuffled)

    train_subjects = set(subjects_shuffled[:n_train_subjects])
    test_subjects = set(subjects_shuffled[n_train_subjects:])

    train_indices = []
    test_indices = []

    for idx in range(len(dataset)):
        subject = dataset.get_subject_for_idx(idx)
        if subject in train_subjects:
            train_indices.append(idx)
        else:
            test_indices.append(idx)


    return train_indices, test_indices


def prepare_mmpd_dataloaders(config, frame_depth=16, train_ratio=0.8, random_seed=42):


    full_dataset = ImprovedMMPDDataset(
        data_path=config.TRAIN.DATA.DATA_PATH,
        frame_depth=frame_depth
    )


    train_indices, test_indices = create_subject_split_indices(
        full_dataset,
        train_ratio=train_ratio,
        random_seed=random_seed
    )


    train_dataset = Subset(full_dataset, train_indices)
    test_dataset = Subset(full_dataset, test_indices)


    train_loader = DataLoader(
        train_dataset,
        batch_size=config.TRAIN.BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.INFERENCE.BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    return train_loader, test_loader