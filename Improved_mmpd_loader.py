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

    def __init__(self, data_path, frame_depth=64, sample_frames = 180,n_chunks_per_segment = 4):



        self.frame_depth = frame_depth
        self.sample_frames = sample_frames
        self.n_chunks_per_segment = n_chunks_per_segment
        self.data_path = data_path


        #Store all chunks info

        self.chunks = []

        # Finding all subject directories
        all_subject_items = glob.glob(os.path.join(self.data_path, 'subject*'))
        subject_dirs = [item for item in all_subject_items if os.path.isdir(item)]
        subject_dirs = sorted(subject_dirs)

        total_videos = 0

        for subject_dir in subject_dirs:
            subject_name = os.path.basename(subject_dir)
            subject_files = glob.glob(os.path.join(subject_dir, '*.mat'))

            for file_path in subject_files:
                total_videos += 1
                video_chunks = self._create_chunks_for_video(file_path, subject_name)
                self.chunks.extend(video_chunks)

        self.subjects = list(set([c['subject'] for c in self.chunks]))

        print(f"\nDataset Statistics:")
        print(f"  Total videos: {total_videos}")
        print(f"  Total chunks: {len(self.chunks)}")


    def _create_equal_spacing_chunks(self,n_chunks):

        if n_chunks == 1:
            return [(0, self.frame_depth)]

        stride = (self.sample_frames - self.frame_depth)/ (n_chunks-1)

        chunks = []

        for i in range(n_chunks):
            start = int(i * stride)
            end = min(start + self.frame_depth, self.sample_frames)
            chunks.append((start, end))

        return chunks

    def _create_chunks_for_video(self, file_path, subject_name):

        chunks = []

        chunk_position = self._create_equal_spacing_chunks(self.n_chunks_per_segment)

        segments = [('random', 'random')]
        # segments = [('start', 0)]

        for seg_name, seg_start in segments:
            for chunk_idx,(chunk_start, chunk_end) in enumerate(chunk_position):
                chunks.append({
                    'path': file_path,
                    'subject': subject_name,
                    'segment_name': seg_name,
                    'segment_start': seg_start,
                    'chunk_idx': chunk_idx,
                    'chunk_start_in_segment': chunk_start,
                    'chunk_end_in_segment': chunk_end
                })

        return chunks

    def __len__(self):
        return len(self.chunks)

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
            chunk_info = self.chunks[idx]

            #load mat file
            mat_data = sio.loadmat(chunk_info['path'])
            all_frames = np.array(mat_data['video'])
            all_bvp = np.array(mat_data['GT_ppg']).T.reshape(-1)

            if chunk_info['segment_start'] == 'random':
                max_start = 1800 - self.sample_frames
                segment_start = np.random.randint(0,max_start+1)
            else:
                segment_start = chunk_info['segment_start']

            # Extract 180-frame segment
            segment_end = segment_start + self.sample_frames
            segment_frames = all_frames[segment_start:segment_end]
            segment_bvp = all_bvp[segment_start:segment_end]

            #Extract chunk from segment
            chunk_start = chunk_info['chunk_start_in_segment']
            chunk_end = chunk_info['chunk_end_in_segment']

            # Apply preprocessing (CLAHE enhancement)
            frames = segment_frames[chunk_start:chunk_end]
            bvp = segment_bvp[chunk_start:chunk_end]

            frames = self.preprocess_frames(frames)
            bvp = self.preprocess_bvp(bvp)

            # Transpose for PyTorch (C, T, H, W)
            frames = frames.transpose(3, 0, 1, 2)

            frames = torch.FloatTensor(frames)
            bvp = torch.FloatTensor(bvp)

            return frames, bvp

        except Exception as e:
            print(f"Error loading file {idx}: {str(e)}")
            frames = torch.zeros((3, self.frame_depth, 80, 60))
            bvp = torch.zeros(self.frame_depth)
            return frames, bvp

    def get_subjects(self):
        return self.subjects

    def get_subject_for_idx(self, idx):
        return self.chunks[idx]['subject']

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


def prepare_mmpd_dataloaders(config, frame_depth=64, train_ratio=0.8, random_seed=42, n_chunks=4):


    full_dataset = ImprovedMMPDDataset(
        data_path=config.TRAIN.DATA.DATA_PATH,
        frame_depth=frame_depth,
        sample_frames = 180,
        n_chunks_per_segment = n_chunks
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