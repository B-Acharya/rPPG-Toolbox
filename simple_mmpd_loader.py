import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import scipy.io as sio
import glob
import os


# This is the dataset class that loads the raw MMPD data directly from the .mat files with subject based splitting as it  is with the original MMPDdataloader. I didn't do the resising here but it will only take a couple of line changes.
class DirectMMPDDataset(Dataset):

    def __init__(self, data_path, frame_depth=16):

        self.frame_depth = frame_depth
        self.data_path = data_path


        self.mat_files = []
        subject_dirs = sorted(glob.glob(os.path.join(data_path, 'subject*')))
        for subject_dir in subject_dirs:
            subject_files = glob.glob(os.path.join(subject_dir, '*.mat'))
            self.mat_files.extend(subject_files)

        print(f"Found {len(self.mat_files)} samples from {len(subject_dirs)} subjects")

    def __len__(self):
        return len(self.mat_files)

    def __getitem__(self, idx):

        try:

            mat_data = sio.loadmat(self.mat_files[idx])


            # (T, H, W, C)
            frames = np.array(mat_data['video'])
            # (T,)
            bvp = np.array(mat_data['GT_ppg']).T.reshape(-1)


            if frames.shape[0] < self.frame_depth:
                print("This case existed!!!")

                repetitions = int(np.ceil(self.frame_depth / frames.shape[0]))
                repeated_frames = np.tile(frames, (repetitions,1,1,1))
                frames = repeated_frames[:self.frame_depth]

                repeated_bvp = np.tile(bvp,repetitions)
                bvp = repeated_bvp[:self.frame_depth]

            else:
                frames = frames[:self.frame_depth]
                bvp = bvp[:self.frame_depth]

            # Normalize frames to [0, 1]
            # frames = frames.astype(np.float32) / 255.0
            # frames = frames.astype(np.float32)

            # frames = (frames - frames.min()) / (frames.max() - frames.min() + 1e-6)
            frames = frames.astype(np.float32)
            # normalise the bvp signals
            bvp = (bvp - np.mean(bvp))/(np.std(bvp) + 1e-8)


            frames = frames.transpose(3, 0, 1, 2)


            frames = torch.FloatTensor(frames)
            bvp = torch.FloatTensor(bvp)

            return frames, bvp

        except Exception as e:
            print(f"Error loading file {self.mat_files[idx]}: {str(e)}")
            # Return a zero sample in case of error
            frames = torch.zeros((3, self.frame_depth, 80, 60))
            bvp = torch.zeros(self.frame_depth)
            return frames, bvp


def prepare_mmpd_dataloaders(config):

    full_dataset = DirectMMPDDataset(data_path=config.TRAIN.DATA.DATA_PATH, frame_depth=config.TRAIN.DATA.PREPROCESS.CHUNK_LENGTH)

    total_samples = len(full_dataset)
    train_size = int(total_samples * 0.8)
    test_size = total_samples - train_size

    print(f"Total samples: {total_samples}")
    print(f"Train samples: {train_size}({train_size/total_samples:.1%})")
    print(f"Test samples: {test_size}({test_size/total_samples:.1%})")



    train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])

    print(f"Actual train_dataset size: {len(train_dataset)}")
    print(f"Actual test_dataset size: {len(test_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=config.TRAIN.BATCH_SIZE, shuffle=True, num_workers=4,pin_memory=True)

    test_loader = DataLoader(test_dataset,batch_size=config.INFERENCE.BATCH_SIZE, shuffle=False, num_workers=4,pin_memory=True)

    return train_loader, test_loader
