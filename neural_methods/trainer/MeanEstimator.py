"""Trainer for TSCAN."""

import os
from collections import OrderedDict

import numpy as np
import torch
import torch.optim as optim

from evaluation.metrics import calculate_metrics
import lightning.pytorch as pl
from neural_methods.trainer.BaseTrainer import BaseTrainer
from evaluation.post_process import calculate_HR


class MeanEstimator(BaseTrainer):

    def __init__(self, config, data_loader, logger):
        """Inits parameters from args and the writer for TensorboardX."""
        super().__init__()
        self.lr = 1e-4
        self.config = config
        self.frame_depth = config.MODEL.EFFICIENTPHYS.FRAME_DEPTH
        self.num_of_gpu = config.NUM_OF_GPU_TRAIN
        self.base_len = self.num_of_gpu * self.frame_depth
        self.chunk_len = config.TRAIN.DATA.PREPROCESS.CHUNK_LENGTH
        self.data_loader = data_loader
        self.logger = logger

    def find_mean_hr(self):
        HR = []
        with torch.no_grad():
            for _, train_batch in enumerate(self.data_loader['train']):
                _, labels_train = train_batch[0].to(
                    self.config.DEVICE), train_batch[1].to(self.config.DEVICE)
                N, D = labels_train.shape
                labels_train = labels_train.view(-1, 1)
                # Add one more frame for EfficientPhys since it does torch.diff for the input
                labels_train = labels_train[:(N * D) // self.base_len * self.base_len]
                HR.append(calculate_HR(labels_train.cpu().reshape(-1,), fs=self.config.TRAIN.DATA.FS ))
        self.HR_mean = np.mean(HR)
    def test_step(self ):
        """ Model evaluation on the testing dataset."""
        predictions = dict()
        labels = dict()
        with torch.no_grad():
            for _, test_batch in enumerate(self.data_loader['test']):
                batch_size = test_batch[0].shape[0]
                data_test, labels_test = test_batch[0].to(
                    self.config.DEVICE), test_batch[1].to(self.config.DEVICE)
                N, D, C, H, W = data_test.shape
                labels_test = labels_test.view(-1, 1)
                # Add one more frame for EfficientPhys since it does torch.diff for the input
                labels_test = labels_test[:(N * D) // self.base_len * self.base_len]
                for idx in range(batch_size):
                    subj_index = test_batch[2][idx]
                    sort_index = int(test_batch[3][idx])
                    if subj_index not in predictions.keys():
                        predictions[subj_index] = dict()
                        labels[subj_index] = dict()
                    predictions[subj_index][sort_index] = labels_test[idx * self.chunk_len:(idx + 1) * self.chunk_len]
                    labels[subj_index][sort_index] = labels_test[idx * self.chunk_len:(idx + 1) * self.chunk_len]

        calculate_metrics(predictions, labels, self.config, self.logger, mean_HR=self.HR_mean)
