"""Trainer for TSCAN."""

import logging
import os
from collections import OrderedDict

import numpy as np
import torch
import torch.optim as optim
from evaluation.metrics import calculate_metrics
from neural_methods.loss.NegPearsonLoss import Neg_Pearson
from neural_methods.model.TS_CAN import TSCAN
from neural_methods.trainer.BaseTrainer import BaseTrainer
from tqdm import tqdm
import lightning.pytorch as pl


class TscanTrainer(pl.LightningModule):

    def __init__(self, config, data_loader):
        """Inits parameters from args and the writer for TensorboardX."""
        super().__init__()
        #self.device = torch.device(config.DEVICE)
        self.frame_depth = config.MODEL.TSCAN.FRAME_DEPTH
        self.max_epoch_num = config.TRAIN.EPOCHS
        self.model_dir = config.MODEL.MODEL_DIR
        self.model_file_name = config.TRAIN.MODEL_FILE_NAME
        self.batch_size = config.TRAIN.BATCH_SIZE
        self.num_of_gpu = config.NUM_OF_GPU_TRAIN
        self.base_len = self.num_of_gpu * self.frame_depth
        self.chunk_len = config.TRAIN.DATA.PREPROCESS.CHUNK_LENGTH
        self.test_chunk_len = config.TEST.DATA.PREPROCESS.CHUNK_LENGTH
        self.config = config
        self.min_valid_loss = None
        self.best_epoch = 0
        self.lr = config.TRAIN.LR
        self.epochs = config.TRAIN.EPOCHS
        self.predictions = dict()
        self.labels = dict()
        # self.save_hyperparameters()


        if config.TOOLBOX_MODE == "train_and_test" or config.TOOLBOX_MODE == "LOO" or config.TOOLBOX_MODE == "LOO_test":
            self.model = TSCAN(frame_depth=self.frame_depth, img_size=config.TRAIN.DATA.PREPROCESS.RESIZE.H).to(self.device)
            # self.model = torch.nn.DataParallel(self.model, device_ids=list(range(config.NUM_OF_GPU_TRAIN)))

            self.num_train_batches = len(data_loader["train"])
            self.criterion = torch.nn.MSELoss()
        elif config.TOOLBOX_MODE == "only_test":
            self.model = TSCAN(frame_depth=self.frame_depth, img_size=config.TEST.DATA.PREPROCESS.RESIZE.H).to(self.device)
            # self.model = torch.nn.DataParallel(self.model, device_ids=list(range(config.NUM_OF_GPU_TRAIN)))
        else:
            raise ValueError("TS-CAN trainer initialized in incorrect toolbox mode!")

    def training_step(self, batch, batch_idx):
        """Training routine for model"""
        if batch is None:
            raise ValueError("No data for train")

        running_loss = 0.0
        train_loss = []
        # Model Training
        data, labels = batch[0].to(
            self.device), batch[1].to(self.device)
        N, D, C, H, W = data.shape
        data = data.view(N * D, C, H, W)
        labels = labels.view(-1, 1)
        data = data[:(N * D) // self.base_len * self.base_len]
        labels = labels[:(N * D) // self.base_len * self.base_len]
        pred_ppg = self.model(data)
        loss = self.criterion(pred_ppg, labels)
        running_loss += loss.item()
        self.log("train_loss", loss, on_step=True, on_epoch=True, batch_size=self.config.TRAIN.BATCH_SIZE, sync_dist=True )
        # self.logger.log_metrics({"train_loss" : loss, }, self.current_epoch)
        #TODO: update the validaiton loop according to the use last epoch stratergy
        # if not self.config.TEST.USE_LAST_EPOCH:
        #     valid_loss = self.valid(data_loader)
        #     print('validation loss: ', valid_loss)
        #     if self.min_valid_loss is None:
        #         self.min_valid_loss = valid_loss
        #         self.best_epoch = epoch
        #         print("Update best model! Best epoch: {}".format(self.best_epoch))
        #     elif (valid_loss < self.min_valid_loss):
        #         self.min_valid_loss = valid_loss
        #         self.best_epoch = epoch
        #         print("Update best model! Best epoch: {}".format(self.best_epoch))
        # if not self.config.TEST.USE_LAST_EPOCH:
        #     print("best trained epoch: {}, min_val_loss: {}".format(self.best_epoch, self.min_valid_loss))
        return loss


    def validation_step(self, batch, batch_idx):
        """ Model evaluation on the validation dataset."""
        if batch is None:
            raise ValueError("No data for valid")

        data_valid, labels_valid = batch[0].to(
                    self.device), batch[1].to(self.device)
        N, D, C, H, W = data_valid.shape
        data_valid = data_valid.view(N * D, C, H, W)
        labels_valid = labels_valid.view(-1, 1)
        data_valid = data_valid[:(N * D) // self.base_len * self.base_len]
        labels_valid = labels_valid[:(N * D) // self.base_len * self.base_len]
        pred_ppg_valid = self.model(data_valid)
        loss = self.criterion(pred_ppg_valid, labels_valid)
        # self.logger.log_metrics({"val_loss" : loss, }, self.current_epoch)
        # self.log("val_loss", loss, on_step=True)
        self.log("val_loss", loss, batch_size=self.config.TRAIN.BATCH_SIZE, on_epoch=True, on_step=True)
        return loss

    def test_step(self, batch, batch_idx):
        """ Model evaluation on the testing dataset."""
        if batch is None:
            raise ValueError("No data for test")

        # self.logger.log_metrics({"testing":1})
        #TODO: check how to pass the model trained based on the condition
        # if self.config.TOOLBOX_MODE == "only_test":
        #     if not os.path.exists(self.config.INFERENCE.MODEL_PATH):
        #         raise ValueError("Inference model path error! Please check INFERENCE.MODEL_PATH in your yaml.")
        #     self.model.load_state_dict(torch.load(self.config.INFERENCE.MODEL_PATH))
        #     print("Testing uses pretrained model!")
        # else:
        #     if self.config.test.use_last_epoch:
        #         last_epoch_model_path = os.path.join(
        #         self.model_dir, self.model_file_name + '_epoch' + str(self.max_epoch_num - 1) + '.pth')
        #         print("testing uses last epoch as non-pretrained model!")
        #         print(last_epoch_model_path)
        #         self.model.load_state_dict(torch.load(last_epoch_model_path))
        #     else:
        #         best_model_path = os.path.join(
        #             self.model_dir, self.model_file_name + '_Epoch' + str(self.best_epoch) + '.pth')
        #         print("Testing uses best epoch selected using model selection as non-pretrained model!")
        #         print(best_model_path)
        #         self.model.load_state_dict(torch.load(best_model_path))

        batch_size = batch[0].shape[0]
        data_test, labels_test = batch[0].to(
                    self.config.DEVICE), batch[1].to(self.config.DEVICE)
        N, D, C, H, W = data_test.shape
        data_test = data_test.view(N * D, C, H, W)
        labels_test = labels_test.view(-1, 1)
        data_test = data_test[:(N * D) // self.base_len * self.base_len]
        labels_test = labels_test[:(N * D) // self.base_len * self.base_len]
        pred_ppg_test = self.model(data_test)

        for idx in range(batch_size):
            subj_index = batch[2][idx]
            sort_index = int(batch[3][idx])
            if subj_index not in self.predictions.keys():
                self.predictions[subj_index] = dict()
                self.labels[subj_index] = dict()
            self.predictions[subj_index][sort_index] = pred_ppg_test[idx * self.test_chunk_len:(idx + 1) * self.test_chunk_len]
            self.labels[subj_index][sort_index] = labels_test[idx * self.test_chunk_len:(idx + 1) * self.test_chunk_len]

    def on_test_end(self) -> None:
        calculate_metrics(self.predictions, self.labels, self.config, self.logger)

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=0)

        # See more details on the OneCycleLR scheduler here: https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.OneCycleLR.html
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=self.lr, epochs=self.epochs, steps_per_epoch=self.num_train_batches)
        return [optimizer], scheduler

    def save_model(self, index):
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        model_path = os.path.join(
            self.model_dir, self.model_file_name + '_Epoch' + str(index) + '.pth')
        torch.save(self.model.state_dict(), model_path)
        print('Saved Model Path: ', model_path)
