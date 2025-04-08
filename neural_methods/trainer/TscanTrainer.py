"""Trainer for TSCAN."""

import logging
import os
from collections import OrderedDict

import numpy as np
import torch
import torch.optim as optim
from evaluation.metrics import calculate_metrics, calculate_metrics_epoch
from neural_methods.loss.NegPearsonLoss import Neg_Pearson
from neural_methods.model.TS_CAN import TSCAN
from neural_methods.trainer.BaseTrainer import BaseTrainer
from tqdm import tqdm
import lightning.pytorch as pl


class TscanTrainer(pl.LightningModule):

    def __init__(self, config, data_loader, dropout_rate1= 0.25 , dropout_rate2=0.5,
                 lr=None,
                 epochs = None,
                 batch_size = None,
                 weight_decay = None,
                 save_dir = None):
        """Inits parameters from args and the writer for TensorboardX."""
        super().__init__()
        #self.device = torch.device(config.DEVICE)
        self.frame_depth = config.MODEL.TSCAN.FRAME_DEPTH
        self.model_dir = config.MODEL.MODEL_DIR
        self.model_file_name = config.TRAIN.MODEL_FILE_NAME

        if batch_size == None:
            self.batch_size = config.TRAIN.BATCH_SIZE
        else:
            self.batch_size = batch_size

        self.num_of_gpu = config.NUM_OF_GPU_TRAIN
        self.base_len = self.num_of_gpu * self.frame_depth
        self.chunk_len = config.TRAIN.DATA.PREPROCESS.CHUNK_LENGTH
        self.test_chunk_len = config.TEST.DATA.PREPROCESS.CHUNK_LENGTH
        self.config = config
        self.min_valid_loss = None
        self.best_epoch = 0

        #added for ray tune to adjust the hyperparamters
        if lr == None:
            self.lr = config.TRAIN.LR
        else:
            self.lr = lr

        if epochs == None:
            self.epochs = config.TRAIN.EPOCHS
        else:
            self.epochs = epochs

        if weight_decay == None:
            self.weight_decay = 0.0
        else:
            self.weight_decay = weight_decay

        if save_dir == None:
            self.save_dir = config.TEST.OUT_SAVE_DIR
        else:
            self.save_dir = save_dir

        self.predictions = dict()
        self.labels = dict()


        self.beta1 = 0.9
        self.beta2 = 0.999

        self.drop_rate1 = dropout_rate1
        self.drop_rate2 = dropout_rate2

        # self.save_hyperparameters()

        if config.TOOLBOX_MODE == "train_and_test" or config.TOOLBOX_MODE == "LOO" or config.TOOLBOX_MODE == "LOO_test" or config.TOOLBOX_MODE == "ENRICH" or config.TOOLBOX_MODE == "train_and_test_enrich" or config.TOOLBOX_MODE=="RAY_LOO" or config.TOOLBOX_MODE=="RAY_LOO_TEST":
            self.model = TSCAN(frame_depth=self.frame_depth, img_size=config.TRAIN.DATA.PREPROCESS.RESIZE.H, dropout_rate1=self.drop_rate1, dropout_rate2=self.drop_rate2).to(self.device)
            # self.model = torch.nn.DataParallel(self.model, device_ids=list(range(config.NUM_OF_GPU_TRAIN)))

            self.num_train_batches = len(data_loader["train"])

            if config.MODEL.LOSS=="MSE":
                self.criterion = torch.nn.MSELoss()
            elif config.MODEL.LOSS == "NEGPEARSON":
                raise NotImplementedError
            else:
                raise NotImplementedError

        elif config.TOOLBOX_MODE == "only_test":
            # self.model = TSCAN(frame_depth=self.frame_depth, img_size=config.TEST.DATA.PREPROCESS.RESIZE.H).to(self.device)
            self.model = TSCAN(frame_depth=self.frame_depth, img_size=config.TRAIN.DATA.PREPROCESS.RESIZE.H, dropout_rate1=self.drop_rate1, dropout_rate2=self.drop_rate2).to(self.device)
            # self.model = torch.nn.DataParallel(self.model, device_ids=list(range(config.NUM_OF_GPU_TRAIN)))
        else:
            raise ValueError("TS-CAN trainer initialized in incorrect toolbox mode!")

    def training_step(self, batch, batch_idx):
        """Training routine for model"""
        if batch is None:
            raise ValueError("No data for train")

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
        self.log("train_loss", loss, on_step=True, on_epoch=True, batch_size=self.config.TRAIN.BATCH_SIZE, sync_dist=True )
        return loss

    def on_validation_epoch_start(self)-> None:
        self.predictions = dict()
        self.labels = dict()

    def on_test_epoch_start(self) -> None:
        self.predictions = dict()
        self.labels = dict()

    def validation_step(self, batch, batch_idx):
        """ Model evaluation on the validation dataset."""

        if batch is None:
            raise ValueError("No data for valid")

        batch_size = batch[0].shape[0]

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

        for idx in range(batch_size):
            subj_index = batch[2][idx]
            sort_index = int(batch[3][idx])
            if subj_index not in self.predictions.keys():
                self.predictions[subj_index] = dict()
                self.labels[subj_index] = dict()
            self.predictions[subj_index][sort_index] = pred_ppg_valid[
                                                       idx * self.test_chunk_len:(idx + 1) * self.test_chunk_len]
            self.labels[subj_index][sort_index] = labels_valid[
                                                      idx * self.test_chunk_len:(idx + 1) * self.test_chunk_len]

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
        if self.config.TEST.DATA.DATASET == "DST":
            #TODO Make dst processing for batch size > 1
            pass
        else:
            labels_test = labels_test[:(N * D) // self.base_len * self.base_len]
        pred_ppg_test = self.model(data_test)

        for idx in range(batch_size):
            subj_index = batch[2][idx]
            sort_index = int(batch[3][idx])
            if subj_index not in self.predictions.keys():
                self.predictions[subj_index] = dict()
                self.labels[subj_index] = dict()
            self.predictions[subj_index][sort_index] = pred_ppg_test[idx * self.test_chunk_len:(idx + 1) * self.test_chunk_len]
            if self.config.TEST.DATA.DATASET == "DST":
                #DST only works for batch size 1
                self.labels[subj_index][sort_index] = labels_test
            else:
                self.labels[subj_index][sort_index] = labels_test[idx * self.test_chunk_len:(idx + 1) * self.test_chunk_len]

    def on_test_end(self) -> None:
        prediction_dict = calculate_metrics(self.predictions, self.labels, self.config, self.logger, save_dir=self.save_dir)

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        # See more details on the OneCycleLR scheduler here: https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.OneCycleLR.html
        if self.config.MODEL.SCHEDULER == "OneCycle":
            print("Using OneCycle HR")
            print("number of steps", self.trainer.estimated_stepping_batches)
            print(" epcohs and batches", self.epochs, self.num_train_batches)
            print("000" * 100)
            scheduler = {
                "scheduler" : torch.optim.lr_scheduler.OneCycleLR(
                optimizer, max_lr = self.lr, total_steps=self.trainer.estimated_stepping_batches),
                "interval": "step"
            }

            return [optimizer], [scheduler]

        elif self.config.MODEL.SCHEDULER == "ReduceOnPlatue":
            raise NotImplementedError
            # return [optimizer], [scheduler]

        else:
            print("No scheduler used")
            return [optimizer]

    def save_model(self, index):
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        model_path = os.path.join(
            self.model_dir, self.model_file_name + '_Epoch' + str(index) + '.pth')
        torch.save(self.model.state_dict(), model_path)
        print('Saved Model Path: ', model_path)

    def load_model(self, path):
        self.model.load_state_dict(self._rename_module(torch.load(path, map_location=self.device )))
        print('Model Created!')

    def _rename_module(self, torch_dict):
        new_dict = OrderedDict()
        keys = torch_dict.keys()
        for key in keys:
            new_key = ".".join(key.split(".")[1:])
            new_dict[new_key] = torch_dict[key]
        return new_dict

    def on_validation_epoch_end(self)-> None:

        MAE, RMSE, MAPE, Pearson, SNR, _ = calculate_metrics_epoch(self.predictions, self.labels, self.config, self.logger)
        print("In validation_epoch_end")
        if self.config.MODEL.SCHEDULER == "OneCycle":
            self.log("lr-step", self.lr_schedulers().get_last_lr()[-1])
            self.log("lr-logged", self.lr)
        self.log("MAE", MAE)
        self.log("RMSE", RMSE)
        self.log("MAPE", MAPE)
        # self.log("Pearson", Pearson) Nans why ?
        self.log("SNR", SNR)

    def check_weights(self):
        for param in self.model.parameters():
            print(param.data)

    def const_init(self, fill=0.0):
        for name, param in self.model.named_parameters():
            param.data.fill_(fill)



