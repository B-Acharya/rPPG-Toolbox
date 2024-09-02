"""PhysNet Trainer."""
import os
from collections import OrderedDict

import numpy as np
import torch
import torch.optim as optim
from evaluation.metrics import calculate_metrics, calculate_metrics_epoch
from neural_methods.loss.PhysNetNegPearsonLoss import Neg_Pearson
from neural_methods.model.PhysNet import PhysNet_padding_Encoder_Decoder_MAX
from neural_methods.trainer.BaseTrainer import BaseTrainer
from torch.autograd import Variable
from tqdm import tqdm
import lightning.pytorch as pl
from lightning.pytorch.utilities import grad_norm



class PhysnetTrainer(pl.LightningModule):

    def __init__(self, config, data_loader):
        """Inits parameters from args and the writer for TensorboardX."""
        super().__init__()
        # self.device = torch.device(config.DEVICE)
        self.max_epoch_num = config.TRAIN.EPOCHS
        self.model_dir = config.MODEL.MODEL_DIR
        self.model_file_name = config.TRAIN.MODEL_FILE_NAME
        self.batch_size = config.TRAIN.BATCH_SIZE
        self.num_of_gpu = config.NUM_OF_GPU_TRAIN
        self.base_len = self.num_of_gpu
        self.config = config
        self.min_valid_loss = None
        self.best_epoch = 0
        self.lr = config.TRAIN.LR
        self.epochs = config.TRAIN.EPOCHS
        self.predictions = dict()
        self.labels = dict()

        self.model = PhysNet_padding_Encoder_Decoder_MAX(
            frames=config.MODEL.PHYSNET.FRAME_NUM).to(self.device)  # [3, T, 128,128]

        if config.TOOLBOX_MODE == "train_and_test" or config.TOOLBOX_MODE == "LOO" or config.TOOLBOX_MODE == "LOO_test" or config.TOOLBOX_MODE == "ENRICH":
            self.num_train_batches = len(data_loader["train"])

            if config.MODEL.PHYSNET.LOSS == "MSE":
                self.loss_model = torch.nn.MSELoss()
            elif config.MODEL.PHYSNET.LOSS == "NEGPEARSON":
                self.loss_model = Neg_Pearson()
            else:
                raise ValueError("Loss not supported")

            # Do we need a scheduler ?
            # self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            #     self.optimizer, max_lr=config.TRAIN.LR, epochs=config.TRAIN.EPOCHS, steps_per_epoch=self.num_train_batches)

        elif config.TOOLBOX_MODE == "only_test":
            pass
        else:
            raise ValueError("PhysNet trainer initialized in incorrect toolbox mode!")

    def training_step(self, batch, batch_idx):
        """Training routine for model"""
        if batch is None:
            raise ValueError("No data for train")

        if torch.isnan(batch[0]).any():
            print(batch)
            print("Error in input")
            raise RuntimeError

        rPPG, x_visual, x_visual3232, x_visual1616 = self.model(
                    batch[0].to(torch.float32).to(self.device))

        #checks to avoid Nans
        if torch.isnan(rPPG).any():
            print("rppg signal out has nan", torch.isnan(rPPG).any())
            print(batch)
            raise RuntimeError

        BVP_label = batch[1].to(
                    torch.float32).to(self.device)

        #checks to avoid Nans
        if torch.isnan(BVP_label).any():
            print("BVP signal is nan")
            raise RuntimeError

        rPPG = (rPPG - torch.mean(rPPG)) / torch.std(rPPG)  # normalize
        BVP_label = (BVP_label - torch.mean(BVP_label)) / \
                            torch.std(BVP_label)  # normalize
        loss = self.loss_model(rPPG, BVP_label)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=self.batch_size)

        return loss

    def on_validation_epoch_start(self)-> None:
        self.predictions = dict()
        self.labels = dict()

    def on_test_epoch_start(self) -> None:
        self.predictions = dict()
        self.labels = dict()

    def validation_step(self, batch, batch_idx):
        """ Runs the model on valid sets."""


        if batch is None:
            raise ValueError("No data for valid")

        batch_size = batch[0].shape[0]

        label = batch[1].to(
                    torch.float32).to(self.device)

        rPPG, x_visual, x_visual3232, x_visual1616 = self.model(
                    batch[0].to(torch.float32).to(self.device))

        rPPG_normalized = (rPPG - torch.mean(rPPG)) / torch.std(rPPG)  # normalize

        BVP_label = (label- torch.mean(label)) / \
                            torch.std(label)  # normalize

        # normalized loss
        loss_ecg = self.loss_model(rPPG_normalized, BVP_label)

        self.log("val_loss", loss_ecg, on_step=True, on_epoch=True, prog_bar=True, batch_size=self.batch_size)

        for idx in range(batch_size):
            subj_index = batch[2][idx]
            sort_index = int(batch[3][idx])

            if subj_index not in self.predictions.keys():
                self.predictions[subj_index] = dict()
                self.labels[subj_index] = dict()

            self.predictions[subj_index][sort_index] = rPPG[idx]
            self.labels[subj_index][sort_index] = label[idx]

        return loss_ecg

    def on_validation_epoch_end(self)-> None:
        MSE, RMSE, MAPE, Pearson, SNR = calculate_metrics_epoch(self.predictions, self.labels, self.config, self.logger)
        print("In validation_epoch_end")
        self.log("MSE", MSE)
        self.log("RMSE", RMSE)
        self.log("MAPE", MAPE)
        # self.log("Pearson", Pearson) Nans why ?
        self.log("SNR", SNR)

    def test_step(self, batch, batch_idx):
        """ Runs the model on test sets."""

        if batch is None:
            raise ValueError("No data for test")
        
        batch_size = batch[0].shape[0]
        data, label = batch[0].to(
                    self.config.DEVICE), batch[1].to(self.config.DEVICE)
        pred_ppg_test, _, _, _ = self.model(data)
        for idx in range(batch_size):
            subj_index = batch[2][idx]
            sort_index = int(batch[3][idx])
            if subj_index not in self.predictions.keys():
                self.predictions[subj_index] = dict()
                self.labels[subj_index] = dict()
            self.predictions[subj_index][sort_index] = pred_ppg_test[idx]
            self.labels[subj_index][sort_index] = label[idx]

    def on_test_end(self) -> None:
        calculate_metrics(self.predictions, self.labels, self.config, self.logger)

    # def on_before_optimizer_step(self, optimizer):
    #     # Compute the 2-norm for each layer
    #     # If using mixed precision, the gradients are already unscaled here
    #     norms = grad_norm(self.layer, norm_type=2)
    #     self.log_dict(norms, prog_bar=True)

    def configure_optimizers(self):
        optimizer = optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=0.0)

        # See more details on the OneCycleLR scheduler here: https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.OneCycleLR.html
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=self.lr, epochs=self.epochs, steps_per_epoch=self.num_train_batches)
        return [optimizer], scheduler
        # return optimizer

    def save_model(self, index):
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        model_path = os.path.join(
            self.model_dir, self.model_file_name + '_Epoch' + str(index) + '.pth')
        torch.save(self.model.state_dict(), model_path)
        print('Saved Model Path: ', model_path)
