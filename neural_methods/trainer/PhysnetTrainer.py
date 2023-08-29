"""PhysNet Trainer."""
import os
from collections import OrderedDict

import numpy as np
import torch
import torch.optim as optim
from evaluation.metrics import calculate_metrics
from neural_methods.loss.PhysNetNegPearsonLoss import Neg_Pearson
from neural_methods.model.PhysNet import PhysNet_padding_Encoder_Decoder_MAX
from neural_methods.trainer.BaseTrainer import BaseTrainer
from torch.autograd import Variable
from tqdm import tqdm
import lightning.pytorch as pl


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

        if config.TOOLBOX_MODE == "train_and_test":
            self.num_train_batches = len(data_loader["train"])
            self.loss_model = Neg_Pearson()
            # self.optimizer = optim.Adam(
            #     self.model.parameters(), lr=config.TRAIN.LR)
            # # See more details on the OneCycleLR scheduler here: https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.OneCycleLR.html
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

        running_loss = 0.0
        train_loss = []
        rPPG, x_visual, x_visual3232, x_visual1616 = self.model(
                    batch[0].to(torch.float32).to(self.device))
        BVP_label = batch[1].to(
                    torch.float32).to(self.device)
        rPPG = (rPPG - torch.mean(rPPG)) / torch.std(rPPG)  # normalize
        BVP_label = (BVP_label - torch.mean(BVP_label)) / \
                            torch.std(BVP_label)  # normalize
        loss = self.loss_model(rPPG, BVP_label)
        running_loss += loss.item()
        self.log("train_loss", loss, on_step=True, on_epoch=True)
        #     if not self.config.TEST.USE_LAST_EPOCH:
        #         valid_loss = self.valid(data_loader)
        #         print('validation loss: ', valid_loss)
        #         if self.min_valid_loss is None:
        #             self.min_valid_loss = valid_loss
        #             self.best_epoch = epoch
        #             print("Update best model! Best epoch: {}".format(self.best_epoch))
        #         elif (valid_loss < self.min_valid_loss):
        #             self.min_valid_loss = valid_loss
        #             self.best_epoch = epoch
        #             print("Update best model! Best epoch: {}".format(self.best_epoch))
        # if not self.config.TEST.USE_LAST_EPOCH:
        #     print("best trained epoch: {}, min_val_loss: {}".format(
        #         self.best_epoch, self.min_valid_loss))
        return loss

    def validation_step(self, batch, batch_idx):
        """ Runs the model on valid sets."""
        if batch is None:
            raise ValueError("No data for valid")

        valid_loss = []
        BVP_label = batch[1].to(
                    torch.float32).to(self.device)
        rPPG, x_visual, x_visual3232, x_visual1616 = self.model(
                    batch[0].to(torch.float32).to(self.device))
        rPPG = (rPPG - torch.mean(rPPG)) / torch.std(rPPG)  # normalize
        BVP_label = (BVP_label - torch.mean(BVP_label)) / \
                            torch.std(BVP_label)  # normalize
        loss_ecg = self.loss_model(rPPG, BVP_label)
        self.log("val_loss", loss_ecg, on_step=True, on_epoch=True)
        return np.mean(valid_loss)

    def test_step(self, batch, batch_idx):
        """ Runs the model on test sets."""
        if batch is None:
            raise ValueError("No data for test")
        

        # if self.config.TOOLBOX_MODE == "only_test":
        #     if not os.path.exists(self.config.INFERENCE.MODEL_PATH):
        #         raise ValueError("Inference model path error! Please check INFERENCE.MODEL_PATH in your yaml.")
        #     self.model.load_state_dict(torch.load(self.config.INFERENCE.MODEL_PATH))
        #     print("Testing uses pretrained model!")
        #     print(self.config.INFERENCE.MODEL_PATH)
        # else:
        #     if self.config.TEST.USE_LAST_EPOCH:
        #         last_epoch_model_path = os.path.join(
        #         self.model_dir, self.model_file_name + '_Epoch' + str(self.max_epoch_num - 1) + '.pth')
        #         print("Testing uses last epoch as non-pretrained model!")
        #         print(last_epoch_model_path)
        #         self.model.load_state_dict(torch.load(last_epoch_model_path))
        #     else:
        #         best_model_path = os.path.join(
        #             self.model_dir, self.model_file_name + '_Epoch' + str(self.best_epoch) + '.pth')
        #         print("Testing uses best epoch selected using model selection as non-pretrained model!")
        #         print(best_model_path)
        #         self.model.load_state_dict(torch.load(best_model_path))

        # self.model = self.model.to(self.config.DEVICE)
        # self.model.eval()
        # with torch.no_grad():
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

    def configure_optimizers(self):
        optimizer = optim.Adam(
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
