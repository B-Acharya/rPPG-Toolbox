"""PhysNet Trainer."""
import os
from collections import OrderedDict

import numpy as np
import torch
import torch.optim as optim
from evaluation.metrics import calculate_metrics
from neural_methods.loss.rPPGNetNegPearsonLoss import Neg_Pearson
from neural_methods.model.rPPGNet import rPPGNet
from neural_methods.trainer.BaseTrainer import BaseTrainer
from torch.autograd import Variable
from tqdm import tqdm
import lightning.pytorch as pl
import torch.nn.functional as F


class rPPGNetTrainer(pl.LightningModule):

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

        self.model = rPPGNet(
            frames=config.MODEL.PHYSNET.FRAME_NUM).to(self.device)  # [3, T, 128,128]

        if config.TOOLBOX_MODE == "train_and_test" or config.TOOLBOX_MODE == "LOO":
            self.num_train_batches = len(data_loader["train"])
            self.criterion = Neg_Pearson()
            self.skin_loss = torch.nn.BCELoss()

        elif config.TOOLBOX_MODE == "only_test":
            pass
        else:
            raise ValueError("PhysNet trainer initialized in incorrect toolbox mode!")

    def training_step(self, batch, batch_idx):
        """Training routine for model"""
        if batch is None:
            raise ValueError("No data for train")

        skin_seg_label = batch[0][:, 3, :, :, :]

        skin_seg_label = torch.nn.functional.interpolate(skin_seg_label,(64, 64) )

        skin_map, rPPG_aux, rPPG, rPPG_SA1, rPPG_SA2, rPPG_SA3, rPPG_SA4, x_visual6464, x_visual3232 = self.model(batch[0][:, 0:3, :, :, :].to(torch.float32).to(self.device))

        #ground truth bvp signal, orginally ecg signal was used
        BVP_label = batch[1].to(
                    torch.float32).to(self.device)

        rPPG = (rPPG - torch.mean(rPPG)) / torch.std(rPPG)  # normalize2
        rPPG_SA1 = (rPPG_SA1 - torch.mean(rPPG_SA1)) / torch.std(rPPG_SA1)  # normalize2
        rPPG_SA2 = (rPPG_SA2 - torch.mean(rPPG_SA2)) / torch.std(rPPG_SA2)  # normalize2
        rPPG_SA3 = (rPPG_SA3 - torch.mean(rPPG_SA3)) / torch.std(rPPG_SA3)  # normalize2
        rPPG_SA4 = (rPPG_SA4 - torch.mean(rPPG_SA4)) / torch.std(rPPG_SA4)  # normalize2
        rPPG_aux = (rPPG_aux - torch.mean(rPPG_aux)) / torch.std(rPPG_aux)  # normalize2

        loss_ecg = self.criterion(rPPG, BVP_label)
        loss_ecg1 = self.criterion(rPPG_SA1, BVP_label)
        loss_ecg2 = self.criterion(rPPG_SA2, BVP_label)
        loss_ecg3 = self.criterion(rPPG_SA3, BVP_label)
        loss_ecg4 = self.criterion(rPPG_SA4, BVP_label)
        loss_ecg_aux = self.criterion(rPPG_aux, BVP_label)

        # skin_seg_label = torch.nan_to_num(skin_seg_label, nan=0.0)
        if torch.isnan(skin_map).any():
            print("Nan in skinmap")
            skin_map = torch.nan_to_num(skin_map)

        if torch.isnan(skin_seg_label).any():
            print("Nan in skinseg")
            skin_seg_label = torch.nan_to_num(skin_seg_label)

        loss_skin = self.skin_loss(skin_map, skin_seg_label)

        loss = 0.1 * loss_skin + 0.5 * (loss_ecg1 + loss_ecg2 + loss_ecg3 + loss_ecg4 + loss_ecg_aux) + loss_ecg


        self.log("train_loss", loss, on_step=True, on_epoch=True, batch_size=self.batch_size, prog_bar=True)
        self.log("train_skin_loss", loss_skin, on_step=True, on_epoch=True, batch_size=self.batch_size, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        """ Runs the model on valid sets."""
        if batch is None:
            raise ValueError("No data for valid")

        skin_seg_label = batch[0][:, 3, :, :, :]

        #batch size
        skin_seg_label = torch.nn.functional.interpolate(skin_seg_label,(64, 64) )
        skin_map, rPPG_aux, rPPG, rPPG_SA1, rPPG_SA2, rPPG_SA3, rPPG_SA4, x_visual6464, x_visual3232 = self.model(batch[0][:, 0:3, :, :, :].to(torch.float32).to(self.device))

        #ground truth bvp signal, orginally ecg signal was used
        BVP_label = batch[1].to(
            torch.float32).to(self.device)

        rPPG = (rPPG - torch.mean(rPPG)) / torch.std(rPPG)  # normalize2
        rPPG_SA1 = (rPPG_SA1 - torch.mean(rPPG_SA1)) / torch.std(rPPG_SA1)  # normalize2
        rPPG_SA2 = (rPPG_SA2 - torch.mean(rPPG_SA2)) / torch.std(rPPG_SA2)  # normalize2
        rPPG_SA3 = (rPPG_SA3 - torch.mean(rPPG_SA3)) / torch.std(rPPG_SA3)  # normalize2
        rPPG_SA4 = (rPPG_SA4 - torch.mean(rPPG_SA4)) / torch.std(rPPG_SA4)  # normalize2
        rPPG_aux = (rPPG_aux - torch.mean(rPPG_aux)) / torch.std(rPPG_aux)  # normalize2

        loss_ecg = self.criterion(rPPG, BVP_label)
        loss_ecg1 = self.criterion(rPPG_SA1, BVP_label)
        loss_ecg2 = self.criterion(rPPG_SA2, BVP_label)
        loss_ecg3 = self.criterion(rPPG_SA3, BVP_label)
        loss_ecg4 = self.criterion(rPPG_SA4, BVP_label)
        loss_ecg_aux = self.criterion(rPPG_aux, BVP_label)

        skin_map = torch.nan_to_num(skin_map)
        loss_skin = self.skin_loss(skin_map, skin_seg_label)

        loss = 0.1 * loss_skin + 0.5 * (loss_ecg1 + loss_ecg2 + loss_ecg3 + loss_ecg4 + loss_ecg_aux) + loss_ecg


        self.log("val_loss", loss, on_step=True, on_epoch=True, batch_size=self.batch_size, prog_bar=True)
        self.log("val_skin_loss", loss_skin, on_step=True, on_epoch=True, batch_size=self.batch_size, prog_bar=True)

        return loss

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
        #pred_ppg_test, _, _, _ = self.model(data[:,0:3,:,:,:])
        skin_map, rPPG_aux, pred_ppg_test, rPPG_SA1, rPPG_SA2, rPPG_SA3, rPPG_SA4, x_visual6464, x_visual3232 = self.model(data[:, 0:3, :, :, :].to(torch.float32).to(self.device))
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
