"""PhysNet Trainer."""
import os

import numpy as np
import torch
import torch.optim as optim
from evaluation.metrics import calculate_metrics
from neural_methods.loss.PhysFormerLossComputer import TorchLossComputer
from neural_methods.model.PhysFormer import ViT_ST_ST_Compact3_TDC_gra_sharp
from neural_methods.loss.PhysNetNegPearsonLoss import Neg_Pearson
import lightning.pytorch as pl
from scipy.signal import welch
import math


class PhysFormerTrainer(pl.LightningModule):

    def __init__(self, config, data_loader):
        """Inits parameters from args and the writer for TensorboardX."""
        super().__init__()
        # self.device = torch.device(config.DEVICE)
        self.max_epoch_num = config.TRAIN.EPOCHS
        self.model_dir = config.MODEL.MODEL_DIR
        self.dropout_rate = config.MODEL.DROP_RATE
        self.patch_size = config.MODEL.PHYSFORMER.PATCH_SIZE
        self.dim = config.MODEL.PHYSFORMER.DIM
        self.ff_dim = config.MODEL.PHYSFORMER.FF_DIM
        self.num_heads = config.MODEL.PHYSFORMER.NUM_HEADS
        self.num_layers = config.MODEL.PHYSFORMER.NUM_LAYERS
        self.theta = config.MODEL.PHYSFORMER.THETA
        self.model_file_name = config.TRAIN.MODEL_FILE_NAME
        self.batch_size = config.TRAIN.BATCH_SIZE
        self.num_of_gpu = config.NUM_OF_GPU_TRAIN
        self.chunk_len = config.TRAIN.DATA.PREPROCESS.CHUNK_LENGTH
        self.frame_rate = config.TRAIN.DATA.FS
        self.config = config
        self.lr = config.TRAIN.LR
        self.min_valid_loss = None
        self.best_epoch = 0

        self.a_start = 1.0
        self.b_start = 1.0
        self.exp_a = 0.5  # Unused
        self.exp_b = 1.0

        self.loss_rPPG_avg = []
        self.loss_peak_avg = []
        self.loss_kl_avg_test = []
        self.loss_hr_mae = []

        self.predictions = dict()
        self.labels = dict()

        self.hrs = []

        if config.TOOLBOX_MODE == "train_and_test" or config.TOOLBOX_MODE == "LOO":
            self.model = ViT_ST_ST_Compact3_TDC_gra_sharp(
                image_size=(
                    self.chunk_len, config.TRAIN.DATA.PREPROCESS.RESIZE.H, config.TRAIN.DATA.PREPROCESS.RESIZE.W),
                patches=(self.patch_size,) * 3, dim=self.dim, ff_dim=self.ff_dim, num_heads=self.num_heads,
                num_layers=self.num_layers,
                dropout_rate=self.dropout_rate, theta=self.theta).to(self.device)
            self.num_train_batches = len(data_loader["train"])
            self.criterion_reg = torch.nn.MSELoss()
            self.criterion_L1loss = torch.nn.L1Loss()
            self.criterion_class = torch.nn.CrossEntropyLoss()
            self.criterion_Pearson = Neg_Pearson()

        elif config.TOOLBOX_MODE == "only_test":
            pass
        else:
            raise ValueError("PhysNet trainer initialized in incorrect toolbox mode!")

    def training_step(self, batch, batch_idx):
        """Training routine for model"""
        if batch is None:
            raise ValueError("No data for train")


        hr_old = torch.tensor([self.get_hr(i.cpu()) for i in batch[1]]).float().to(self.device)
        hr = torch.tensor([i for i in batch[-1]]).to(self.device)
        print(hr, hr_old)
        data, label = batch[0].float().to(self.device), batch[1].float().to(self.device)

        gra_sharp = 2.0
        rPPG, _, _, _ = self.model(data, gra_sharp)
        rPPG = (rPPG - torch.mean(rPPG, axis=-1).view(-1, 1)) / torch.std(rPPG, axis=-1).view(-1,
                                                                                              1)  # normalize
        loss_rPPG = self.criterion_Pearson(rPPG, label)

        fre_loss = 0.0
        kl_loss = 0.0
        train_mae = 0.0

        for bb in range(data.shape[0]):
            loss_distribution_kl, \
                fre_loss_temp, \
                train_mae_temp = TorchLossComputer.cross_entropy_power_spectrum_DLDL_softmax2(
                rPPG[bb],
                hr[bb],
                self.frame_rate,
                std=1.0
            )
            fre_loss = fre_loss + fre_loss_temp
            kl_loss = kl_loss + loss_distribution_kl
            train_mae = train_mae + train_mae_temp
        fre_loss /= data.shape[0]
        kl_loss /= data.shape[0]
        train_mae /= data.shape[0]

        if self.current_epoch > 10:
            a = 0.05
            b = 5.0
        else:
            a = self.a_start
            # exp ascend
            b = self.b_start * math.pow(self.exp_b, self.current_epoch / 10.0)

        loss = a * loss_rPPG + b * (fre_loss + kl_loss)

        n = data.size(0)
        self.loss_rPPG_avg.append(float(loss_rPPG.data))
        self.loss_peak_avg.append(float(fre_loss.data))
        self.loss_kl_avg_test.append(float(kl_loss.data))
        self.loss_hr_mae.append(float(train_mae))
        if batch_idx % 100 == 99:  # print every 100 mini-batches
            print(f'\nepoch:{self.current_epoch}, batch:{batch_idx+ 1}, total:{self.num_train_batches // self.batch_size}, '
                  f'lr:0.0001, sharp:{gra_sharp:.3f}, a:{a:.3f}, NegPearson:{np.mean(self.loss_rPPG_avg[-2000:]):.4f}, '
                  f'\nb:{b:.3f}, kl:{np.mean(self.loss_kl_avg_test[-2000:]):.3f}, fre_CEloss:{np.mean(self.loss_peak_avg[-2000:]):.3f}, '
                  f'hr_mae:{np.mean(self.loss_hr_mae[-2000:]):.3f}')

        self.log("train_loss", loss, on_step=True, on_epoch=True, batch_size=self.batch_size)
        # if not self.config.TEST.USE_LAST_EPOCH:
        #     valid_loss = self.valid(data_loader)
        #     print(f'Validation RMSE:{valid_loss:.3f}, batch:{idx + 1}')
        #     if self.min_valid_loss is None:
        #         self.min_valid_loss = valid_loss
        #         self.best_epoch = epoch
        #         print("Update best model! Best epoch: {}".format(self.best_epoch))
        #     elif (valid_loss < self.min_valid_loss):
        #         self.min_valid_loss = valid_loss
        #         self.best_epoch = epoch
        #         print("Update best model! Best epoch: {}".format(self.best_epoch))
        # if not self.config.TEST.USE_LAST_EPOCH:
        #     print("best trained epoch: {}, min_val_loss: {}".format(
        #         self.best_epoch, self.min_valid_loss))

        return loss

    def validation_step(self, batch, batch_idx):
        """ Runs the model on valid sets."""
        if batch is None:
            raise ValueError("No data for valid")
        data, label = batch[0].float().to(self.device), batch[1].float().to(self.device)
        gra_sharp = 2.0
        rPPG, _, _, _ = self.model(data, gra_sharp)
        rPPG = (rPPG - torch.mean(rPPG, axis=-1).view(-1, 1)) / torch.std(rPPG).view(-1, 1)
        for _1, _2 in zip(rPPG, label):
             self.hrs.append((self.get_hr(_1.cpu().detach().numpy()), self.get_hr(_2.cpu().detach().numpy())))

        loss_rPPG = self.criterion_Pearson(rPPG, label)
        self.log("val_loss", loss_rPPG, on_step=True ,on_epoch=True, batch_size=self.batch_size)

        return loss_rPPG

    def on_validation_epoch_end(self) -> None:
        RMSE = np.mean([(i - j) ** 2 for i, j in self.hrs]) ** 0.5
        self.log("val_rmse", RMSE,  on_epoch=True, batch_size=self.batch_size)
        self.hrs = []

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
        #             self.model_dir, self.model_file_name + '_Epoch' + str(self.max_epoch_num - 1) + '.pth')
        #         print("Testing uses last epoch as non-pretrained model!")
        #         print(last_epoch_model_path)
        #         self.model.load_state_dict(torch.load(last_epoch_model_path))
        #     else:
        #         best_model_path = os.path.join(
        #             self.model_dir, self.model_file_name + '_Epoch' + str(self.best_epoch) + '.pth')
        #         print("Testing uses best epoch selected using model selection as non-pretrained model!")
        #         print(best_model_path)
        #         self.model.load_state_dict(torch.load(best_model_path))

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

        batch_size = batch[0].shape[0]
        data, label = batch[0].to(
            self.config.DEVICE), batch[1].to(self.config.DEVICE)
        gra_sharp = 2.0
        pred_ppg_test, _, _, _ = self.model(data, gra_sharp)
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
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=0.00005)
        # See more details on the OneCycleLR scheduler here: https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.OneCycleLR.html
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
        return [optimizer], scheduler

    def save_model(self, index):
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        model_path = os.path.join(
            self.model_dir, self.model_file_name + '_Epoch' + str(index) + '.pth')
        torch.save(self.model.state_dict(), model_path)
        print('Saved Model Path: ', model_path)

    def get_hr(self, y, sr=30, min=30, max=180):
        p, q = welch(y, sr, nfft=1e5 / sr, nperseg=np.min((len(y) - 1, 256)))
        return p[(p > min / 60) & (p < max / 60)][np.argmax(q[(p > min / 60) & (p < max / 60)])] * 60
