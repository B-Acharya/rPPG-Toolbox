""" The main function of rPPG deep learning pipeline."""

import argparse
import random
import time
from comet_ml.integration.ray import CometTrainLoggerCallback, comet_worker_logger

import numpy as np
import pandas as pd
import torch
from config import get_config
from dataset import data_loader
from neural_methods import trainer
from unsupervised_methods.unsupervised_predictor import unsupervised_predict, unsupervised_HR_predict
from torch.utils.data import DataLoader
import lightning.pytorch as pl
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import CometLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.strategies import DDPStrategy
import matplotlib.pyplot as plt
import pathlib
from sklearn.model_selection import KFold, train_test_split
from pathlib import Path

from ray.tune.search import ConcurrencyLimiter
from ray.tune.search.optuna import OptunaSearch

from ray.tune.search.ax import AxSearch

RANDOM_SEED = 100
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# Create a general generator for use with the validation dataloader,
# the test dataloader, and the unsupervised dataloader
general_generator = torch.Generator()
general_generator.manual_seed(RANDOM_SEED)
# Create a training generator to isolate the train dataloader from
# other dataloaders and better control non-deterministic behavior
train_generator = torch.Generator()
train_generator.manual_seed(RANDOM_SEED)
torch.set_float32_matmul_precision("high")

from ray.air.integrations.comet import CometLoggerCallback
import ray

from ray.util import inspect_serializability
import threading

lock = threading.Lock()

#ray tune imports
from ray.train.torch import TorchTrainer
from ray.train import RunConfig, ScalingConfig, CheckpointConfig

from ray.train.lightning import (
    RayDDPStrategy,
    RayLightningEnvironment,
    RayTrainReportCallback,
    prepare_trainer,
)
from ray import tune
from ray.tune.schedulers import ASHAScheduler


#to get accurate stack
# import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"



def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def trail_name(trial):
    return f"{trial.trial_id}_{trial.experiment_tag}"

def get_participant_index(pathfile):
    "Pandas parsing for geting the index from the full path"
    if "CMBP" in pathfile:
        path = pathfile.split("/")[-1]
        index = path = path.split("_")[0]
        return index

    else:
        path = pathfile.split("/")[-1]
        if "v" in path:
            index = path.split("v")[0]
            return int(index)
        elif "_" in path:
            index = path.split("_")[0]
            if len(index) == 3:
                return int(index[0])
            else:
                return int(index[:2])

def create_path_list(listdf, train, test, i, j=0, outer=True):
    train_df = listdf[listdf['id'].isin(train)]
    test_df = listdf[listdf['id'].isin(test)]
    if outer:

        train_path = pathlib.Path(listdf["input_files"][0]).parent / f"fold_{i}_train.csv"
        test_path = pathlib.Path(listdf["input_files"][0]).parent / f"fold_{i}_test.csv"
        train_df.to_csv(str(train_path))
        test_df.to_csv(str(test_path))
    else:
        train_path = pathlib.Path(listdf["input_files"][0]).parent / f"fold_{i}_{j}_train.csv"
        test_path = pathlib.Path(listdf["input_files"][0]).parent / f"fold_{i}_{j}_valid.csv"
        train_df.to_csv(str(train_path))
        test_df.to_csv(str(test_path))
    return train_path, test_path

def add_args(parser):
    """Adds arguments for parser."""
    parser.add_argument('--config_file', required=False,
                        default="configs/train_configs/PURE_PURE_UBFC-rPPG_TSCAN_BASIC.yaml", type=str, help="The name of the model.")
    '''Neural Method Sample YAML LIST:
      SCAMPS_SCAMPS_UBFC-rPPG_TSCAN_BASIC.yaml
      SCAMPS_SCAMPS_UBFC-rPPG_DEEPPHYS_BASIC.yaml
      SCAMPS_SCAMPS_UBFC-rPPG_PHYSNET_BASIC.yaml
      SCAMPS_SCAMPS_PURE_DEEPPHYS_BASIC.yaml
      SCAMPS_SCAMPS_PURE_TSCAN_BASIC.yaml
      SCAMPS_SCAMPS_PURE_PHYSNET_BASIC.yaml
      PURE_PURE_UBFC-rPPG_TSCAN_BASIC.yaml
      PURE_PURE_UBFC-rPPG_DEEPPHYS_BASIC.yaml
      PURE_PURE_UBFC-rPPG_PHYSNET_BASIC.yaml
      PURE_PURE_MMPD_TSCAN_BASIC.yaml
      UBFC-rPPG_UBFC-rPPG_PURE_TSCAN_BASIC.yaml
      UBFC-rPPG_UBFC-rPPG_PURE_DEEPPHYS_BASIC.yaml
      UBFC-rPPG_UBFC-rPPG_PURE_PHYSNET_BASIC.yaml
      MMPD_MMPD_UBFC-rPPG_TSCAN_BASIC.yaml
    Unsupervised Method Sample YAML LIST:
      PURE_UNSUPERVISED.yaml
      UBFC-rPPG_UNSUPERVISED.yaml
    '''
    return parser
class Struct(object):
    #stackoverflow code here to convert a dict to an object
    def __init__(self, d):
        for k, v in d.items():
            if isinstance(k, (list, tuple)):
                setattr(self, k, [Struct(x) if isinstance(x, dict) else x for x in v])
            else:
                setattr(self, k, Struct(v) if isinstance(v, dict) else v)
def train_func(config_ray):
    """trains the model."""

    args = ray.train.get_context().get_metadata()

    config = Struct(args)

    print(config)

    print(config_ray)

    save_dir = ray.train.get_context().get_trial_dir()
    print(save_dir)

    data_loader_dict = dict() # dictionary of data loaders

    if config.TRAIN.DATA.DATASET == "CMBP":
        loader = data_loader.CMBPLoader.CMBPLoader
        basepath = pathlib.Path(config.TRAIN.DATA.DATA_PATH)
    else:
        raise NotImplementedError

    train_data = loader(
        name="train",
        data_path=config.TRAIN.DATA.DATA_PATH,
        config_data=config.TRAIN.DATA,
        model=config.MODEL.NAME
    )

    data_loader_dict['train'] = DataLoader(
            dataset=train_data,
            num_workers=2,
            batch_size=config.TRAIN.BATCH_SIZE,
            # batch_size=config_ray['batch_size'],
            shuffle=True,
            worker_init_fn=seed_worker,
            # generator=train_generator,
    )

    valid_data = loader(
            name="valid",
            data_path=config.VALID.DATA.DATA_PATH,
            config_data=config.VALID.DATA,
            model=config.MODEL.NAME
        )

    data_loader_dict["valid"] = DataLoader(
            dataset=valid_data,
            num_workers=2,
            batch_size=config.TRAIN.BATCH_SIZE,  # batch size for val is the same as train
            # batch_size=config_ray['batch_size'],
            shuffle=False,
            worker_init_fn=seed_worker,
            # generator=general_generator,
        )

    test_data = loader(
        name="test",
        data_path=config.TEST.DATA.DATA_PATH,
        config_data=config.TEST.DATA,
        model=config.MODEL.NAME
    )

    data_loader_dict["test"] = DataLoader(
        dataset=test_data,
        num_workers=2,
        batch_size=config.INFERENCE.BATCH_SIZE,
        # batch_size=config_ray['batch_size'],
        shuffle=False,
        worker_init_fn=seed_worker,
        # generator=general_generator,
    )

    # from torch_lr_finder import LRFinder, TrainDataLoaderIter, ValDataLoaderIter
    # from neural_methods.model.PhysNet import PhysNet_padding_Encoder_Decoder_MAX
    #
    # class CustomTrainIter(TrainDataLoaderIter):
    #     def inputs_labels_from_batch(self, batch_data):
    #         return batch_data[0], batch_data[1]
    #
    # class CustomValIter(ValDataLoaderIter):
    #     def inputs_labels_from_batch(self, batch_data):
    #         return batch_data[0], batch_data[1]
    #
    # model = PhysNet_padding_Encoder_Decoder_MAX(
    #     frames=config.MODEL.PHYSNET.FRAME_NUM)  # [3, T, 128,128]
    #
    # loss = torch.nn.MSELoss()
    # train_iter = CustomTrainIter(data_loader_dict['train'])
    # val_iter = CustomValIter(data_loader_dict['valid'])
    # print(train_iter)
    # print(val_iter)
    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-2)
    # lr_finder = LRFinder(model, optimizer, loss, device="cuda")
    # # lr_finder.range_test(train_iter, val_iter, end_lr=1, num_iter=100, step_mode="linear")
    # lr_finder.range_test(train_iter,  end_lr=10, num_iter=100)
    # lr_finder.plot(log_lr=True)
    # plt.savefig("/homes/bacharya/lr-test.png")
    # # plt.show()

    if config.MODEL.NAME == "Physnet":
        model_trainer = trainer.PhysnetTrainer.PhysnetTrainer(config, data_loader_dict)
    elif config.MODEL.NAME == "iBVPNet":
        model_trainer = trainer.iBVPNetTrainer.iBVPNetTrainer(config, data_loader_dict)
    elif config.MODEL.NAME == "Tscan":
        model_trainer = trainer.TscanTrainer.TscanTrainer(config,
                                                          data_loader_dict,
                                                          dropout_rate1=config_ray['dropout_rate1'],
                                                          dropout_rate2=config_ray['dropout_rate2'],
                                                          lr=config_ray['lr'],
                                                          epochs=config_ray['epochs'],
                                                          batch_size=config_ray['batch_size'],
                                                          weight_decay=config_ray['weight_decay'],
                                                          save_dir = save_dir)
    elif config.MODEL.NAME == "EfficientPhys":
        model_trainer = trainer.EfficientPhysTrainer.EfficientPhysTrainer(config, data_loader_dict)
    elif config.MODEL.NAME == 'DeepPhys':
        model_trainer = trainer.DeepPhysTrainer.DeepPhysTrainer(config, data_loader_dict)
    elif config.MODEL.NAME == 'BigSmall':
        model_trainer = trainer.BigSmallTrainer.BigSmallTrainer(config, data_loader_dict)
    elif config.MODEL.NAME == 'rPPGNet':
        model_trainer = trainer.rPPGNetTrainer.rPPGNetTrainer(config, data_loader_dict)
    elif config.MODEL.NAME == 'PhysFormer':
        model_trainer = trainer.PhysFormerTrainer.PhysFormerTrainer(config, data_loader_dict)
    elif config.MODEL.NAME == 'Mean':
        model_trainer = trainer.MeanEstimator.MeanEstimator(config, data_loader_dict, comet_logger)
        model_trainer.find_mean_hr()
        model_trainer.test_step()
        return True
    else:
        raise ValueError('your model is not supported  yet!')


    #update hyperparameters from search space
    # Additional parameters need to be added here
    #Now set above , when the class is instanciated
    model_trainer.max_epoch_num = config_ray['epochs']
    # model_trainer.epochs = config_ray['epochs']
    # model_trainer.batch_size = config_ray['batch_size']
    # model_trainer.weight_decay = config_ray['weight_decay']

    #create checkpoint every epoch and track the validation loss
    if config.TEST.USE_LAST_EPOCH:
        checkpoint_callback = ModelCheckpoint(dirpath=config.MODEL.MODEL_DIR,
                                              filename=f"{config.TRAIN.MODEL_FILE_NAME}" + "_{epoch}",
                                              save_last=True,
                                              )
    else:
        checkpoint_callback = ModelCheckpoint(every_n_epochs=1,
                                              save_top_k=2,
                                              dirpath=config.MODEL.MODEL_DIR,
                                              filename=f"{config.TRAIN.MODEL_FILE_NAME}"+"_{epoch}",
                                              monitor="val_loss_epoch",
                                              mode='min')
    lr_monitor = LearningRateMonitor(logging_interval='step')
    if config.EARLY_STOPPING.TRAIN:
        early_stop_callback = EarlyStopping(monitor="train_loss_epoch", min_delta=config.EARLY_STOPPING.DELTA, patience=config.EARLY_STOPPING.PATIENCE, verbose=True,
                                                mode="min")
        # comet_logger.add_tag("early_stopping")
        trainer_light= pl.Trainer(accelerator='gpu', default_root_dir=config.MODEL.MODEL_DIR, callbacks=[early_stop_callback, checkpoint_callback, lr_monitor], logger=comet_logger, max_epochs=config.TRAIN.EPOCHS)
        # trainer_light = pl.Trainer(limit_train_batches=0.1, limit_val_batches=0.5,
        #                            default_root_dir=config.MODEL.MODEL_DIR, logger=comet_logger,
        #                            max_epochs=config.TRAIN.EPOCHS, callbacks=[early_stop_callback,lr_monitor,checkpoint_callback])
    elif config.EARLY_STOPPING.VALID:
        print("In early stopping valid function")
        early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=config.EARLY_STOPPING.DELTA, patience=config.EARLY_STOPPING.PATIENCE, verbose=True,
                                            mode="min")
        trainer_light = pl.Trainer(accelerator='auto', devices="auto", strategy=RayDDPStrategy(),
                                   default_root_dir=config.MODEL.MODEL_DIR,
                                   max_epochs=config_ray["epochs"],
                                   callbacks=[RayTrainReportCallback(), early_stop_callback, lr_monitor],
                                   plugins=[RayLightningEnvironment()],
                                   enable_progress_bar=False)

        trainer_light = prepare_trainer(trainer_light)

    else:
        if not config.CLUSTER :
            # trainer_light = pl.Trainer(limit_train_batches=0.1, limit_val_batches=0.5, default_root_dir=config.MODEL.MODEL_DIR, logger=comet_logger, max_epochs=config.TRAIN.EPOCHS, callbacks=[checkpoint_callback])
            #trainer_light = pl.Trainer(default_root_dir=config.MODEL.MODEL_DIR, logger=comet_logger, max_epochs=config.TRAIN.EPOCHS, callbacks=[checkpoint_callback, lr_monitor], detect_anomaly=True)
            #trainer_light = pl.Trainer(default_root_dir=config.MODEL.MODEL_DIR, logger=comet_logger, max_epochs=config.TRAIN.EPOCHS, callbacks=[checkpoint_callback, lr_monitor], gradient_clip_val=0.5, gradient_clip_algorithm='value', detect_anomaly=True)
            print("Running on pycharm")
            trainer_light = pl.Trainer(accelerator='auto', devices="auto",
                                        strategy=RayDDPStrategy(),
                                   default_root_dir=config.MODEL.MODEL_DIR,
                                   # max_epochs=config.TRAIN.EPOCHS,
                                   max_epochs=config_ray["epochs"],
                                   # logger=logger,
                                   callbacks=[RayTrainReportCallback(), lr_monitor], plugins=[RayLightningEnvironment()],
                                   enable_progress_bar=False)

            trainer_light = prepare_trainer(trainer_light)
        # slurm settings
        else:
            if config.MODEL.NAME == "rPPGNet":
                # import os
                # os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
                # ddp_find was added for rppgnet
                # trainer_light = pl.Trainer(devices=1, num_nodes=1,strategy=DDPStrategy(find_unused_parameters=True), accelerator='gpu', default_root_dir=config.MODEL.MODEL_DIR, logger=comet_logger, max_epochs=config.TRAIN.EPOCHS, callbacks=[checkpoint_callback])
                trainer_light = pl.Trainer(accelerator='gpu',default_root_dir=config.MODEL.MODEL_DIR, logger=comet_logger,
                                           max_epochs=config.TRAIN.EPOCHS, callbacks=[checkpoint_callback, lr_monitor])
            else:
                #trainer_light = pl.Trainer(devices=1, num_nodes=1, strategy='ddp', accelerator='gpu', default_root_dir=config.MODEL.MODEL_DIR, logger=comet_logger, max_epochs=config.TRAIN.EPOCHS, callbacks=[checkpoint_callback])
                # trainer_light = pl.Trainer(accelerator='gpu', default_root_dir=config.MODEL.MODEL_DIR, logger=comet_logger, max_epochs=config.TRAIN.EPOCHS, callbacks=[checkpoint_callback, lr_monitor])

                # ray tune added here
                trainer_light = pl.Trainer(accelerator='auto', devices="auto", strategy= RayDDPStrategy(),
                                           default_root_dir=config.MODEL.MODEL_DIR,
                                           # max_epochs=config.TRAIN.EPOCHS,
                                           max_epochs=config_ray["epochs"],
                                           # logger=logger,
                                           callbacks=[RayTrainReportCallback(), lr_monitor ], plugins=[RayLightningEnvironment()],
                                           enable_progress_bar=False)

                trainer_light = prepare_trainer(trainer_light)

    # Find the best lr
    if config.TRAIN.USE_LR_FINDER:
        # finding the best lr
        from lightning.pytorch.tuner.tuning import Tuner
        tuner = Tuner(trainer_light)
        if config.TEST.USE_LAST_EPOCH:
            lr_finder = tuner.lr_find(model_trainer, train_dataloaders=data_loader_dict['train'])
        else:
            # lr_finder = tuner.lr_find(model_trainer, train_dataloaders=data_loader_dict['train'],
            #                       val_dataloaders=data_loader_dict['valid'])
            lr_finder = tuner.lr_find(model_trainer, train_dataloaders=data_loader_dict['train'],
                                      val_dataloaders=data_loader_dict['test'])
        fig = lr_finder.plot(suggest=True, show=True )
        plt.savefig("/homes/bacharya/lr_learningrate.png")
        plt.close()
        # comet_logger.log_figure(figure=fig, figure_name="lr-curve")

        new_lr = lr_finder.suggestion()

        model_trainer.lr = new_lr

        # comet_logger.add_tags(
        #     ['lr-finder'])

        print(model_trainer.lr)

    else:
        print("Using lr:", model_trainer.lr)
        print("Using batch:", model_trainer.batch_size)
        print("Using epochs:", model_trainer.max_epoch_num)
        print("Weight decay set to :", model_trainer.weight_decay)

    # if config.MODEL.NAME == "MEAN":
    #     trainer_light.test( dataloaders=data_loader_dict['test'])

    if config.TEST.USE_LAST_EPOCH:
        trainer_light.fit(model_trainer, data_loader_dict['train'])
        comet_logger.add_tag("last_epoch")
        trainer_light.test(ckpt_path="last", dataloaders=data_loader_dict['test'])

    else:
        trainer_light.fit(model_trainer, train_dataloaders=data_loader_dict['train'],  val_dataloaders=data_loader_dict['valid'])
        # trainer_light.fit(model_trainer, data_loader_dict['train'], data_loader_dict['test'])
        # comet_logger.add_tag("best_epoch")
        trainer_light.test(ckpt_path="best", dataloaders=data_loader_dict['test'])

    # #load and test the model on the best epoch / last epoch
    # experiment_key = comet_logger.experiment.get_key()
    # comet_logger = cometlogger(experiment_key=experiment_key)
    # trainer_light = pl.trainer(logger=comet_logger)

def LOO(comet_logger, config, data_loader_dict, outer):
    """trains the model."""


    if config.MODEL.NAME == "Physnet":
        model_trainer = trainer.PhysnetTrainer.PhysnetTrainer(config, data_loader_dict)
    elif config.MODEL.NAME == "iBVPNet":
        model_trainer = trainer.iBVPNetTrainer.iBVPNetTrainer(config, data_loader_dict)
    elif config.MODEL.NAME == "Tscan":
        model_trainer = trainer.TscanTrainer.TscanTrainer(config, data_loader_dict)
    elif config.MODEL.NAME == "EfficientPhys":
        model_trainer = trainer.EfficientPhysTrainer.EfficientPhysTrainer(config, data_loader_dict)
    elif config.MODEL.NAME == 'DeepPhys':
        model_trainer = trainer.DeepPhysTrainer.DeepPhysTrainer(config, data_loader_dict)
    elif config.MODEL.NAME == 'BigSmall':
        model_trainer = trainer.BigSmallTrainer.BigSmallTrainer(config, data_loader_dict)
    elif config.MODEL.NAME == 'rPPGNet':
        model_trainer = trainer.rPPGNetTrainer.rPPGNetTrainer(config, data_loader_dict)
    elif config.MODEL.NAME == 'PhysFormer':
        model_trainer = trainer.PhysFormerTrainer.PhysFormerTrainer(config, data_loader_dict)
    elif config.MODEL.NAME == 'Mean':
        model_trainer = trainer.MeanEstimator.MeanEstimator(config, data_loader_dict, comet_logger)
        model_trainer.find_mean_hr()
        model_trainer.test_step()
        return True
    else:
        raise ValueError('your model is not supported  yet!')


        #create checkpoint every epoch and track the validation loss
    if config.TEST.USE_LAST_EPOCH:
        checkpoint_callback = ModelCheckpoint(dirpath=config.MODEL.MODEL_DIR,
                                              filename=f"{config.TRAIN.MODEL_FILE_NAME}_{outer}" + "_{epoch}",
                                              save_last=True,
                                              )
    else:
        checkpoint_callback = ModelCheckpoint(every_n_epochs=1,
                                              save_top_k=2,
                                              dirpath=config.MODEL.MODEL_DIR,
                                              filename=f"{config.TRAIN.MODEL_FILE_NAME}_{outer}"+"_{epoch}",
                                              monitor="val_loss_epoch",
                                              mode='min')
    lr_monitor = LearningRateMonitor(logging_interval='step')
    if config.EARLY_STOPPING.TRAIN:
        early_stop_callback = EarlyStopping(monitor="train_loss_epoch", min_delta=config.EARLY_STOPPING.DELTA, patience=config.EARLY_STOPPING.PATIENCE, verbose=True,
                                                mode="min")
        comet_logger.experiment.add_tag("early_stopping")
        trainer_light= pl.Trainer(accelerator='gpu', default_root_dir=config.MODEL.MODEL_DIR, callbacks=[early_stop_callback, checkpoint_callback, lr_monitor], logger=comet_logger, max_epochs=config.TRAIN.EPOCHS)
        # trainer_light = pl.Trainer(limit_train_batches=0.1, limit_val_batches=0.5,
        #                            default_root_dir=config.MODEL.MODEL_DIR, logger=comet_logger,
        #                            max_epochs=config.TRAIN.EPOCHS, callbacks=[early_stop_callback,lr_monitor,checkpoint_callback])
    elif config.EARLY_STOPPING.VALID:
        early_stop_callback = EarlyStopping(monitor="val_loss_epoch", min_delta=config.EARLY_STOPPING.DELTA, patience=config.EARLY_STOPPING.PATIENCE, verbose=True,
                                            mode="min")
        comet_logger.experiment.add_tag("early_stopping")
        trainer_light= pl.Trainer(accelerator='gpu', default_root_dir=config.MODEL.MODEL_DIR, callbacks=[early_stop_callback, checkpoint_callback, lr_monitor], logger=comet_logger, max_epochs=config.TRAIN.EPOCHS)
    else:
        if not config.CLUSTER :
            trainer_light = pl.Trainer(limit_train_batches=0.1, limit_val_batches=0.5, default_root_dir=config.MODEL.MODEL_DIR, logger=comet_logger, max_epochs=config.TRAIN.EPOCHS, callbacks=[checkpoint_callback])
            #trainer_light = pl.Trainer(default_root_dir=config.MODEL.MODEL_DIR, logger=comet_logger, max_epochs=config.TRAIN.EPOCHS, callbacks=[checkpoint_callback, lr_monitor], detect_anomaly=True)
            #trainer_light = pl.Trainer(default_root_dir=config.MODEL.MODEL_DIR, logger=comet_logger, max_epochs=config.TRAIN.EPOCHS, callbacks=[checkpoint_callback, lr_monitor], gradient_clip_val=0.5, gradient_clip_algorithm='value', detect_anomaly=True)
            print("Running on pycharm")
        # slurm settings
        else:
            if config.MODEL.NAME == "rPPGNet":
                # import os
                # os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
                # ddp_find was added for rppgnet
                # trainer_light = pl.Trainer(devices=1, num_nodes=1,strategy=DDPStrategy(find_unused_parameters=True), accelerator='gpu', default_root_dir=config.MODEL.MODEL_DIR, logger=comet_logger, max_epochs=config.TRAIN.EPOCHS, callbacks=[checkpoint_callback])
                trainer_light = pl.Trainer(accelerator='gpu',default_root_dir=config.MODEL.MODEL_DIR, logger=comet_logger,
                                           max_epochs=config.TRAIN.EPOCHS, callbacks=[checkpoint_callback, lr_monitor])
            else:
                #trainer_light = pl.Trainer(devices=1, num_nodes=1, strategy='ddp', accelerator='gpu', default_root_dir=config.MODEL.MODEL_DIR, logger=comet_logger, max_epochs=config.TRAIN.EPOCHS, callbacks=[checkpoint_callback])
                # trainer_light = pl.Trainer(accelerator='gpu', default_root_dir=config.MODEL.MODEL_DIR, logger=comet_logger, max_epochs=config.TRAIN.EPOCHS, callbacks=[checkpoint_callback, lr_monitor])

                # ray tune added here
                trainer_light = pl.Trainer(accelerator='auto', devices="auto", strategy=RayDDPStrategy(),
                                           default_root_dir=config.MODEL.MODEL_DIR, logger=comet_logger, max_epochs=config.TRAIN.EPOCHS,
                                           callbacks=[RayTrainReportCallback], plugins=[RayLightningEnvironment()],
                                           enable_progress_bar=False)

                trainer_light = prepare_trainer(trainer_light)

    if config.TOOLBOX_MODE == "LOO" or config.TOOLBOX_MODE == "ENRICH":

        # Find the best lr
        if config.TRAIN.USE_LR_FINDER:
            # finding the best lr
            from lightning.pytorch.tuner.tuning import Tuner
            tuner = Tuner(trainer_light)
            if config.TEST.USE_LAST_EPOCH:
                lr_finder = tuner.lr_find(model_trainer, train_dataloaders=data_loader_dict['train'])
            else:
                # lr_finder = tuner.lr_find(model_trainer, train_dataloaders=data_loader_dict['train'],
                #                       val_dataloaders=data_loader_dict['valid'])
                lr_finder = tuner.lr_find(model_trainer, train_dataloaders=data_loader_dict['train'],
                                          val_dataloaders=data_loader_dict['test'])
            fig = lr_finder.plot(suggest=True)
            comet_logger.experiment.log_figure(figure=fig, figure_name="lr-curve")
            plt.close()
            new_lr = lr_finder.suggestion()
            model_trainer.lr = new_lr
            comet_logger.experiment.add_tags(
                ['lr-finder'])
            print(model_trainer.lr)
        else:
            print("Using lr:", config.TRAIN.LR)

        # if config.MODEL.NAME == "MEAN":
        #     trainer_light.test( dataloaders=data_loader_dict['test'])

        if config.TEST.USE_LAST_EPOCH:
            trainer_light.fit(model_trainer, data_loader_dict['train'])
            comet_logger.experiment.add_tag("last_epoch")
            trainer_light.test(ckpt_path="last", dataloaders=data_loader_dict['test'])
        else:
            trainer_light.fit(model_trainer, train_dataloaders=data_loader_dict['train'],  val_dataloaders=data_loader_dict['valid'])
            # trainer_light.fit(model_trainer, data_loader_dict['train'], data_loader_dict['test'])
            comet_logger.experiment.add_tag("best_epoch")
            trainer_light.test(ckpt_path="best", dataloaders=data_loader_dict['valid'])

    elif config.TOOLBOX_MODE == "LOO_test":
        trainer_light.test(model_trainer, ckpt_path=config.INFERENCE.MODEL_PATH, dataloaders=data_loader_dict['test'])

        # #load and test the model on the best epoch / last epoch
        # experiment_key = comet_logger.experiment.get_key()
        # comet_logger = cometlogger(experiment_key=experiment_key)
        # trainer_light = pl.trainer(logger=comet_logger)

def train_and_test(comet_logger,config, data_loader_dict):
    """trains the model."""

    if config.MODEL.NAME == "Physnet":
        model_trainer = trainer.PhysnetTrainer.PhysnetTrainer(config, data_loader_dict)
    elif config.MODEL.NAME == "Tscan":
        model_trainer = trainer.TscanTrainer.TscanTrainer(config, data_loader_dict)
    elif config.MODEL.NAME == "Efficientphys":
        model_trainer = trainer.EfficientPhystrainer.EfficientPhystrainer(config, data_loader_dict)
    elif config.MODEL.NAME == 'DeepPhys':
        model_trainer = trainer.DeepPhysTrainer.DeepPhysTrainer(config, data_loader_dict)
    elif config.MODEL.NAME == 'rPPGNet':
        model_trainer = trainer.rPPGNetTrainer.rPPGNetTrainer(config, data_loader_dict)
    elif config.MODEL.NAME == 'PhysFormer':
        model_trainer = trainer.PhysFormerTrainer.PhysFormerTrainer(config, data_loader_dict)
    else:
        raise ValueError('your model is not supported  yet!')

    #create checkpoint every epoch and track the validation loss
    if config.TEST.USE_LAST_EPOCH:
        checkpoint_callback = ModelCheckpoint(dirpath=config.MODEL.MODEL_DIR,
                                                  filename=f"{config.TRAIN.MODEL_FILE_NAME}"+"_{epoch}",
                                                  save_last='link'
                                                  )
    else:
        checkpoint_callback = ModelCheckpoint(every_n_epochs=1,
                                              save_top_k=2,
                                              dirpath=config.MODEL.MODEL_DIR,
                                              #filename=f"{config.TRAIN.MODEL_FILE_NAME}"+"_{epoch}",
                                              filename=f"{config.TRAIN.MODEL_FILE_NAME}"+"_{epoch}",
                                              monitor="val_loss_epoch",
                                              mode='min')

    lr_monitor = LearningRateMonitor(logging_interval='step')
    if config.EARLY_STOPPING.TRAIN:
        early_stop_callback = EarlyStopping(monitor="train_loss_epoch", min_delta=config.EARLY_STOPPING.DELTA, patience=config.EARLY_STOPPING.PATIENCE, verbose=True,
                                            mode="min")
        comet_logger.experiment.add_tag("early_stopping")
        if not config.CLUSTER:
            trainer_light = pl.Trainer(limit_train_batches=0.1, limit_val_batches=0.1, default_root_dir=config.MODEL.MODEL_DIR, logger=comet_logger, max_epochs=config.TRAIN.EPOCHS, callbacks=[early_stop_callback,checkpoint_callback])
        else:
            trainer_light= pl.Trainer(accelerator='gpu', default_root_dir=config.MODEL.MODEL_DIR, callbacks=[early_stop_callback, checkpoint_callback, lr_monitor], logger=comet_logger, max_epochs=config.TRAIN.EPOCHS)
    elif config.EARLY_STOPPING.VALID:
        early_stop_callback = EarlyStopping(monitor="val_loss_epoch", min_delta=config.EARLY_STOPPING.DELTA,
                                            patience=config.EARLY_STOPPING.PATIENCE, verbose=True,
                                            mode="min")
        comet_logger.experiment.add_tag("early_stopping")
        trainer_light = pl.Trainer(accelerator='gpu', default_root_dir=config.MODEL.MODEL_DIR,
                                   callbacks=[early_stop_callback, checkpoint_callback, lr_monitor],
                                   logger=comet_logger, max_epochs=config.TRAIN.EPOCHS)
    else:
        if not config.CLUSTER:
            trainer_light = pl.Trainer(limit_train_batches=0.1, limit_val_batches=0.1, default_root_dir=config.MODEL.MODEL_DIR, logger=comet_logger, max_epochs=config.TRAIN.EPOCHS, callbacks=[checkpoint_callback])
            # trainer_light = pl.Trainer(default_root_dir=config.MODEL.MODEL_DIR, logger=comet_logger, max_epochs=config.TRAIN.EPOCHS, callbacks=[checkpoint_callback])
            #trainer_light = pl.Trainer(limit_train_batches=0.1, limit_val_batches=0.5, default_root_dir=config.MODEL.MODEL_DIR, logger=comet_logger, max_epochs=config.TRAIN.EPOCHS, callbacks=[checkpoint_callback])
        else:
            #slurm settings
            #trainer_light = pl.Trainer(devices=1, num_nodes=1,strategy='ddp', accelerator='gpu', default_root_dir=config.MODEL.MODEL_DIR, logger=comet_logger, max_epochs=config.TRAIN.EPOCHS, callbacks=[checkpoint_callback])
            trainer_light = pl.Trainer(accelerator='gpu', default_root_dir=config.MODEL.MODEL_DIR, logger=comet_logger, max_epochs=config.TRAIN.EPOCHS, num_sanity_val_steps=0, callbacks=[checkpoint_callback, lr_monitor])

        #finding the best lr
    if config.TRAIN.USE_LR_FINDER:
        from lightning.pytorch.tuner.tuning import Tuner
        tuner = Tuner(trainer_light)
        if config.TEST.USE_LAST_EPOCH:
            lr_finder = tuner.lr_find(model_trainer, train_dataloaders=data_loader_dict['train'] )
        else:
            lr_finder = tuner.lr_find(model_trainer, train_dataloaders=data_loader_dict['train'],
                                          val_dataloaders=data_loader_dict['valid'])
        fig = lr_finder.plot(suggest=True)
        comet_logger.experiment.log_figure(figure=fig, figure_name="lr-curve")
        plt.close()
        new_lr = lr_finder.suggestion()
        model_trainer.lr = new_lr
        comet_logger.experiment.add_tags(
                 ['lr-finder'])
        print(model_trainer.lr)

    else:
        print("Using lr:", config.TRAIN.LR)

    if config.TEST.USE_LAST_EPOCH:
        trainer_light.fit(model_trainer, data_loader_dict['train'] )
        comet_logger.experiment.add_tag("last_epoch")
        trainer_light.test(ckpt_path="last", dataloaders=data_loader_dict['test'])
    else:
        trainer_light.fit(model_trainer, data_loader_dict['train'], data_loader_dict['valid'])
        comet_logger.experiment.add_tag("best_epoch")
        trainer_light.test(ckpt_path="best", dataloaders=data_loader_dict['test'])

def test(config, data_loader_dict):
    """tests the model."""
    if config.MODEL.NAME == "Physnet":
        model_trainer = trainer.PhysnetTrainer.PhysnetTrainer(config, data_loader_dict)
    elif config.MODEL.NAME == "iBVPNet":
        model_trainer = trainer.iBVPNetTrainer.iBVPNetTrainer(config, data_loader_dict)
    elif config.MODEL.NAME == "Tscan":
        model_trainer = trainer.TscanTrainer.TscanTrainer(config, data_loader_dict)
    elif config.MODEL.NAME == "EfficientPhys":
        model_trainer = trainer.EfficientPhysTrainer.EfficientPhysTrainer(config, data_loader_dict)
    elif config.MODEL.NAME == 'DeepPhys':
        model_trainer = trainer.DeepPhysTrainer.DeepPhysTrainer(config, data_loader_dict)
    elif config.MODEL.NAME == 'rPPGNet':
        model_trainer = trainer.rPPGNetTrainer.rPPGNetTrainer(config, data_loader_dict)
    elif config.MODEL.NAME == 'BigSmall':
        model_trainer = trainer.BigSmallTrainer.BigSmallTrainer(config, data_loader_dict)
    elif config.MODEL.NAME == 'PhysFormer':
        model_trainer = trainer.PhysFormerTrainer.PhysFormerTrainer(config, data_loader_dict)
    else:
        raise ValueError('Your Model is Not Supported  Yet!')
    # model_trainer.test(data_loader_dict)

    comet_logger = CometLogger(api_key="V1x7OI9PoIRM8yze4prM2FPcE",
                               # project_name="Strat-exp1",
                               project_name="Exp3-cmbp-on-public-Start2",
                               workspace="b-acharya",
                               experiment_name= f"{config.MODEL.NAME}_{config.TEST.DATA.DATASET}_{config.TRAIN.MODEL_FILE_NAME}",
                               log_code=True
                               )
    hyper_parameters = {
        "learning_rate": config.TRAIN.LR,
        "epochs": config.TRAIN.EPOCHS
    }


    trainer_light = pl.Trainer(default_root_dir=config.MODEL.MODEL_DIR, logger=comet_logger, max_epochs=config.TRAIN.EPOCHS)

    # model_trainer.const_init(0.0)
    # model_trainer.check_weights()
    if Path(config.INFERENCE.MODEL_PATH).suffix == ".pth" :
        model_trainer.load_model(config.INFERENCE.MODEL_PATH)
        model_trainer.check_weights()
        trainer_light.test(model_trainer, dataloaders=data_loader_dict['test'])
    else:
        trainer_light.test(model_trainer, ckpt_path=config.INFERENCE.MODEL_PATH, dataloaders=data_loader_dict['test'])


def unsupervised_HR(config, data_loader):

    if not config.UNSUPERVISED.METHOD:
        raise ValueError("Please set unsupervised method in yaml!")
    for unsupervised_method in config.UNSUPERVISED.METHOD:
        run_name = unsupervised_method
        comet_logger = CometLogger(api_key="V1x7OI9PoIRM8yze4prM2FPcE",
                                   #project_name="unsupervised-methods",
                                   project_name="SIT-dataset-unsupervised",
                                   workspace="b-acharya",
                                   experiment_name=f"{run_name}",
                                   log_code=False
                                   )
        comet_logger.experiment.add_tag(f"{config.UNSUPERVISED.DATA.DATASET}")
        comet_logger.experiment.add_tag(f"{config.INFERENCE.EVALUATION_METHOD}")
        if config.INFERENCE.EVALUATION_WINDOW.USE_SMALLER_WINDOW:
            comet_logger.experiment.add_tag(f"Smaller_windows size:{config.INFERENCE.EVALUATION_WINDOW.WINDOW_SIZE}")

        if unsupervised_method == "POS":
            unsupervised_HR_predict(config, data_loader, "POS", comet_logger)
        elif unsupervised_method == "CHROM":
            unsupervised_HR_predict(config, data_loader, "CHROM", comet_logger)
        elif unsupervised_method == "ICA":
            unsupervised_HR_predict(config, data_loader, "ICA", comet_logger)
        elif unsupervised_method == "GREEN":
            unsupervised_HR_predict(config, data_loader, "GREEN", comet_logger)
        elif unsupervised_method == "BLUE":
            unsupervised_HR_predict(config, data_loader, "BLUE", comet_logger)
        elif unsupervised_method == "RED":
            unsupervised_HR_predict(config, data_loader, "RED", comet_logger)
        elif unsupervised_method == "LGI":
            unsupervised_HR_predict(config, data_loader, "LGI", comet_logger)
        elif unsupervised_method == "PBV":
            unsupervised_HR_predict(config, data_loader, "PBV", comet_logger)
        elif unsupervised_method == "dummy":
            unsupervised_HR_predict(config, data_loader, "dummy", comet_logger)
        elif unsupervised_method == "random":
            unsupervised_HR_predict(config, data_loader, "random", comet_logger)
        else:
            raise ValueError("Not supported unsupervised method!")

def unsupervised_method_inference(config, data_loader):

    if not config.UNSUPERVISED.METHOD:
        raise ValueError("Please set unsupervised method in yaml!")
    for unsupervised_method in config.UNSUPERVISED.METHOD:
        run_name = unsupervised_method
        comet_logger = CometLogger(api_key="V1x7OI9PoIRM8yze4prM2FPcE",
                                   #project_name="unsupervised-methods",
                                   project_name="DST-dataset-unsupervised",
                                   workspace="b-acharya",
                                   experiment_name=f"{run_name}",
                                   log_code=False
                                   )
        comet_logger.experiment.add_tag(f"{config.UNSUPERVISED.DATA.DATASET}")
        comet_logger.experiment.add_tag(f"{config.INFERENCE.EVALUATION_METHOD}")
        if config.INFERENCE.EVALUATION_WINDOW.USE_SMALLER_WINDOW:
            comet_logger.experiment.add_tag(f"Smaller_windows size:{config.INFERENCE.EVALUATION_WINDOW.WINDOW_SIZE}")

        if unsupervised_method == "POS":
            unsupervised_predict(config, data_loader, "POS", comet_logger)
        elif unsupervised_method == "CHROM":
            unsupervised_predict(config, data_loader, "CHROM", comet_logger)
        elif unsupervised_method == "ICA":
            unsupervised_predict(config, data_loader, "ICA", comet_logger)
        elif unsupervised_method == "GREEN":
            unsupervised_predict(config, data_loader, "GREEN", comet_logger)
        elif unsupervised_method == "BLUE":
            unsupervised_predict(config, data_loader, "BLUE", comet_logger)
        elif unsupervised_method == "RED":
            unsupervised_predict(config, data_loader, "RED", comet_logger)
        elif unsupervised_method == "LGI":
            unsupervised_predict(config, data_loader, "LGI", comet_logger)
        elif unsupervised_method == "PBV":
            unsupervised_predict(config, data_loader, "PBV", comet_logger)
        elif unsupervised_method == "dummy":
            unsupervised_predict(config, data_loader, "dummy", comet_logger)
        elif unsupervised_method == "random":
            unsupervised_predict(config, data_loader, "random", comet_logger)
        else:
            raise ValueError("Not supported unsupervised method!")


if __name__ == "__main__":
    # parse arguments.
    parser = argparse.ArgumentParser()
    parser = add_args(parser)
    parser = trainer.BaseTrainer.BaseTrainer.add_trainer_args(parser)
    parser = data_loader.BaseLoader.BaseLoader.add_data_loader_args(parser)
    parser.add_argument("--FOLD", default=None, type=int)
    args = parser.parse_args()

    # configurations.
    config = get_config(args)
    print('Configuration:')
    print(config, end='\n\n')
    global CONFIG_FILE_NAME
    CONFIG_FILE_NAME = args.config_file
    data_loader_dict = dict() # dictionary of data loaders
    if config.TOOLBOX_MODE == "train_and_test" or config.TOOLBOX_MODE == "train_and_test_enrich":
        # train_loader
        if config.TRAIN.DATA.DATASET == "UBFC":
            train_loader = data_loader.UBFCLoader.UBFCLoader
        elif config.TRAIN.DATA.DATASET == "PURE":
            train_loader = data_loader.PURELoader.PURELoader
        elif config.TRAIN.DATA.DATASET == "SCAMPS":
            train_loader = data_loader.SCAMPSLoader.SCAMPSLoader
        elif config.TRAIN.DATA.DATASET == "MMPD":
            train_loader = data_loader.MMPDLoader.MMPDLoader
        elif config.TRAIN.DATA.DATASET == "BP4DPlus":
            train_loader = data_loader.BP4DPlusLoader.BP4DPlusLoader
        elif config.TRAIN.DATA.DATASET == "CMBP":
            train_loader = data_loader.CMBPLoader.CMBPLader
        elif config.TRAIN.DATA.DATASET == "VIPL":
            train_loader = data_loader.VIPLLoader.VIPLLoader
        elif config.TRAIN.DATA.DATASET == "COHFACE":
            train_loader = data_loader.COHFACELoader.COHFACELoader
        else:
            raise ValueError("Unsupported dataset! Currently supporting UBFC, PURE, MMPD, and SCAMPS.")

        # Create and initialize the train dataloader given the correct toolbox mode,
        # a supported dataset name, and a valid dataset path
        if (config.TRAIN.DATA.DATASET and config.TRAIN.DATA.DATA_PATH):
            train_data_loader = train_loader(
                name="train",
                data_path=config.TRAIN.DATA.DATA_PATH,
                config_data=config.TRAIN.DATA,
                model = config.MODEL.NAME
            )
            data_loader_dict['train'] = DataLoader(
                dataset=train_data_loader,
                num_workers=16,
                batch_size=config.TRAIN.BATCH_SIZE,
                shuffle=True,
                worker_init_fn=seed_worker,
                generator=train_generator,
            )
        else:
            data_loader_dict['train'] = None

        # valid_loader
        if config.VALID.DATA.DATASET == "UBFC":
            valid_loader = data_loader.UBFCLoader.UBFCLoader
        elif config.VALID.DATA.DATASET == "PURE":
            valid_loader = data_loader.PURELoader.PURELoader
        elif config.VALID.DATA.DATASET == "SCAMPS":
            valid_loader = data_loader.SCAMPSLoader.SCAMPSLoader
        elif config.VALID.DATA.DATASET == "MMPD":
            valid_loader = data_loader.MMPDLoader.MMPDLoader
        elif config.VALID.DATA.DATASET == "CMBP":
            valid_loader = data_loader.CMBPLoader.CMBPLoader
        elif config.VALID.DATA.DATASET == "BP4DPlus":
            valid_loader = data_loader.BP4DPlusLoader.BP4DPlusLoader
        elif config.VALID.DATA.DATASET == "VIPL":
            valid_loader = data_loader.VIPLLoader.VIPLLoader
        elif config.VALID.DATA.DATASET == "COHFACE":
            valid_loader = data_loader.COHFACELoader.COHFACELoader
        elif config.VALID.DATA.DATASET == "BP4DPlusBigSmall":
            valid_loader = data_loader.BP4DPlusBigSmallLoader.BP4DPlusBigSmallLoader
        elif config.VALID.DATA.DATASET == "UBFC-PHYS":
            valid_loader = data_loader.UBFCPHYSLoader.UBFCPHYSLoader
        elif config.VALID.DATA.DATASET == "iBVP":
            valid_loader = data_loader.iBVPLoader.iBVPLoader
        elif config.VALID.DATA.DATASET is None and not config.TEST.USE_LAST_EPOCH:
            raise ValueError("Validation dataset not specified despite USE_LAST_EPOCH set to False!")
        elif config.TEST.USE_LAST_EPOCH:
            print("Using lat epoch no validation set and will be set to None")
        else:
            raise ValueError("Unsupported dataset! Currently supporting UBFC, PURE, MMPD, and SCAMPS.")

        # Create and initialize the valid dataloader given the correct toolbox mode,
        # a supported dataset name, and a valid dataset path
        if (config.VALID.DATA.DATASET and config.VALID.DATA.DATA_PATH and not config.TEST.USE_LAST_EPOCH) or config.TOOLBOX_MODE == "train_and_test_enrich":
            valid_data = valid_loader(
                name="valid",
                data_path=config.VALID.DATA.DATA_PATH,
                config_data=config.VALID.DATA,
                model=config.MODEL.NAME
            )
            data_loader_dict["valid"] = DataLoader(
                dataset=valid_data,
                num_workers=16,
                batch_size=config.TRAIN.BATCH_SIZE,  # batch size for val is the same as train
                shuffle=False,
                worker_init_fn=seed_worker,
                generator=general_generator,
            )
        else:
            data_loader_dict['valid'] = None

        #enrichment of dataset used to train two datasets together
        if config.TOOLBOX_MODE == "train_and_test_enrich":
            enriched_train_set = torch.utils.data.ConcatDataset([train_data_loader, valid_data])
            if config.MODEL.NAME == 'DeepPhys' or config.MODEL.NAME == "Tscan":

                a_len = len(train_data_loader)
                ab_len = a_len + len(valid_data)

                a_indices = list(range(a_len))
                b_indices = list(range(a_len, ab_len))

                class MyBatchSampler(torch.utils.data.Sampler):
                    def __init__(self, a_indices, b_indices, batch_size):
                        self.a_indices = a_indices
                        self.b_indices = b_indices
                        self.batch_size = batch_size
                    def __iter__(self):
                        random.shuffle(self.a_indices)
                        random.shuffle(self.b_indices)
                        a_batches = self.chunk(self.a_indices, self.batch_size)
                        b_batches = self.chunk(self.b_indices, self.batch_size)
                        all_batches = list(a_batches + b_batches)
                        all_batches = [batch.tolist() for batch in all_batches]
                        random.shuffle(all_batches)
                        return iter(all_batches)
                    def __len__(self):
                        return (len(self.a_indices) + len(self.b_indices)) // self.batch_size
                    @staticmethod
                    def chunk(indices, size):
                        return torch.split(torch.tensor(indices), size)

                batch_sampler = MyBatchSampler(a_indices, b_indices, config.TRAIN.BATCH_SIZE)

                dl = DataLoader(enriched_train_set, batch_sampler=batch_sampler)

                data_loader_dict['train'] = DataLoader(
                dataset= enriched_train_set,
                num_workers=16,
                batch_sampler=batch_sampler,
                worker_init_fn=seed_worker,
                generator=train_generator,
                )
            else:
                data_loader_dict['train'] = DataLoader(
                    dataset= enriched_train_set,
                    num_workers=16,
                    batch_size=config.TRAIN.BATCH_SIZE,
                    shuffle=True,
                    worker_init_fn=seed_worker,
                    generator=train_generator,
                )


    if config.TOOLBOX_MODE == "preprocess":
            print("done with preprocessing")

    if config.TOOLBOX_MODE == "train_and_test" or config.TOOLBOX_MODE == "only_test" or config.TOOLBOX_MODE=="train_and_test_enrich":
        # test_loader
        if config.TEST.DATA.DATASET == "UBFC":
            test_loader = data_loader.UBFCLoader.UBFCLoader
        elif config.TEST.DATA.DATASET == "PURE":
            test_loader = data_loader.PURELoader.PURELoader
        elif config.TEST.DATA.DATASET == "SCAMPS":
            test_loader = data_loader.SCAMPSLoader.SCAMPSLoader
        elif config.TEST.DATA.DATASET == "MMPD":
            test_loader = data_loader.MMPDLoader.MMPDLoader
        elif config.TEST.DATA.DATASET == "CMBP":
            test_loader = data_loader.CMBPLoader.CMBPLoader
        elif config.TEST.DATA.DATASET == "BP4DPlus":
            test_loader = data_loader.BP4DPlusLoader.BP4DPlusLoader
        elif config.TEST.DATA.DATASET == "VIPL":
            test_loader = data_loader.VIPLLoader.VIPLLoader
        elif config.TEST.DATA.DATASET == "COHFACE":
            test_loader = data_loader.COHFACELoader.COHFACELoader
        elif config.TEST.DATA.DATASET == "DST":
            test_loader = data_loader.DSTLoader.DSTLoader
        else:
            raise ValueError("Unsupported dataset! Currently supporting UBFC, PURE, MMPD, and SCAMPS.")

        if config.TOOLBOX_MODE == "train_and_test" and config.TEST.USE_LAST_EPOCH:
            print("Testing uses last epoch, validation dataset is not required.", end='\n\n')   

        # Create and initialize the test dataloader given the correct toolbox mode,
        # a supported dataset name, and a valid dataset path
        if config.TEST.DATA.DATASET and config.TEST.DATA.DATA_PATH:
            test_data = test_loader(
                name="test",
                data_path=config.TEST.DATA.DATA_PATH,
                config_data=config.TEST.DATA,
                model=config.MODEL.NAME
            )
            data_loader_dict["test"] = DataLoader(
                dataset=test_data,
                num_workers=16,
                batch_size=config.INFERENCE.BATCH_SIZE,
                shuffle=False,
                worker_init_fn=seed_worker,
                generator=general_generator,
            )
        else:
            data_loader_dict['test'] = None

    elif config.TOOLBOX_MODE == "unsupervised_method" or config.TOOLBOX_MODE == "unsupervised_predict":
        # unsupervised method dataloader
        if config.UNSUPERVISED.DATA.DATASET == "COHFACE":
            unsupervised_loader = data_loader.COHFACELoader.COHFACELoader
        elif config.UNSUPERVISED.DATA.DATASET == "UBFC":
            unsupervised_loader = data_loader.UBFCLoader.UBFCLoader
        elif config.UNSUPERVISED.DATA.DATASET == "PURE":
            unsupervised_loader = data_loader.PURELoader.PURELoader
        elif config.UNSUPERVISED.DATA.DATASET == "SCAMPS":
            unsupervised_loader = data_loader.SCAMPSLoader.SCAMPSLoader
        elif config.UNSUPERVISED.DATA.DATASET == "MMPD":
            unsupervised_loader = data_loader.MMPDLoader.MMPDLoader
        elif config.UNSUPERVISED.DATA.DATASET == "CMBP":
            unsupervised_loader = data_loader.CMBPLoader.CMBPLoader
        elif config.UNSUPERVISED.DATA.DATASET == "VIPL":
            unsupervised_loader = data_loader.VIPLLoader.VIPLLoader
        elif config.UNSUPERVISED.DATA.DATASET == "DST":
            unsupervised_loader = data_loader.DSTLoader.DSTLoader
        elif config.UNSUPERVISED.DATA.DATASET == "BP4DPlus":
            unsupervised_loader = data_loader.BP4DPlusLoader.BP4DPlusLoader
        elif config.UNSUPERVISED.DATA.DATASET == "UBFC-PHYS":
            unsupervised_loader = data_loader.UBFCPHYSLoader.UBFCPHYSLoader
        elif config.UNSUPERVISED.DATA.DATASET == "iBVP":
            unsupervised_loader = data_loader.iBVPLoader.iBVPLoader
        elif config.UNSUPERVISED.DATA.DATASET == "SIT":
            unsupervised_loader = data_loader.SITLoader.SITLoader
        else:
            raise ValueError("Unsupported dataset! Currently supporting UBFC-rPPG, PURE, MMPD, \
                             SCAMPS, BP4D+, UBFC-PHYS and iBVP.")

        unsupervised_data = unsupervised_loader(
            name="unsupervised",
            data_path=config.UNSUPERVISED.DATA.DATA_PATH,
            config_data=config.UNSUPERVISED.DATA,
            model="unsupervised")
        data_loader_dict["unsupervised"] = DataLoader(
            dataset=unsupervised_data,
            num_workers=0,
            batch_size=1,
            shuffle=False,
            worker_init_fn=seed_worker,
            generator=general_generator
        )

    elif config.TOOLBOX_MODE == "LOO" or config.TOOLBOX_MODE == "ENRICH":


        # Number of folds for k fold validation
        K_fold = 10

        #dataloader for the training set
        if config.TRAIN.DATA.DATASET == "PURE":
            loader = data_loader.PURELoader.PURELoader
            participants = np.arange(1, 11)
            kf_inner = KFold(n_splits=9, shuffle=True, random_state=42)

        elif config.TRAIN.DATA.DATASET == "CMBP":
            loader = data_loader.CMBPLoader.CMBPLoader
            basepath = pathlib.Path(config.TRAIN.DATA.DATA_PATH)
            # basepath = pathlib.Path("/homes/bacharya/rPPG-Toolbox/CMBP/")
            # import os
            # participants = [ name for name in os.listdir(str(basepath)) if os.path.isdir(os.path.join(basepath, name)) ]
            kf_inner = KFold(n_splits=9, shuffle=True, random_state=42)

        elif config.TRAIN.DATA.DATASET == "VIPL":
            loader = data_loader.VIPLLoader.VIPLLoader
            participants = np.arange(1, 108)

        elif config.TRAIN.DATA.DATASET == "COHFACE":
            loader = data_loader.COHFACELoader.COHFACELoader
            participants = np.arange(1, 41)
            kf_inner = KFold(n_splits=9, shuffle=True, random_state=42)

        else:
            raise ValueError("Unsupported dataset! Currently supporting UBFC, PURE, MMPD, and SCAMPS.")

        #dataloader for training
        if (config.TRAIN.DATA.DATASET and config.TRAIN.DATA.DATA_PATH):
            train_data_loader = loader(
                name="train",
                data_path=config.TRAIN.DATA.DATA_PATH,
                config_data=config.TRAIN.DATA,
                model=config.MODEL.NAME
            )
            data_loader_dict['train'] = DataLoader(
                dataset=train_data_loader,
                num_workers=16,
                batch_size=config.TRAIN.BATCH_SIZE,
                shuffle=False,
                worker_init_fn=seed_worker,
                generator=train_generator,
            )
        else:
            raise NotImplementedError

        if config.TOOLBOX_MODE == "ENRICH":
            # dataloader for the seconddataset set
            if config.VALID.DATA.DATASET == "PURE":
                loader_valid = data_loader.PURELoader.PURELoader
            elif config.VALID.DATA.DATASET == "CMBP":
                loader_valid = data_loader.CMBPLoader.CMBPLoader
            elif config.VALID.DATA.DATASET == "VIPL":
                loader_valid = data_loader.VIPLLoader.VIPLLoader
            elif config.VALID.DATA.DATASET == "COHFACE":
                loader_valid = data_loader.COHFACELoader.COHFACELoader
            else:
                raise ValueError("Unsupported dataset! Currently supporting UBFC, PURE, MMPD, and SCAMPS.")
            if (config.VALID.DATA.DATASET and config.VALID.DATA.DATA_PATH):
                valid_data_loader = loader_valid(
                    name="valid",
                    data_path=config.VALID.DATA.DATA_PATH,
                    config_data=config.VALID.DATA,
                    model = config.MODEL.NAME
                )
            else:
                raise NotImplementedError

        kf = KFold(n_splits=K_fold, shuffle=True, random_state=42)
        file_list_path = config.TRAIN.DATA.FILE_LIST_PATH
        list_file = pd.read_csv(file_list_path)

        list_file['id'] = list_file['input_files'].apply(get_participant_index)
        participants = sorted(list(set(list_file['id'])))


        for test_i, (train_index, test_index) in enumerate(kf.split(participants)):

            # Uncommet this if you want to re-run a particular fold
            if test_i != int(config.FOLD):
                print("skipping:", test_i)
                continue

            # splits for dataloaders for Kfold cross validation
            train_participants = np.array([participants[i] for i in train_index])
            test_participants = np.array([participants[i] for i in test_index])
            train_path, test_path = create_path_list(list_file, train_participants, test_participants, test_i)


            # TODO: Old code for nested cross validation is this still needed ?
            for valid_j, (train_index_inner, valid_index_inner) in enumerate(kf_inner.split(train_participants)):

                if config.TOOLBOX_MODE == "LOO_test":
                    if j != config.INNER_FOLD:
                        continue

            #split the training set into train and validation sets
                train_participants_inner, valid_participants_inner = train_test_split(train_participants)

            #
                train_participants_inner = [train_participants[i] for i in train_index_inner]
                valid_participants_inner = [train_participants[i] for i in valid_index_inner]
                train_path, valid_path = create_path_list(list_file, train_participants_inner, valid_participants_inner, test_i, valid_j, outer=False)

                train_df = pd.read_csv(train_path)
                valid_df = pd.read_csv(valid_path)
                test_df = pd.read_csv(test_path)

                print("-"*100)
                print(train_participants_inner)
                print("-"*100)
                print( valid_participants_inner)
                print("-"*100)
                print(test_participants)

                config.defrost()
                config.TRAIN.DATA.FILE_LIST_PATH = str(train_path)
                config.VALID.DATA.FILE_LIST_PATH = str(valid_path)
                config.TEST.DATA.FILE_LIST_PATH = str(test_path)
                print(config.TEST.DATA.FILE_LIST_PATH)
                print(config.TRAIN.DATA.FILE_LIST_PATH)
                print(config.VALID.DATA.FILE_LIST_PATH)
                config.TRAIN.DATA.DO_PREPROCESS = False
                config.VALID.DATA.DO_PREPROCESS = False
                config.TEST.DATA.DO_PREPROCESS = False
                config.freeze()

                # comet_logger = CometLogger(api_key="V1x7OI9PoIRM8yze4prM2FPcE",
                #                            project_name="loss-functions",
                #                            workspace="b-acharya",
                #                            experiment_name= f"{config.MODEL.NAME}_{config.TRAIN.DATA.DATASET}_{config.VALID.DATA.DATASET}_{config.TEST.DATA.DATASET}",
                                           # experiment_name= f"{config.TRAIN.DATA.DATASET}_{config.MODEL.NAME}_FOLD_{test_i}_valid_{valid_j}",
                                           # log_code=True
                                           # )
                # hyper_parameters = {
                #         "Learning_rate": config.TRAIN.LR,
                #         "epochs": config.TRAIN.EPOCHS,
                # }

                # config_tune = {"lr": 1e-3, "batch_size": 64, "epochs": 20}




    # get it? relu but ray_loo XD
    elif config.TOOLBOX_MODE == "RAY_LOO":
        K_fold = 10

        if config.TRAIN.DATA.DATASET == "CMBP":
            loader = data_loader.CMBPLoader.CMBPLoader
            basepath = pathlib.Path(config.TRAIN.DATA.DATA_PATH)
            kf_inner = KFold(n_splits=9, shuffle=True, random_state=42)

        else:
            raise ValueError("Unsupported dataset! Currently supporting UBFC, PURE, MMPD, and SCAMPS.")

        #dataloader for training
        if (config.TRAIN.DATA.DATASET and config.TRAIN.DATA.DATA_PATH):
            train_data_loader = loader(
                name="train",
                data_path=config.TRAIN.DATA.DATA_PATH,
                config_data=config.TRAIN.DATA,
                model=config.MODEL.NAME
            )
            data_loader_dict['train'] = DataLoader(
                dataset=train_data_loader,
                num_workers=16,
                batch_size=config.TRAIN.BATCH_SIZE,
                shuffle=False,
                worker_init_fn=seed_worker,
                generator=train_generator,
            )
        else:
            raise NotImplementedError

        kf = KFold(n_splits=K_fold, shuffle=True, random_state=42)
        file_list_path = config.TRAIN.DATA.FILE_LIST_PATH
        list_file = pd.read_csv(file_list_path)

        list_file['id'] = list_file['input_files'].apply(get_participant_index)
        participants = sorted(list(set(list_file['id'])))


        for test_i, (train_index, test_index) in enumerate(kf.split(participants)):

            # Uncommet this if you want to re-run a particular fold
            if test_i != int(config.FOLD):
                print(f"skipping {test_i}")
                continue

            # splits for dataloaders for Kfold cross validation
            train_participants = np.array([participants[i] for i in train_index])
            test_participants = np.array([participants[i] for i in test_index])
            train_path, test_path = create_path_list(list_file, train_participants, test_participants, test_i)


            #split the training set into train and validation sets
            train_participants_inner, valid_participants_inner = train_test_split(train_participants, random_state=42, test_size=0.2)

            # train_participants_inner = [train_participants[i] for i in train_participants_inner]
            # valid_participants_inner = [train_participants[i] for i in valid_participants_inner]

            train_path, valid_path = create_path_list(list_file, train_participants_inner, valid_participants_inner, test_i, outer=False)

            train_df = pd.read_csv(train_path)
            valid_df = pd.read_csv(valid_path)
            test_df = pd.read_csv(test_path)

            print("-"*100)
            print(train_participants_inner)
            print("-"*100)
            print( valid_participants_inner)
            print("-"*100)
            print(test_participants)

            config.defrost()
            config.TRAIN.DATA.FILE_LIST_PATH = str(train_path)
            config.VALID.DATA.FILE_LIST_PATH = str(valid_path)
            config.TEST.DATA.FILE_LIST_PATH = str(test_path)
            print(config.TEST.DATA.FILE_LIST_PATH)
            print(config.TRAIN.DATA.FILE_LIST_PATH)
            print(config.VALID.DATA.FILE_LIST_PATH)
            config.TRAIN.DATA.DO_PREPROCESS = False
            config.VALID.DATA.DO_PREPROCESS = False
            config.TEST.DATA.DO_PREPROCESS = False
            config.freeze()


            #TODO: If we use a generator to initialize the dataloader, the line below will crash it.
            inspect_serializability(train_func, name="train_func")

            tags = []
            ############################
            # configure a search space #
            ############################

            if config.MODEL.SCHEDULER == "OneCycle":
                print("One cycle schedular used")
                search_space = {
                    "lr": tune.loguniform(1e-7, 1.0),
                    "epochs": tune.qrandint(10, config.TRAIN.EPOCHS), #min number of epochs is 10 and max is 50, we search this space to find the best epochs. This needs to be optimized as the scheduler would need this.
                    "weight_decay": tune.choice([1e-4, 1e-3, 1e-5, 1e-6, 1e-2, 1e-1, 0.0]),
                }
                tags.append("OneCycle")
            elif config.MODEL.SCHEDULER == "ReduceOnPlatue":
                pass
                tags.append("ReduceOnPlatue")

            elif config.EARLY_STOPPING.VALID:
                print("Early stopping via pytorch lighting is used here")

                if config.MODEL.NAME == "Tscan" and config.MODEL.OPTIMIZE_DROP_RATE:
                    print("drop rate in hyperparameter search")
                    search_space = {
                        "lr": tune.loguniform(1e-7, 1.0),
                        # "epochs": tune.choice([50]),
                        "epochs": tune.choice([1]),
                        "weight_decay": tune.choice([1e-4, 1e-3, 1e-5, 1e-6, 1e-2, 1e-1, 0.0]),
                        "dropout_rate1": tune.choice([0.1, 0.2, 0.3, 0.4, 0.5]),
                        "dropout_rate2": tune.choice([0.1, 0.2, 0.3, 0.4, 0.5])
                    }
                    tags.append("Drop-rate")

                elif config.MODEl.NAME == "Physnet" and config.MODEL.OPTIMIZE_DROP_RATE:

                    search_space = {
                        "lr": tune.loguniform(1e-7, 1.0),
                        "epochs": tune.choice([50]),
                        "weight_decay": tune.choice([1e-4, 1e-3, 1e-5, 1e-6, 1e-2, 1e-1, 0.0]),
                    }
                    tags.append("Drop-rate")

                else:
                    print("drop rate was not optimized")
                    search_space = {
                        "lr": tune.loguniform(1e-7, 1.0),
                        "epochs": tune.choice([50]),
                        "weight_decay": tune.choice([1e-4, 1e-3, 1e-5, 1e-6, 1e-2, 1e-1, 0.0]),
                    }

                tags.append("EarlyStopping-Valid")

            else:
                print("scheduler was not used")

                if config.MODEL.NAME == "Tscan" and config.MODEL.OPTIMIZE_DROP_RATE:
                    print("drop rate in hyperparameter search")
                    search_space = {
                        "lr": tune.loguniform(1e-7, 1.0),
                        "epochs": tune.choice([50]),
                        "weight_decay": tune.choice([1e-4, 1e-3, 1e-5, 1e-6, 1e-2, 1e-1, 0.0]),
                        "dropout_rate1": tune.choice([0.1, 0.2, 0.3, 0.4, 0.5]),
                        "dropout_rate2": tune.choice([0.1, 0.2, 0.3, 0.4, 0.5])
                    }
                    tags.append("Drop-rate")
                elif config.MODEl.NAME == "Physnet" and config.MODEL.OPTIMIZE_DROP_RATE:

                    search_space = {
                        "lr": tune.loguniform(1e-7, 1.0),
                        "epochs": tune.choice([50]),
                        "weight_decay": tune.choice([1e-4, 1e-3, 1e-5, 1e-6, 1e-2, 1e-1, 0.0]),
                    }
                    tags.append("Drop-rate")

                else:
                    print("drop rate was not optimized")
                    search_space = {
                        "lr": tune.loguniform(1e-7, 1.0),
                        "epochs": tune.choice([50]),
                        "weight_decay": tune.choice([1e-4, 1e-3, 1e-5, 1e-6, 1e-2, 1e-1, 0.0]),
                    }

                #scheduler for the trails should be only used here
                num_epochs = config.TRAIN.EPOCHS
                scheduler = ASHAScheduler(max_t=num_epochs, grace_period=10, reduction_factor=2)
                tags.append('ASHA')

            ############################
            #  Optimization algo       #
            ############################
            search_algo = OptunaSearch()

            ##############################
            #  gpu training and scalling #
            ##############################

            scaling_config = ScalingConfig(
                num_workers=1,
                use_gpu=True,
                # placement_strategy="STRICT_SPREAD",
                resources_per_worker={"CPU": 20, "GPU": 1}
            )

            if config.MODEL.OPTIMIZE_DROP_RATE:
                project_name = config.MODEL.NAME + "_fold_" + str(test_i) + "_loss_" + config.MODEL.LOSS + "_pseudo_label_" + str(config.TRAIN.DATA.PREPROCESS.USE_PSUEDO_PPG_LABEL) + "_Scheduler_" + str(config.MODEL.SCHEDULER) + "drop_rate" + str(config.MODEL.OPTIMIZE_DROP_RATE)
            else:
                project_name = config.MODEL.NAME + "_fold_" + str(test_i) + "_loss_" + config.MODEL.LOSS + "_pseudo_label_" + str(config.TRAIN.DATA.PREPROCESS.USE_PSUEDO_PPG_LABEL) + "_Scheduler_" + str(config.MODEL.SCHEDULER)


            comet_callback = CometLoggerCallback(
                True,
                workspace='loss-functions',
                api_key='V1x7OI9PoIRM8yze4prM2FPcE',
                project_name=project_name,
                tags=tags,
            )

            # TODO: who takes care of checkpointintg
            storage_path = "/vol/rPPG/loss-functions/" + config.MODEL.NAME
            checkpoint = CheckpointConfig(num_to_keep=1, checkpoint_score_order="min")
            run_config = RunConfig(storage_path= storage_path, callbacks=[comet_callback], checkpoint_config=checkpoint)

            #default configs which wont be used
            train_loop_config = {"num_epochs": 20, "lr": 0.01, "batch_size": 32}

            ray_trainer = TorchTrainer(
                train_func,
                train_loop_config=train_loop_config,
                scaling_config=scaling_config,
                run_config=run_config,
                metadata=config
            )

            # trainable_with_gpu = tune.with_resources(ray_trainer, {"gpu": 1})

            if config.MODEL.SCHEDULER == "OneCycle" or config.MODEL.SCHEDULER == "ReduceOnPlatue" or config.EARLY_STOPPING.VALID:
                tuner = tune.Tuner(
                    ray_trainer,
                    # trainable_with_gpu,
                    param_space={"train_loop_config": search_space},
                    tune_config=tune.TuneConfig(
                        metric="val_loss_epoch",
                        mode="min",
                        num_samples=100,
                        search_alg=search_algo,
                        trial_name_creator=trail_name,
                    ),
                )

            else:
                tuner = tune.Tuner(
                    ray_trainer,
                    # trainable_with_gpu,
                    param_space={"train_loop_config": search_space},
                    tune_config=tune.TuneConfig(
                        metric="val_loss_epoch",
                        mode="min",
                        num_samples=100,
                        search_alg=search_algo,
                        scheduler=scheduler,
                        trial_name_creator=trail_name,
                    ),
                )

            result = tuner.fit()

            print(f"-------fold-{test_i}-----------")
            print(result.get_best_result(metric="val_loss_epoch", mode="min"))

    else:
        raise ValueError("Unsupported toolbox_mode! Currently support train_and_test or only_test or unsupervised_method.")

    if config.TOOLBOX_MODE == "train_and_test" or config.TOOLBOX_MODE == "train_and_test_enrich":

        comet_logger = CometLogger(api_key="V1x7OI9PoIRM8yze4prM2FPcE",
                                   # project_name="Exp3-cmbp-on-public-Start2",
                                   # project_name="pure_vipl_dst_tscan",
                                   project_name="Physnet-loss-funciton-analysis",
                                   # project_name="Exp2-public-on-CMBP-start2",
                                   workspace="b-acharya",
                                   experiment_name=f"{config.TRAIN.DATA.DATASET}_{config.TEST.DATA.DATASET}_{config.MODEL.NAME}",
                                   log_code=True
                                   )
        hyper_parameters = {
            "Learning_rate": config.TRAIN.LR,
            "epochs": config.TRAIN.EPOCHS
        }

        comet_logger.log_hyperparams(hyper_parameters)
        if config.MODEL.NAME == "Physnet":
            comet_logger.experiment.add_tags(
                [config.MODEL.NAME, config.TRAIN.DATA.PREPROCESS.CHUNK_LENGTH, config.TRAIN.DATA.DATASET,
                 config.TRAIN.DATA.PREPROCESS.LABEL_TYPE, config.TRAIN.BATCH_SIZE, config.MODEL.PHYSNET.LOSS])
        else:
            comet_logger.experiment.add_tags(
                [config.MODEL.NAME, config.TRAIN.DATA.PREPROCESS.CHUNK_LENGTH, config.TRAIN.DATA.DATASET,
                config.TRAIN.DATA.PREPROCESS.LABEL_TYPE, config.TRAIN.BATCH_SIZE])

        comet_logger.experiment.log_asset(CONFIG_FILE_NAME)

        train_and_test(comet_logger, config, data_loader_dict)

    elif config.TOOLBOX_MODE == "only_test":
        test(config, data_loader_dict)
    elif config.TOOLBOX_MODE == "unsupervised_method":
        unsupervised_method_inference(config, data_loader_dict)
    elif config.TOOLBOX_MODE == "unsupervised_predict":
        unsupervised_HR(config, data_loader_dict)
    else:
        print("TOOLBOX_MODE only support train_and_test or only_test !", end='\n\n')
