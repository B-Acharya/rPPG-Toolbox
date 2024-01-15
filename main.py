""" The main function of rPPG deep learning pipeline."""

import argparse
import random
import time

import numpy as np
import pandas as pd
import torch
from config import get_config
from dataset import data_loader
from neural_methods import trainer
from unsupervised_methods.unsupervised_predictor import unsupervised_predict
from torch.utils.data import DataLoader
import lightning.pytorch as pl
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import CometLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.strategies import DDPStrategy
import matplotlib.pyplot as plt
import pathlib
from sklearn.model_selection import KFold

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

#to get accurate stack
# import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def add_args(parser):
    """Adds arguments for parser."""
    parser.add_argument('--config_file', required=False,
                        default="configs/train_configs/PURE_PURE_UBFC_TSCAN_BASIC.yaml", type=str, help="The name of the model.")
    '''Neural Method Sample YAMSL LIST:
      SCAMPS_SCAMPS_UBFC_TSCAN_BASIC.yaml
      SCAMPS_SCAMPS_UBFC_DEEPPHYS_BASIC.yaml
      SCAMPS_SCAMPS_UBFC_PHYSNET_BASIC.yaml
      SCAMPS_SCAMPS_PURE_DEEPPHYS_BASIC.yaml
      SCAMPS_SCAMPS_PURE_TSCAN_BASIC.yaml
      SCAMPS_SCAMPS_PURE_PHYSNET_BASIC.yaml
      PURE_PURE_UBFC_TSCAN_BASIC.yaml
      PURE_PURE_UBFC_DEEPPHYS_BASIC.yaml
      PURE_PURE_UBFC_PHYSNET_BASIC.yaml
      PURE_PURE_MMPD_TSCAN_BASIC.yaml
      UBFC_UBFC_PURE_TSCAN_BASIC.yaml
      UBFC_UBFC_PURE_DEEPPHYS_BASIC.yaml
      UBFC_UBFC_PURE_PHYSNET_BASIC.yaml
      MMPD_MMPD_UBFC_TSCAN_BASIC.yaml
    Unsupervised Method Sample YAMSL LIST:
      PURE_UNSUPERVISED.yaml
      UBFC_UNSUPERVISED.yaml
    '''
    return parser

def LOO(comet_logger, config, data_loader_dict, outer):
    """trains the model."""


    if config.MODEL.NAME == "Physnet":
        model_trainer = trainer.PhysnetTrainer.PhysnetTrainer(config, data_loader_dict)
    elif config.MODEL.NAME == "Tscan":
        model_trainer = trainer.TscanTrainer.TscanTrainer(config, data_loader_dict)
    elif config.MODEL.NAME == "EfficientPhys":
        model_trainer = trainer.EfficientPhysTrainer.EfficientPhysTrainer(config, data_loader_dict)
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
        early_stop_callback = EarlyStopping(monitor="train_loss_epoch", min_delta=0.00, patience=3, verbose=False,
                                                mode="min")
        comet_logger.experiment.add_tag("early_stopping")
        trainer_light= pl.Trainer(default_root_dir=config.MODEL.MODEL_DIR, callbacks=[early_stop_callback, checkpoint_callback], logger=comet_logger, max_epochs=config.TRAIN.EPOCHS)
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
                #trainer_light = pl.Trainer(devices=1, num_nodes=1,strategy=DDPStrategy(find_unused_parameters=True), accelerator='gpu', default_root_dir=config.MODEL.MODEL_DIR, logger=comet_logger, max_epochs=config.TRAIN.EPOCHS, callbacks=[checkpoint_callback])
                trainer_light = pl.Trainer(accelerator='gpu',default_root_dir=config.MODEL.MODEL_DIR, logger=comet_logger,
                                           max_epochs=config.TRAIN.EPOCHS, callbacks=[checkpoint_callback, lr_monitor])
            else:
                #trainer_light = pl.Trainer(devices=1, num_nodes=1, strategy='ddp', accelerator='gpu', default_root_dir=config.MODEL.MODEL_DIR, logger=comet_logger, max_epochs=config.TRAIN.EPOCHS, callbacks=[checkpoint_callback])
                trainer_light = pl.Trainer(accelerator='gpu', default_root_dir=config.MODEL.MODEL_DIR, logger=comet_logger, max_epochs=config.TRAIN.EPOCHS, callbacks=[checkpoint_callback])



    if config.TOOLBOX_MODE == "LOO":
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
        # #
        new_lr = lr_finder.suggestion()
        model_trainer.lr = new_lr
        comet_logger.experiment.add_tags(
            ['lr-finder'])
        print(model_trainer.lr)
        if config.TEST.USE_LAST_EPOCH:
            trainer_light.fit(model_trainer, data_loader_dict['train'])
            comet_logger.experiment.add_tag("last_epoch")
            trainer_light.test(ckpt_path="last", dataloaders=data_loader_dict['test'])
        else:
            #trainer_light.fit(model_trainer, data_loader_dict['train'], data_loader_dict['valid'])
            trainer_light.fit(model_trainer, data_loader_dict['train'], data_loader_dict['test'])
            comet_logger.experiment.add_tag("best_epoch")
            trainer_light.test(ckpt_path="best", dataloaders=data_loader_dict['test'])

    elif config.TOOLBOX_MODE == "LOO_test":
        trainer_light.test(model_trainer, ckpt_path=config.INFERENCE.MODEL_PATH, dataloaders=data_loader_dict['test'])


        # #load and test the model on the best epoch / last epoch
        # experiment_key = comet_logger.experiment.get_key()
        # comet_logger = cometlogger(experiment_key=experiment_key)
        # trainer_light = pl.trainer(logger=comet_logger)


def train_and_test(comet_logger,config, data_loader_dict):
    """trains the model."""

    # comet_logger = CometLogger(api_key="V1x7OI9PoIRM8yze4prM2FPcE",
    #     project_name="EXP-2-rppg",
    #     workspace="b-acharya",
    #     experiment_name= f"{config.MODEL.NAME}_{config.TRAIN.DATA.DATASET}_{config.VALID.DATA.DATASET}_{config.TEST.DATA.DATASET}",
    #     log_code=True
    # # )
    # hyper_parameters = {
    #     "learning_rate": config.TRAIN.LR,
    #     "epochs": config.TRAIN.EPOCHS
    # }

    # comet_logger.log_hyperparams(hyper_parameters)
    # # comet_logger.experiment.add_tags(["final",config.model.name,config.train.data.preprocess.chunk_length, config.train.data.dataset, config.train.data.preprocess.label_type])
    # comet_logger.experiment.log_asset(CONFIG_FILE_NAME)

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

    if config.TOOLBOX_MODE == "only_test":
        # todo: load the appropriate model with checkpoint path
        # if not os.path.exists(self.config.inference.model_path):
        #     raise valueerror("inference model path error! please check inference.model_path in your yaml.")
        # self.model.load_state_dict(torch.load(self.config.inference.model_path))
        # print("testing uses pretrained model!")
        pass
    else:
        #create checkpoint every epoch and track the validation loss
        if config.TEST.USE_LAST_EPOCH:
            checkpoint_callback = ModelCheckpoint(dirpath=config.MODEL.MODEL_DIR,
                                                  filename=f"{config.TRAIN.MODEL_FILE_NAME}"+"_{epoch}",
                                                  save_last=True,
                                                  )
        else:
            checkpoint_callback = ModelCheckpoint(every_n_epochs=1,
                                              save_top_k=2,
                                              dirpath=config.MODEL.MODEL_DIR,
                                              #filename=f"{config.TRAIN.MODEL_FILE_NAME}"+"_{epoch}",
                                              filename=f"{config.TRAIN.MODEL_FILE_NAME}"+"_{epoch}",
                                              monitor="val_loss_epoch",
                                              mode='min')

        if config.EARLY_STOPPING.TRAIN:
            early_stop_callback = EarlyStopping(monitor="train_loss_epoch", min_delta=0.00, patience=3, verbose=false,
                                                mode="min")
            comet_logger.experiment.add_tag("early_stopping")
            trainer_light= pl.trainer(default_root_dir=config.MODEL.MODEL_DIR, callbacks=[early_stop_callback, checkpoint_callback], logger=comet_logger, max_epochs=config.TRAIN.EPOCHS)
        else:
            if not config.CLUSTER:
                # trainer_light = pl.trainer(limit_train_batches=0.1, limit_val_batches=0.1, default_root_dir=config.model.model_dir, logger=comet_logger, max_epochs=config.train.epochs, callbacks=[checkpoint_callback])
                trainer_light = pl.Trainer(default_root_dir=config.MODEL.MODEL_DIR, logger=comet_logger, max_epochs=config.TRAIN.EPOCHS, callbacks=[checkpoint_callback])
            else:
                #slurm settings
                #trainer_light = pl.Trainer(devices=1, num_nodes=1,strategy='ddp', accelerator='gpu', default_root_dir=config.MODEL.MODEL_DIR, logger=comet_logger, max_epochs=config.TRAIN.EPOCHS, callbacks=[checkpoint_callback])
                trainer_light = pl.Trainer(accelerator='gpu', default_root_dir=config.MODEL.MODEL_DIR, logger=comet_logger, max_epochs=config.TRAIN.EPOCHS, callbacks=[checkpoint_callback])

            #finding the best lr
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
            # #
            new_lr = lr_finder.suggestion()
            model_trainer.lr = new_lr
            comet_logger.experiment.add_tags(
                 ['lr-finder'])
            print(model_trainer.lr)


        # #load and test the model on the best epoch / last epoch
        # experiment_key = comet_logger.experiment.get_key()
        # comet_logger = cometlogger(experiment_key=experiment_key)
        # trainer_light = pl.trainer(logger=comet_logger)

        if config.TEST.USE_LAST_EPOCH:
            trainer_light.fit(model_trainer, data_loader_dict['train'])
            comet_logger.experiment.add_tag("last_epoch")
            trainer_light.test(ckpt_path="last", dataloaders=data_loader_dict['test'])
        else:
            trainer_light.fit(model_trainer, data_loader_dict['train'], data_loader_dict['valid'])
            comet_logger.experiment.add_tag("best_epoch")
            trainer_light.test(ckpt_path="best", dataloaders=data_loader_dict['test'])


        # trainer_light.test(model_trainer, ckpt_path=config.INFERENCE.MODEL_PATH, dataloaders=data_loader_dict['test'])


def test(config, data_loader_dict):
    """tests the model."""
    if config.MODEL.NAME == "Physnet":
        model_trainer = trainer.PhysnetTrainer.PhysnetTrainer(config, data_loader_dict)
    elif config.MODEL.NAME == "Tscan":
        model_trainer = trainer.TscanTrainer.TscanTrainer(config, data_loader_dict)
    elif config.MODEL.NAME == "EfficientPhys":
        model_trainer = trainer.EfficientPhysTrainer.EfficientPhysTrainer(config, data_loader_dict)
    elif config.MODEL.NAME == 'DeepPhys':
        model_trainer = trainer.DeepPhysTrainer.DeepPhysTrainer(config, data_loader_dict)
    elif config.MODEL.NAME == 'rPPGNet':
        model_trainer = trainer.rPPGNetTrainer.rPPGNetTrainer(config, data_loader_dict)
    elif config.MODEL.NAME == 'PhysFormer':
        model_trainer = trainer.PhysFormerTrainer.PhysFormerTrainer(config, data_loader_dict)
    else:
        raise ValueError('Your Model is Not Supported  Yet!')
    # model_trainer.test(data_loader_dict)

    comet_logger = CometLogger(api_key="V1x7OI9PoIRM8yze4prM2FPcE",
                               project_name="EXP-2-rppg",
                               workspace="b-acharya",
                               experiment_name= f"{config.MODEL.NAME}_{config.TRAIN.DATA.DATASET}_{config.VALID.DATA.DATASET}_{config.TEST.DATA.DATASET}",
                               log_code=True
                               )
    hyper_parameters = {
        "learning_rate": config.TRAIN.LR,
        "epochs": config.TRAIN.EPOCHS
    }
    trainer_light = pl.Trainer(default_root_dir=config.MODEL.MODEL_DIR, logger=comet_logger, max_epochs=config.TRAIN.EPOCHS)

    trainer_light.test(model_trainer, ckpt_path="/data/rppg_23_bi_video_nt_lab/processed/Models/CMBP_SizeW128_SizeH128_ClipLength160_DataTypeRaw_DataAugNone_LabelTypeStandardized_Crop_faceTrue_Large_boxTrue_Large_size1.5_Dyamic_DetFalse_det_len1_Median_face_boxFalse/CMBP_CMBP_CMBP_physformer_epoch=11.ckpt", dataloaders=data_loader_dict['test'])


def unsupervised_method_inference(config, data_loader):

    if not config.UNSUPERVISED.METHOD:
        raise ValueError("Please set unsupervised method in yaml!")
    for unsupervised_method in config.UNSUPERVISED.METHOD:
        run_name = unsupervised_method
        comet_logger = CometLogger(api_key="V1x7OI9PoIRM8yze4prM2FPcE",
                                   #project_name="unsupervised-methods",
                                   project_name="test-hr",
                                   workspace="b-acharya",
                                   experiment_name=f"{run_name}",
                                   log_code=False
                                   )
        comet_logger.experiment.add_tag(f"{config.UNSUPERVISED.DATA.DATASET}")
        comet_logger.experiment.add_tag(f"{config.INFERENCE.EVALUATION_METHOD}")
        if unsupervised_method == "POS":
            unsupervised_predict(config, data_loader, "POS", comet_logger)
        elif unsupervised_method == "CHROM":
            unsupervised_predict(config, data_loader, "CHROM", comet_logger)
        elif unsupervised_method == "ICA":
            unsupervised_predict(config, data_loader, "ICA", comet_logger)
        elif unsupervised_method == "GREEN":
            unsupervised_predict(config, data_loader, "GREEN", comet_logger)
        elif unsupervised_method == "LGI":
            unsupervised_predict(config, data_loader, "LGI", comet_logger)
        elif unsupervised_method == "PBV":
            unsupervised_predict(config, data_loader, "PBV", comet_logger)
        elif unsupervised_method == "dummy":
            unsupervised_predict(config, data_loader, "dummy", comet_logger)
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
    if config.TOOLBOX_MODE == "train_and_test":
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
            train_loader = data_loader.CMBPLoader.CMBPLoader
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
        elif config.VALID.DATA.DATASET is None and not config.TEST.USE_LAST_EPOCH:
                raise ValueError("Validation dataset not specified despite USE_LAST_EPOCH set to False!")
        else:
            raise ValueError("Unsupported dataset! Currently supporting UBFC, PURE, MMPD, and SCAMPS.")      

        # Create and initialize the valid dataloader given the correct toolbox mode,
        # a supported dataset name, and a valid dataset path
        if (config.VALID.DATA.DATASET and config.VALID.DATA.DATA_PATH and not config.TEST.USE_LAST_EPOCH):
            valid_data = valid_loader(
                name="valid",
                data_path=config.VALID.DATA.DATA_PATH,
                config_data=config.VALID.DATA,
                model = config.MODEL.NAME
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

    if config.TOOLBOX_MODE == "preprocess":
            print("done with preprocessing")

    if config.TOOLBOX_MODE == "train_and_test" or config.TOOLBOX_MODE == "only_test":
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

    elif config.TOOLBOX_MODE == "unsupervised_method":
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
        else:
            raise ValueError("Unsupported dataset! Currently supporting UBFC, PURE, MMPD, and SCAMPS.")

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
    elif config.TOOLBOX_MODE == "leave_one_out":
        # unsupervised method dataloader
        if config.LOO.DATA.DATASET == "COHFACE":
            # unsupervised_loader = data_loader.COHFACELoader.COHFACELoader
            raise ValueError("Unsupported dataset! Currently supporting UBFC, PURE, MMPD, and SCAMPS.")
        elif config.LOO.DATA.DATASET == "UBFC":
            loo_loader = data_loader.UBFCLoader.UBFCLoader
        elif config.LOO.DATA.DATASET == "PURE":
            loo_loader = data_loader.PURELoader.PURELoader
        elif config.LOO.DATA.DATASET == "SCAMPS":
            loo_loader = data_loader.SCAMPSLoader.SCAMPSLoader
        elif config.LOO.DATA.DATASET == "MMPD":
            loo_loader = data_loader.MMPDLoader.MMPDLoader
        elif config.LOO.DATA.DATASET == "CMBP":
            loo_loader = data_loader.CMBPLoader.CMBPLoader
        else:
            raise ValueError("Unsupported dataset! Currently supporting UBFC, PURE, MMPD, and SCAMPS.")


    elif config.TOOLBOX_MODE == "LOO" or config.TOOLBOX_MODE == "LOO_test":
        K_fold = 10
        #train_loader
        if config.TRAIN.DATA.DATASET == "UBFC":
            loader = data_loader.UBFCLoader.UBFCLoader
        elif config.TRAIN.DATA.DATASET == "PURE":
            loader = data_loader.PURELoader.PURELoader
            participants = np.arange(1, 11)
            kf_inner = KFold(n_splits=9, shuffle=True, random_state=42)
        elif config.TRAIN.DATA.DATASET == "SCAMPS":
            loader = data_loader.SCAMPSLoader.SCAMPSLoader
        elif config.TRAIN.DATA.DATASET == "MMPD":
            loader = data_loader.MMPDLoader.MMPDLoader
        elif config.TRAIN.DATA.DATASET == "BP4DPlus":
            loader = data_loader.BP4DPlusLoader.BP4DPlusLoader
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

        #Initial processing of Dataset
        if (config.TRAIN.DATA.DATASET and config.TRAIN.DATA.DATA_PATH):

            train_data_loader = loader(
                name="train",
                data_path=config.TRAIN.DATA.DATA_PATH,
                config_data=config.TRAIN.DATA,
                model = config.MODEL.NAME
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


        list_file['id'] = list_file['input_files'].apply(get_participant_index)
        participants = list(set(list_file['id']))

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

        for i, (train_index, test_index) in enumerate(kf.split(participants)):
            if i != int(config.FOLD):
                print(i)
                continue
            train_participants = np.array([participants[i] for i in train_index])
            test_participants = np.array([participants[i] for i in test_index])
            train_path, test_path = create_path_list(list_file, train_participants, test_participants, i)

            # for j, (train_index_inner, valid_index_inner) in enumerate(kf_inner.split(train_participants)):
            #
            #     if config.TOOLBOX_MODE == "LOO_test":
            #         if j != config.INNER_FOLD:
            #             continue
            #     train_participants_inner = [train_participants[i] for i in train_index_inner]
            #     valid_participants_inner = [train_participants[i] for i in valid_index_inner]
            #     train_path, valid_path = create_path_list(list_file, train_participants_inner, valid_participants_inner, i, j, outer=False)
            #
            #     print(train_path, test_path)

            train_df = pd.read_csv(train_path)
                # valid_df = pd.read_csv(valid_path)
            test_df = pd.read_csv(test_path)

            config.defrost()
            config.TRAIN.DATA.FILE_LIST_PATH = str(train_path)
            # config.VALID.DATA.FILE_LIST_PATH = str(valid_path)
            config.TEST.DATA.FILE_LIST_PATH = str(test_path)
            print(config.TEST.DATA.FILE_LIST_PATH)
            print(config.TRAIN.DATA.FILE_LIST_PATH)
            config.TRAIN.DATA.DO_PREPROCESS = False
            config.VALID.DATA.DO_PREPROCESS = False
            config.TEST.DATA.DO_PREPROCESS = False
            config.freeze()

            if config.TOOLBOX_MODE == "LOO" :

                comet_logger = CometLogger(api_key="V1x7OI9PoIRM8yze4prM2FPcE",
                                           project_name="Exp1-Retraining-Best-epcoh",
                                           workspace="b-acharya",
                                           #experiment_name= f"{config.MODEL.NAME}_{config.TRAIN.DATA.DATASET}_{config.VALID.DATA.DATASET}_{config.TEST.DATA.DATASET}",
                                           experiment_name= f"{config.TRAIN.DATA.DATASET}_{config.MODEL.NAME}_FOLD_{i}",
                                           log_code=True
                                           )
                hyper_parameters = {
                    "Learning_rate": config.TRAIN.LR,
                    "epochs": config.TRAIN.EPOCHS
                }

            elif config.TOOLBOX_MODE == "LOO_test":

                comet_logger = CometLogger(api_key="V1x7OI9PoIRM8yze4prM2FPcE",
                                               project_name="Exp1-Leave-One-Out-Welch",
                                               workspace="b-acharya",
                                               #experiment_name= f"{config.MODEL.NAME}_{config.TRAIN.DATA.DATASET}_{config.VALID.DATA.DATASET}_{config.TEST.DATA.DATASET}",
                                               experiment_name= f"{config.TRAIN.DATA.DATASET}_{config.MODEL.NAME}_LOO__FOLD_{i}",
                                               log_code=True
                                               )
                hyper_parameters = {
                        "Learning_rate": config.TRAIN.LR,
                        "epochs": config.TRAIN.EPOCHS
                }
            else:
                raise NotImplementedError

            comet_logger.log_hyperparams(hyper_parameters)
            comet_logger.experiment.add_tags(
                    [config.MODEL.NAME, config.TRAIN.DATA.PREPROCESS.CHUNK_LENGTH, config.TRAIN.DATA.DATASET,
                     config.TRAIN.DATA.PREPROCESS.LABEL_TYPE, config.TRAIN.BATCH_SIZE])
            comet_logger.experiment.add_tag(f"Outer_loop:{i}")
            comet_logger.experiment.log_asset(train_path)
            comet_logger.experiment.log_asset(test_path)
            # comet_logger.experiment.log_asset(valid_path)
            comet_logger.experiment.log_asset(CONFIG_FILE_NAME)

                #ALL run on the same dataset and dont need a asepearte loader
            train_data = loader(
                    name="train",
                    data_path=config.TRAIN.DATA.DATA_PATH,
                    config_data=config.TRAIN.DATA,
                    model = config.MODEL.NAME
                )

                #same as test data loader only difference is that paths
                #Has left out set is used for both testing and validation
                # valid_data = loader(
                #     name="valid",
                #     data_path=config.VALID.DATA.DATA_PATH,
                #     config_data=config.VALID.DATA,
                #     model = config.MODEL.NAME
                # )

            test_data = loader(
                    name="test",
                    data_path=config.TEST.DATA.DATA_PATH,
                    config_data=config.TEST.DATA,
                    model=config.MODEL.NAME
            )

            data_loader_dict['train'] = DataLoader(
                    dataset=train_data,
                    num_workers=16,
                    batch_size=config.TRAIN.BATCH_SIZE,
                    shuffle=True,
                    worker_init_fn=seed_worker,
                    generator=train_generator,
            )
                # data_loader_dict["valid"] = DataLoader(
                #     dataset=valid_data,
                #     num_workers=16,
                #     batch_size=config.TRAIN.BATCH_SIZE,  # batch size for val is the same as train
                #     shuffle=False,
                #     worker_init_fn=seed_worker,
                #     generator=general_generator,
                # )
                #
            data_loader_dict["test"] = DataLoader(
                    dataset=test_data,
                    num_workers=16,
                    batch_size=config.INFERENCE.BATCH_SIZE,
                    shuffle=False,
                    worker_init_fn=seed_worker,
                    generator=general_generator,
            )

            LOO(comet_logger, config, data_loader_dict, i)



    else:
        raise ValueError("Unsupported toolbox_mode! Currently support train_and_test or only_test or unsupervised_method.")

    if config.TOOLBOX_MODE == "train_and_test":
        #cross validated evaluation of the methods on MBP-PPG dataset
        # kf = KFold(n_splits=10, shuffle=True, random_state=42)
        #
        # file_list_path = config.TRAIN.DATA.FILE_LIST_PATH
        # list_file = pd.read_csv(file_list_path)
        #
        # def get_participant_index(pathfile):
        #     "Pandas parsing for geting the index from the full path"
        #     if "CMBP" in pathfile:
        #         path = pathfile.split("/")[-1]
        #         index = path = path.split("_")[0]
        #         return index
        #
        #     else:
        #         path = pathfile.split("/")[-1]
        #         if "v" in path:
        #             index = path.split("v")[0]
        #             return int(index)
        #         elif "_" in path:
        #             index = path.split("_")[0]
        #             if len(index) == 3:
        #                 return int(index[0])
        #             else:
        #                 return int(index[:2])
        #
        #
        # list_file['id'] = list_file['input_files'].apply(get_participant_index)
        # participants = list(set(list_file['id']))
        #
        # def create_path_list(listdf, train, test, i, j=0, outer=True):
        #     train_df = listdf[listdf['id'].isin(train)]
        #     test_df = listdf[listdf['id'].isin(test)]
        #     if outer:
        #         train_path = pathlib.Path(listdf["input_files"][0]).parent / f"fold_{i}_train.csv"
        #         test_path = pathlib.Path(listdf["input_files"][0]).parent / f"fold_{i}_test.csv"
        #         train_df.to_csv(str(train_path))
        #         test_df.to_csv(str(test_path))
        #     else:
        #         train_path = pathlib.Path(listdf["input_files"][0]).parent / f"fold_{i}_{j}_train.csv"
        #         test_path = pathlib.Path(listdf["input_files"][0]).parent / f"fold_{i}_{j}_valid.csv"
        #         train_df.to_csv(str(train_path))
        #         test_df.to_csv(str(test_path))
        #
        #     return train_path, test_path
        #
        #
        # for i, (train_index, valid_index) in enumerate(kf.split(participants)):
        #     if i != int(config.FOLD):
        #         print(i)
        #         continue
        #
        #     train_participants = np.array([participants[i] for i in train_index])
        #     valid_participants = np.array([participants[i] for i in valid_index])
        #
        #     train_path, valid_path = create_path_list(list_file, train_participants, valid_participants, i, outer=False)
        #
        #     print(train_path, valid_path)
        #     train_df = pd.read_csv(train_path)
        #     valid_df = pd.read_csv(valid_path)
        #
        #     config.defrost()
        #     config.TRAIN.DATA.FILE_LIST_PATH = str(train_path)
        #     config.VALID.DATA.FILE_LIST_PATH = str(valid_path)
        #     print(config.VALID.DATA.FILE_LIST_PATH)
        #     print(config.TRAIN.DATA.FILE_LIST_PATH)
        #     config.TRAIN.DATA.DO_PREPROCESS = False
        #     config.VALID.DATA.DO_PREPROCESS = False
        #     config.TEST.DATA.DO_PREPROCESS = False
        #     config.freeze()
        #
        # comet_logger = CometLogger(api_key="V1x7OI9PoIRM8yze4prM2FPcE",
        #                                project_name="Exp2-public-on-cmbp-best-epoch",
        #                                workspace="b-acharya",
        #                                experiment_name=f"{config.TRAIN.DATA.DATASET}_{config.TEST.DATA.DATASET}_{config.MODEL.NAME}",
        #                                log_code=True
        #                                )

        comet_logger = CometLogger(api_key="V1x7OI9PoIRM8yze4prM2FPcE",
                                   project_name="Exp3-cmbp-on-public-best-epoch",
                                   workspace="b-acharya",
                                   experiment_name=f"{config.TRAIN.DATA.DATASET}_{config.TEST.DATA.DATASET}_{config.MODEL.NAME}",
                                   log_code=True
                                   )
        hyper_parameters = {
            "Learning_rate": config.TRAIN.LR,
            "epochs": config.TRAIN.EPOCHS
        }

        comet_logger.log_hyperparams(hyper_parameters)
        comet_logger.experiment.add_tags(
            [config.MODEL.NAME, config.TRAIN.DATA.PREPROCESS.CHUNK_LENGTH, config.TRAIN.DATA.DATASET,
            config.TRAIN.DATA.PREPROCESS.LABEL_TYPE, config.TRAIN.BATCH_SIZE])
        # comet_logger.experiment.add_tag(f"Outer_loop:{i}")
        # comet_logger.experiment.log_asset(train_path)
        # comet_logger.experiment.log_asset(test())
        comet_logger.experiment.log_asset(CONFIG_FILE_NAME)

        #         # ALL run on the same dataset and dont need a asepearte loader
        #     train_data = train_loader(
        #         name="train",
        #         data_path=config.TRAIN.DATA.DATA_PATH,
        #         config_data=config.TRAIN.DATA,
        #         model=config.MODEL.NAME
        #     )
        #
        #         # same as test data loader only difference is that paths
        #         # Has left out set is used for both testing and validation
        #     valid_data = valid_loader(
        #         name="valid",
        #         data_path=config.VALID.DATA.DATA_PATH,
        #         config_data=config.VALID.DATA,
        #         model=config.MODEL.NAME
        #     )

            # data_loader_dict['train'] = DataLoader(
            #     dataset=train_data,
            #     num_workers=16,
            #     batch_size=config.TRAIN.BATCH_SIZE,
            #     shuffle=True,
            #     worker_init_fn=seed_worker,
            #     generator=train_generator,
            # )
            #
            # data_loader_dict["valid"] = DataLoader(
            #     dataset=valid_data,
            #     num_workers=16,
            #     batch_size=config.TRAIN.BATCH_SIZE,  # batch size for val is the same as train
            #     shuffle=False,
            #     worker_init_fn=seed_worker,
            #     generator=general_generator,
            # )

        train_and_test(comet_logger, config, data_loader_dict)
    elif config.TOOLBOX_MODE == "only_test":
        test(config, data_loader_dict)
    elif config.TOOLBOX_MODE == "unsupervised_method":
        unsupervised_method_inference(config, data_loader_dict)
    elif config.TOOLBOX_MODE == "leave_one_out":
        num_of_participants = loo_loader.num_of_participants
        exp_name = config.LOO.DATA.DATASET + "LOO" + "MBP"
        #mlflow experiment id
        try:
            experiment_id = mlflow.create_experiment(exp_name)
        except:
            experiment_id = mlflow.get_experiment_by_name(exp_name).experiment_id

        for i in range(1, num_of_participants+1):
            #run_name = "dummy" + str(i)
            run_name = "particiapnt_" + str(i)
            mlflow_run = mlflow.start_run(experiment_id=experiment_id, run_name=run_name)
            config.defrost()
            config.LOO.DATA.PARTICIPANT_IDS = [j for j in range(1,num_of_participants+1) if j != i]
            print(config.LOO.DATA.PARTICIPANT_IDS)
            config.TRAIN.MODEL_FILE_NAME = "LOO_ID_" + str(i)
            config.freeze()
            loo_train = loo_loader(
                name="train",
                data_path=config.LOO.DATA.DATA_PATH,
                config_data=config.LOO.DATA)
            data_loader_dict["train"] = DataLoader(
                dataset=loo_train,
                num_workers=12,
                batch_size=config.TRAIN.BATCH_SIZE,
                shuffle=True,
                worker_init_fn=seed_worker,
                generator=train_generator
            )
            config.defrost()
            config.LOO.DATA.PARTICIPANT_IDS = [j for j in range(1,num_of_participants+1) if j == i]
            config.freeze()

            loo_test = loo_loader(
                name="loo_valid",
                data_path=config.LOO.DATA.DATA_PATH,
                config_data=config.LOO.DATA)
            data_loader_dict["valid"] = DataLoader(
                dataset=loo_test,
                num_workers=12,
                batch_size=1,
                shuffle=False,
                worker_init_fn=seed_worker,
                generator=general_generator
            )
            test_loader = data_loader.CMBPLoader.CMBPLoader
            test = test_loader(
                name="test",
                data_path=config.TEST.DATA.DATA_PATH,
                config_data=config.TEST.DATA)
            data_loader_dict["test"] = DataLoader(
                dataset=test,
                num_workers=12,
                batch_size=1,
                shuffle=False,
                worker_init_fn=seed_worker,
                generator=general_generator
            )

            with mlflow_run:
                mlflow.log_param("participant left out", str(i))

    else:
        print("TOOLBOX_MODE only support train_and_test or only_test !", end='\n\n')
