""" The main function of rPPG deep learning pipeline."""

import argparse
import random
import time

import numpy as np
import torch
from config import get_config
from dataset import data_loader
from neural_methods import trainer
from unsupervised_methods.unsupervised_predictor import unsupervised_predict
from torch.utils.data import DataLoader
import lightning.pytorch as pl
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import CometLogger
from lightning.pytorch.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt

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


def train_and_test(config, data_loader_dict):
    """Trains the model."""

    comet_logger = CometLogger(api_key="V1x7OI9PoIRM8yze4prM2FPcE",
        project_name="rppg",
        workspace="b-acharya",
        experiment_name= f"{config.MODEL.NAME}_{config.TRAIN.DATA.DATASET}_{config.VALID.DATA.DATASET}_{config.TEST.DATA.DATASET}",
        log_code=False
    )
    # Overfitting testing
    # comet_logger = CometLogger(api_key="V1x7OI9PoIRM8yze4prM2FPcE",
    #     project_name="rppg",
    #     workspace="b-acharya",
    #     experiment_name= "overfit-testing",
    #     log_code=False
    # )
    # comet_logger.add_tag("train_validation_test")
    hyper_parameters = {
        "Learning_rate": config.TRAIN.LR,
        "epochs": config.TRAIN.EPOCHS
    }

    comet_logger.log_hyperparams(hyper_parameters)
    comet_logger.experiment.add_tags(["final",config.MODEL.NAME,config.TRAIN.DATA.PREPROCESS.CHUNK_LENGTH, config.TRAIN.DATA.DATASET, config.TRAIN.DATA.PREPROCESS.LABEL_TYPE])
    comet_logger.experiment.log_asset(CONFIG_FILE_NAME)

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

    if config.TOOLBOX_MODE == "only_test":
        # TODO: load the appropriate model with checkpoint path
        # if not os.path.exists(self.config.INFERENCE.MODEL_PATH):
        #     raise ValueError("Inference model path error! Please check INFERENCE.MODEL_PATH in your yaml.")
        # self.model.load_state_dict(torch.load(self.config.INFERENCE.MODEL_PATH))
        # print("Testing uses pretrained model!")
        pass
    else:
        #create checkpoint every epoch and track the validation loss
        checkpoint_callback = ModelCheckpoint(every_n_epochs=1,
                                              save_top_k=-1,
                                              dirpath=config.MODEL.MODEL_DIR,
                                              filename=f"{config.TRAIN.MODEL_FILE_NAME}"+"_{epoch}",
                                              monitor="val_loss_epoch",
                                              mode='min')
        if config.EARLY_STOPPING.TRAIN:
            early_stop_callback = EarlyStopping(monitor="train_loss_epoch", min_delta=0.00, patience=3, verbose=False,
                                                mode="min")
            comet_logger.experiment.add_tag("Early_stopping")
            trainer_light= pl.Trainer(default_root_dir=config.MODEL.MODEL_DIR, callbacks=[early_stop_callback, checkpoint_callback], logger=comet_logger, max_epochs=config.TRAIN.EPOCHS)
        else:
            #trainer_light = pl.Trainer(limit_train_batches=0.1, limit_val_batches=0.1, default_root_dir=config.MODEL.MODEL_DIR, logger=comet_logger, max_epochs=config.TRAIN.EPOCHS, callbacks=[checkpoint_callback])
            # trainer_light = pl.Trainer(default_root_dir=config.MODEL.MODEL_DIR, logger=comet_logger, max_epochs=config.TRAIN.EPOCHS, callbacks=[checkpoint_callback])


            #Slurm settings
            trainer_light = pl.Trainer(devices=1, num_nodes=1,strategy='ddp_find_unused_parameters_true', accelerator='gpu', default_root_dir=config.MODEL.MODEL_DIR, logger=comet_logger, max_epochs=config.TRAIN.EPOCHS, callbacks=[checkpoint_callback])

            #finding the best LR
            # from lightning.pytorch.tuner import Tuner
            # tuner = Tuner(trainer_light)
            # lr_finder = tuner.lr_find(model_trainer, train_dataloaders=data_loader_dict['train'], val_dataloaders=data_loader_dict['valid'])
            # fig = lr_finder.plot(suggest=True)
            # comet_logger.experiment.log_figure(figure=fig, figure_name="Lr-curve")
            # plt.close()
            # #
            # new_lr = lr_finder.suggestion()
            # model_trainer.lr = new_lr
            # comet_logger.experiment.add_tags(
            #     ['lr-finder'])
            # print(model_trainer.lr)


        trainer_light.fit(model_trainer, data_loader_dict['train'], data_loader_dict['valid'])

        # #load and test the model on the best epoch / last epoch
        # experiment_key = comet_logger.experiment.get_key()
        # comet_logger = CometLogger(experiment_key=experiment_key)
        # trainer_light = pl.Trainer(logger=comet_logger)

        if config.TEST.USE_LAST_EPOCH:
            comet_logger.experiment.add_tag("Last_epoch")
            trainer_light.test(ckpt_path="last", dataloaders=data_loader_dict['test'])
        else:
            comet_logger.experiment.add_tag("Best_epoch")
            trainer_light.test(ckpt_path="best", dataloaders=data_loader_dict['test'])


def test(config, data_loader_dict):
    """Tests the model."""
    if config.MODEL.NAME == "Physnet":
        model_trainer = trainer.PhysnetTrainer.PhysnetTrainer(config, data_loader_dict)
    elif config.MODEL.NAME == "Tscan":
        model_trainer = trainer.TscanTrainer.TscanTrainer(config, data_loader_dict)
    elif config.MODEL.NAME == "EfficientPhys":
        model_trainer = trainer.EfficientPhysTrainer.EfficientPhysTrainer(config, data_loader_dict)
    elif config.MODEL.NAME == 'DeepPhys':
        model_trainer = trainer.DeepPhysTrainer.DeepPhysTrainer(config, data_loader_dict)
    else:
        raise ValueError('Your Model is Not Supported  Yet!')
    model_trainer.test(data_loader_dict)


def unsupervised_method_inference(config, data_loader):

    exp_name = config.TOOLBOX_MODE + config.UNSUPERVISED.DATA.DATASET
    # mlflow experiment id
    try:
        experiment_id = mlflow.create_experiment(exp_name)
    except:
        experiment_id = mlflow.get_experiment_by_name(exp_name).experiment_id


    if not config.UNSUPERVISED.METHOD:
        raise ValueError("Please set unsupervised method in yaml!")
    for unsupervised_method in config.UNSUPERVISED.METHOD:
        run_name = unsupervised_method
        mlflow_run = mlflow.start_run(experiment_id=experiment_id, run_name=run_name)
        with mlflow_run:
            if unsupervised_method == "POS":
                unsupervised_predict(config, data_loader, "POS")
            elif unsupervised_method == "CHROM":
                unsupervised_predict(config, data_loader, "CHROM")
            elif unsupervised_method == "ICA":
                unsupervised_predict(config, data_loader, "ICA")
            elif unsupervised_method == "GREEN":
                unsupervised_predict(config, data_loader, "GREEN")
            elif unsupervised_method == "LGI":
                unsupervised_predict(config, data_loader, "LGI")
            elif unsupervised_method == "PBV":
                unsupervised_predict(config, data_loader, "PBV")
            else:
                raise ValueError("Not supported unsupervised method!")


if __name__ == "__main__":
    # parse arguments.
    parser = argparse.ArgumentParser()
    parser = add_args(parser)
    parser = trainer.BaseTrainer.BaseTrainer.add_trainer_args(parser)
    parser = data_loader.BaseLoader.BaseLoader.add_data_loader_args(parser)
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
                config_data=config.TRAIN.DATA)
            data_loader_dict['train'] = DataLoader(
                dataset=train_data_loader,
                num_workers=16,
                batch_size=config.TRAIN.BATCH_SIZE,
                shuffle=True,
                worker_init_fn=seed_worker,
                generator=train_generator
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
                config_data=config.VALID.DATA)
            data_loader_dict["valid"] = DataLoader(
                dataset=valid_data,
                num_workers=16,
                batch_size=config.TRAIN.BATCH_SIZE,  # batch size for val is the same as train
                shuffle=False,
                worker_init_fn=seed_worker,
                generator=general_generator
            )
        else:
            data_loader_dict['valid'] = None

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
                config_data=config.TEST.DATA)
            data_loader_dict["test"] = DataLoader(
                dataset=test_data,
                num_workers=16,
                batch_size=config.INFERENCE.BATCH_SIZE,
                shuffle=False,
                worker_init_fn=seed_worker,
                generator=general_generator
            )
        else:
            data_loader_dict['test'] = None

    elif config.TOOLBOX_MODE == "unsupervised_method":
        # unsupervised method dataloader
        if config.UNSUPERVISED.DATA.DATASET == "COHFACE":
            # unsupervised_loader = data_loader.COHFACELoader.COHFACELoader
            raise ValueError("Unsupported dataset! Currently supporting UBFC, PURE, MMPD, and SCAMPS.")
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
            config_data=config.UNSUPERVISED.DATA)
        data_loader_dict["unsupervised"] = DataLoader(
            dataset=unsupervised_data,
            num_workers=16,
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
    else:
        raise ValueError("Unsupported toolbox_mode! Currently support train_and_test or only_test or unsupervised_method.")

    if config.TOOLBOX_MODE == "train_and_test":
        train_and_test(config, data_loader_dict)
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
