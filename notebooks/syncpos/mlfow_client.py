import yaml
import os
import sys
sys.path.append("/homes/bacharya/rPPG-Toolbox")
from config import get_config
from dataset import data_loader
import argparse
from torch.utils.data import DataLoader
from notebooks import syncpos
import random
import torch
import numpy as np
from mlflow import MlflowClient
import pathlib

import mlflow
# mlflow.pytorch.autolog()
# mlflow.set_tracking_uri("file:///homes/bacharya/mlruns")

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

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

if __name__ == "__main__":

    # config_file_path = "/homes/bacharya/rPPG-Toolbox/configs/train_configs/PURE_PURE_PURE_DEEPPHYS_FINGER_PPG.yaml"
    parser = argparse.ArgumentParser()

    parser.add_argument('--config_file', required=True, type=str, help="The config file.")
    args = parser.parse_args()
    args.FOLD = 0
    config = get_config(args)


    #Load the pure dataset
    test_loader = data_loader.PURELoader.PURELoader(
                    name="test",
                    data_path=config.TEST.DATA.DATA_PATH,
                    config_data=config.TEST.DATA,
                    model=config.MODEL.NAME
                )

    # Create your data loaders
    test_dataloader = DataLoader(
            dataset=test_loader,
            num_workers=16,
            batch_size=config.INFERENCE.BATCH_SIZE,
            shuffle=False,
            worker_init_fn = seed_worker
    )

    # Define your experiment name
    if config.TRAIN.DATA.PREPROCESS.USE_PSUEDO_PPG_LABEL:
        EXPERIMENT_NAME = "Phase_2_PURE_PURE_FACE_PPG"
    else:
        EXPERIMENT_NAME = "Phase_2_PURE_PURE_FINGER_PPG"

    MLFLOW_TRACKING_URI = "./mlruns"  # Path where MLflow stores artifacts

    # Initialize MLflow client
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()

    # Get Experiment ID
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        raise ValueError(f"Experiment {EXPERIMENT_NAME} not found!")

    experiment_id = experiment.experiment_id

    # Get all runs in this experiment
    runs = client.search_runs(experiment_id)

    # Select a specific run (e.g., the one with a particular learning rate)
    # lrs = [1.0, 0.1, 0.01, 0.001, 0.0001, 0.5, 0.05, 0.005]
    lrs = [0.5]
    # lrs = [1.0]

    for selected_lr in lrs:
        selected_run = None

        for run in runs:
            run_name = run.data.tags['mlflow.runName']

            if run_name.split("_")[0] == str(selected_lr):

                print(f"Selected Run ID: {run.info.run_id}")

                 # Load the model
                model_path = f"/homes/bacharya/rPPG-Toolbox/notebooks/syncpos/mlruns/{experiment_id}/{run.info.run_id}/artifacts/model/checkpoints/"
                model_path = list(pathlib.Path(model_path).rglob("*.ckpt"))

                print(model_path[0])

                model = syncpos.DeepPhysOnlyMotionTrainer.DeepPhysOnlyMotionTrainer(config, test_loader, lr=selected_lr)

                # Initialize the custom trainer
                trainer = syncpos.CustomTrainer.ConvergenceThenFixedEpochsTrainer(
                     model=model,
                     train_dataloader=None,
                     test_dataloader=test_dataloader,
                     config=config,
                     post_convergence_epochs=48,
                     lr=selected_lr
                 )

                # Final evaluation on best model
                print(trainer.final_evaluation(str(model_path[0])))