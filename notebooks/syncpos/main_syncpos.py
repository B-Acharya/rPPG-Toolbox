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

    parser.add_argument('--config_file', required=True, type=str, help="The name of the model.")
    parser.add_argument('--lr', required=True, type=float, help="The initial learning rate.")
    # config_file_path = "/homes/bacharya/rPPG-Toolbox/configs/train_configs/PURE_PURE_PURE_DEEPPHYS_FACE_PPG.yaml"
    # args = argparse.Namespace()
    args = parser.parse_args()
    args.FOLD = 0
    config = get_config(args)

    #Load the pure dataset
    train_loader = data_loader.PURELoader.PURELoader(
                    name="train",
                    data_path=config.TRAIN.DATA.DATA_PATH,
                    config_data=config.TRAIN.DATA,
                    model = config.MODEL.NAME
                )

    test_loader = data_loader.PURELoader.PURELoader(
                    name="test",
                    data_path=config.TEST.DATA.DATA_PATH,
                    config_data=config.TEST.DATA,
                    model=config.MODEL.NAME
                )

    # Create your data loaders
    train_dataloader = DataLoader(
            dataset=train_loader,
            num_workers=16,
            batch_size=config.TRAIN.BATCH_SIZE,
            shuffle=True,
            generator=general_generator,
            worker_init_fn=seed_worker
    )

    test_dataloader = DataLoader(
            dataset=test_loader,
            num_workers=16,
            batch_size=config.INFERENCE.BATCH_SIZE,
            shuffle=False,
            worker_init_fn = seed_worker
    )

    # Create your model
    model = syncpos.DeepPhysOnlyMotionTrainer.DeepPhysOnlyMotionTrainer(config, train_dataloader, lr=args.lr)

    # Initialize the custom trainer
    trainer = syncpos.CustomTrainer.ConvergenceThenFixedEpochsTrainer(
        model=model,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        config=config,
        post_convergence_epochs=48,
        lr=args.lr
    )

    # Train the model
    converged_model_path, epoch = trainer.train()

    post_convergence_model_path = trainer.retrain(converged_model_path, epoch, 16)

    # Final evaluation on best model
    print(trainer.final_evaluation(post_convergence_model_path))