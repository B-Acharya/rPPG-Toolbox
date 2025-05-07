import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import logging
import os
import glob
import scipy.io as sio
from torch.utils.tensorboard import SummaryWriter
import datetime
import os
from types import SimpleNamespace
from config import get_config,_C
import argparse

from VAE_model_16 import RPPGVAE_16,VAE_16_Trainer,init_rppg_vae
from simple_mmpd_loader import prepare_mmpd_dataloaders

def prepare_data_for_vae(data, labels, device):

    return data.to(device), labels.to(device)


def train_rppg_vae(model, config, train_loader, valid_loader=None, device='cuda'):


    trainer = VAE_16_Trainer(model, config, device)
    num_epochs = config.TRAIN.EPOCHS
    best_valid_loss = float('inf')


    best_model_dir = "saved_models"
    os.makedirs(best_model_dir, exist_ok=True)
    best_model_path = os.path.join(best_model_dir, "VAE16_best_model.pth")


    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join("tensorboard_logs", f"rppg_vae_{timestamp}")
    writer = SummaryWriter(log_dir=log_dir)
    print(f"TensorBoard logs will be saved to {log_dir}")

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    patience = 50
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        train_losses = []

        pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}')
        for batch_idx, (frames, bvp) in enumerate(pbar):
            frames, bvp = prepare_data_for_vae(frames, bvp, device)

            losses = trainer.train_step(frames, bvp)
            train_losses.append(losses)

            pbar.set_postfix({
                'loss': f"{losses['total_loss']:.2f}",
                'bvp': f"{losses['bvp_loss']:.2f}",
                'temp': f"{losses['bvp_temporal_loss']:.2f}",
            })

        avg_train_losses = {
            k: np.mean([x[k] for x in train_losses])
            for k in train_losses[0].keys()
        }


        writer.add_scalar('Loss/train/total', avg_train_losses["total_loss"], epoch)
        writer.add_scalar('Loss/train/bvp', avg_train_losses["bvp_loss"], epoch)
        writer.add_scalar('Loss/train/bvp_temporal', avg_train_losses["bvp_temporal_loss"], epoch)
        writer.add_scalar('Loss/train/frame', avg_train_losses["frame_loss"], epoch)
        writer.add_scalar('Loss/train/kl', avg_train_losses["kl_loss"], epoch)

        if valid_loader is not None:
            model.eval()
            valid_losses = []

            with torch.no_grad():
                for frames, bvp in valid_loader:
                    frames, bvp = prepare_data_for_vae(frames, bvp, device)

                    losses = trainer.validate_step(frames, bvp)
                    valid_losses.append(losses)

            avg_valid_losses = {
                k: np.mean([x[k] for x in valid_losses])
                for k in valid_losses[0].keys()
            }


            writer.add_scalar('Loss/valid/total', avg_valid_losses["total_loss"], epoch)
            writer.add_scalar('Loss/valid/bvp', avg_valid_losses["bvp_loss"], epoch)
            writer.add_scalar('Loss/valid/bvp_temporal', avg_valid_losses["bvp_temporal_loss"], epoch)
            writer.add_scalar('Loss/valid/frame', avg_valid_losses["frame_loss"], epoch)
            writer.add_scalar('Loss/valid/kl', avg_valid_losses["kl_loss"], epoch)


            best_combined_loss = avg_valid_losses['bvp_loss'] + avg_valid_losses['bvp_temporal_loss']
            current_valid_loss = best_combined_loss

            if current_valid_loss < best_valid_loss:
                patience_counter = 0
                best_valid_loss = current_valid_loss

                # Save the best model to your manually specified path
                print(f"Saving best model to {best_model_path} with  combined BVP loss {best_valid_loss:.4f}")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': trainer.optimizer.state_dict(),
                    'loss': best_valid_loss,
                }, best_model_path)

                # Log the best model epoch
                writer.add_text('Training Info',
                                f'New best model at epoch {epoch + 1} with validation loss: {best_valid_loss:.4f}',
                                epoch)
            else:
                patience_counter += 1

            logger.info(
                f'Epoch {epoch + 1}/{num_epochs} - '
                f'Train Loss: {avg_train_losses["total_loss"]:.4f} - '
                f'Valid Loss: {avg_valid_losses["total_loss"]:.4f} - '
                f'BVP Loss: {avg_valid_losses["bvp_loss"]:.4f}'
                f'BVP Temporal Loss: {avg_valid_losses["bvp_temporal_loss"]:.4f}'
            )
        else:
            patience_counter += 1
            logger.info(
                f'Epoch {epoch + 1}/{num_epochs} - '
                f'Train Loss: {avg_train_losses["total_loss"]:.4f} - '
                f'BVP Loss: {avg_train_losses["bvp_loss"]:.4f}'
                f'BVP Temporal Loss: {avg_train_losses["bvp_temporal_loss"]:.4f}'
            )


        checkpoints_dir = os.path.join(best_model_dir, "checkpoints")
        os.makedirs(checkpoints_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoints_dir, f'rppg_vae_epoch_{epoch + 1}.pth')

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': trainer.optimizer.state_dict(),
            'loss': avg_train_losses["total_loss"],
        }, checkpoint_path)


        writer.add_scalar('Learning Rate', trainer.optimizer.param_groups[0]['lr'], epoch)


        if patience_counter >= patience:
            logger.info(f"Early stopping triggered after {epoch + 1} epochs")
            break


    writer.close()

    print(f"Training complete. Best model saved at: {best_model_path}")
    print(f"To view logs, run: tensorboard --logdir={log_dir}")

    return model

def main(config):

    device = torch.device(config.DEVICE if torch.cuda.is_available() else 'cpu')


    model = init_rppg_vae()


    train_loader, valid_loader = prepare_mmpd_dataloaders(config)

    model = train_rppg_vae(
        model=model,
        config=config,
        train_loader=train_loader,
        valid_loader=valid_loader,
        device=device
    )

    return model


if __name__ == '__main__':
    args = SimpleNamespace()
    args.config_file = 'configs/train_configs/PURE_PURE_MMPD_VAE.yaml'

    config = get_config(args)

    model = main(config)
