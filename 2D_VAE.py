import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader, Subset
import pandas as pd
from collections import defaultdict
import random
import os
from tqdm import tqdm
import logging


class FaceVAE(nn.Module):
    

    def __init__(self, latent_dim=16):
        super().__init__()
        self.latent_dim = latent_dim

        
        self.encoder = nn.Sequential(
            
            nn.Conv2d(3, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            nn.Conv2d(128, 256, 3, stride=2, padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )

       
        self.fc_mu = nn.Linear(256 * 4 * 4, latent_dim)
        self.fc_logvar = nn.Linear(256 * 4 * 4, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, 256 * 4 * 4)

        
        self.decoder = nn.Sequential(
            
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def encode(self, x):
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.fc_decode(z)
        h = h.view(h.size(0), 256, 4, 4)
        return self.decoder(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

    def loss_function(self, recon, x, mu, logvar, beta=1.0):
       
        recon_loss = F.mse_loss(recon, x, reduction='mean')
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        total_loss = recon_loss + beta * kl_loss

        return {
            'total': total_loss,
            'recon': recon_loss,
            'kl': kl_loss
        }


class FrameDataset(Dataset):
    

    def __init__(self, csv_file_path, frames_per_file=15, frame_depth=90, seed=42):
        
        self.frames_per_file = frames_per_file
        self.frame_depth = frame_depth
        self.rng = random.Random(seed)

        
        df = pd.read_csv(csv_file_path)
        if 'input_files' not in df.columns:
            raise ValueError("CSV must have 'input_files' column")

        self.input_files = df['input_files'].tolist()
        self.label_files = [f.replace("_input", "_label") for f in self.input_files]

        
        self.samples = []
        self.subject_to_indices = defaultdict(list)

        print(f"Preparing dataset from {len(self.input_files)} files...")
        for file_idx, input_file in enumerate(tqdm(self.input_files, desc="Scanning files")):
            try:
                
                frames = np.load(input_file, mmap_mode='r')
                if frames.ndim == 4:  # (T, H, W, C)
                    num_frames = frames.shape[0]
                    if num_frames >= 1:
                        # Sample up to frames_per_file random frames
                        n_samples = min(self.frames_per_file, num_frames)
                        frame_indices = self.rng.sample(range(num_frames), n_samples)

                        for frame_idx in frame_indices:

                            self.samples.append((file_idx, frame_idx))


            except Exception as e:
                print(f"Error loading {input_file}: {e}")
                continue


        print(f"Created dataset with {len(self.samples)} samples")



    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_idx, frame_idx = self.samples[idx]
        input_file = self.input_files[file_idx]

        try:
            
            frames = np.load(input_file, mmap_mode='r')

            if frames.ndim == 4:  
                frame = frames[frame_idx]  
                
                frame = frame.transpose(2, 0, 1)
                frame = frame.astype(np.float32) / 255.0
                return torch.FloatTensor(frame)
            else:
                raise ValueError(f"Unexpected shape: {frames.shape}")

        except Exception as e:
            print(f"Error loading frame {frame_idx} from {input_file}: {e}")
            
            return torch.zeros((3, 72, 72))



class Trainer:
    

    def __init__(self, model, device=None, lr=1e-3):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )

    def train_epoch(self, train_loader, beta=1.0):
        self.model.train()
        total_loss = 0
        total_recon = 0
        total_kl = 0

        for batch in tqdm(train_loader, desc="Training"):
            x = batch.to(self.device)

            self.optimizer.zero_grad()
            recon, mu, logvar = self.model(x)
            losses = self.model.loss_function(recon, x, mu, logvar, beta)

            losses['total'].backward()
            
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += losses['total'].item()
            total_recon += losses['recon'].item()
            total_kl += losses['kl'].item()

        n = len(train_loader)
        return {
            'total': total_loss / n,
            'recon': total_recon / n,
            'kl': total_kl / n
        }

    @torch.no_grad()
    def validate(self, val_loader, beta=1.0):
        self.model.eval()
        total_loss = 0
        total_recon = 0
        total_kl = 0

        for batch in tqdm(val_loader, desc="Validation"):
            x = batch.to(self.device)

            recon, mu, logvar = self.model(x)
            losses = self.model.loss_function(recon, x, mu, logvar, beta)

            total_loss += losses['total'].item()
            total_recon += losses['recon'].item()
            total_kl += losses['kl'].item()

        n = len(val_loader)
        return {
            'total': total_loss / n,
            'recon': total_recon / n,
            'kl': total_kl / n
        }

    def train(self, train_loader, val_loader, num_epochs=50, start_beta=0.0, end_beta=1.0):
        best_val_loss = float('inf')

        for epoch in range(num_epochs):
            
            beta = start_beta + (end_beta - start_beta) * (epoch / num_epochs)

            train_losses = self.train_epoch(train_loader, beta)
            val_losses = self.validate(val_loader, beta)

            print(f"Epoch {epoch + 1}/{num_epochs}")
            print(f"  Train - Total: {train_losses['total']:.4f}, "
                  f"Recon: {train_losses['recon']:.4f}, KL: {train_losses['kl']:.4f}")
            print(f"  Val   - Total: {val_losses['total']:.4f}, "
                  f"Recon: {val_losses['recon']:.4f}, KL: {val_losses['kl']:.4f}")
            print(f"  Beta: {beta:.4f}")

            
            self.scheduler.step(val_losses['total'])

            
            if val_losses['total'] < best_val_loss:
                best_val_loss = val_losses['total']
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': best_val_loss,
                }, 'simple_vae_best.pth')
                print(f"  New best model saved!")
            print()


def create_dataloaders(csv_file_path, batch_size=32, train_ratio=0.8,
                                 frames_per_file=5, num_workers=8, seed=42):
    
    
    dataset = FrameDataset(
        csv_file_path=csv_file_path,
        frames_per_file=frames_per_file,
        seed=seed
    )

    
    all_indices = list(range(len(dataset)))
    random.Random(seed).shuffle(all_indices)

    n_train = int(len(all_indices) * train_ratio)
    train_indices = all_indices[:n_train]
    val_indices = all_indices[n_train:]

    print(f"Train samples: {len(train_indices)}, Val samples: {len(val_indices)}")

    
    train_subset = Subset(dataset, train_indices)
    val_subset = Subset(dataset, val_indices)

    
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0
    )

    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0
    )

    return train_loader, val_loader



if __name__ == "__main__":
    
    csv_path = "/data/rppg_23_mmpd_video_nt_lab/processed/DataFileLists/MMPD_SizeW72_SizeH72_ClipLength180_DataTypeRaw_DataAugNone_LabelTypeRaw_Crop_faceTrue_BackendY5F_Large_boxFalse_Large_size1.5_Dyamic_DetTrue_det_len1_Median_face_boxFalse_PSEUDO_LABELFalse_0.0_1.0.csv"

    
    print("Creating dataloaders...")
    train_loader, val_loader = create_dataloaders(
        csv_path,
        batch_size=64,  
        frames_per_file=5,  
        num_workers=8
    )

    
    model = FaceVAE(latent_dim=16)
    trainer = Trainer(model, lr=1e-3)

    print("\nTesting model dimensions...")
    test_input = torch.randn(2, 3, 72, 72).to(trainer.device)
    test_model = FaceVAE(latent_dim=16).to(trainer.device)
    test_recon, _, _ = test_model(test_input)
    print(f"Input shape: {test_input.shape}")
    print(f"Output shape: {test_recon.shape}")
    assert test_input.shape == test_recon.shape, "Dimension mismatch!"
    print("Dimension test passed!\n")

    
    print("Starting training...")
    trainer.train(train_loader, val_loader, num_epochs=50, start_beta=0.0, end_beta=0.1)

    print("Training complete!")