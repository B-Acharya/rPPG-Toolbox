import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class RPPGVAE_16(nn.Module):

    def __init__(self, frame_depth = 16, latent_dim=32, hidden_dims = 2048, input_height = 80, input_width = 60):
        super(RPPGVAE_16, self).__init__()
        self.frame_depth = frame_depth
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.input_height = input_height
        self.input_width = input_width

        # By calculation after two Maxpool operations with Stride(1,2,2), the spatial dims are reduced by 1/4
        # change it if you change the stride
        self.spatial_h = input_height // 4
        self.spatial_w = input_width // 4
        self.spatial_channels = 128
        self.spatial_features_dim = self.spatial_channels*self.spatial_h*self.spatial_w

        self.spatial_encoder = nn.Sequential(
            nn.Conv3d(3,32,(1,3,3),padding=(0,1,1)),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.Conv3d(32,64,(1,3,3),padding=(0,1,1)),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d((1,2,2), stride=(1,2,2)),

            nn.Conv3d(64,self.spatial_channels,(1,3,3),padding=(0,1,1)),
            nn.BatchNorm3d(self.spatial_channels),
            nn.ReLU(),
            nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2))
        )
        self.temporal_encoder = nn.Sequential(
            nn.Conv3d(self.spatial_channels, self.spatial_channels, kernel_size=(3, 1, 1), padding=(1, 0, 0)),
            nn.BatchNorm3d(self.spatial_channels),
            nn.ReLU(),
            nn.Conv3d(self.spatial_channels, self.spatial_channels, kernel_size=(5, 1, 1), padding=(2, 0, 0)),
            nn.BatchNorm3d(self.spatial_channels),
            nn.ReLU(),
            nn.Conv3d(self.spatial_channels, self.spatial_channels, kernel_size=(7, 1, 1), padding=(3, 0, 0)),
            nn.BatchNorm3d(self.spatial_channels),
            nn.ReLU()
        )

        self.frame_flatten = nn.Flatten(2,4)


        self.fc_frame = nn.Sequential(
            nn.Linear(self.spatial_features_dim, self.hidden_dims),
            nn.BatchNorm1d(self.hidden_dims),
            nn.ReLU(),
            nn.Dropout(0.4),

            nn.Linear(self.hidden_dims, self.hidden_dims//4),
            nn.BatchNorm1d(self.hidden_dims//4),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(self.hidden_dims//4,self.hidden_dims//16),
            nn.BatchNorm1d(self.hidden_dims//16),
            nn.ReLU()
        )

        self.final_hidden_dim = self.hidden_dims//16

        self.fc_mu = nn.Linear(self.final_hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.final_hidden_dim, latent_dim)

        self.frame_decoder_fc = nn.Sequential(
            nn.Linear(latent_dim, self.final_hidden_dim),
            nn.BatchNorm1d(self.final_hidden_dim),
            nn.ReLU(),

            nn.Linear(self.final_hidden_dim, self.hidden_dims // 4),
            nn.BatchNorm1d(self.hidden_dims // 4),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(self.hidden_dims // 4, self.hidden_dims),
            nn.BatchNorm1d(self.hidden_dims),
            nn.ReLU(),
            nn.Dropout(0.4)
        )

        self.frame_spatial_fc = nn.Linear(self.hidden_dims, self.spatial_features_dim)


        self.spatial_decoder = nn.Sequential(
            nn.ConvTranspose3d(self.spatial_channels, 64, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.Upsample(scale_factor=(1, 2, 2)),

            nn.ConvTranspose3d(64, 32, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.Upsample(scale_factor=(1, 2, 2)),

            nn.ConvTranspose3d(32, 3, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
        )


        self.bvp_decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def encode(self, x):

        batch_size, channels, seq_len, height, width = x.size()


        spatial_features = self.spatial_encoder(x)

        temporal_features = self.temporal_encoder(spatial_features)


        temporal_features = temporal_features.permute(0, 2, 1, 3, 4)


        flattened = temporal_features.reshape(batch_size, seq_len, -1)


        features_dim = flattened.shape[2]
        flattened_reshape = flattened.reshape(-1, features_dim)  # Combine batch and seq dimensions
        processed_features = self.fc_frame(flattened_reshape)  # Apply FC layers with BatchNorm1d
        frame_features = processed_features.reshape(batch_size, seq_len, -1)  # Restore original shape


        mu_input = frame_features.reshape(-1, self.final_hidden_dim)
        logvar_input = frame_features.reshape(-1, self.final_hidden_dim)

        mu = self.fc_mu(mu_input).reshape(batch_size, seq_len, self.latent_dim)
        logvar = self.fc_logvar(logvar_input).reshape(batch_size, seq_len, self.latent_dim)

        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode_frames(self, z):

        batch_size, seq_len, latent_dim = z.size()


        z_flat = z.reshape(-1, latent_dim)
        frame_features = self.frame_decoder_fc(z_flat)
        frame_features = frame_features.reshape(batch_size, seq_len, -1)


        spatial_flat = self.frame_spatial_fc(frame_features.reshape(-1, self.hidden_dims))
        spatial_features = spatial_flat.reshape(batch_size, seq_len, self.spatial_channels, self.spatial_h,
                                                self.spatial_w)


        spatial_features = spatial_features.permute(0, 2, 1, 3, 4)


        frames_recon = self.spatial_decoder(spatial_features)

        return frames_recon

    def decode_bvp(self, z):

        batch_size, seq_len, latent_dim = z.size()


        z_flat = z.reshape(-1, latent_dim)
        bvp_values = self.bvp_decoder(z_flat)
        bvp_recon = bvp_values.reshape(batch_size, seq_len)

        return bvp_recon

    def forward(self, x):
        batch_size, channels, seq_len, height, width = x.size()


        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)


        frames_recon = self.decode_frames(z)
        bvp_recon = self.decode_bvp(z)

        return frames_recon, bvp_recon, mu, logvar


class VAE_16_Trainer:
    def __init__(self, model, config, device):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.TRAIN.LR,
            weight_decay=1e-5
        )


        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )

    def compute_loss(self, frames_recon, bvp_recon, frames, bvp, mu, logvar):

        frame_loss = F.mse_loss(frames_recon, frames, reduction='mean')


        bvp_loss = F.mse_loss(bvp_recon, bvp, reduction='mean')


        if bvp.size(1) > 1:
            bvp_temp_diff_orig = bvp[:, 1:] - bvp[:, :-1]
            bvp_temp_diff_recon = bvp_recon[:, 1:] - bvp_recon[:, :-1]
            bvp_temporal_loss = F.mse_loss(bvp_temp_diff_recon, bvp_temp_diff_orig)
        else:
            bvp_temporal_loss = torch.tensor(0.0, device=self.device)


        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        kl_loss = kl_loss / (frames.size(0) * frames.size(2))


        total_loss = 0.1 * frame_loss + 0.5 * bvp_loss + 1.0 * bvp_temporal_loss + 0.01 * kl_loss

        return total_loss, frame_loss, bvp_loss, kl_loss, bvp_temporal_loss

    def train_step(self, frames, bvp):
        self.model.train()
        self.optimizer.zero_grad()

        frames_recon, bvp_recon, mu, logvar = self.model(frames)

        total_loss, frame_loss, bvp_loss, kl_loss, bvp_temporal_loss = self.compute_loss(
            frames_recon, bvp_recon, frames, bvp, mu, logvar
        )

        total_loss.backward()


        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

        self.optimizer.step()

        return {
            'total_loss': total_loss.item(),
            'frame_loss': frame_loss.item(),
            'bvp_loss': bvp_loss.item(),
            'kl_loss': kl_loss.item(),
            'bvp_temporal_loss': bvp_temporal_loss.item()
        }

    def validate_step(self, frames, bvp):
        self.model.eval()
        with torch.no_grad():
            frames_recon, bvp_recon, mu, logvar = self.model(frames)
            total_loss, frame_loss, bvp_loss, kl_loss, bvp_temporal_loss = self.compute_loss(
                frames_recon, bvp_recon, frames, bvp, mu, logvar
            )

        return {
            'total_loss': total_loss.item(),
            'frame_loss': frame_loss.item(),
            'bvp_loss': bvp_loss.item(),
            'kl_loss': kl_loss.item(),
            'bvp_temporal_loss': bvp_temporal_loss.item()
        }

    def update_scheduler(self, val_loss):

        self.scheduler.step(val_loss)


def init_rppg_vae():

    model = RPPGVAE_16(
        frame_depth=16,
        latent_dim=32,
        hidden_dims=2048,
        input_height=80,
        input_width=60
    )
    return model
        


