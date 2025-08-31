import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class BVP_Decoder(nn.Module):

    def __init__(self, latent_dim, seq_len=64):
        super().__init__()

        self.expand = nn.Sequential(
            nn.Linear(latent_dim, 16),
            nn.LayerNorm(16),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(16, 32),
            nn.LayerNorm(32),
            nn.GELU()
        )

        self.temporal = nn.Sequential(
            nn.Conv1d(32, 16, kernel_size=5, padding=2),
            nn.BatchNorm1d(16),
            nn.GELU(),
            nn.Conv1d(16, 1, kernel_size=3, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.expand(x)
        x = x.transpose(1, 2)
        output = self.temporal(x)
        return output.squeeze(1)


class RPPGVAE_64(nn.Module):

    def __init__(self, frame_depth = 90, latent_dim=8, hidden_dims = 256, input_height = 72, input_width = 72, beta = 4.0):
        super(RPPGVAE_64, self).__init__()
        self.frame_depth = frame_depth
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.input_height = input_height
        self.input_width = input_width
        self.beta = beta

        # By calculation after two Maxpool operations with Stride(1,2,2), the spatial dims are reduced by 1/4
        # change it if you change the stride
        self.spatial_h = input_height // 4
        self.spatial_w = input_width // 4
        self.spatial_channels = 32
        self.spatial_features_dim = self.spatial_channels*self.spatial_h*self.spatial_w

        self.spatial_encoder = nn.Sequential(
            nn.Conv3d(3,16,(1,3,3),padding=(0,1,1)),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.Conv3d(16,32,(1,3,3),padding=(0,1,1)),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d((1,2,2), stride=(1,2,2)),

            nn.Conv3d(32,self.spatial_channels,(1,3,3),padding=(0,1,1)),
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
            nn.ReLU(),
            nn.Conv3d(self.spatial_channels, self.spatial_channels,kernel_size=(11, 1, 1), padding=(5, 0, 0)),
            nn.BatchNorm3d(self.spatial_channels),
            nn.ReLU()
        )

        self.frame_flatten = nn.Flatten(2,4)


        # self.fc_frame = nn.Sequential(
        #     nn.Linear(self.spatial_features_dim, self.hidden_dims),
        #     nn.BatchNorm1d(self.hidden_dims),
        #     nn.GELU(),
        #     nn.Dropout(0.4),
        #
        #     nn.Linear(self.hidden_dims, self.hidden_dims//4),
        #     nn.BatchNorm1d(self.hidden_dims//4),
        #     nn.GELU(),
        #     nn.Dropout(0.3),
        #
        #     nn.Linear(self.hidden_dims//4,self.hidden_dims//16),
        #     nn.BatchNorm1d(self.hidden_dims//16),
        #     nn.GELU()
        # )

        self.fc_frame = nn.Sequential(
            nn.Linear(self.spatial_features_dim, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.3),

            nn.Linear(128,64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(0.4),

            nn.Linear(64,32),
            nn.BatchNorm1d(32),
            nn.GELU()
        )

        self.final_hidden_dim = 32

        self.fc_mu = nn.Linear(self.final_hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.final_hidden_dim, latent_dim)

        self.frame_decoder_fc = nn.Sequential(
            nn.Linear(latent_dim, self.final_hidden_dim),
            nn.BatchNorm1d(self.final_hidden_dim),
            nn.GELU(),

            nn.Linear(self.final_hidden_dim, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(0.3),

            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.4)
        )

        self.frame_spatial_fc = nn.Linear(128, self.spatial_features_dim)


        self.spatial_decoder = nn.Sequential(
            nn.ConvTranspose3d(self.spatial_channels, 32, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.Upsample(scale_factor=(1, 2, 2)),

            nn.ConvTranspose3d(32, 16, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(16),
            nn.GELU(),
            nn.Upsample(scale_factor=(1, 2, 2)),

            nn.ConvTranspose3d(16, 3, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
        )


        # self.bvp_decoder = nn.Sequential(
        #     nn.Linear(latent_dim, 64),
        #     nn.ReLU(),
        #     nn.Linear(64, 32),
        #     nn.ReLU(),
        #     nn.Linear(32, 1)
        # )
        self.bvp_decoder = BVP_Decoder(latent_dim=latent_dim,seq_len=90)

        self.free_bits = 0.3

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
        # batch_size, seq_len, latent_dim = z.size()
        #
        # z_flat = z.reshape(-1, latent_dim)
        # bvp_values = self.bvp_decoder(z_flat)
        # bvp_recon = bvp_values.reshape(batch_size, seq_len)
        bvp_recon = self.bvp_decoder(z)

        return bvp_recon

    def forward(self, x):
        batch_size, channels, seq_len, height, width = x.size()


        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)


        frames_recon = self.decode_frames(z)
        bvp_recon = self.decode_bvp(z)

        return frames_recon, bvp_recon, mu, logvar


class VAE_64_Trainer:
    def __init__(self, model, config, device):
        self.model = model.to(device)
        self.device = device

        # self.optimizer = torch.optim.AdamW([
        #
        #     {'params': model.spatial_encoder.parameters(), 'lr': config.TRAIN.LR},
        #     {'params': model.temporal_encoder.parameters(), 'lr': config.TRAIN.LR},
        #
        #
        #     {'params': model.fc_frame.parameters(), 'lr': config.TRAIN.LR},
        #     {'params': model.frame_decoder_fc.parameters(), 'lr': config.TRAIN.LR},
        #     {'params': model.frame_spatial_fc.parameters(), 'lr': config.TRAIN.LR},
        #
        #
        #     {'params': model.spatial_decoder.parameters(), 'lr': config.TRAIN.LR * 1.5},
        #     {'params': model.bvp_decoder.parameters(), 'lr': config.TRAIN.LR * 1.5, 'weight_decay': 1e-3},
        #
        #
        #     {'params': model.fc_mu.parameters(), 'lr': config.TRAIN.LR * 0.5},
        #     {'params': model.fc_logvar.parameters(), 'lr': config.TRAIN.LR * 0.5},
        # ], lr=config.TRAIN.LR, weight_decay=1e-5)

        self.optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-4)





        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=10,
            min_lr=1e-7,
            verbose=True
        )
        # self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        #     self.optimizer, T_0=20, T_mult=2, eta_min=1e-6
        # )

        self.current_epoch = 0
        self.beta_max = model.beta
        # Length of each cycle after warmup
        self.cycle_length = 8
        # warmup period
        self.warmup_epochs = 10
        self.min_beta = 0.5

    def get_beta(self):
        epoch_factor = min(self.current_epoch / 100, 1.0)
        current_beta_max = 1.0 + (self.beta_max - 1.0) * epoch_factor

        if self.current_epoch < self.warmup_epochs:
            beta = current_beta_max * (self.current_epoch / self.warmup_epochs)
        else:
            t = (self.current_epoch - self.warmup_epochs) % self.cycle_length
            beta = current_beta_max * (1 + np.cos(np.pi * t / self.cycle_length)) / 2

        # Simply ensure beta never goes below min_beta
        return max(beta, self.min_beta)

    def pearson_correlation_loss(self, pred, target):

        pred_mean = pred.mean(dim=1, keepdim=True)
        target_mean = target.mean(dim=1, keepdim=True)

        pred_centered = pred - pred_mean
        target_centered = target - target_mean

        numerator = (pred_centered * target_centered).sum(dim=1)
        pred_norm = torch.sqrt((pred_centered ** 2).sum(dim=1) + 1e-8)
        target_norm = torch.sqrt((target_centered ** 2).sum(dim=1) + 1e-8)

        correlation = numerator / (pred_norm * target_norm)

        correlation = torch.clamp(correlation, -1.0, 1.0)


        return (1 - correlation).mean()

    def compute_kl_loss_with_free_bits(self, mu, logvar):

        kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())

        kl_per_dim = torch.maximum(kl_per_dim, torch.tensor(self.model.free_bits).to(self.device))

        kl_loss = kl_per_dim.sum()/(mu.size(0)*mu.size(1))
        return kl_loss


    def compute_loss(self, frames_recon, bvp_recon, frames, bvp, mu, logvar):

        frame_mse = F.mse_loss(frames_recon, frames, reduction='mean')

        frame_l1 = F.l1_loss(frames_recon, frames)
        frame_loss = 0.7 * frame_mse + 0.3 * frame_l1

        bvp_mse = F.mse_loss(bvp_recon, bvp, reduction='mean')
        bvp_corr_loss = self.pearson_correlation_loss(bvp_recon, bvp)


        if bvp.size(1) > 1:
            bvp_temp_diff_orig = bvp[:, 1:] - bvp[:, :-1]
            bvp_temp_diff_recon = bvp_recon[:, 1:] - bvp_recon[:, :-1]
            bvp_temporal_loss = F.mse_loss(bvp_temp_diff_recon, bvp_temp_diff_orig)
        else:
            bvp_temporal_loss = torch.tensor(0.0, device=self.device)


        kl_loss = self.compute_kl_loss_with_free_bits(mu, logvar)
        kl_weight = self.get_beta()

        if self.current_epoch < 10:
            # Early: Focus on BVP but maintain minimum frame reconstruction
            frame_weight = 0.3  # Not too low - need spatial awareness
            bvp_mse_weight = 0.5
            bvp_corr_weight = 1.2  # Primary focus
            bvp_temporal_weight = 0.7
        elif self.current_epoch < 30:
            # Middle: Gradual transition
            progress = (self.current_epoch - 10) / 20
            frame_weight = 0.2 + 0.1 * progress  # 0.15 → 0.3
            bvp_mse_weight = 0.4 - 0.1 * progress  # 0.5 → 0.3
            bvp_corr_weight = 0.8 - 0.2 * progress  # 1.0 → 0.8
            bvp_temporal_weight = 0.4 - 0.1 * progress  # 0.5 → 0.3
        else:
            # Later: Balanced approach
            frame_weight = 0.3
            bvp_mse_weight = 0.3
            bvp_corr_weight = 0.6
            bvp_temporal_weight = 0.3

            # KL weight from cyclical beta
        kl_weight = self.get_beta()


        total_loss = frame_weight * frame_loss + bvp_mse_weight * bvp_mse +  bvp_corr_weight *bvp_corr_loss + bvp_temporal_weight* bvp_temporal_loss + kl_weight * kl_loss

        return total_loss, frame_loss, bvp_mse, kl_loss,bvp_corr_loss, bvp_temporal_loss

    def train_step(self, frames, bvp):
        self.model.train()
        self.optimizer.zero_grad()

        frames_recon, bvp_recon, mu, logvar = self.model(frames)

        total_loss, frame_loss, bvp_loss, kl_loss,bvp_corr_loss, bvp_temporal_loss = self.compute_loss(
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
            'bvp_corr_loss': bvp_corr_loss.item(),
            'bvp_temporal_loss': bvp_temporal_loss.item()
        }

    def validate_step(self, frames, bvp):
        self.model.eval()
        with torch.no_grad():
            frames_recon, bvp_recon, mu, logvar = self.model(frames)
            total_loss, frame_loss, bvp_loss, kl_loss,bvp_corr_loss, bvp_temporal_loss = self.compute_loss(
                frames_recon, bvp_recon, frames, bvp, mu, logvar
            )

        return {
            'total_loss': total_loss.item(),
            'frame_loss': frame_loss.item(),
            'bvp_loss': bvp_loss.item(),
            'kl_loss': kl_loss.item(),
            'bvp_corr_loss': bvp_corr_loss.item(),
            'bvp_temporal_loss': bvp_temporal_loss.item()
        }

    def update_scheduler(self, val_loss = None):

        self.current_epoch += 1

        if val_loss is not None:
            self.scheduler.step(val_loss)


def init_rppg_vae():

    model = RPPGVAE_64(
        frame_depth=64,
        latent_dim=8,
        hidden_dims=256,
        input_height=80,
        input_width=60
    )
    return model
