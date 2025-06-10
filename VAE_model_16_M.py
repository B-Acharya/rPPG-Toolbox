import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class RPPGVAE_16(nn.Module):

    def __init__(self, frame_depth=16, latent_dim=16, input_height=80, input_width=60):
        super(RPPGVAE_16, self).__init__()
        self.frame_depth = frame_depth
        self.latent_dim = latent_dim
        self.input_height = input_height
        self.input_width = input_width
        self.feature_encoder = nn.Sequential(
            nn.Conv3d(3, 16, (1, 5, 5), (1, 2, 2), padding=(0, 2, 2)),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.Dropout3d(0.1),

            nn.Conv3d(16, 32, (1, 3, 3), (1, 2, 2), padding=(0, 1, 1)),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.Dropout3d(0.1),

            # temporal processing
            nn.Conv3d(32, 32, (3, 1, 1), padding=(1, 0, 0)),
            nn.BatchNorm3d(32),
            nn.ReLU()
        )

        self.global_pool = nn.AdaptiveAvgPool3d((frame_depth,1,1))

        self.shared_features = nn.Sequential(
            nn.Linear(32,64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64,32),
            nn.ReLU(),
        )

        self.fc_mu = nn.Linear(32, latent_dim)
        self.fc_logvar = nn.Linear(32, latent_dim)

        self.bvp_decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

        # Temporal smoothing for BVP
        self.bvp_smooth = nn.Conv1d(1, 1, kernel_size=3, padding=1)

        self.frame_decoder_fc = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 512),
            nn.ReLU()
        )

        self.decoder_h = input_height//4
        self.decoder_w = input_width//4
        self.decoder_channels = 32

        self.frame_decoder_reshape = nn.Linear(512, self.decoder_h * self.decoder_w * self.decoder_channels)

        self.frame_decoder_conv = nn.Sequential(
            nn.ConvTranspose3d(32, 16, (1, 3, 3), (1, 2, 2), padding=(0, 1, 1), output_padding=(0, 1, 1)),
            nn.BatchNorm3d(16),
            nn.ReLU(),

            nn.ConvTranspose3d(16,3,(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1), output_padding=(0, 1, 1)),
            nn.Sigmoid()
        )

        # self._initialize_weights()

    # def _initialize_weights(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.Conv3d) or isinstance(m, nn.ConvTranspose3d):
    #             nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    #             if m.bias is not None:
    #                 nn.init.constant_(m.bias, 0)
    #         elif isinstance(m, nn.BatchNorm3d):
    #             nn.init.constant_(m.weight, 1)
    #             nn.init.constant_(m.bias, 0)
    #         elif isinstance(m, nn.Linear):
    #             nn.init.normal_(m.weight, 0, 0.01)
    #             nn.init.constant_(m.bias, 0)

    def encode(self, x):
        batch_size = x.size(0)

        feature = self.feature_encoder(x)

        pooled = self.global_pool(feature)
        pooled = pooled.squeeze(-1).squeeze(-1)
        pooled = pooled.permute(0, 2, 1)

        pooled_flat = pooled.reshape(-1,32)
        shared = self.shared_features(pooled_flat)

        mu = self.fc_mu(shared).reshape(batch_size,self.frame_depth, self.latent_dim)
        logvar = self.fc_logvar(shared).reshape(batch_size, self.frame_depth,  self.latent_dim)

        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode_frames(self, z):
        batch_size = z.size(0)

        z_mean = z.mean(dim = 1, keepdim = True)
        z_repeated = z_mean.repeat(1, self.frame_depth, 1)

        z_flat = z_repeated.reshape(-1,self.latent_dim)
        frame_features = self.frame_decoder_fc(z_flat)

        spatial_features = self.frame_decoder_reshape(frame_features)
        spatial_features = spatial_features.reshape(batch_size,self.frame_depth, self.decoder_channels, self. decoder_h, self.decoder_w)

        spatial_features_permute = spatial_features.permute(0, 2, 1, 3, 4)
        frames_recon = self.frame_decoder_conv(spatial_features_permute)

        return frames_recon

    def decode_bvp(self, z):
        batch_size = z.size(0)

        z_flat = z.reshape(-1, self.latent_dim)
        bvp_values = self.bvp_decoder(z_flat)
        bvp = bvp_values.reshape(batch_size,self.frame_depth)

        bvp_smooth = self.bvp_smooth(bvp.unsqueeze(1)).squeeze(1)

        return bvp_smooth

    def forward(self, x):

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
            lr=config.TRAIN.LR * 0.1,
            weight_decay=1e-4
        )

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,T_0=10,T_mult=2
        )

        self.kl_weights = 0.0
        self.kl_anneal_weights = 1.0/50

    def correlation_loss(self,pred,target):

        pred_centered = pred - pred.mean(dim=1, keepdim=True)
        target_centered = target - target.mean(dim=1, keepdim=True)

        correlation = F.cosine_similarity(pred_centered, target_centered, dim=1)

        return (1-correlation).mean()

    def compute_loss(self, frames_recon, bvp_recon, frames, bvp, mu, logvar):

        frame_loss = F.mse_loss(frames_recon, frames, reduction='mean')

        bvp_loss = F.mse_loss(bvp_recon, bvp, reduction='mean')
        bvp_corr_loss = self.correlation_loss(bvp_recon, bvp)

        if bvp.size(1) > 1:
            # First-order derivative
            bvp_diff_orig = bvp[:, 1:] - bvp[:, :-1]
            bvp_diff_recon = bvp_recon[:, 1:] - bvp_recon[:, :-1]
            bvp_temporal_loss = F.mse_loss(bvp_diff_recon, bvp_diff_orig)

            # Second-order derivative for smoother signals
            if bvp.size(1) > 2:
                bvp_diff2_orig = bvp_diff_orig[:, 1:] - bvp_diff_orig[:, :-1]
                bvp_diff2_recon = bvp_diff_recon[:, 1:] - bvp_diff_recon[:, :-1]
                bvp_temporal_loss += 0.5 * F.mse_loss(bvp_diff2_recon, bvp_diff2_orig)
        else:
            bvp_temporal_loss = torch.tensor(0.0, device=self.device)

        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        kl_loss = kl_loss / (frames.size(0) * frames.size(2))

        self.Kl_weights = min(self.kl_weights + self.kl_anneal_weights, 0.001)

        total_loss = 0.1 * frame_loss + 1.0 * bvp_loss + 2.0* bvp_corr_loss + 1.0 * bvp_temporal_loss + self.Kl_weights * kl_loss

        return total_loss, frame_loss, bvp_loss,bvp_corr_loss, kl_loss, bvp_temporal_loss

    def train_step(self, frames, bvp):

        self.model.train()
        self.optimizer.zero_grad()

        frames_recon, bvp_recon, mu, logvar = self.model(frames)

        total_loss, frame_loss, bvp_loss,bvp_corr_loss, kl_loss, bvp_temporal_loss = self.compute_loss(
            frames_recon, bvp_recon, frames, bvp, mu, logvar
        )

        total_loss.backward()

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

        self.optimizer.step()

        return {
            'total_loss': total_loss.item(),
            'frame_loss': frame_loss.item(),
            'bvp_loss': bvp_loss.item(),
            'bvp_corr_loss': bvp_corr_loss.item(),
            'kl_loss': kl_loss.item(),
            'bvp_temporal_loss': bvp_temporal_loss.item()
        }

    def validate_step(self, frames, bvp):
        self.model.eval()
        with torch.no_grad():
            frames_recon, bvp_recon, mu, logvar = self.model(frames)
            total_loss, frame_loss, bvp_loss,bvp_corr_loss, kl_loss, bvp_temporal_loss = self.compute_loss(
                frames_recon, bvp_recon, frames, bvp, mu, logvar
            )

        return {
            'total_loss': total_loss.item(),
            'frame_loss': frame_loss.item(),
            'bvp_loss': bvp_loss.item(),
            'bvp_corr_loss': bvp_corr_loss.item(),
            'kl_loss': kl_loss.item(),
            'bvp_temporal_loss': bvp_temporal_loss.item()
        }

    def update_scheduler(self, val_loss):

        self.scheduler.step()


def init_rppg_vae():
    model = RPPGVAE_16(
        frame_depth=16,
        latent_dim=16,
        input_height=80,
        input_width=60
    )
    return model

