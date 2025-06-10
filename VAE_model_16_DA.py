import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class RPPGVAE_16(nn.Module):

    def __init__(self, frame_depth=16,appearance_dim=16, pulse_dim=16, input_height=80, input_width=60):
        super(RPPGVAE_16, self).__init__()
        self.frame_depth = frame_depth
        self.appearance_dim = appearance_dim
        self.pulse_dim = pulse_dim
        self.latent_dim = appearance_dim + pulse_dim
        self.input_height = input_height
        self.input_width = input_width


        self.encoder_scale1 = nn.Sequential(
            nn.Conv3d(3, 8, (1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1)),
            nn.BatchNorm3d(8),
            nn.ReLU(),
            nn.MaxPool3d((1, 2, 2))
        )

        self.encoder_scale2 = nn.Sequential(
            nn.Conv3d(3, 8, (1, 5, 5), stride=(1, 2, 2), padding=(0, 2, 2)),
            nn.BatchNorm3d(8),
            nn.ReLU()
        )

        self.encoder_scale3 = nn.Sequential(
            nn.Conv3d(3, 8, (1, 7, 7), stride=(1, 4, 4), padding=(0, 3, 3)),
            nn.BatchNorm3d(8),
            nn.ReLU()
        )


        self.fusion = nn.Sequential(
            nn.Conv3d(24, 32, (1, 1, 1)),  # Fuse multi-scale features
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.Dropout3d(0.1),


            nn.Conv3d(32, 32, (3, 1, 1), padding=(1, 0, 0)),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.Conv3d(32, 32, (5, 1, 1), padding=(2, 0, 0)),
            nn.BatchNorm3d(32),
            nn.ReLU()
        )


        self.global_pool = nn.AdaptiveAvgPool3d((frame_depth, 1, 1))


        self.shared_features = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU()
        )


        self.fc_mu_appearance = nn.Linear(32, appearance_dim)
        self.fc_logvar_appearance = nn.Linear(32, appearance_dim)
        self.fc_mu_pulse = nn.Linear(32, pulse_dim)
        self.fc_logvar_pulse = nn.Linear(32, pulse_dim)


        self.bvp_decoder = nn.Sequential(
            nn.Linear(pulse_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )


        self.bvp_smooth = nn.Conv1d(1, 1, kernel_size=3, padding=1)


        self.frame_decoder_fc = nn.Sequential(
            nn.Linear(self.latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 512),
            nn.ReLU()
        )


        self.dec_h = input_height // 4
        self.dec_w = input_width // 4
        self.dec_channels = 32

        self.frame_decoder_reshape = nn.Linear(512, self.dec_channels * self.dec_h * self.dec_w)

        self.frame_decoder_conv = nn.Sequential(
            nn.ConvTranspose3d(32, 16, (1, 3, 3), stride=(1, 2, 2),
                              padding=(0, 1, 1), output_padding=(0, 1, 1)),
            nn.BatchNorm3d(16),
            nn.ReLU(),

            nn.ConvTranspose3d(16, 3, (1, 3, 3), stride=(1, 2, 2),
                              padding=(0, 1, 1), output_padding=(0, 1, 1)),
            nn.Sigmoid()
        )

        # self._initialize_weights()

    # def _initialize_weights(self):
    #     for m in self.modules():
    #         if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
    #             nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    #             if m.bias is not None:
    #                 nn.init.constant_(m.bias, 0)
    #         elif isinstance(m, (nn.BatchNorm3d, nn.BatchNorm1d)):
    #             nn.init.constant_(m.weight, 1)
    #             nn.init.constant_(m.bias, 0)
    #         elif isinstance(m, nn.Linear):
    #             nn.init.normal_(m.weight, 0, 0.01)
    #             nn.init.constant_(m.bias, 0)

    def encode(self, x):
        batch_size = x.size(0)


        scale1 = self.encoder_scale1(x)
        scale2 = self.encoder_scale2(x)
        scale3 = self.encoder_scale3(x)


        target_h = scale3.size(3)
        target_w = scale3.size(4)
        scale1 = F.interpolate(scale1, size=(scale1.size(2), target_h, target_w),
                              mode='trilinear', align_corners=False)
        scale2 = F.interpolate(scale2, size=(scale2.size(2), target_h, target_w),
                              mode='trilinear', align_corners=False)


        multi_scale = torch.cat([scale1, scale2, scale3], dim=1)


        features = self.fusion(multi_scale)


        pooled = self.global_pool(features)
        pooled = pooled.squeeze(-1).squeeze(-1).permute(0, 2, 1)

        # Shared processing
        pooled_flat = pooled.reshape(-1, 32)
        shared = self.shared_features(pooled_flat)


        mu_appearance = self.fc_mu_appearance(shared).reshape(batch_size, self.frame_depth, self.appearance_dim)
        logvar_appearance = self.fc_logvar_appearance(shared).reshape(batch_size, self.frame_depth, self.appearance_dim)
        mu_pulse = self.fc_mu_pulse(shared).reshape(batch_size, self.frame_depth, self.pulse_dim)
        logvar_pulse = self.fc_logvar_pulse(shared).reshape(batch_size, self.frame_depth, self.pulse_dim)


        mu = torch.cat([mu_appearance, mu_pulse], dim=-1)
        logvar = torch.cat([logvar_appearance, logvar_pulse], dim=-1)

        return mu, logvar, mu_appearance, logvar_appearance, mu_pulse, logvar_pulse

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode_bvp(self, z_pulse):
        batch_size = z_pulse.size(0)


        z_flat = z_pulse.reshape(-1, self.pulse_dim)
        bvp = self.bvp_decoder(z_flat)
        bvp = bvp.reshape(batch_size, self.frame_depth)


        bvp_smooth = self.bvp_smooth(bvp.unsqueeze(1)).squeeze(1)

        return bvp_smooth

    def decode_frames(self, z):
        batch_size = z.size(0)


        z_mean = z.mean(dim=1, keepdim=True)
        z_repeated = z_mean.repeat(1, self.frame_depth, 1)


        z_flat = z_repeated.reshape(-1, self.latent_dim)
        frame_features = self.frame_decoder_fc(z_flat)


        spatial_features = self.frame_decoder_reshape(frame_features)
        spatial_features = spatial_features.reshape(
            batch_size, self.frame_depth, self.dec_channels, self.dec_h, self.dec_w
        )


        spatial_features = spatial_features.permute(0, 2, 1, 3, 4)
        frames_recon = self.frame_decoder_conv(spatial_features)

        return frames_recon

    def forward(self, x):
        mu, logvar, mu_app, logvar_app, mu_pulse, logvar_pulse = self.encode(x)


        z = self.reparameterize(mu, logvar)
        z_pulse = z[:, :, self.appearance_dim:]


        frames_recon = self.decode_frames(z)
        bvp_recon = self.decode_bvp(z_pulse)

        return frames_recon, bvp_recon, mu, logvar



class VAE_16_Trainer:
    def __init__(self, model, config, device):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=1e-3,
            weight_decay=1e-5
        )

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=10,
            min_lr=1e-6
        )

        self.kl_weights = 0.0
        self.kl_anneal_rate = 1.0/20
        self.kl_target = 0.01

    def compute_frequency_loss(self, pred, target):

        # Compute FFT
        pred_fft = torch.fft.rfft(pred, dim=1)
        target_fft = torch.fft.rfft(target, dim=1)

        # Magnitude spectrum
        pred_mag = torch.abs(pred_fft)
        target_mag = torch.abs(target_fft)

        # Focus on physiological frequency range (0.7-4 Hz)
        # Assuming 30 fps, this corresponds to indices ~1-8
        freq_mask = torch.zeros_like(pred_mag)
        freq_mask[:, 1:9] = 1.0

        pred_mag_masked = pred_mag * freq_mask
        target_mag_masked = target_mag * freq_mask

        return F.mse_loss(pred_mag_masked, target_mag_masked)

    def pearson_correlation_loss(self, pred, target):

        batch_size = pred.size(0)

        # Standardize
        pred_mean = pred.mean(dim=1, keepdim=True)
        target_mean = target.mean(dim=1, keepdim=True)

        pred_std = pred.std(dim=1, keepdim=True) + 1e-8
        target_std = target.std(dim=1, keepdim=True) + 1e-8

        pred_norm = (pred - pred_mean) / pred_std
        target_norm = (target - target_mean) / target_std

        # Compute correlation
        correlation = (pred_norm * target_norm).mean(dim=1)

        # Loss is 1 - correlation (we want to maximize correlation)
        return (1 - correlation).mean()

    def compute_loss(self, frames_recon, bvp_recon, frames, bvp, mu, logvar):

        frame_loss = F.mse_loss(frames_recon, frames, reduction='mean')

        bvp_loss = F.mse_loss(bvp_recon, bvp, reduction='mean')
        bvp_corr_loss = self.pearson_correlation_loss(bvp_recon, bvp)
        bvp_freq_loss = self.compute_frequency_loss(bvp_recon, bvp)


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

        self.kl_weights = min(self.kl_weights + self.kl_anneal_rate, self.kl_target)

        total_loss = 0.5 * frame_loss + 1.0 * bvp_loss + 3.0* bvp_corr_loss + 2.0 * bvp_temporal_loss +1.0*bvp_freq_loss + self.kl_weights* kl_loss

        return total_loss, frame_loss, bvp_loss,bvp_corr_loss,bvp_freq_loss, kl_loss, bvp_temporal_loss

    def train_step(self, frames, bvp):

        self.model.train()
        self.optimizer.zero_grad()

        frames_recon, bvp_recon, mu, logvar = self.model(frames)

        total_loss, frame_loss, bvp_loss,bvp_corr_loss,bvp_freq_loss, kl_loss, bvp_temporal_loss = self.compute_loss(
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
            'bvp_freq_loss': bvp_freq_loss.item(),
            'kl_loss': kl_loss.item(),
            'bvp_temporal_loss': bvp_temporal_loss.item()
        }

    def validate_step(self, frames, bvp):
        self.model.eval()
        with torch.no_grad():
            frames_recon, bvp_recon, mu, logvar = self.model(frames)
            total_loss, frame_loss, bvp_loss,bvp_corr_loss, bvp_freq_loss, kl_loss, bvp_temporal_loss = self.compute_loss(
                frames_recon, bvp_recon, frames, bvp, mu, logvar
            )

        return {
            'total_loss': total_loss.item(),
            'frame_loss': frame_loss.item(),
            'bvp_loss': bvp_loss.item(),
            'bvp_corr_loss': bvp_corr_loss.item(),
            'bvp_freq_loss': bvp_freq_loss.item(),
            'kl_loss': kl_loss.item(),
            'bvp_temporal_loss': bvp_temporal_loss.item()
        }

    def update_scheduler(self, val_loss):

        self.scheduler.step()


def init_rppg_vae():
    model = RPPGVAE_16(
        frame_depth=16,
        appearance_dim=16,
        pulse_dim=16,
        input_height=80,
        input_width=60
    )
    return model
