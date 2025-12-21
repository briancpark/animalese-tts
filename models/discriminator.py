"""
Discriminators for Animalese TTS adversarial training.

Implements Multi-Period Discriminator (MPD) and Multi-Scale Discriminator (MSD)
following the HiFi-GAN architecture for high-quality audio generation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple


class PeriodDiscriminator(nn.Module):
    """Period discriminator for a single period."""

    def __init__(
        self,
        period: int,
        kernel_size: int = 5,
        stride: int = 3,
    ):
        super().__init__()
        self.period = period

        # Convolutional layers
        self.convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(1, 32, (kernel_size, 1), (stride, 1), padding=(2, 0)),
                    nn.LeakyReLU(0.1),
                ),
                nn.Sequential(
                    nn.Conv2d(32, 128, (kernel_size, 1), (stride, 1), padding=(2, 0)),
                    nn.LeakyReLU(0.1),
                ),
                nn.Sequential(
                    nn.Conv2d(128, 512, (kernel_size, 1), (stride, 1), padding=(2, 0)),
                    nn.LeakyReLU(0.1),
                ),
                nn.Sequential(
                    nn.Conv2d(512, 1024, (kernel_size, 1), (stride, 1), padding=(2, 0)),
                    nn.LeakyReLU(0.1),
                ),
                nn.Sequential(
                    nn.Conv2d(1024, 1024, (kernel_size, 1), 1, padding=(2, 0)),
                    nn.LeakyReLU(0.1),
                ),
            ]
        )

        self.conv_post = nn.Conv2d(1024, 1, (3, 1), 1, padding=(1, 0))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Args:
            x: Audio waveform, shape (batch, 1, audio_len)

        Returns:
            - Discriminator output
            - List of feature maps for feature matching loss
        """
        feature_maps = []

        # Reshape to 2D based on period
        batch, channels, length = x.shape

        # Pad if necessary
        if length % self.period != 0:
            pad_len = self.period - (length % self.period)
            x = F.pad(x, (0, pad_len), "reflect")
            length = x.shape[2]

        # Reshape: (batch, 1, length) -> (batch, 1, length // period, period)
        x = x.view(batch, channels, length // self.period, self.period)

        for conv in self.convs:
            x = conv(x)
            feature_maps.append(x)

        x = self.conv_post(x)
        feature_maps.append(x)
        x = x.flatten(1, -1)

        return x, feature_maps


class MultiPeriodDiscriminator(nn.Module):
    """Multi-period discriminator with multiple periods."""

    def __init__(self, periods: List[int] = [2, 3, 5, 7, 11]):
        super().__init__()
        self.discriminators = nn.ModuleList(
            [PeriodDiscriminator(p) for p in periods]
        )

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[List[torch.Tensor], List[List[torch.Tensor]]]:
        """
        Args:
            x: Audio waveform, shape (batch, 1, audio_len)

        Returns:
            - List of discriminator outputs
            - List of feature map lists for each discriminator
        """
        outputs = []
        feature_maps = []

        for disc in self.discriminators:
            out, fmaps = disc(x)
            outputs.append(out)
            feature_maps.append(fmaps)

        return outputs, feature_maps


class ScaleDiscriminator(nn.Module):
    """Scale discriminator for a single scale."""

    def __init__(self, use_spectral_norm: bool = False):
        super().__init__()

        norm_fn = nn.utils.spectral_norm if use_spectral_norm else nn.utils.weight_norm

        self.convs = nn.ModuleList(
            [
                norm_fn(nn.Conv1d(1, 128, 15, 1, padding=7)),
                norm_fn(nn.Conv1d(128, 128, 41, 2, groups=4, padding=20)),
                norm_fn(nn.Conv1d(128, 256, 41, 2, groups=16, padding=20)),
                norm_fn(nn.Conv1d(256, 512, 41, 4, groups=16, padding=20)),
                norm_fn(nn.Conv1d(512, 1024, 41, 4, groups=16, padding=20)),
                norm_fn(nn.Conv1d(1024, 1024, 41, 1, groups=16, padding=20)),
                norm_fn(nn.Conv1d(1024, 1024, 5, 1, padding=2)),
            ]
        )
        self.conv_post = norm_fn(nn.Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Args:
            x: Audio waveform, shape (batch, 1, audio_len)

        Returns:
            - Discriminator output
            - List of feature maps
        """
        feature_maps = []

        for conv in self.convs:
            x = conv(x)
            x = F.leaky_relu(x, 0.1)
            feature_maps.append(x)

        x = self.conv_post(x)
        feature_maps.append(x)
        x = x.flatten(1, -1)

        return x, feature_maps


class MultiScaleDiscriminator(nn.Module):
    """Multi-scale discriminator with downsampling."""

    def __init__(self):
        super().__init__()

        self.discriminators = nn.ModuleList(
            [
                ScaleDiscriminator(use_spectral_norm=True),
                ScaleDiscriminator(),
                ScaleDiscriminator(),
            ]
        )

        self.downsamplers = nn.ModuleList(
            [
                nn.AvgPool1d(4, 2, padding=2),
                nn.AvgPool1d(4, 2, padding=2),
            ]
        )

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[List[torch.Tensor], List[List[torch.Tensor]]]:
        """
        Args:
            x: Audio waveform, shape (batch, 1, audio_len)

        Returns:
            - List of discriminator outputs
            - List of feature map lists
        """
        outputs = []
        feature_maps = []

        for i, disc in enumerate(self.discriminators):
            if i > 0:
                x = self.downsamplers[i - 1](x)
            out, fmaps = disc(x)
            outputs.append(out)
            feature_maps.append(fmaps)

        return outputs, feature_maps


def discriminator_loss(
    disc_real_outputs: List[torch.Tensor],
    disc_fake_outputs: List[torch.Tensor],
) -> torch.Tensor:
    """
    Compute discriminator loss.

    Args:
        disc_real_outputs: Discriminator outputs for real audio
        disc_fake_outputs: Discriminator outputs for fake audio

    Returns:
        Total discriminator loss
    """
    loss = 0
    for real, fake in zip(disc_real_outputs, disc_fake_outputs):
        real_loss = torch.mean((1 - real) ** 2)
        fake_loss = torch.mean(fake ** 2)
        loss += real_loss + fake_loss
    return loss


def generator_loss(disc_outputs: List[torch.Tensor]) -> torch.Tensor:
    """
    Compute generator adversarial loss.

    Args:
        disc_outputs: Discriminator outputs for generated audio

    Returns:
        Generator loss
    """
    loss = 0
    for output in disc_outputs:
        loss += torch.mean((1 - output) ** 2)
    return loss


def feature_matching_loss(
    real_feature_maps: List[List[torch.Tensor]],
    fake_feature_maps: List[List[torch.Tensor]],
) -> torch.Tensor:
    """
    Compute feature matching loss.

    Args:
        real_feature_maps: Feature maps from discriminator for real audio
        fake_feature_maps: Feature maps from discriminator for fake audio

    Returns:
        Feature matching loss
    """
    loss = 0
    for real_fmaps, fake_fmaps in zip(real_feature_maps, fake_feature_maps):
        for real_fmap, fake_fmap in zip(real_fmaps, fake_fmaps):
            loss += torch.mean(torch.abs(real_fmap - fake_fmap))
    return loss * 2  # Scale factor from HiFi-GAN
