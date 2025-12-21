"""
Neural Vocoder for Animalese TTS.

A lightweight HiFi-GAN style vocoder that converts mel spectrograms to audio waveforms.
Optimized for the simpler Animalese audio characteristics.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


def get_padding(kernel_size: int, dilation: int = 1) -> int:
    """Calculate padding for 'same' convolution."""
    return int((kernel_size * dilation - dilation) / 2)


class ResBlock(nn.Module):
    """Residual block with dilated convolutions."""

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dilations: List[int] = [1, 3, 5],
    ):
        super().__init__()

        self.convs1 = nn.ModuleList()
        self.convs2 = nn.ModuleList()

        for dilation in dilations:
            self.convs1.append(
                nn.Sequential(
                    nn.LeakyReLU(0.1),
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        dilation=dilation,
                        padding=get_padding(kernel_size, dilation),
                    ),
                )
            )
            self.convs2.append(
                nn.Sequential(
                    nn.LeakyReLU(0.1),
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        dilation=1,
                        padding=get_padding(kernel_size, 1),
                    ),
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for conv1, conv2 in zip(self.convs1, self.convs2):
            residual = x
            x = conv1(x)
            x = conv2(x)
            x = x + residual
        return x


class MultiReceptiveFieldFusion(nn.Module):
    """Multi-receptive field fusion module."""

    def __init__(
        self,
        channels: int,
        resblock_kernel_sizes: List[int] = [3, 7, 11],
        resblock_dilation_sizes: List[List[int]] = [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
    ):
        super().__init__()

        self.resblocks = nn.ModuleList()
        for kernel_size, dilations in zip(resblock_kernel_sizes, resblock_dilation_sizes):
            self.resblocks.append(ResBlock(channels, kernel_size, dilations))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = None
        for resblock in self.resblocks:
            if out is None:
                out = resblock(x)
            else:
                out = out + resblock(x)
        return out / len(self.resblocks)


class AnimalVocoder(nn.Module):
    """
    HiFi-GAN style vocoder for converting mel spectrograms to audio.

    Uses a stack of transposed convolutions for upsampling, with
    multi-receptive field fusion blocks for quality enhancement.
    """

    def __init__(
        self,
        n_mels: int = 80,
        upsample_rates: List[int] = [8, 8, 2, 2],
        upsample_kernel_sizes: List[int] = [16, 16, 4, 4],
        upsample_initial_channel: int = 512,
        resblock_kernel_sizes: List[int] = [3, 7, 11],
        resblock_dilation_sizes: List[List[int]] = [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
    ):
        super().__init__()

        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)

        # Initial convolution
        self.conv_pre = nn.Conv1d(n_mels, upsample_initial_channel, 7, padding=3)

        # Upsampling layers
        self.ups = nn.ModuleList()
        self.mrfs = nn.ModuleList()

        ch = upsample_initial_channel
        for i, (rate, kernel_size) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(
                nn.ConvTranspose1d(
                    ch,
                    ch // 2,
                    kernel_size,
                    rate,
                    padding=(kernel_size - rate) // 2,
                )
            )
            ch = ch // 2
            self.mrfs.append(
                MultiReceptiveFieldFusion(ch, resblock_kernel_sizes, resblock_dilation_sizes)
            )

        # Final convolution
        self.conv_post = nn.Conv1d(ch, 1, 7, padding=3)

        # Weight initialization
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d)):
            nn.init.normal_(m.weight, 0.0, 0.01)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mel: Mel spectrogram, shape (batch, n_mels, mel_len)

        Returns:
            Audio waveform, shape (batch, 1, audio_len)
        """
        x = self.conv_pre(mel)

        for up, mrf in zip(self.ups, self.mrfs):
            x = F.leaky_relu(x, 0.1)
            x = up(x)
            x = mrf(x)

        x = F.leaky_relu(x, 0.1)
        x = self.conv_post(x)
        x = torch.tanh(x)

        return x

    def remove_weight_norm(self):
        """Remove weight normalization for inference."""
        for up in self.ups:
            try:
                nn.utils.remove_weight_norm(up)
            except ValueError:
                pass

        for mrf in self.mrfs:
            for resblock in mrf.resblocks:
                for conv in resblock.convs1:
                    try:
                        nn.utils.remove_weight_norm(conv[1])
                    except ValueError:
                        pass
                for conv in resblock.convs2:
                    try:
                        nn.utils.remove_weight_norm(conv[1])
                    except ValueError:
                        pass
