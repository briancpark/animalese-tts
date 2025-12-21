"""
Mel Spectrogram Decoder for Animalese TTS.

Converts the variance-adapted hidden representations into mel spectrograms.
Uses a Feed-Forward Transformer architecture similar to the encoder.
"""

import torch
import torch.nn as nn
import math
from typing import Optional


class PostNet(nn.Module):
    """
    Post-processing network to refine mel spectrogram predictions.

    Uses a stack of 1D convolutions with residual connections.
    """

    def __init__(
        self,
        n_mels: int = 80,
        hidden_dim: int = 512,
        n_layers: int = 5,
        kernel_size: int = 5,
        dropout: float = 0.1,
    ):
        super().__init__()

        padding = (kernel_size - 1) // 2

        layers = []

        # First layer: n_mels -> hidden_dim
        layers.append(
            nn.Sequential(
                nn.Conv1d(n_mels, hidden_dim, kernel_size, padding=padding),
                nn.BatchNorm1d(hidden_dim),
                nn.Tanh(),
                nn.Dropout(dropout),
            )
        )

        # Middle layers: hidden_dim -> hidden_dim
        for _ in range(n_layers - 2):
            layers.append(
                nn.Sequential(
                    nn.Conv1d(hidden_dim, hidden_dim, kernel_size, padding=padding),
                    nn.BatchNorm1d(hidden_dim),
                    nn.Tanh(),
                    nn.Dropout(dropout),
                )
            )

        # Last layer: hidden_dim -> n_mels
        layers.append(
            nn.Sequential(
                nn.Conv1d(hidden_dim, n_mels, kernel_size, padding=padding),
                nn.Dropout(dropout),
            )
        )

        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Mel spectrogram, shape (batch, n_mels, mel_len)

        Returns:
            Refined mel spectrogram, shape (batch, n_mels, mel_len)
        """
        out = x
        for layer in self.layers:
            out = layer(out)
        return out


class FFTBlock(nn.Module):
    """Feed-Forward Transformer block for the decoder."""

    def __init__(
        self,
        hidden_dim: int,
        n_heads: int,
        ff_dim: int = 2048,
        kernel_size: int = 9,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.self_attn = nn.MultiheadAttention(
            hidden_dim, n_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(hidden_dim)

        padding = (kernel_size - 1) // 2
        self.conv1 = nn.Conv1d(hidden_dim, ff_dim, kernel_size, padding=padding)
        self.conv2 = nn.Conv1d(ff_dim, hidden_dim, kernel_size, padding=padding)
        self.norm2 = nn.LayerNorm(hidden_dim)

        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: Input tensor, shape (batch, seq_len, hidden_dim)
            mask: Padding mask

        Returns:
            Output tensor, shape (batch, seq_len, hidden_dim)
        """
        # Self-attention
        residual = x
        x, _ = self.self_attn(x, x, x, key_padding_mask=mask)
        x = self.dropout(x)
        x = self.norm1(x + residual)

        # Convolutional feed-forward
        residual = x
        x = x.transpose(1, 2)  # (batch, hidden_dim, seq_len)
        x = self.conv1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = x.transpose(1, 2)  # (batch, seq_len, hidden_dim)
        x = self.dropout(x)
        x = self.norm2(x + residual)

        return x


class MelDecoder(nn.Module):
    """
    Decodes hidden representations to mel spectrograms.

    The decoder uses FFT blocks similar to FastSpeech2 to convert
    the variance-adapted representations into mel spectrograms.
    """

    def __init__(
        self,
        hidden_dim: int = 512,
        n_mels: int = 80,
        n_layers: int = 4,
        n_heads: int = 4,
        ff_dim: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.n_mels = n_mels

        # FFT blocks
        self.layers = nn.ModuleList(
            [
                FFTBlock(hidden_dim, n_heads, ff_dim, dropout=dropout)
                for _ in range(n_layers)
            ]
        )

        self.norm = nn.LayerNorm(hidden_dim)

        # Linear projection to mel spectrogram
        self.mel_linear = nn.Linear(hidden_dim, n_mels)

        # PostNet for refinement
        self.postnet = PostNet(n_mels, hidden_dim, n_layers=5, dropout=dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> tuple:
        """
        Args:
            x: Variance-adapted hidden states, shape (batch, mel_len, hidden_dim)
            mask: Padding mask

        Returns:
            - mel_output: Mel spectrogram after postnet, shape (batch, n_mels, mel_len)
            - mel_output_before: Mel spectrogram before postnet
        """
        # Pass through decoder layers
        for layer in self.layers:
            x = layer(x, mask)

        x = self.norm(x)

        # Project to mel dimensions
        mel_output_before = self.mel_linear(x)

        # (batch, mel_len, n_mels) -> (batch, n_mels, mel_len)
        mel_output_before = mel_output_before.transpose(1, 2)

        # Apply postnet
        mel_residual = self.postnet(mel_output_before)
        mel_output = mel_output_before + mel_residual

        return mel_output, mel_output_before
