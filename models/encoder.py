"""
Character Encoder for Animalese TTS.

Encodes text characters into hidden representations using a Transformer encoder.
Animalese maps each character to a distinct phoneme-like sound, so we need
character-level encoding rather than phoneme-based encoding.
"""

import torch
import torch.nn as nn
import math
from typing import Optional


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )

        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (seq_len, batch, embed_dim)
        """
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)


class ConvBlock(nn.Module):
    """Convolutional block with layer norm and activation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)
        self.norm = nn.LayerNorm(out_channels)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch, seq_len, channels)
        """
        # (batch, seq_len, channels) -> (batch, channels, seq_len)
        x = x.transpose(1, 2)
        x = self.conv(x)
        # (batch, channels, seq_len) -> (batch, seq_len, channels)
        x = x.transpose(1, 2)
        x = self.norm(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x


class FFTBlock(nn.Module):
    """Feed-Forward Transformer block."""

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

        # Convolutional feed-forward
        self.conv1 = ConvBlock(hidden_dim, ff_dim, kernel_size, dropout)
        self.conv2 = ConvBlock(ff_dim, hidden_dim, kernel_size, dropout)
        self.norm2 = nn.LayerNorm(hidden_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch, seq_len, hidden_dim)
            mask: Optional attention mask
        """
        # Self-attention with residual
        residual = x
        x, _ = self.self_attn(x, x, x, key_padding_mask=mask)
        x = self.dropout(x)
        x = self.norm1(x + residual)

        # Feed-forward with residual
        residual = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.norm2(x + residual)

        return x


class CharacterEncoder(nn.Module):
    """
    Encodes text characters for Animalese TTS.

    Each ASCII character is embedded and processed through transformer layers
    to produce hidden representations that capture character identity and context.
    """

    def __init__(
        self,
        vocab_size: int = 128,
        embed_dim: int = 256,
        hidden_dim: int = 512,
        n_layers: int = 4,
        n_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim

        # Character embedding
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        # Project to hidden dimension
        self.prenet = nn.Linear(embed_dim, hidden_dim)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(hidden_dim, dropout=dropout)

        # Transformer encoder layers
        self.layers = nn.ModuleList(
            [
                FFTBlock(hidden_dim, n_heads, hidden_dim * 4, dropout=dropout)
                for _ in range(n_layers)
            ]
        )

        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        chars: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            chars: Character indices, shape (batch, seq_len)
            mask: Padding mask, shape (batch, seq_len), True for padding

        Returns:
            Hidden representations, shape (batch, seq_len, hidden_dim)
        """
        # Embed characters
        x = self.embedding(chars)

        # Project to hidden dimension
        x = self.prenet(x)

        # Add positional encoding
        x = x.transpose(0, 1)  # (seq_len, batch, hidden_dim)
        x = self.pos_encoder(x)
        x = x.transpose(0, 1)  # (batch, seq_len, hidden_dim)

        # Pass through transformer layers
        for layer in self.layers:
            x = layer(x, mask)

        x = self.norm(x)

        return x
