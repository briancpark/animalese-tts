"""
Variance Adaptor for Animalese TTS.

Predicts and applies duration, pitch, and energy variations to the encoder output.
This is crucial for Animalese as each character has specific timing and pitch characteristics.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class VariancePredictor(nn.Module):
    """Predicts a single variance parameter (duration, pitch, or energy)."""

    def __init__(
        self,
        hidden_dim: int = 256,
        kernel_size: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()

        padding = (kernel_size - 1) // 2

        self.layers = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size, padding=padding),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size, padding=padding),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
        )

        self.output = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Input tensor, shape (batch, seq_len, hidden_dim)
            mask: Padding mask, shape (batch, seq_len)

        Returns:
            Predicted values, shape (batch, seq_len)
        """
        # (batch, seq_len, hidden_dim) -> (batch, hidden_dim, seq_len)
        out = x.transpose(1, 2)

        for layer in self.layers:
            if isinstance(layer, nn.LayerNorm):
                out = out.transpose(1, 2)
                out = layer(out)
                out = out.transpose(1, 2)
            else:
                out = layer(out)

        # (batch, hidden_dim, seq_len) -> (batch, seq_len, hidden_dim)
        out = out.transpose(1, 2)
        out = self.output(out).squeeze(-1)

        if mask is not None:
            out = out.masked_fill(mask, 0.0)

        return out


class LengthRegulator(nn.Module):
    """Expands encoder output based on predicted durations."""

    def __init__(self):
        super().__init__()

    def forward(
        self,
        x: torch.Tensor,
        durations: torch.Tensor,
        max_len: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input tensor, shape (batch, seq_len, hidden_dim)
            durations: Duration for each position, shape (batch, seq_len)
            max_len: Maximum output length

        Returns:
            Expanded tensor and output lengths
        """
        batch_size, seq_len, hidden_dim = x.shape
        durations = durations.long()

        # Calculate output lengths
        output_lengths = durations.sum(dim=1)

        if max_len is None:
            max_len = output_lengths.max().item()

        # Expand each sequence
        expanded = []
        for i in range(batch_size):
            seq_expanded = []
            for j in range(seq_len):
                dur = durations[i, j].item()
                if dur > 0:
                    seq_expanded.append(x[i, j : j + 1].expand(dur, -1))

            if seq_expanded:
                seq_expanded = torch.cat(seq_expanded, dim=0)
                # Pad or truncate to max_len
                if seq_expanded.size(0) < max_len:
                    padding = torch.zeros(
                        max_len - seq_expanded.size(0),
                        hidden_dim,
                        device=x.device,
                        dtype=x.dtype,
                    )
                    seq_expanded = torch.cat([seq_expanded, padding], dim=0)
                else:
                    seq_expanded = seq_expanded[:max_len]
            else:
                seq_expanded = torch.zeros(max_len, hidden_dim, device=x.device, dtype=x.dtype)

            expanded.append(seq_expanded)

        expanded = torch.stack(expanded, dim=0)

        return expanded, output_lengths


class PitchEmbedding(nn.Module):
    """Converts pitch values to embeddings using continuous representation."""

    def __init__(self, hidden_dim: int = 256, n_bins: int = 256):
        super().__init__()
        self.n_bins = n_bins
        self.embedding = nn.Embedding(n_bins, hidden_dim)

    def forward(self, pitch: torch.Tensor, pitch_min: float = 50.0, pitch_max: float = 400.0) -> torch.Tensor:
        """
        Args:
            pitch: Pitch values in Hz, shape (batch, seq_len)

        Returns:
            Pitch embeddings, shape (batch, seq_len, hidden_dim)
        """
        # Normalize and quantize pitch to bins
        pitch_normalized = (pitch - pitch_min) / (pitch_max - pitch_min)
        pitch_normalized = torch.clamp(pitch_normalized, 0, 1)
        pitch_bins = (pitch_normalized * (self.n_bins - 1)).long()

        return self.embedding(pitch_bins)


class EnergyEmbedding(nn.Module):
    """Converts energy values to embeddings."""

    def __init__(self, hidden_dim: int = 256, n_bins: int = 256):
        super().__init__()
        self.n_bins = n_bins
        self.embedding = nn.Embedding(n_bins, hidden_dim)

    def forward(self, energy: torch.Tensor, energy_min: float = 0.0, energy_max: float = 1.0) -> torch.Tensor:
        """
        Args:
            energy: Energy values, shape (batch, seq_len)

        Returns:
            Energy embeddings, shape (batch, seq_len, hidden_dim)
        """
        # Normalize and quantize energy to bins
        energy_normalized = (energy - energy_min) / (energy_max - energy_min + 1e-8)
        energy_normalized = torch.clamp(energy_normalized, 0, 1)
        energy_bins = (energy_normalized * (self.n_bins - 1)).long()

        return self.embedding(energy_bins)


class VarianceAdaptor(nn.Module):
    """
    Adapts encoder output with duration, pitch, and energy information.

    For Animalese, this is particularly important as:
    - Duration controls the length of each character sound
    - Pitch creates the characteristic varying tones
    - Energy affects the prominence of each sound
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        kernel_size: int = 3,
        dropout: float = 0.1,
        pitch_bins: int = 256,
        energy_bins: int = 256,
    ):
        super().__init__()

        self.duration_predictor = VariancePredictor(hidden_dim, kernel_size, dropout)
        self.pitch_predictor = VariancePredictor(hidden_dim, kernel_size, dropout)
        self.energy_predictor = VariancePredictor(hidden_dim, kernel_size, dropout)

        self.length_regulator = LengthRegulator()

        self.pitch_embedding = PitchEmbedding(hidden_dim, pitch_bins)
        self.energy_embedding = EnergyEmbedding(hidden_dim, energy_bins)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        duration_target: Optional[torch.Tensor] = None,
        pitch_target: Optional[torch.Tensor] = None,
        energy_target: Optional[torch.Tensor] = None,
        max_len: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Encoder output, shape (batch, seq_len, hidden_dim)
            mask: Padding mask
            duration_target: Ground truth durations for training
            pitch_target: Ground truth pitch for training
            energy_target: Ground truth energy for training
            max_len: Maximum output length

        Returns:
            - Adapted output (batch, mel_len, hidden_dim)
            - Duration predictions
            - Pitch predictions
            - Energy predictions
            - Output lengths
        """
        # Predict durations
        duration_pred = self.duration_predictor(x, mask)

        # Use targets during training, predictions during inference
        if duration_target is not None:
            duration_rounded = duration_target
        else:
            duration_rounded = torch.clamp(torch.round(torch.exp(duration_pred) - 1), min=0)

        # Apply length regulation
        x, output_lengths = self.length_regulator(x, duration_rounded, max_len)

        # Create expanded mask
        if max_len is None:
            max_len = x.size(1)
        expanded_mask = self._create_expanded_mask(output_lengths, max_len, x.device)

        # Predict pitch
        pitch_pred = self.pitch_predictor(x, expanded_mask)

        if pitch_target is not None:
            pitch = pitch_target
        else:
            # Convert log-pitch prediction to Hz
            pitch = torch.exp(pitch_pred) * 200  # Base frequency around 200 Hz

        # Add pitch embedding
        pitch_emb = self.pitch_embedding(pitch)
        x = x + pitch_emb

        # Predict energy
        energy_pred = self.energy_predictor(x, expanded_mask)

        if energy_target is not None:
            energy = energy_target
        else:
            energy = torch.sigmoid(energy_pred)

        # Add energy embedding
        energy_emb = self.energy_embedding(energy)
        x = x + energy_emb

        return x, duration_pred, pitch_pred, energy_pred, output_lengths

    def _create_expanded_mask(
        self,
        lengths: torch.Tensor,
        max_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Create mask for expanded sequences."""
        batch_size = lengths.size(0)
        mask = torch.arange(max_len, device=device).unsqueeze(0).expand(batch_size, -1)
        mask = mask >= lengths.unsqueeze(1)
        return mask
