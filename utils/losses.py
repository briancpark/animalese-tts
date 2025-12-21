"""
Loss functions for Animalese TTS training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


class AnimaleseLoss(nn.Module):
    """
    Combined loss for Animalese TTS training.

    Includes:
    - Mel spectrogram reconstruction loss (L1)
    - Duration prediction loss (MSE)
    - Pitch prediction loss (MSE)
    - Energy prediction loss (MSE)
    """

    def __init__(
        self,
        mel_weight: float = 1.0,
        duration_weight: float = 1.0,
        pitch_weight: float = 1.0,
        energy_weight: float = 1.0,
    ):
        super().__init__()
        self.mel_weight = mel_weight
        self.duration_weight = duration_weight
        self.pitch_weight = pitch_weight
        self.energy_weight = energy_weight

        self.mel_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()

    def forward(
        self,
        mel_output: torch.Tensor,
        mel_output_before: torch.Tensor,
        mel_target: torch.Tensor,
        duration_pred: torch.Tensor,
        duration_target: torch.Tensor,
        pitch_pred: torch.Tensor,
        pitch_target: torch.Tensor,
        energy_pred: torch.Tensor,
        energy_target: torch.Tensor,
        mel_lens: Optional[torch.Tensor] = None,
        char_lens: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute combined loss.

        Args:
            mel_output: Mel spectrogram output (after postnet)
            mel_output_before: Mel spectrogram output (before postnet)
            mel_target: Target mel spectrogram
            duration_pred: Predicted durations (log scale)
            duration_target: Target durations
            pitch_pred: Predicted pitch
            pitch_target: Target pitch
            energy_pred: Predicted energy
            energy_target: Target energy
            mel_lens: Mel spectrogram lengths for masking
            char_lens: Character sequence lengths for masking

        Returns:
            Total loss and dictionary of individual losses
        """
        # Create masks
        if mel_lens is not None:
            mel_mask = self._create_mask(mel_lens, mel_target.size(2))
        else:
            mel_mask = None

        if char_lens is not None:
            char_mask = self._create_mask(char_lens, duration_target.size(1))
        else:
            char_mask = None

        # Mel loss (with postnet)
        mel_loss = self._masked_l1_loss(mel_output, mel_target, mel_mask)

        # Mel loss (before postnet)
        mel_loss_before = self._masked_l1_loss(mel_output_before, mel_target, mel_mask)

        # Duration loss (in log space)
        duration_target_log = torch.log(duration_target.float() + 1)
        duration_loss = self._masked_mse_loss(duration_pred, duration_target_log, char_mask)

        # Pitch loss
        pitch_loss = self._masked_mse_loss(pitch_pred, pitch_target, mel_mask)

        # Energy loss
        energy_loss = self._masked_mse_loss(energy_pred, energy_target, mel_mask)

        # Combined loss
        total_loss = (
            self.mel_weight * (mel_loss + mel_loss_before)
            + self.duration_weight * duration_loss
            + self.pitch_weight * pitch_loss
            + self.energy_weight * energy_loss
        )

        loss_dict = {
            "total": total_loss,
            "mel": mel_loss,
            "mel_before": mel_loss_before,
            "duration": duration_loss,
            "pitch": pitch_loss,
            "energy": energy_loss,
        }

        return total_loss, loss_dict

    def _create_mask(self, lengths: torch.Tensor, max_len: int) -> torch.Tensor:
        """Create boolean mask from lengths."""
        batch_size = lengths.size(0)
        mask = torch.arange(max_len, device=lengths.device).unsqueeze(0).expand(batch_size, -1)
        mask = mask < lengths.unsqueeze(1)
        return mask

    def _masked_l1_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute masked L1 loss."""
        if mask is None:
            return F.l1_loss(pred, target)

        # Expand mask to match mel dimensions (batch, n_mels, mel_len)
        mask = mask.unsqueeze(1).expand_as(pred)
        loss = F.l1_loss(pred * mask, target * mask, reduction="sum")
        return loss / mask.sum().clamp(min=1)

    def _masked_mse_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute masked MSE loss."""
        if mask is None:
            return F.mse_loss(pred, target)

        # Handle different tensor shapes
        if pred.dim() != mask.dim():
            if pred.dim() > mask.dim():
                mask = mask.unsqueeze(-1).expand_as(pred)

        loss = F.mse_loss(pred * mask, target * mask, reduction="sum")
        return loss / mask.sum().clamp(min=1)


class MultiResolutionSTFTLoss(nn.Module):
    """
    Multi-resolution STFT loss for waveform reconstruction.

    Helps ensure the generated audio matches the target at multiple
    time-frequency resolutions.
    """

    def __init__(
        self,
        fft_sizes: list = [512, 1024, 2048],
        hop_sizes: list = [50, 120, 240],
        win_lengths: list = [240, 600, 1200],
    ):
        super().__init__()
        self.fft_sizes = fft_sizes
        self.hop_sizes = hop_sizes
        self.win_lengths = win_lengths

    def forward(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute multi-resolution STFT loss.

        Args:
            y_pred: Predicted waveform, shape (batch, 1, audio_len)
            y_true: Target waveform, shape (batch, 1, audio_len)

        Returns:
            Spectral convergence loss and magnitude loss
        """
        sc_loss = 0.0
        mag_loss = 0.0

        for fft_size, hop_size, win_length in zip(
            self.fft_sizes, self.hop_sizes, self.win_lengths
        ):
            sc, mag = self._stft_loss(
                y_pred.squeeze(1), y_true.squeeze(1), fft_size, hop_size, win_length
            )
            sc_loss += sc
            mag_loss += mag

        sc_loss /= len(self.fft_sizes)
        mag_loss /= len(self.fft_sizes)

        return sc_loss, mag_loss

    def _stft_loss(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        fft_size: int,
        hop_size: int,
        win_length: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute STFT loss for a single resolution."""
        window = torch.hann_window(win_length, device=y_pred.device)

        # Compute STFT
        pred_stft = torch.stft(
            y_pred,
            n_fft=fft_size,
            hop_length=hop_size,
            win_length=win_length,
            window=window,
            return_complex=True,
        )
        true_stft = torch.stft(
            y_true,
            n_fft=fft_size,
            hop_length=hop_size,
            win_length=win_length,
            window=window,
            return_complex=True,
        )

        # Magnitude
        pred_mag = torch.abs(pred_stft)
        true_mag = torch.abs(true_stft)

        # Spectral convergence
        sc_loss = torch.norm(true_mag - pred_mag, p="fro") / torch.norm(true_mag, p="fro").clamp(min=1e-8)

        # Log magnitude
        pred_log = torch.log(pred_mag.clamp(min=1e-5))
        true_log = torch.log(true_mag.clamp(min=1e-5))
        mag_loss = F.l1_loss(pred_log, true_log)

        return sc_loss, mag_loss
