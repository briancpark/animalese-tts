"""
Animalese TTS - Main Model

End-to-end text-to-speech model for generating Animalese (Animal Crossing style speech).
Combines character encoder, variance adaptor, mel decoder, and neural vocoder.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple

from .encoder import CharacterEncoder
from .variance_adaptor import VarianceAdaptor
from .decoder import MelDecoder
from .vocoder import AnimalVocoder


class AnimaleseTTS(nn.Module):
    """
    Animalese Text-to-Speech Model.

    This model converts text input to Animalese-style audio output.
    The architecture follows a non-autoregressive design similar to FastSpeech2,
    but adapted for the unique characteristics of Animalese:

    1. Character-level encoding (each character maps to a distinct sound)
    2. Pitch variation (creating the characteristic melodic speech)
    3. Fast, babbling-like delivery
    4. Simple phoneme-like sounds

    Components:
    - Character Encoder: Encodes ASCII characters into hidden representations
    - Variance Adaptor: Predicts and applies duration, pitch, and energy
    - Mel Decoder: Generates mel spectrograms from adapted representations
    - Vocoder: Converts mel spectrograms to audio waveforms
    """

    def __init__(
        self,
        # Encoder params
        vocab_size: int = 128,
        encoder_embed_dim: int = 256,
        encoder_hidden_dim: int = 512,
        encoder_n_layers: int = 4,
        encoder_n_heads: int = 4,
        # Variance adaptor params
        variance_hidden_dim: int = 256,
        variance_kernel_size: int = 3,
        # Decoder params
        decoder_hidden_dim: int = 512,
        decoder_n_layers: int = 4,
        decoder_n_heads: int = 4,
        # Audio params
        n_mels: int = 80,
        # Vocoder params
        vocoder_upsample_rates: list = [8, 8, 2, 2],
        vocoder_upsample_kernels: list = [16, 16, 4, 4],
        vocoder_resblock_kernels: list = [3, 7, 11],
        vocoder_resblock_dilations: list = [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        vocoder_initial_channel: int = 512,
        # General
        dropout: float = 0.1,
    ):
        super().__init__()

        self.n_mels = n_mels

        # Character encoder
        self.encoder = CharacterEncoder(
            vocab_size=vocab_size,
            embed_dim=encoder_embed_dim,
            hidden_dim=encoder_hidden_dim,
            n_layers=encoder_n_layers,
            n_heads=encoder_n_heads,
            dropout=dropout,
        )

        # Variance adaptor
        self.variance_adaptor = VarianceAdaptor(
            hidden_dim=encoder_hidden_dim,
            kernel_size=variance_kernel_size,
            dropout=dropout,
        )

        # Projection layer if encoder and decoder dimensions differ
        if encoder_hidden_dim != decoder_hidden_dim:
            self.enc_to_dec_proj = nn.Linear(encoder_hidden_dim, decoder_hidden_dim)
        else:
            self.enc_to_dec_proj = None

        # Mel decoder
        self.decoder = MelDecoder(
            hidden_dim=decoder_hidden_dim,
            n_mels=n_mels,
            n_layers=decoder_n_layers,
            n_heads=decoder_n_heads,
            dropout=dropout,
        )

        # Vocoder
        self.vocoder = AnimalVocoder(
            n_mels=n_mels,
            upsample_rates=vocoder_upsample_rates,
            upsample_kernel_sizes=vocoder_upsample_kernels,
            upsample_initial_channel=vocoder_initial_channel,
            resblock_kernel_sizes=vocoder_resblock_kernels,
            resblock_dilation_sizes=vocoder_resblock_dilations,
        )

    def forward(
        self,
        chars: torch.Tensor,
        char_lens: Optional[torch.Tensor] = None,
        duration_target: Optional[torch.Tensor] = None,
        pitch_target: Optional[torch.Tensor] = None,
        energy_target: Optional[torch.Tensor] = None,
        mel_lens: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for training.

        Args:
            chars: Character indices, shape (batch, max_char_len)
            char_lens: Character sequence lengths, shape (batch,)
            duration_target: Ground truth durations, shape (batch, max_char_len)
            pitch_target: Ground truth pitch, shape (batch, max_mel_len)
            energy_target: Ground truth energy, shape (batch, max_mel_len)
            mel_lens: Mel spectrogram lengths, shape (batch,)

        Returns:
            Dictionary containing:
            - mel_output: Predicted mel spectrogram (with postnet)
            - mel_output_before: Predicted mel spectrogram (before postnet)
            - duration_pred: Predicted durations
            - pitch_pred: Predicted pitch
            - energy_pred: Predicted energy
            - audio: Generated audio waveform
        """
        # Create padding mask
        if char_lens is not None:
            max_len = chars.size(1)
            mask = torch.arange(max_len, device=chars.device).unsqueeze(0) >= char_lens.unsqueeze(1)
        else:
            mask = None

        # Encode characters
        encoder_output = self.encoder(chars, mask)

        # Apply variance adaptor
        max_mel_len = mel_lens.max().item() if mel_lens is not None else None
        adapted_output, duration_pred, pitch_pred, energy_pred, output_lens = self.variance_adaptor(
            encoder_output,
            mask=mask,
            duration_target=duration_target,
            pitch_target=pitch_target,
            energy_target=energy_target,
            max_len=max_mel_len,
        )

        # Project to decoder dimension if needed
        if self.enc_to_dec_proj is not None:
            adapted_output = self.enc_to_dec_proj(adapted_output)

        # Create decoder mask
        if output_lens is not None:
            max_out_len = adapted_output.size(1)
            decoder_mask = torch.arange(max_out_len, device=chars.device).unsqueeze(0) >= output_lens.unsqueeze(1)
        else:
            decoder_mask = None

        # Decode to mel spectrogram
        mel_output, mel_output_before = self.decoder(adapted_output, decoder_mask)

        # Generate audio
        audio = self.vocoder(mel_output)

        return {
            "mel_output": mel_output,
            "mel_output_before": mel_output_before,
            "duration_pred": duration_pred,
            "pitch_pred": pitch_pred,
            "energy_pred": energy_pred,
            "audio": audio,
            "output_lens": output_lens,
        }

    @torch.no_grad()
    def infer(
        self,
        chars: torch.Tensor,
        char_lens: Optional[torch.Tensor] = None,
        duration_scale: float = 1.0,
        pitch_scale: float = 1.0,
        energy_scale: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        """
        Inference mode for generating Animalese audio.

        Args:
            chars: Character indices, shape (batch, max_char_len)
            char_lens: Character sequence lengths
            duration_scale: Scale factor for predicted durations
            pitch_scale: Scale factor for predicted pitch
            energy_scale: Scale factor for predicted energy

        Returns:
            Dictionary containing generated audio and mel spectrogram
        """
        self.eval()

        # Create padding mask
        if char_lens is not None:
            max_len = chars.size(1)
            mask = torch.arange(max_len, device=chars.device).unsqueeze(0) >= char_lens.unsqueeze(1)
        else:
            mask = None

        # Encode characters
        encoder_output = self.encoder(chars, mask)

        # Apply variance adaptor (no targets - uses predictions)
        adapted_output, duration_pred, pitch_pred, energy_pred, output_lens = self.variance_adaptor(
            encoder_output,
            mask=mask,
        )

        # Apply scaling
        # Note: scaling is applied during variance adaptor in a more sophisticated implementation
        # Here we just use the raw predictions

        # Project to decoder dimension if needed
        if self.enc_to_dec_proj is not None:
            adapted_output = self.enc_to_dec_proj(adapted_output)

        # Create decoder mask
        if output_lens is not None:
            max_out_len = adapted_output.size(1)
            decoder_mask = torch.arange(max_out_len, device=chars.device).unsqueeze(0) >= output_lens.unsqueeze(1)
        else:
            decoder_mask = None

        # Decode to mel spectrogram
        mel_output, _ = self.decoder(adapted_output, decoder_mask)

        # Generate audio
        audio = self.vocoder(mel_output)

        return {
            "audio": audio,
            "mel_output": mel_output,
            "duration_pred": duration_pred,
            "output_lens": output_lens,
        }

    def text_to_audio(
        self,
        text: str,
        device: torch.device = torch.device("cpu"),
    ) -> torch.Tensor:
        """
        Convert text string directly to audio.

        Args:
            text: Input text string
            device: Device to run inference on

        Returns:
            Audio waveform tensor
        """
        # Convert text to character indices
        chars = torch.tensor(
            [[ord(c) for c in text]],
            dtype=torch.long,
            device=device
        )
        char_lens = torch.tensor([len(text)], device=device)

        # Generate audio
        output = self.infer(chars, char_lens)

        return output["audio"].squeeze(0)

    def get_num_params(self) -> Dict[str, int]:
        """Get the number of parameters for each component."""
        def count_params(module):
            return sum(p.numel() for p in module.parameters())

        return {
            "encoder": count_params(self.encoder),
            "variance_adaptor": count_params(self.variance_adaptor),
            "decoder": count_params(self.decoder),
            "vocoder": count_params(self.vocoder),
            "total": count_params(self),
        }
