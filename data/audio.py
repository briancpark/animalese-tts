"""
Audio processing utilities for Animalese TTS.

Handles mel spectrogram extraction, pitch estimation, and energy computation.
"""

import torch
import torch.nn.functional as F
import numpy as np
import librosa
from typing import Optional, Tuple, Dict
from scipy.interpolate import interp1d


class AudioProcessor:
    """
    Audio processing for Animalese TTS training and inference.

    Extracts mel spectrograms, pitch (F0), and energy from audio.
    """

    def __init__(
        self,
        sample_rate: int = 22050,
        n_fft: int = 1024,
        hop_length: int = 256,
        win_length: int = 1024,
        n_mels: int = 80,
        mel_fmin: float = 0.0,
        mel_fmax: float = 8000.0,
        pitch_min: float = 50.0,
        pitch_max: float = 400.0,
    ):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_mels = n_mels
        self.mel_fmin = mel_fmin
        self.mel_fmax = mel_fmax
        self.pitch_min = pitch_min
        self.pitch_max = pitch_max

        # Mel filterbank
        self.mel_basis = librosa.filters.mel(
            sr=sample_rate,
            n_fft=n_fft,
            n_mels=n_mels,
            fmin=mel_fmin,
            fmax=mel_fmax,
        )

    def load_audio(self, path: str) -> np.ndarray:
        """Load audio file and resample if necessary."""
        audio, sr = librosa.load(path, sr=self.sample_rate)
        return audio

    def get_mel_spectrogram(self, audio: np.ndarray) -> np.ndarray:
        """
        Compute mel spectrogram from audio.

        Args:
            audio: Audio waveform, shape (audio_len,)

        Returns:
            Mel spectrogram, shape (n_mels, mel_len)
        """
        # STFT
        stft = librosa.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window="hann",
            center=True,
            pad_mode="reflect",
        )

        # Magnitude spectrogram
        mag = np.abs(stft)

        # Mel spectrogram
        mel = np.dot(self.mel_basis, mag)

        # Log scale
        mel = np.log(np.clip(mel, a_min=1e-5, a_max=None))

        return mel

    def get_pitch(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract pitch (F0) from audio using PYIN algorithm.

        Args:
            audio: Audio waveform, shape (audio_len,)

        Returns:
            Pitch contour, shape (mel_len,)
        """
        # Use librosa's pyin for pitch extraction
        f0, voiced_flag, voiced_probs = librosa.pyin(
            audio,
            fmin=self.pitch_min,
            fmax=self.pitch_max,
            sr=self.sample_rate,
            hop_length=self.hop_length,
        )

        # Replace NaN with 0 (unvoiced frames)
        f0 = np.nan_to_num(f0, nan=0.0)

        # Interpolate unvoiced regions for continuous pitch
        if np.any(f0 > 0):
            nonzero_ids = np.where(f0 > 0)[0]
            if len(nonzero_ids) > 1:
                interp_fn = interp1d(
                    nonzero_ids,
                    f0[nonzero_ids],
                    kind="linear",
                    fill_value=(f0[nonzero_ids[0]], f0[nonzero_ids[-1]]),
                    bounds_error=False,
                )
                f0_interp = interp_fn(np.arange(len(f0)))
                f0 = f0_interp

        return f0

    def get_energy(self, audio: np.ndarray) -> np.ndarray:
        """
        Compute energy (RMS) from audio.

        Args:
            audio: Audio waveform, shape (audio_len,)

        Returns:
            Energy contour, shape (mel_len,)
        """
        # STFT
        stft = librosa.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
        )

        # Energy as L2 norm of magnitude spectrogram
        energy = np.linalg.norm(np.abs(stft), axis=0)

        return energy

    def process_audio(self, audio: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extract all features from audio.

        Args:
            audio: Audio waveform, shape (audio_len,)

        Returns:
            Dictionary containing mel, pitch, and energy
        """
        mel = self.get_mel_spectrogram(audio)
        pitch = self.get_pitch(audio)
        energy = self.get_energy(audio)

        # Ensure all have same length
        min_len = min(mel.shape[1], len(pitch), len(energy))
        mel = mel[:, :min_len]
        pitch = pitch[:min_len]
        energy = energy[:min_len]

        return {
            "mel": mel,
            "pitch": pitch,
            "energy": energy,
        }

    def mel_to_audio(
        self,
        mel: np.ndarray,
        n_iter: int = 32,
    ) -> np.ndarray:
        """
        Convert mel spectrogram back to audio using Griffin-Lim.
        (Used for debugging - neural vocoder is preferred)

        Args:
            mel: Mel spectrogram, shape (n_mels, mel_len)
            n_iter: Number of Griffin-Lim iterations

        Returns:
            Reconstructed audio waveform
        """
        # Exp to undo log
        mel = np.exp(mel)

        # Invert mel filterbank
        mel_inv = np.linalg.pinv(self.mel_basis)
        mag = np.maximum(1e-10, np.dot(mel_inv, mel))

        # Griffin-Lim
        audio = librosa.griffinlim(
            mag,
            n_iter=n_iter,
            hop_length=self.hop_length,
            win_length=self.win_length,
        )

        return audio

    def compute_duration(
        self,
        char_len: int,
        mel_len: int,
    ) -> np.ndarray:
        """
        Compute uniform durations for characters.
        Used for synthetic data or when duration alignment is not available.

        Args:
            char_len: Number of characters
            mel_len: Number of mel frames

        Returns:
            Duration array, shape (char_len,)
        """
        # Simple uniform distribution
        base_duration = mel_len // char_len
        remainder = mel_len % char_len

        durations = np.full(char_len, base_duration, dtype=np.int32)
        # Distribute remainder to first few characters
        durations[:remainder] += 1

        return durations

    @staticmethod
    def normalize_audio(audio: np.ndarray, target_db: float = -20.0) -> np.ndarray:
        """Normalize audio to target dB level."""
        rms = np.sqrt(np.mean(audio ** 2))
        if rms > 0:
            target_rms = 10 ** (target_db / 20)
            audio = audio * (target_rms / rms)
        return np.clip(audio, -1.0, 1.0)
