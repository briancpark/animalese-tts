"""
Dataset and data loading utilities for Animalese TTS.

Handles loading text-audio pairs and preprocessing for training.
"""

import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Tuple
from pathlib import Path

from .audio import AudioProcessor


class AnimaleseDataset(Dataset):
    """
    Dataset for Animalese TTS training.

    Expected data format:
    - data_dir/
      - metadata.json  (list of {"text": "...", "audio": "audio_file.wav"})
      - audio/
        - audio_file1.wav
        - audio_file2.wav
        - ...

    Or for pre-processed data:
    - data_dir/
      - metadata.json
      - mel/
        - file1.npy
      - pitch/
        - file1.npy
      - energy/
        - file1.npy
      - duration/
        - file1.npy
    """

    def __init__(
        self,
        data_dir: str,
        audio_processor: AudioProcessor,
        max_text_len: int = 200,
        max_mel_len: int = 1000,
        precomputed: bool = False,
    ):
        """
        Args:
            data_dir: Path to dataset directory
            audio_processor: AudioProcessor instance
            max_text_len: Maximum text length
            max_mel_len: Maximum mel spectrogram length
            precomputed: If True, load pre-computed features
        """
        self.data_dir = Path(data_dir)
        self.audio_processor = audio_processor
        self.max_text_len = max_text_len
        self.max_mel_len = max_mel_len
        self.precomputed = precomputed

        # Load metadata
        metadata_path = self.data_dir / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, "r") as f:
                self.metadata = json.load(f)
        else:
            # Try to auto-discover audio files
            self.metadata = self._auto_discover()

        print(f"Loaded {len(self.metadata)} samples from {data_dir}")

    def _auto_discover(self) -> List[Dict]:
        """Auto-discover audio files and create metadata."""
        audio_dir = self.data_dir / "audio"
        if not audio_dir.exists():
            audio_dir = self.data_dir

        metadata = []
        for audio_file in sorted(audio_dir.glob("*.wav")):
            # Try to find corresponding text file
            text_file = audio_file.with_suffix(".txt")
            if text_file.exists():
                with open(text_file, "r") as f:
                    text = f.read().strip()
            else:
                # Use filename as text (for synthetic data)
                text = audio_file.stem.replace("_", " ")

            metadata.append({
                "text": text,
                "audio": str(audio_file.relative_to(self.data_dir)),
                "id": audio_file.stem,
            })

        return metadata

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.metadata[idx]
        text = item["text"]
        item_id = item.get("id", str(idx))

        if self.precomputed:
            return self._load_precomputed(item_id, text)
        else:
            return self._process_raw(item, text)

    def _load_precomputed(self, item_id: str, text: str) -> Dict[str, torch.Tensor]:
        """Load pre-computed features."""
        mel = np.load(self.data_dir / "mel" / f"{item_id}.npy")
        pitch = np.load(self.data_dir / "pitch" / f"{item_id}.npy")
        energy = np.load(self.data_dir / "energy" / f"{item_id}.npy")
        duration = np.load(self.data_dir / "duration" / f"{item_id}.npy")

        return self._create_sample(text, mel, pitch, energy, duration)

    def _process_raw(self, item: Dict, text: str) -> Dict[str, torch.Tensor]:
        """Process raw audio file."""
        audio_path = self.data_dir / item["audio"]
        audio = self.audio_processor.load_audio(str(audio_path))

        # Extract features
        features = self.audio_processor.process_audio(audio)
        mel = features["mel"]
        pitch = features["pitch"]
        energy = features["energy"]

        # Compute duration (uniform distribution)
        duration = self.audio_processor.compute_duration(len(text), mel.shape[1])

        return self._create_sample(text, mel, pitch, energy, duration, audio)

    def _create_sample(
        self,
        text: str,
        mel: np.ndarray,
        pitch: np.ndarray,
        energy: np.ndarray,
        duration: np.ndarray,
        audio: Optional[np.ndarray] = None,
    ) -> Dict[str, torch.Tensor]:
        """Create a sample dictionary."""
        # Convert text to character indices
        chars = [ord(c) for c in text[:self.max_text_len]]

        # Truncate mel if necessary
        mel_len = min(mel.shape[1], self.max_mel_len)
        mel = mel[:, :mel_len]
        pitch = pitch[:mel_len]
        energy = energy[:mel_len]

        # Adjust duration if necessary
        char_len = len(chars)
        if len(duration) != char_len:
            duration = self.audio_processor.compute_duration(char_len, mel_len)

        sample = {
            "chars": torch.tensor(chars, dtype=torch.long),
            "char_len": torch.tensor(char_len, dtype=torch.long),
            "mel": torch.tensor(mel, dtype=torch.float32),
            "mel_len": torch.tensor(mel_len, dtype=torch.long),
            "pitch": torch.tensor(pitch, dtype=torch.float32),
            "energy": torch.tensor(energy, dtype=torch.float32),
            "duration": torch.tensor(duration, dtype=torch.long),
        }

        if audio is not None:
            # Compute expected audio length from mel
            audio_len = mel_len * self.audio_processor.hop_length
            audio = audio[:audio_len]
            if len(audio) < audio_len:
                audio = np.pad(audio, (0, audio_len - len(audio)))
            sample["audio"] = torch.tensor(audio, dtype=torch.float32)

        return sample


class AnimaleseCollator:
    """
    Collates samples into batches with proper padding.
    """

    def __init__(self, n_mels: int = 80):
        self.n_mels = n_mels

    def __call__(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        Collate a list of samples into a batch.

        Args:
            batch: List of sample dictionaries

        Returns:
            Batched dictionary with padded tensors
        """
        # Get max lengths
        max_char_len = max(s["char_len"].item() for s in batch)
        max_mel_len = max(s["mel_len"].item() for s in batch)
        batch_size = len(batch)

        # Initialize tensors
        chars = torch.zeros(batch_size, max_char_len, dtype=torch.long)
        char_lens = torch.zeros(batch_size, dtype=torch.long)
        mels = torch.zeros(batch_size, self.n_mels, max_mel_len, dtype=torch.float32)
        mel_lens = torch.zeros(batch_size, dtype=torch.long)
        pitches = torch.zeros(batch_size, max_mel_len, dtype=torch.float32)
        energies = torch.zeros(batch_size, max_mel_len, dtype=torch.float32)
        durations = torch.zeros(batch_size, max_char_len, dtype=torch.long)

        has_audio = "audio" in batch[0]
        if has_audio:
            hop_length = 256  # Default hop length
            max_audio_len = max_mel_len * hop_length
            audios = torch.zeros(batch_size, max_audio_len, dtype=torch.float32)

        # Fill tensors
        for i, sample in enumerate(batch):
            char_len = sample["char_len"].item()
            mel_len = sample["mel_len"].item()

            chars[i, :char_len] = sample["chars"]
            char_lens[i] = char_len
            mels[i, :, :mel_len] = sample["mel"]
            mel_lens[i] = mel_len
            pitches[i, :mel_len] = sample["pitch"]
            energies[i, :mel_len] = sample["energy"]
            durations[i, :char_len] = sample["duration"]

            if has_audio:
                audio_len = len(sample["audio"])
                audios[i, :audio_len] = sample["audio"]

        result = {
            "chars": chars,
            "char_lens": char_lens,
            "mels": mels,
            "mel_lens": mel_lens,
            "pitches": pitches,
            "energies": energies,
            "durations": durations,
        }

        if has_audio:
            result["audios"] = audios

        return result


def create_dataloader(
    data_dir: str,
    audio_processor: AudioProcessor,
    batch_size: int = 32,
    num_workers: int = 4,
    shuffle: bool = True,
    precomputed: bool = False,
) -> DataLoader:
    """
    Create a DataLoader for training.

    Args:
        data_dir: Path to dataset directory
        audio_processor: AudioProcessor instance
        batch_size: Batch size
        num_workers: Number of data loading workers
        shuffle: Whether to shuffle the data
        precomputed: Whether to use pre-computed features

    Returns:
        DataLoader instance
    """
    dataset = AnimaleseDataset(
        data_dir=data_dir,
        audio_processor=audio_processor,
        precomputed=precomputed,
    )

    collator = AnimaleseCollator(n_mels=audio_processor.n_mels)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collator,
        pin_memory=True,
        drop_last=True,
    )

    return dataloader
