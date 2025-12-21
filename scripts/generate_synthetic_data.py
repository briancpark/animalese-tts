"""
Generate synthetic Animalese training data.

This script creates synthetic Animalese-style audio from text by:
1. Mapping each character to a simple vocal sound
2. Adding pitch variation based on character position
3. Controlling timing to create the characteristic fast speech

This can be used to bootstrap training before fine-tuning on real Animalese samples.

Usage:
    python scripts/generate_synthetic_data.py --output-dir data/train --num-samples 1000
"""

import os
import json
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
import soundfile as sf
from typing import List, Tuple


# Character to phoneme mapping (simplified)
CHAR_TO_PHONEME = {
    'a': 'ah', 'b': 'buh', 'c': 'kuh', 'd': 'duh', 'e': 'eh',
    'f': 'fuh', 'g': 'guh', 'h': 'huh', 'i': 'ee', 'j': 'juh',
    'k': 'kuh', 'l': 'luh', 'm': 'muh', 'n': 'nuh', 'o': 'oh',
    'p': 'puh', 'q': 'kuh', 'r': 'ruh', 's': 'suh', 't': 'tuh',
    'u': 'oo', 'v': 'vuh', 'w': 'wuh', 'x': 'eks', 'y': 'yuh',
    'z': 'zuh', ' ': 'pause',
}

# Vowel formant frequencies (F1, F2) for synthesis
VOWEL_FORMANTS = {
    'a': (800, 1200),
    'e': (500, 1800),
    'i': (300, 2200),
    'o': (500, 900),
    'u': (350, 700),
}

# Sample sentences for training data
SAMPLE_TEXTS = [
    "Hello, how are you today?",
    "Welcome to my island!",
    "Would you like to buy something?",
    "The weather is nice today.",
    "I found a fossil!",
    "Let's go fishing together.",
    "My house is full of furniture.",
    "Have you seen Tom Nook?",
    "I need more bells for my loan.",
    "The museum is open now.",
    "Blathers loves bugs and fish.",
    "Isabelle is very helpful.",
    "The flowers are blooming.",
    "I caught a sea bass!",
    "Time to water the plants.",
    "The shop closes at ten.",
    "I like your outfit!",
    "Want to trade items?",
    "The concert starts at eight.",
    "Happy birthday to you!",
]


class AnimaleseSynthesizer:
    """
    Synthesizes Animalese-style audio from text.

    Uses simple additive synthesis to create characteristic
    vocal sounds with pitch variation.
    """

    def __init__(
        self,
        sample_rate: int = 22050,
        char_duration_ms: int = 70,
        base_freq: float = 200.0,
        pitch_range: float = 12.0,  # semitones
    ):
        self.sample_rate = sample_rate
        self.char_duration = int(sample_rate * char_duration_ms / 1000)
        self.base_freq = base_freq
        self.pitch_range = pitch_range

    def text_to_audio(self, text: str) -> Tuple[np.ndarray, List[int]]:
        """
        Convert text to Animalese audio.

        Args:
            text: Input text

        Returns:
            Tuple of (audio waveform, list of character durations in samples)
        """
        audio_segments = []
        durations = []

        text = text.lower()

        for i, char in enumerate(text):
            # Get pitch for this character
            pitch = self._get_pitch_for_char(char, i, len(text))

            # Generate audio for character
            segment = self._synthesize_char(char, pitch)
            audio_segments.append(segment)
            durations.append(len(segment))

        if audio_segments:
            audio = np.concatenate(audio_segments)
        else:
            audio = np.zeros(self.char_duration)

        # Normalize
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val * 0.9

        return audio.astype(np.float32), durations

    def _get_pitch_for_char(self, char: str, pos: int, total_len: int) -> float:
        """
        Determine pitch for a character based on its properties and position.
        """
        # Base pitch variation from character
        char_offset = (ord(char) % 12) - 6  # -6 to +5 semitones

        # Add position-based variation (slight rise at end of sentences)
        pos_factor = pos / max(total_len, 1)
        pos_offset = pos_factor * 2 - 1  # -1 to +1

        # Punctuation affects pitch
        if char in '.!':
            return self.base_freq * 0.7  # Lower for sentence end
        elif char == '?':
            return self.base_freq * 1.3  # Higher for questions
        elif char == ' ':
            return self.base_freq

        # Calculate final pitch
        semitone_offset = char_offset * 0.5 + pos_offset * 2
        pitch = self.base_freq * (2 ** (semitone_offset / 12))

        return pitch

    def _synthesize_char(self, char: str, pitch: float) -> np.ndarray:
        """
        Synthesize audio for a single character.
        """
        duration = self.char_duration

        # Handle spaces and punctuation
        if char == ' ':
            # Short pause
            return np.zeros(duration // 2)
        elif char in '.,!?':
            # Longer pause after punctuation
            return np.zeros(duration)

        # Time array
        t = np.linspace(0, duration / self.sample_rate, duration)

        # Determine vowel character for formants
        vowel = self._get_vowel_for_char(char)
        f1, f2 = VOWEL_FORMANTS.get(vowel, (500, 1500))

        # Generate harmonic series (vocal-like sound)
        audio = np.zeros(duration)

        # Fundamental and harmonics
        for harmonic in range(1, 8):
            freq = pitch * harmonic
            amplitude = 1.0 / harmonic  # Decreasing amplitude for higher harmonics

            # Add formant emphasis
            formant_amp = self._formant_response(freq, f1, f2)
            amplitude *= formant_amp

            audio += amplitude * np.sin(2 * np.pi * freq * t)

        # Add some noise for consonants
        if char not in 'aeiou':
            noise = np.random.randn(duration) * 0.1
            noise *= np.exp(-t * 30)  # Quick decay
            audio += noise

        # Apply envelope
        envelope = self._create_envelope(duration)
        audio *= envelope

        return audio

    def _get_vowel_for_char(self, char: str) -> str:
        """Get the vowel sound associated with a character."""
        if char in 'aeiou':
            return char
        # For consonants, use the following vowel pattern
        vowel_pattern = {'b': 'u', 'c': 'a', 'd': 'u', 'f': 'u', 'g': 'u',
                        'h': 'a', 'j': 'a', 'k': 'a', 'l': 'u', 'm': 'u',
                        'n': 'u', 'p': 'u', 'q': 'u', 'r': 'u', 's': 'u',
                        't': 'u', 'v': 'u', 'w': 'u', 'x': 'e', 'y': 'i',
                        'z': 'u'}
        return vowel_pattern.get(char, 'a')

    def _formant_response(self, freq: float, f1: float, f2: float) -> float:
        """Calculate formant filter response."""
        # Simple Gaussian formant approximation
        bandwidth = 100
        resp1 = np.exp(-((freq - f1) ** 2) / (2 * bandwidth ** 2))
        resp2 = np.exp(-((freq - f2) ** 2) / (2 * bandwidth ** 2))
        return 0.3 + 0.7 * max(resp1, resp2)

    def _create_envelope(self, duration: int) -> np.ndarray:
        """Create an ADSR-like envelope."""
        attack = int(duration * 0.1)
        decay = int(duration * 0.2)
        sustain_level = 0.7
        release = int(duration * 0.3)

        envelope = np.ones(duration)

        # Attack
        envelope[:attack] = np.linspace(0, 1, attack)

        # Decay
        envelope[attack:attack + decay] = np.linspace(1, sustain_level, decay)

        # Release
        release_start = duration - release
        envelope[release_start:] = np.linspace(sustain_level, 0, release)

        return envelope


def generate_more_texts(num_samples: int) -> List[str]:
    """Generate additional training texts."""
    texts = SAMPLE_TEXTS.copy()

    # Word lists for generating random sentences
    nouns = ["fish", "bug", "flower", "tree", "house", "shop", "museum", "island",
             "friend", "neighbor", "fossil", "star", "cloud", "rain", "sun"]
    verbs = ["found", "caught", "planted", "built", "bought", "sold", "saw", "met",
             "visited", "cleaned", "decorated", "watered", "dug", "shook"]
    adjectives = ["big", "small", "pretty", "rare", "common", "lovely", "nice",
                  "amazing", "wonderful", "cute", "shiny", "colorful"]

    while len(texts) < num_samples:
        template = np.random.choice([
            f"I {np.random.choice(verbs)} a {np.random.choice(adjectives)} {np.random.choice(nouns)}!",
            f"The {np.random.choice(nouns)} is {np.random.choice(adjectives)}.",
            f"Would you like to see my {np.random.choice(nouns)}?",
            f"I love {np.random.choice(adjectives)} {np.random.choice(nouns)}s!",
            f"Have you {np.random.choice(verbs)} the {np.random.choice(nouns)}?",
        ])
        texts.append(template)

    return texts[:num_samples]


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic Animalese data")
    parser.add_argument(
        "--output-dir", type=str, default="data/train",
        help="Output directory for generated data"
    )
    parser.add_argument(
        "--num-samples", type=int, default=1000,
        help="Number of samples to generate"
    )
    parser.add_argument(
        "--sample-rate", type=int, default=22050,
        help="Audio sample rate"
    )
    parser.add_argument(
        "--char-duration", type=int, default=70,
        help="Duration per character in milliseconds"
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    audio_dir = output_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    # Initialize synthesizer
    synthesizer = AnimaleseSynthesizer(
        sample_rate=args.sample_rate,
        char_duration_ms=args.char_duration,
    )

    # Generate texts
    texts = generate_more_texts(args.num_samples)

    # Generate audio
    metadata = []
    for i, text in enumerate(tqdm(texts, desc="Generating samples")):
        audio, durations = synthesizer.text_to_audio(text)

        # Save audio
        audio_filename = f"sample_{i:05d}.wav"
        audio_path = audio_dir / audio_filename
        sf.write(str(audio_path), audio, args.sample_rate)

        # Save text
        text_path = audio_dir / f"sample_{i:05d}.txt"
        with open(text_path, "w") as f:
            f.write(text)

        metadata.append({
            "id": f"sample_{i:05d}",
            "text": text,
            "audio": f"audio/{audio_filename}",
        })

    # Save metadata
    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nGenerated {len(metadata)} samples in {output_dir}")
    print(f"Metadata saved to {metadata_path}")


if __name__ == "__main__":
    main()
