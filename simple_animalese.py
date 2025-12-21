"""
Simple Rule-Based Animalese Generator

This creates authentic-sounding Animalese by:
1. Mapping each character to a short vocal-like sound
2. Varying pitch based on the character
3. Playing sounds in rapid succession

This is how the actual Animal Crossing game does it!
"""

import numpy as np
import soundfile as sf
from scipy import signal
from typing import Optional
import argparse


class SimpleAnimalese:
    """
    Generates Animalese audio using simple synthesis.

    Each character maps to a short "phoneme" sound with:
    - Vowels: longer, more open sounds
    - Consonants: shorter, more percussive sounds
    - Pitch varies based on character value
    """

    def __init__(
        self,
        sample_rate: int = 22050,
        base_duration_ms: int = 60,
        base_pitch: float = 220.0,  # A3
        pitch_variation: float = 0.3,  # 30% variation
        speed: float = 1.0,
    ):
        self.sample_rate = sample_rate
        self.base_duration = int(sample_rate * base_duration_ms / 1000)
        self.base_pitch = base_pitch
        self.pitch_variation = pitch_variation
        self.speed = speed

        # Vowel formant frequencies (F1, F2, F3) for more realistic sounds
        self.vowel_formants = {
            'a': [800, 1200, 2500],
            'e': [400, 2000, 2800],
            'i': [300, 2300, 3000],
            'o': [500, 900, 2500],
            'u': [350, 700, 2400],
        }

        # Map consonants to their "vowel-like" sounds (like Japanese phonetics)
        self.consonant_vowels = {
            'b': 'a', 'c': 'e', 'd': 'o', 'f': 'u', 'g': 'a',
            'h': 'a', 'j': 'i', 'k': 'a', 'l': 'u', 'm': 'u',
            'n': 'u', 'p': 'u', 'q': 'u', 'r': 'u', 's': 'u',
            't': 'o', 'v': 'i', 'w': 'u', 'x': 'e', 'y': 'i', 'z': 'u',
        }

    def _get_pitch_for_char(self, char: str, position: int, total_length: int) -> float:
        """Calculate pitch for a character."""
        # Base variation from character
        char_val = ord(char.lower()) if char.isalpha() else ord(' ')
        char_offset = ((char_val % 12) - 6) / 12.0  # -0.5 to 0.5

        # Slight melodic contour (rise slightly, then fall at end)
        pos_ratio = position / max(total_length, 1)
        if pos_ratio < 0.7:
            contour = pos_ratio * 0.1
        else:
            contour = 0.07 - (pos_ratio - 0.7) * 0.3

        # Question marks go up
        pitch_mult = 1.0 + char_offset * self.pitch_variation + contour

        return self.base_pitch * pitch_mult

    def _generate_formant_wave(
        self,
        duration_samples: int,
        pitch: float,
        formants: list,
        is_consonant: bool = False,
    ) -> np.ndarray:
        """Generate a formant-filtered sound."""
        t = np.arange(duration_samples) / self.sample_rate

        # Generate source signal (glottal pulse train approximation)
        # Use a mix of harmonics for a richer sound
        source = np.zeros(duration_samples)
        for harmonic in range(1, 12):
            amp = 1.0 / (harmonic ** 1.2)  # Harmonic rolloff
            source += amp * np.sin(2 * np.pi * pitch * harmonic * t)

        # Add slight noise for breathiness
        noise = np.random.randn(duration_samples) * 0.05
        source += noise

        # Apply formant filtering
        output = np.zeros(duration_samples)
        for formant_freq in formants:
            # Bandpass filter around each formant
            bandwidth = formant_freq * 0.1
            low = max(formant_freq - bandwidth, 20)
            high = min(formant_freq + bandwidth, self.sample_rate / 2 - 100)

            if low < high:
                b, a = signal.butter(2, [low, high], btype='band', fs=self.sample_rate)
                filtered = signal.filtfilt(b, a, source)
                output += filtered

        # Normalize
        max_val = np.max(np.abs(output))
        if max_val > 0:
            output = output / max_val

        # Consonants get a sharper attack
        if is_consonant:
            attack = int(duration_samples * 0.05)
            output[:attack] *= np.linspace(0, 1, attack) ** 0.5

        return output

    def _create_envelope(self, duration_samples: int, char: str) -> np.ndarray:
        """Create amplitude envelope."""
        envelope = np.ones(duration_samples)

        # Attack
        attack_len = int(duration_samples * 0.08)
        envelope[:attack_len] = np.linspace(0, 1, attack_len)

        # Decay/sustain
        decay_start = int(duration_samples * 0.2)
        decay_end = int(duration_samples * 0.6)
        envelope[decay_start:decay_end] = np.linspace(1, 0.7, decay_end - decay_start)

        # Release
        release_len = int(duration_samples * 0.3)
        release_start = duration_samples - release_len
        envelope[release_start:] = np.linspace(0.7, 0, release_len)

        # Vowels sustain longer
        if char.lower() in 'aeiou':
            envelope[decay_start:release_start] *= 1.1
            np.clip(envelope, 0, 1, out=envelope)

        return envelope

    def _synthesize_char(self, char: str, pitch: float) -> np.ndarray:
        """Synthesize a single character sound."""
        char_lower = char.lower()

        # Handle spaces and punctuation
        if char == ' ':
            return np.zeros(int(self.base_duration * 0.5))
        elif char in '.,!?':
            return np.zeros(int(self.base_duration * 0.8))
        elif not char.isalpha():
            return np.zeros(int(self.base_duration * 0.3))

        # Determine duration (vowels slightly longer)
        if char_lower in 'aeiou':
            duration = int(self.base_duration * 1.2)
            formants = self.vowel_formants[char_lower]
            is_consonant = False
        else:
            duration = self.base_duration
            vowel = self.consonant_vowels.get(char_lower, 'a')
            formants = self.vowel_formants[vowel]
            is_consonant = True

        # Adjust for speed
        duration = int(duration / self.speed)

        # Generate the sound
        wave = self._generate_formant_wave(duration, pitch, formants, is_consonant)

        # Apply envelope
        envelope = self._create_envelope(duration, char)
        wave *= envelope

        return wave

    def synthesize(self, text: str) -> np.ndarray:
        """
        Convert text to Animalese audio.

        Args:
            text: Input text string

        Returns:
            Audio waveform as numpy array
        """
        segments = []
        text_len = len(text)

        for i, char in enumerate(text):
            pitch = self._get_pitch_for_char(char, i, text_len)
            segment = self._synthesize_char(char, pitch)
            segments.append(segment)

            # Small gap between characters (creates the babbling effect)
            gap = np.zeros(int(self.base_duration * 0.05))
            segments.append(gap)

        if segments:
            audio = np.concatenate(segments)
        else:
            audio = np.zeros(1000)

        # Normalize final output
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val * 0.8

        return audio.astype(np.float32)

    def save(self, text: str, output_path: str):
        """Generate and save Animalese audio."""
        audio = self.synthesize(text)
        sf.write(output_path, audio, self.sample_rate)
        print(f"Saved: {output_path} ({len(audio)/self.sample_rate:.2f}s)")


def main():
    parser = argparse.ArgumentParser(description="Generate Animalese audio")
    parser.add_argument("--text", type=str, required=True, help="Text to convert")
    parser.add_argument("--output", type=str, default="animalese_output.wav", help="Output file")
    parser.add_argument("--pitch", type=float, default=220.0, help="Base pitch in Hz")
    parser.add_argument("--speed", type=float, default=1.0, help="Speed multiplier")
    parser.add_argument("--duration", type=int, default=60, help="Base duration per char (ms)")
    args = parser.parse_args()

    synth = SimpleAnimalese(
        base_pitch=args.pitch,
        speed=args.speed,
        base_duration_ms=args.duration,
    )

    synth.save(args.text, args.output)


if __name__ == "__main__":
    main()
