"""
Clean Animalese Generator - No static/noise

Pure sine-based synthesis for crystal clear Animalese sounds.
"""

import numpy as np
import soundfile as sf
from scipy import signal
import argparse


class CleanAnimalese:
    """
    Clean Animalese synthesizer with no background noise.
    Uses pure sine waves shaped by formant filters.
    """

    def __init__(
        self,
        sample_rate: int = 44100,
        char_duration_ms: int = 55,
        base_pitch: float = 240.0,
        voice_type: str = "normal",
    ):
        self.sample_rate = sample_rate
        self.char_duration = int(sample_rate * char_duration_ms / 1000)

        # Voice presets
        if voice_type == "high":
            self.base_pitch = base_pitch * 1.4
            self.formant_shift = 1.2
        elif voice_type == "low":
            self.base_pitch = base_pitch * 0.7
            self.formant_shift = 0.85
        else:
            self.base_pitch = base_pitch
            self.formant_shift = 1.0

        # Clean vowel formants
        self.vowels = {
            'a': [750, 1200, 2600],
            'e': [450, 1900, 2700],
            'i': [300, 2200, 3000],
            'o': [500, 850, 2500],
            'u': [350, 750, 2400],
        }

        # Map consonants to vowels
        self.consonant_to_vowel = {
            'b': 'a', 'c': 'e', 'd': 'o', 'f': 'u', 'g': 'a',
            'h': 'a', 'j': 'i', 'k': 'a', 'l': 'u', 'm': 'u',
            'n': 'u', 'p': 'u', 'q': 'u', 'r': 'u', 's': 'u',
            't': 'o', 'v': 'i', 'w': 'u', 'x': 'e', 'y': 'i', 'z': 'u',
        }

    def _generate_harmonic_tone(self, duration: int, pitch: float, formants: list) -> np.ndarray:
        """Generate clean harmonic tone shaped by formants."""
        t = np.arange(duration) / self.sample_rate
        wave = np.zeros(duration)

        # Generate harmonics
        for h in range(1, 15):
            freq = pitch * h
            if freq > self.sample_rate / 2:
                break

            # Calculate amplitude based on formant proximity
            amp = 0.0
            for f in formants:
                f_shifted = f * self.formant_shift
                # Gaussian response around formant
                distance = abs(freq - f_shifted) / f_shifted
                amp += np.exp(-distance * distance * 8)

            # Natural harmonic rolloff
            amp *= 1.0 / (h ** 0.8)

            # Add harmonic
            wave += amp * np.sin(2 * np.pi * freq * t)

        return wave

    def _create_smooth_envelope(self, duration: int, is_vowel: bool = True) -> np.ndarray:
        """Create smooth attack-sustain-release envelope."""
        env = np.ones(duration)

        # Attack (smooth sine curve)
        attack_len = int(duration * 0.08)
        if attack_len > 0:
            env[:attack_len] = (1 - np.cos(np.linspace(0, np.pi, attack_len))) / 2

        # Release (smooth sine curve)
        release_len = int(duration * 0.20)
        if release_len > 0:
            env[-release_len:] = (1 + np.cos(np.linspace(0, np.pi, release_len))) / 2

        # Vowels sustain more
        if is_vowel:
            sustain_level = 0.9
        else:
            sustain_level = 0.75

        # Apply sustain decay
        mid_start = attack_len
        mid_end = duration - release_len
        if mid_end > mid_start:
            env[mid_start:mid_end] *= np.linspace(1.0, sustain_level, mid_end - mid_start)

        return env

    def _get_pitch(self, char: str, pos: int, total: int) -> float:
        """Calculate pitch with melodic variation."""
        # Character-based variation
        if char.isalpha():
            char_val = ord(char.lower()) - ord('a')
            semitones = ((char_val % 5) - 2) * 0.8  # Subtle variation
        else:
            semitones = 0

        # Intonation curve
        progress = pos / max(total - 1, 1)
        if progress < 0.6:
            intonation = progress * 1.5
        else:
            intonation = 0.9 - (progress - 0.6) * 2

        total_semitones = semitones + intonation
        return self.base_pitch * (2 ** (total_semitones / 12))

    def _synthesize_char(self, char: str, pitch: float) -> np.ndarray:
        """Synthesize a single character."""
        char_lower = char.lower()

        # Silence for spaces/punctuation
        if char == ' ':
            return np.zeros(int(self.char_duration * 0.35))
        elif char in '.!':
            return np.zeros(int(self.char_duration * 0.5))
        elif char == ',':
            return np.zeros(int(self.char_duration * 0.25))
        elif char == '?':
            return np.zeros(int(self.char_duration * 0.4))
        elif not char.isalpha():
            return np.zeros(int(self.char_duration * 0.15))

        # Get formants
        if char_lower in self.vowels:
            formants = self.vowels[char_lower]
            duration = int(self.char_duration * 1.1)
            is_vowel = True
        else:
            vowel = self.consonant_to_vowel.get(char_lower, 'a')
            formants = self.vowels[vowel]
            duration = int(self.char_duration * 0.85)
            is_vowel = False

        # Generate tone
        wave = self._generate_harmonic_tone(duration, pitch, formants)

        # Apply envelope
        envelope = self._create_smooth_envelope(duration, is_vowel)
        wave *= envelope

        # Normalize
        max_val = np.max(np.abs(wave))
        if max_val > 0:
            wave = wave / max_val

        return wave

    def synthesize(self, text: str) -> np.ndarray:
        """Convert text to clean Animalese audio."""
        segments = []

        for i, char in enumerate(text):
            pitch = self._get_pitch(char, i, len(text))
            segment = self._synthesize_char(char, pitch)
            segments.append(segment)

        if not segments:
            return np.zeros(1000, dtype=np.float32)

        audio = np.concatenate(segments)

        # Soft limiting (no harsh clipping)
        audio = np.tanh(audio * 1.2) * 0.85

        # Gentle low-pass to smooth any remaining harshness
        nyq = self.sample_rate / 2
        cutoff = min(10000, nyq - 100)
        sos = signal.butter(2, cutoff, btype='low', fs=self.sample_rate, output='sos')
        audio = signal.sosfilt(sos, audio)

        return audio.astype(np.float32)

    def save(self, text: str, output_path: str):
        """Generate and save audio."""
        audio = self.synthesize(text)
        sf.write(output_path, audio, self.sample_rate)
        print(f"✓ {output_path} ({len(audio)/self.sample_rate:.2f}s)")


def main():
    parser = argparse.ArgumentParser(description="Clean Animalese Generator")
    parser.add_argument("--text", type=str, required=True)
    parser.add_argument("--output", type=str, default="animalese_clean.wav")
    parser.add_argument("--voice", choices=["high", "normal", "low"], default="normal")
    parser.add_argument("--pitch", type=float, default=240.0)
    parser.add_argument("--speed", type=int, default=55, help="Duration per char in ms")
    args = parser.parse_args()

    synth = CleanAnimalese(
        base_pitch=args.pitch,
        voice_type=args.voice,
        char_duration_ms=args.speed,
    )
    synth.save(args.text, args.output)


if __name__ == "__main__":
    main()
