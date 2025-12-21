"""
Improved Animalese Generator v2

This version uses:
1. Better vowel synthesis with proper formants
2. Softer, more "cute" sound character
3. Proper pitch contours for natural speech
4. Randomized micro-variations for natural feel
"""

import numpy as np
import soundfile as sf
from scipy import signal
from scipy.io import wavfile
import argparse


class AnimaleseV2:
    """
    Improved Animalese synthesizer that sounds more like the real game.
    """

    def __init__(
        self,
        sample_rate: int = 44100,  # Higher sample rate for better quality
        char_duration_ms: int = 55,  # Shorter = faster talking
        base_pitch: float = 240.0,
        voice_type: str = "normal",  # normal, high, low
    ):
        self.sample_rate = sample_rate
        self.char_duration = int(sample_rate * char_duration_ms / 1000)
        self.base_pitch = base_pitch

        # Voice type presets
        voice_presets = {
            "high": {"pitch_mult": 1.4, "formant_shift": 1.2},  # Cute/child
            "normal": {"pitch_mult": 1.0, "formant_shift": 1.0},
            "low": {"pitch_mult": 0.7, "formant_shift": 0.85},  # Deep voice
        }
        preset = voice_presets.get(voice_type, voice_presets["normal"])
        self.base_pitch *= preset["pitch_mult"]
        self.formant_shift = preset["formant_shift"]

        # Phoneme definitions with formants (F1, F2, F3) and characteristics
        self.phonemes = {
            # Vowels - clear, sustained sounds
            'a': {'formants': [700, 1200, 2600], 'duration': 1.1, 'voiced': True},
            'e': {'formants': [400, 2100, 2800], 'duration': 1.0, 'voiced': True},
            'i': {'formants': [280, 2300, 3100], 'duration': 0.9, 'voiced': True},
            'o': {'formants': [450, 800, 2500], 'duration': 1.1, 'voiced': True},
            'u': {'formants': [320, 700, 2500], 'duration': 1.0, 'voiced': True},
            # Consonants mapped to vowel-ish sounds (Japanese style)
            'b': {'formants': [600, 1000, 2400], 'duration': 0.7, 'voiced': True, 'attack': 0.02},
            'c': {'formants': [400, 1800, 2700], 'duration': 0.8, 'voiced': True},
            'd': {'formants': [500, 1100, 2500], 'duration': 0.7, 'voiced': True, 'attack': 0.02},
            'f': {'formants': [350, 900, 2500], 'duration': 0.7, 'voiced': False},
            'g': {'formants': [600, 1100, 2500], 'duration': 0.7, 'voiced': True, 'attack': 0.02},
            'h': {'formants': [500, 1200, 2600], 'duration': 0.6, 'voiced': False},
            'j': {'formants': [300, 2100, 3000], 'duration': 0.8, 'voiced': True},
            'k': {'formants': [500, 1500, 2600], 'duration': 0.6, 'voiced': False, 'attack': 0.01},
            'l': {'formants': [350, 1100, 2500], 'duration': 0.8, 'voiced': True},
            'm': {'formants': [300, 1000, 2400], 'duration': 0.9, 'voiced': True, 'nasal': True},
            'n': {'formants': [300, 1200, 2500], 'duration': 0.9, 'voiced': True, 'nasal': True},
            'p': {'formants': [400, 1000, 2500], 'duration': 0.6, 'voiced': False, 'attack': 0.01},
            'q': {'formants': [450, 1100, 2500], 'duration': 0.7, 'voiced': False},
            'r': {'formants': [350, 1300, 2500], 'duration': 0.8, 'voiced': True},
            's': {'formants': [400, 1800, 2800], 'duration': 0.6, 'voiced': False, 'fricative': True},
            't': {'formants': [400, 1400, 2600], 'duration': 0.6, 'voiced': False, 'attack': 0.01},
            'v': {'formants': [350, 1100, 2500], 'duration': 0.7, 'voiced': True},
            'w': {'formants': [320, 700, 2400], 'duration': 0.8, 'voiced': True},
            'x': {'formants': [400, 1600, 2700], 'duration': 0.7, 'voiced': False},
            'y': {'formants': [290, 2200, 3000], 'duration': 0.8, 'voiced': True},
            'z': {'formants': [350, 1500, 2700], 'duration': 0.7, 'voiced': True, 'fricative': True},
        }

    def _generate_glottal_pulse(self, duration: int, pitch: float) -> np.ndarray:
        """Generate a more natural glottal pulse waveform."""
        t = np.arange(duration) / self.sample_rate

        # LF model approximation for glottal pulses
        wave = np.zeros(duration)
        period_samples = int(self.sample_rate / pitch)

        for start in range(0, duration - period_samples, period_samples):
            # Create one glottal pulse
            pulse_len = min(period_samples, duration - start)
            pulse_t = np.linspace(0, 1, pulse_len)

            # Rising then sharply falling (LF model simplified)
            pulse = np.sin(np.pi * pulse_t) ** 2
            pulse *= np.exp(-3 * pulse_t)  # Decay

            wave[start:start + pulse_len] += pulse

        # Add harmonics for richness
        for h in range(2, 8):
            harmonic = np.sin(2 * np.pi * pitch * h * t) / (h ** 1.5)
            wave += harmonic * 0.3

        return wave

    def _apply_formants(self, source: np.ndarray, formants: list) -> np.ndarray:
        """Apply formant filtering to source signal."""
        output = np.zeros_like(source)

        for i, f in enumerate(formants):
            # Shift formants based on voice type
            f_shifted = f * self.formant_shift

            # Bandwidth varies by formant
            bandwidth = f_shifted * (0.08 + i * 0.02)

            # Create bandpass filter
            low = max(f_shifted - bandwidth, 50)
            high = min(f_shifted + bandwidth, self.sample_rate / 2 - 100)

            if low < high:
                try:
                    sos = signal.butter(2, [low, high], btype='band',
                                       fs=self.sample_rate, output='sos')
                    filtered = signal.sosfilt(sos, source)

                    # Weight formants (F1 and F2 are most important)
                    weight = 1.0 / (1 + i * 0.3)
                    output += filtered * weight
                except:
                    pass

        return output

    def _create_envelope(self, duration: int, attack: float = 0.05) -> np.ndarray:
        """Create smooth amplitude envelope."""
        env = np.ones(duration)

        # Smooth attack
        attack_samples = int(duration * attack)
        if attack_samples > 0:
            env[:attack_samples] = np.sin(np.linspace(0, np.pi/2, attack_samples)) ** 2

        # Smooth release
        release_samples = int(duration * 0.15)
        if release_samples > 0:
            env[-release_samples:] = np.cos(np.linspace(0, np.pi/2, release_samples)) ** 2

        # Slight decay in middle
        mid_start = attack_samples
        mid_end = duration - release_samples
        if mid_end > mid_start:
            decay = np.linspace(1.0, 0.85, mid_end - mid_start)
            env[mid_start:mid_end] *= decay

        return env

    def _get_pitch(self, char: str, pos: int, total: int, prev_pitch: float) -> float:
        """Calculate pitch with natural variation."""
        # Base pitch variation from character
        if char.isalpha():
            char_val = ord(char.lower()) - ord('a')
            semitones = (char_val % 7) - 3  # -3 to +3 semitones
        else:
            semitones = 0

        # Sentence intonation
        progress = pos / max(total - 1, 1)
        if progress < 0.3:
            intonation = progress * 2  # Rise
        elif progress < 0.7:
            intonation = 0.6  # Sustain
        else:
            intonation = 0.6 - (progress - 0.7) * 2  # Fall

        # Small random variation for natural feel
        random_var = (np.random.random() - 0.5) * 0.5

        # Combine
        total_semitones = semitones * 0.5 + intonation * 2 + random_var

        # Smooth transition from previous pitch
        target_pitch = self.base_pitch * (2 ** (total_semitones / 12))
        if prev_pitch > 0:
            pitch = prev_pitch * 0.3 + target_pitch * 0.7
        else:
            pitch = target_pitch

        return pitch

    def _synthesize_phoneme(self, char: str, pitch: float) -> np.ndarray:
        """Synthesize a single phoneme."""
        char_lower = char.lower()

        # Handle non-letters
        if char == ' ':
            return np.zeros(int(self.char_duration * 0.4))
        elif char in '.!':
            return np.zeros(int(self.char_duration * 0.6))
        elif char == ',':
            return np.zeros(int(self.char_duration * 0.3))
        elif char == '?':
            # Questions get a little chirp
            return np.zeros(int(self.char_duration * 0.4))
        elif not char.isalpha():
            return np.zeros(int(self.char_duration * 0.2))

        # Get phoneme info
        phoneme = self.phonemes.get(char_lower, self.phonemes['a'])
        formants = phoneme['formants']
        duration_mult = phoneme.get('duration', 1.0)
        voiced = phoneme.get('voiced', True)
        attack = phoneme.get('attack', 0.05)

        duration = int(self.char_duration * duration_mult)

        # Generate source
        if voiced:
            source = self._generate_glottal_pulse(duration, pitch)
        else:
            # Noise for unvoiced
            source = np.random.randn(duration) * 0.5
            # Add slight tonal component
            t = np.arange(duration) / self.sample_rate
            source += np.sin(2 * np.pi * pitch * 0.5 * t) * 0.2

        # Apply formants
        wave = self._apply_formants(source, formants)

        # Apply envelope
        envelope = self._create_envelope(duration, attack)
        wave *= envelope

        # Add slight breathiness
        if voiced:
            breath = np.random.randn(duration) * 0.03
            wave += breath * envelope

        # Normalize
        max_val = np.max(np.abs(wave))
        if max_val > 0:
            wave = wave / max_val

        return wave

    def synthesize(self, text: str) -> np.ndarray:
        """Convert text to Animalese audio."""
        segments = []
        prev_pitch = 0

        for i, char in enumerate(text):
            pitch = self._get_pitch(char, i, len(text), prev_pitch)
            segment = self._synthesize_phoneme(char, pitch)
            segments.append(segment)
            prev_pitch = pitch

            # Tiny gap for separation
            gap_len = int(self.char_duration * 0.03)
            segments.append(np.zeros(gap_len))

        if not segments:
            return np.zeros(1000)

        audio = np.concatenate(segments)

        # Final processing
        # Slight compression for consistent volume
        audio = np.tanh(audio * 1.5) * 0.7

        # Gentle low-pass for warmth
        try:
            sos = signal.butter(2, 8000, btype='low', fs=self.sample_rate, output='sos')
            audio = signal.sosfilt(sos, audio)
        except:
            pass

        return audio.astype(np.float32)

    def save(self, text: str, output_path: str):
        """Generate and save audio."""
        audio = self.synthesize(text)
        sf.write(output_path, audio, self.sample_rate)
        duration = len(audio) / self.sample_rate
        print(f"✓ Saved: {output_path} ({duration:.2f}s)")


def main():
    parser = argparse.ArgumentParser(description="Generate Animalese v2")
    parser.add_argument("--text", type=str, required=True)
    parser.add_argument("--output", type=str, default="animalese_v2.wav")
    parser.add_argument("--voice", choices=["high", "normal", "low"], default="normal")
    parser.add_argument("--pitch", type=float, default=240.0)
    parser.add_argument("--speed", type=int, default=55, help="Char duration in ms (lower=faster)")
    args = parser.parse_args()

    synth = AnimaleseV2(
        base_pitch=args.pitch,
        voice_type=args.voice,
        char_duration_ms=args.speed,
    )
    synth.save(args.text, args.output)


if __name__ == "__main__":
    main()
