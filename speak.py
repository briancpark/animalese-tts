#!/usr/bin/env python3
"""
Real-time Animalese TTS CLI tool
Usage:
    echo "Hello!" | python speak.py
    python speak.py "Hello world!"
    python speak.py --stdin < file.txt
"""

import sys
import argparse
import tempfile
import os
import subprocess
from animalese_clean import CleanAnimalese
import numpy as np
import soundfile as sf

# Try to import sounddevice for direct playback
try:
    import sounddevice as sd
    HAS_SOUNDDEVICE = True
except (ImportError, OSError):
    HAS_SOUNDDEVICE = False


def speak_text(text, voice_type="normal", speed=1.0, use_file=False):
    """Convert text to Animalese and play it immediately."""
    synthesizer = CleanAnimalese(
        char_duration_ms=int(55 / speed),
        voice_type=voice_type
    )

    audio = synthesizer.synthesize(text)

    # Try direct playback first, fall back to file-based playback
    if HAS_SOUNDDEVICE and not use_file:
        try:
            sd.play(audio, samplerate=synthesizer.sample_rate)
            sd.wait()
            return
        except Exception as e:
            print(f"Direct playback failed: {e}, falling back to file playback", file=sys.stderr)

    # File-based playback (works on WSL2)
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
        tmp_path = tmp.name

    try:
        sf.write(tmp_path, audio, synthesizer.sample_rate)

        # Try various players (works on WSL2 with Windows)
        players = [
            ['powershell.exe', '-c', f'(New-Object Media.SoundPlayer "{tmp_path}").PlaySync()'],  # WSL2 -> Windows
            ['aplay', tmp_path],  # Linux ALSA
            ['paplay', tmp_path],  # Linux PulseAudio
            ['ffplay', '-nodisp', '-autoexit', tmp_path],  # ffmpeg
        ]

        for player_cmd in players:
            try:
                subprocess.run(player_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                break
            except (subprocess.CalledProcessError, FileNotFoundError):
                continue
        else:
            print(f"No audio player found. Audio saved to: {tmp_path}", file=sys.stderr)
            print("Install 'aplay' (alsa-utils) or enable WSL2 audio", file=sys.stderr)
    finally:
        # Clean up temp file after a short delay
        try:
            os.remove(tmp_path)
        except:
            pass


def main():
    parser = argparse.ArgumentParser(description="Speak text as Animalese")
    parser.add_argument('text', nargs='*', help='Text to speak')
    parser.add_argument('--stdin', action='store_true', help='Read from stdin')
    parser.add_argument('--voice', choices=['high', 'normal', 'low'],
                       default='normal', help='Voice type')
    parser.add_argument('--speed', type=float, default=1.0,
                       help='Speech speed (0.5 = slow, 2.0 = fast)')

    args = parser.parse_args()

    # Get text from arguments or stdin
    if args.stdin or not sys.stdin.isatty():
        text = sys.stdin.read().strip()
    elif args.text:
        text = ' '.join(args.text)
    else:
        parser.print_help()
        sys.exit(1)

    if text:
        speak_text(text, voice_type=args.voice, speed=args.speed)


if __name__ == "__main__":
    main()
