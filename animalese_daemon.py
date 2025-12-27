#!/usr/bin/env python3
"""
Animalese TTS Daemon - Real-time streaming speech
Listens on a Unix socket or named pipe and speaks text as it arrives.

Usage:
    # Start daemon
    python animalese_daemon.py

    # Send text to daemon (in another terminal)
    echo "Hello!" | nc -U /tmp/animalese.sock
    echo "I can speak in real-time!" | nc -U /tmp/animalese.sock
"""

import socket
import os
import sys
import threading
import queue
from animalese_clean import CleanAnimalese
import sounddevice as sd
import numpy as np


class AnimaleseDaemon:
    def __init__(self, socket_path="/tmp/animalese.sock", voice_type="normal", speed=1.0):
        self.socket_path = socket_path
        self.voice_type = voice_type
        self.speed = speed
        self.synthesizer = CleanAnimalese(
            char_duration_ms=int(55 / speed),
            voice_type=voice_type
        )
        self.audio_queue = queue.Queue()
        self.running = True

    def audio_player_thread(self):
        """Thread that plays audio from the queue."""
        print("Audio player thread started", file=sys.stderr)
        while self.running:
            try:
                audio = self.audio_queue.get(timeout=1)
                if audio is not None:
                    sd.play(audio, samplerate=self.synthesizer.sample_rate)
                    sd.wait()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Audio playback error: {e}", file=sys.stderr)

    def synthesize_and_queue(self, text):
        """Synthesize text and add to playback queue."""
        if not text.strip():
            return

        try:
            audio = self.synthesizer.synthesize(text)
            self.audio_queue.put(audio)
            print(f"Queued: {text[:50]}...", file=sys.stderr)
        except Exception as e:
            print(f"Synthesis error: {e}", file=sys.stderr)

    def start(self):
        """Start the daemon."""
        # Remove existing socket
        if os.path.exists(self.socket_path):
            os.remove(self.socket_path)

        # Start audio player thread
        player = threading.Thread(target=self.audio_player_thread, daemon=True)
        player.start()

        # Create Unix socket
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.bind(self.socket_path)
        sock.listen(5)

        print(f"Animalese daemon listening on {self.socket_path}", file=sys.stderr)
        print(f"Voice: {self.voice_type}, Speed: {self.speed}x", file=sys.stderr)
        print("Send text with: echo 'text' | nc -U /tmp/animalese.sock", file=sys.stderr)

        try:
            while self.running:
                try:
                    conn, _ = sock.accept()
                    data = conn.recv(4096).decode('utf-8')
                    if data:
                        self.synthesize_and_queue(data)
                    conn.close()
                except KeyboardInterrupt:
                    break
                except Exception as e:
                    print(f"Connection error: {e}", file=sys.stderr)
        finally:
            self.running = False
            sock.close()
            os.remove(self.socket_path)
            print("Daemon stopped", file=sys.stderr)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Animalese TTS Daemon")
    parser.add_argument('--socket', default='/tmp/animalese.sock',
                       help='Unix socket path')
    parser.add_argument('--voice', choices=['high', 'normal', 'low'],
                       default='normal', help='Voice type')
    parser.add_argument('--speed', type=float, default=1.0,
                       help='Speech speed')

    args = parser.parse_args()

    daemon = AnimaleseDaemon(
        socket_path=args.socket,
        voice_type=args.voice,
        speed=args.speed
    )
    daemon.start()


if __name__ == "__main__":
    main()
