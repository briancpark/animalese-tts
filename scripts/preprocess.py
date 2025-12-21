"""
Preprocess audio data for Animalese TTS training.

This script extracts mel spectrograms, pitch, and energy from audio files
and saves them as numpy arrays for faster training.

Usage:
    python scripts/preprocess.py --input-dir data/raw --output-dir data/processed
"""

import os
import json
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from data.audio import AudioProcessor


def main():
    parser = argparse.ArgumentParser(description="Preprocess audio data")
    parser.add_argument(
        "--input-dir", type=str, required=True,
        help="Input directory containing audio files"
    )
    parser.add_argument(
        "--output-dir", type=str, required=True,
        help="Output directory for preprocessed features"
    )
    parser.add_argument(
        "--sample-rate", type=int, default=22050,
        help="Target sample rate"
    )
    parser.add_argument(
        "--n-mels", type=int, default=80,
        help="Number of mel channels"
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    # Create output directories
    (output_dir / "mel").mkdir(parents=True, exist_ok=True)
    (output_dir / "pitch").mkdir(parents=True, exist_ok=True)
    (output_dir / "energy").mkdir(parents=True, exist_ok=True)
    (output_dir / "duration").mkdir(parents=True, exist_ok=True)

    # Initialize audio processor
    audio_processor = AudioProcessor(
        sample_rate=args.sample_rate,
        n_mels=args.n_mels,
    )

    # Load or create metadata
    metadata_path = input_dir / "metadata.json"
    if metadata_path.exists():
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
    else:
        # Auto-discover
        metadata = []
        audio_dir = input_dir / "audio" if (input_dir / "audio").exists() else input_dir
        for audio_file in sorted(audio_dir.glob("*.wav")):
            text_file = audio_file.with_suffix(".txt")
            if text_file.exists():
                with open(text_file, "r") as f:
                    text = f.read().strip()
            else:
                text = audio_file.stem.replace("_", " ")

            metadata.append({
                "id": audio_file.stem,
                "text": text,
                "audio": str(audio_file.relative_to(input_dir)),
            })

    print(f"Processing {len(metadata)} audio files...")

    processed_metadata = []
    for item in tqdm(metadata):
        audio_path = input_dir / item["audio"]

        try:
            # Load audio
            audio = audio_processor.load_audio(str(audio_path))

            # Extract features
            features = audio_processor.process_audio(audio)

            # Compute duration (uniform)
            text = item["text"]
            duration = audio_processor.compute_duration(len(text), features["mel"].shape[1])

            # Save features
            item_id = item["id"]
            np.save(output_dir / "mel" / f"{item_id}.npy", features["mel"])
            np.save(output_dir / "pitch" / f"{item_id}.npy", features["pitch"])
            np.save(output_dir / "energy" / f"{item_id}.npy", features["energy"])
            np.save(output_dir / "duration" / f"{item_id}.npy", duration)

            processed_metadata.append({
                "id": item_id,
                "text": text,
                "mel_len": features["mel"].shape[1],
            })

        except Exception as e:
            print(f"Error processing {audio_path}: {e}")
            continue

    # Save processed metadata
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(processed_metadata, f, indent=2)

    print(f"\nPreprocessed {len(processed_metadata)} files to {output_dir}")


if __name__ == "__main__":
    main()
