"""
Inference script for Animalese TTS.

Generate Animalese audio from text using a trained model.

Usage:
    python inference.py --checkpoint checkpoints/best.pt --text "Hello, how are you?"
    python inference.py --checkpoint checkpoints/best.pt --input texts.txt --output-dir outputs/
"""

import os
import argparse
import torch
import numpy as np
import soundfile as sf
from pathlib import Path
from typing import Optional

from models import AnimaleseTTS
from utils import load_config


class AnimaleseGenerator:
    """Generate Animalese audio from text."""

    def __init__(
        self,
        checkpoint_path: str,
        device: str = "cuda",
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        config = checkpoint["config"]

        # Initialize model
        model_cfg = config["model"]
        self.model = AnimaleseTTS(
            vocab_size=model_cfg["encoder"]["vocab_size"],
            encoder_embed_dim=model_cfg["encoder"]["embed_dim"],
            encoder_hidden_dim=model_cfg["encoder"]["hidden_dim"],
            encoder_n_layers=model_cfg["encoder"]["n_layers"],
            encoder_n_heads=model_cfg["encoder"]["n_heads"],
            variance_hidden_dim=model_cfg["variance_adaptor"]["hidden_dim"],
            variance_kernel_size=model_cfg["variance_adaptor"]["kernel_size"],
            decoder_hidden_dim=model_cfg["decoder"]["hidden_dim"],
            decoder_n_layers=model_cfg["decoder"]["n_layers"],
            decoder_n_heads=model_cfg["decoder"]["n_heads"],
            n_mels=config["audio"]["n_mels"],
            vocoder_upsample_rates=model_cfg["vocoder"]["upsample_rates"],
            vocoder_upsample_kernels=model_cfg["vocoder"]["upsample_kernel_sizes"],
            vocoder_resblock_kernels=model_cfg["vocoder"]["resblock_kernel_sizes"],
            vocoder_resblock_dilations=model_cfg["vocoder"]["resblock_dilation_sizes"],
            vocoder_initial_channel=model_cfg["vocoder"]["initial_channel"],
        ).to(self.device)

        self.model.load_state_dict(checkpoint["model_state"])
        self.model.eval()

        self.sample_rate = config["audio"]["sample_rate"]
        self.hop_length = config["audio"]["hop_length"]

        print(f"Loaded model from {checkpoint_path}")
        print(f"Sample rate: {self.sample_rate}")

    @torch.no_grad()
    def generate(
        self,
        text: str,
        duration_scale: float = 1.0,
        pitch_scale: float = 1.0,
        energy_scale: float = 1.0,
    ) -> np.ndarray:
        """
        Generate Animalese audio from text.

        Args:
            text: Input text
            duration_scale: Scale factor for duration (1.0 = normal, >1 = slower)
            pitch_scale: Scale factor for pitch variation
            energy_scale: Scale factor for energy/loudness variation

        Returns:
            Audio waveform as numpy array
        """
        # Convert text to character indices
        chars = torch.tensor(
            [[ord(c) for c in text]],
            dtype=torch.long,
            device=self.device
        )
        char_lens = torch.tensor([len(text)], device=self.device)

        # Generate
        output = self.model.infer(
            chars=chars,
            char_lens=char_lens,
            duration_scale=duration_scale,
            pitch_scale=pitch_scale,
            energy_scale=energy_scale,
        )

        audio = output["audio"].squeeze().cpu().numpy()

        # Normalize
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val * 0.9

        return audio

    def generate_to_file(
        self,
        text: str,
        output_path: str,
        duration_scale: float = 1.0,
        pitch_scale: float = 1.0,
        energy_scale: float = 1.0,
    ):
        """Generate and save audio to file."""
        audio = self.generate(
            text=text,
            duration_scale=duration_scale,
            pitch_scale=pitch_scale,
            energy_scale=energy_scale,
        )

        sf.write(output_path, audio, self.sample_rate)
        print(f"Saved audio to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate Animalese audio")
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--text", type=str, default=None,
        help="Text to convert to Animalese"
    )
    parser.add_argument(
        "--input", type=str, default=None,
        help="Input file with texts (one per line)"
    )
    parser.add_argument(
        "--output", type=str, default="output.wav",
        help="Output audio file or directory"
    )
    parser.add_argument(
        "--duration-scale", type=float, default=1.0,
        help="Duration scale factor"
    )
    parser.add_argument(
        "--pitch-scale", type=float, default=1.0,
        help="Pitch scale factor"
    )
    parser.add_argument(
        "--energy-scale", type=float, default=1.0,
        help="Energy scale factor"
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Device (cuda/cpu)"
    )
    args = parser.parse_args()

    # Initialize generator
    generator = AnimaleseGenerator(args.checkpoint, args.device)

    if args.text:
        # Single text
        generator.generate_to_file(
            text=args.text,
            output_path=args.output,
            duration_scale=args.duration_scale,
            pitch_scale=args.pitch_scale,
            energy_scale=args.energy_scale,
        )

    elif args.input:
        # Multiple texts from file
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)

        with open(args.input, "r") as f:
            texts = [line.strip() for line in f if line.strip()]

        for i, text in enumerate(texts):
            output_path = output_dir / f"output_{i:04d}.wav"
            generator.generate_to_file(
                text=text,
                output_path=str(output_path),
                duration_scale=args.duration_scale,
                pitch_scale=args.pitch_scale,
                energy_scale=args.energy_scale,
            )

    else:
        # Interactive mode
        print("\nEnter text to convert to Animalese (Ctrl+C to exit):")
        try:
            while True:
                text = input("> ").strip()
                if text:
                    audio = generator.generate(
                        text=text,
                        duration_scale=args.duration_scale,
                        pitch_scale=args.pitch_scale,
                        energy_scale=args.energy_scale,
                    )
                    sf.write(args.output, audio, generator.sample_rate)
                    print(f"Saved to {args.output}")
        except KeyboardInterrupt:
            print("\nExiting...")


if __name__ == "__main__":
    main()
