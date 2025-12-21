# Animalese TTS

Generate Animalese speech (the cute gibberish language from Animal Crossing) from any text input.

## Features

- **Clean Synthesizer** - Rule-based Animalese generator with crystal-clear audio
- **Multiple Voice Types** - High (peppy), normal, and low (cranky) voices
- **Adjustable Speed** - Control how fast characters speak
- **Neural TTS Model** - PyTorch-based model for training on custom data (experimental)

## Quick Start

### Installation

```bash
# Create conda environment
conda create -n animalese-tts python=3.10 -y
conda activate animalese-tts

# Install dependencies
pip install -r requirements.txt
```

### Generate Animalese Audio

```bash
# Basic usage
python animalese_clean.py --text "Hello, welcome to my island!" --output hello.wav

# Different voice types
python animalese_clean.py --text "Your text" --voice high --output cute.wav    # Peppy villager
python animalese_clean.py --text "Your text" --voice normal --output normal.wav
python animalese_clean.py --text "Your text" --voice low --output grumpy.wav   # Cranky villager

# Adjust speed (lower = faster)
python animalese_clean.py --text "Excited!" --speed 45 --output fast.wav
python animalese_clean.py --text "Slow speech" --speed 70 --output slow.wav

# Adjust base pitch
python animalese_clean.py --text "Higher pitch" --pitch 300 --output high_pitch.wav
```

## Synthesizer Options

| Option | Description | Default |
|--------|-------------|---------|
| `--text` | Text to convert to Animalese | (required) |
| `--output` | Output WAV file path | `animalese_clean.wav` |
| `--voice` | Voice type: `high`, `normal`, `low` | `normal` |
| `--pitch` | Base pitch in Hz | `240.0` |
| `--speed` | Duration per character in ms (lower = faster) | `55` |

## Project Structure

```
animalese-tts/
├── animalese_clean.py      # Main synthesizer (recommended)
├── animalese_v2.py         # Alternative synthesizer with more features
├── simple_animalese.py     # Basic synthesizer
├── train.py                # Neural TTS training script
├── inference.py            # Neural TTS inference script
├── configs/
│   ├── default.yaml        # Full training config
│   └── demo.yaml           # Quick demo config
├── models/                 # Neural network components
│   ├── animalese_tts.py    # Main TTS model
│   ├── encoder.py          # Character encoder
│   ├── decoder.py          # Mel decoder
│   ├── vocoder.py          # HiFi-GAN vocoder
│   └── ...
├── data/                   # Dataset utilities
├── scripts/                # Data generation scripts
└── utils/                  # Training utilities
```

## Neural TTS (Experimental)

The project also includes a neural TTS model based on FastSpeech2 + HiFi-GAN architecture. This requires training on Animalese audio samples.

### Generate Training Data

```bash
python scripts/generate_synthetic_data.py --output-dir data/train --num-samples 1000
```

### Train the Model

```bash
python train.py --config configs/default.yaml --data-dir data/train
```

### Run Inference

```bash
python inference.py --checkpoint checkpoints/best.pt --text "Hello!" --output output.wav
```

> **Note:** The neural model requires significant training time (hundreds of epochs) and ideally real Animalese samples to produce good results. For quick results, use `animalese_clean.py`.

## How It Works

### Clean Synthesizer (`animalese_clean.py`)

The rule-based synthesizer creates Animalese by:

1. **Character Mapping** - Each letter maps to a vowel-like sound
2. **Formant Synthesis** - Uses vocal formant frequencies for natural speech quality
3. **Pitch Variation** - Varies pitch based on character and position for melodic speech
4. **Smooth Envelopes** - Attack-sustain-release for each phoneme

This mimics how the actual Animal Crossing games generate Animalese - each character produces a short vocal sound with pitch variation.

### Neural TTS

The neural approach uses:
- **Character Encoder** - Transformer-based encoding of input text
- **Variance Adaptor** - Predicts duration, pitch, and energy
- **Mel Decoder** - Generates mel spectrograms
- **HiFi-GAN Vocoder** - Converts mel spectrograms to audio

## Examples

```bash
# Villager greeting
python animalese_clean.py --text "Hello! How are you today?" --voice high --output greeting.wav

# Tom Nook style
python animalese_clean.py --text "Yes, yes! That will be fifty thousand bells." --voice normal --pitch 200 --output nook.wav

# Excited discovery
python animalese_clean.py --text "Oh wow! I found a rare fossil!" --voice high --speed 45 --output excited.wav

# Long dialogue
python animalese_clean.py --text "Let me tell you about my day..." --voice low --speed 60 --output story.wav
```

## License

This project is for educational and research purposes.

## Acknowledgments

- Animalese is a trademark of Nintendo
- Neural architecture inspired by FastSpeech2 and HiFi-GAN
