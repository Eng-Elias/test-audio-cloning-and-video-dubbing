# MARS5-TTS CLI

A command-line interface for MARS5-TTS (Multilingual Adaptive Refined Synthesis), a text-to-speech system developed by Cambridge AI.

## Overview

MARS5-TTS allows you to synthesize speech that mimics a reference voice. The CLI provides an easy way to use this technology with your own reference audio and text.

## Features

- Voice cloning from a reference audio sample
- Deep cloning option when reference transcript is provided
- Customizable synthesis parameters
- Support for both safetensors and PyTorch checkpoint formats

## Installation

1. use Linux or WSL
2. Clone this repository
3. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Basic usage:

```bash
python mars5_tts_cli.py --ref_audio <reference_audio.wav> --text "Text to synthesize" --output output.wav
```

Deep cloning (requires reference transcript):

```bash
python mars5_tts_cli.py --ref_audio <reference_audio.wav> --ref_transcript "Transcript of the reference audio" --text "Text to synthesize" --deep_clone --output output_deep.wav
```

### Example

```bash
python mars5_tts_cli.py --ref_audio eg9_ref.wav --ref_transcript "The long pepper is less aromatic than the black, but its oil is more pungent." --text "Then the wizard said softly, don't make a sound." --deep_clone --output output_deep.wav
```

### Command-line Arguments

- `--ref_audio`: Path to reference WAV file (1-12s, 24kHz)
- `--ref_transcript`: Transcript of reference audio (required for deep clone)
- `--text`: Text to synthesize
- `--deep_clone`: Enable deep cloning mode (requires reference transcript)
- `--output`: Output WAV filename (default: output.wav)
- `--ckpt_format`: Checkpoint format to load: safetensors or pt (default: safetensors)

## Requirements

- Python 3.6+
- PyTorch
- torchaudio
- librosa
- vocos
- encodec
- safetensors
- regex
- soundfile

## License

Please refer to the Cambridge AI license for MARS5-TTS.

## Acknowledgements

This CLI tool is built on top of the MARS5-TTS model developed by Cambridge AI.
