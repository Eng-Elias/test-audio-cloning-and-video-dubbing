# MARS Audio and Video Tools

This repository contains a collection of tools for audio synthesis and video dubbing using Camb.AI MARS5-TTS technology and other state-of-the-art AI models.

## Available Tools

### 1. MARS5 TTS CLI

A command-line interface for MARS5-TTS (Multilingual Adaptive Refined Synthesis) that allows you to synthesize speech that mimics a reference voice.

- **Key Features**: Voice cloning, deep cloning, customizable synthesis parameters
- **File**: `mars5_tts_cli.py`
- **Documentation**: [README_MARS5_TTS.md](README_MARS5_TTS.md)

### 2. MARS5 Video Dubber

A tool for dubbing videos into English using Whisper for transcription/translation and MARS5-TTS for high-quality voice synthesis.

- **Key Features**: Voice cloning, translation, deep cloning, timing preservation
- **File**: `mars5_video_dubber.py`
- **Documentation**: [README_MARS5_VIDEO_DUBBER.md](README_MARS5_VIDEO_DUBBER.md)

### 3. Coqui Video Dubber

A powerful tool for dubbing videos into English using Whisper for transcription/translation, Demucs for music extraction, and Coqui.ai TTS for high-quality speech synthesis.

- **Key Features**: Music preservation, voice customization, translation, GPU acceleration
- **File**: `coqui_video_dubber.py`
- **Documentation**: [README_COQUI_DUBBER.md](README_COQUI_DUBBER.md)

## Quick Start

### MARS5 TTS CLI

```bash
python mars5_tts_cli.py --ref_audio reference.wav --text "Text to synthesize" --output output.wav
```

### MARS5 Video Dubber

```bash
python mars5_video_dubber.py --video input.mp4 --ref_audio reference.wav --output dubbed.mp4
```

### Coqui Video Dubber

```bash
python coqui_video_dubber.py --video input.mp4 --output dubbed.mp4
```

## Requirements

Each tool has its own specific requirements. Please refer to the individual README files for detailed installation instructions.

## Acknowledgements

These tools combine various AI technologies:

- Camb.AI's MARS5-TTS
- OpenAI's Whisper
- Coqui.ai's TTS
- Facebook's Demucs
