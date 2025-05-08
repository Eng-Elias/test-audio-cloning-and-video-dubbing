# MARS5 Video Dubber

A powerful tool for dubbing videos into English using Whisper for transcription/translation and MARS5-TTS for high-quality voice synthesis.

## Overview

MARS5 Video Dubber automates the process of dubbing videos into English while preserving the voice characteristics of a reference speaker. The tool uses:

- **OpenAI Whisper**: For accurate transcription and translation of the original audio
- **MARS5-TTS**: For high-quality voice synthesis that can clone a reference voice

## Features

- Extract audio from video files
- Transcribe audio in multiple languages using Whisper
- Translate non-English audio to English
- Synthesize English speech with MARS5-TTS voice cloning
- Merge synthesized audio with the original video
- Support for deep voice cloning with reference transcript
- Preserve timing of original speech segments

## Installation

1. Clone this repository
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

3. Ensure you have FFmpeg installed on your system:
   - **Windows**: Download from [FFmpeg website](https://ffmpeg.org/download.html) and add to PATH
   - **macOS**: `brew install ffmpeg`
   - **Linux**: `apt-get install ffmpeg` or equivalent for your distribution

## Usage

Basic usage:

```bash
python video_dubber.py --video input_video.mp4 --ref_audio voice_reference.wav --output dubbed_video.mp4
```

With translation from another language:

```bash
python video_dubber.py --video french_video.mp4 --ref_audio voice_reference.wav --translate --output french_video_english.mp4
```

With deep voice cloning:

```bash
python video_dubber.py --video input_video.mp4 --ref_audio voice_reference.wav --ref_transcript "This is the transcript of the reference audio." --deep_clone --output dubbed_video_deep.mp4
```

### Command-line Arguments

- `--video`: Path to input video file (required)
- `--ref_audio`: Path to reference WAV file for voice cloning (required)
- `--ref_transcript`: Transcript of reference audio (for deep clone)
- `--deep_clone`: Use deep clone (requires --ref_transcript)
- `--output`: Output video filename (default: input_dubbed.mp4)
- `--whisper_model`: Whisper model size to use (choices: tiny, base, small, medium, large; default: base)
- `--translate`: Translate source audio to English (if not already in English)
- `--source_language`: Source language code (e.g., "fr" for French). If not specified, Whisper will auto-detect
- `--ckpt_format`: Checkpoint format for MARS5 (choices: safetensors, pt; default: safetensors)
- `--keep_temp_files`: Keep temporary files (extracted audio, etc.)

## How It Works

1. **Audio Extraction**: The script extracts the audio track from the input video
2. **Transcription/Translation**: Whisper processes the audio to generate timestamped text segments
3. **Voice Synthesis**: Each text segment is synthesized using MARS5-TTS with the reference voice
4. **Audio Alignment**: Synthesized speech is aligned with the original timing
5. **Video Merging**: The new audio track is merged with the original video

## Requirements

- Python 3.6+
- FFmpeg
- PyTorch
- torchaudio
- librosa
- OpenAI Whisper
- MoviePy
- MARS5-TTS dependencies (vocos, encodec, safetensors, etc.)

## Tips for Best Results

- Use a clear reference audio sample (1-12 seconds) with minimal background noise
- For deep cloning, ensure the reference transcript is accurate
- Larger Whisper models (medium, large) provide better transcription but require more resources
- For languages with significant pauses or speech patterns different from English, results may vary

## License

Please refer to the Cambridge AI license for MARS5-TTS.

## Acknowledgements

This tool combines OpenAI's Whisper and Cambridge AI's MARS5-TTS to create a seamless video dubbing solution.
