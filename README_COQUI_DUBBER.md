# Coqui Video Dubber

A powerful tool for dubbing videos into English using Whisper for transcription/translation and Coqui.ai TTS for high-quality speech synthesis.

## Overview

Coqui Video Dubber automates the process of dubbing videos into English by combining two state-of-the-art AI technologies:

- **OpenAI Whisper**: For accurate transcription and translation of the original audio
- **Coqui.ai TTS**: For high-quality, natural-sounding speech synthesis

This tool is optimized for NVIDIA GPUs, including the RTX 2060, providing efficient processing for both transcription and speech synthesis tasks.

## Features

- Extract audio from video files
- Transcribe audio in multiple languages using Whisper
- Translate non-English audio to English
- Synthesize English speech with Coqui.ai TTS
- Support for multiple TTS models, speakers, and languages
- Preserve timing of original speech segments
- GPU acceleration for faster processing
- Comprehensive model and speaker listing capabilities

## Installation

1. Clone this repository
2. Install the required dependencies:

```bash
pip install -r requirements_coqui.txt
```

3. Ensure you have FFmpeg installed on your system:
   - **Windows**: Download from [FFmpeg website](https://ffmpeg.org/download.html) and add to PATH
   - **macOS**: `brew install ffmpeg`
   - **Linux**: `apt-get install ffmpeg` or equivalent for your distribution

## Usage

### Basic Usage

```bash
python coqui_video_dubber.py --video input_video.mp4 --output dubbed_video.mp4
```

### With Translation from Another Language

```bash
python coqui_video_dubber.py --video french_video.mp4 --translate --output french_video_english.mp4
```

### Using a Specific TTS Model and Speaker

```bash
python coqui_video_dubber.py --video input_video.mp4 --tts_model tts_models/en/vctk/vits --speaker p326
```

### Listing Available Models and Speakers

List all available TTS models:
```bash
python coqui_video_dubber.py --list_models
```

List speakers for a specific model:
```bash
python coqui_video_dubber.py --list_speakers tts_models/en/vctk/vits
```

List languages for a multilingual model:
```bash
python coqui_video_dubber.py --list_languages tts_models/multilingual/multi-dataset/xtts_v2
```

## Command-line Arguments

### Main Options
- `--video`: Path to input video file
- `--output`: Output video filename (default: input_dubbed.mp4)
- `--translate`: Translate source audio to English (if not already in English)
- `--source_language`: Source language code (e.g., "fr" for French). If not specified, Whisper will auto-detect
- `--keep_temp_files`: Keep temporary files (extracted audio, etc.)

### Whisper Options
- `--whisper_model`: Whisper model size (choices: tiny, base, small, medium, large; default: base)

### TTS Options
- `--tts_model`: Coqui TTS model to use (default: tts_models/en/ljspeech/tacotron2-DDC)
- `--speaker`: Speaker ID for multi-speaker TTS models
- `--language`: Language ID for multi-language TTS models
- `--cpu`: Force CPU usage for TTS (default: use GPU if available)

### Information Commands
- `--list_models`: List all available Coqui TTS models and exit
- `--list_speakers MODEL`: List available speakers for the specified model and exit
- `--list_languages MODEL`: List available languages for the specified model and exit

## Recommended Models for RTX 2060

For your NVIDIA RTX 2060, we recommend the following configurations:

### Whisper Model
- For shorter videos (< 10 minutes): `small` or `medium`
- For longer videos: `base` (good balance of speed and accuracy)

### TTS Models
- For speed: `tts_models/en/ljspeech/fast_pitch`
- For quality: `tts_models/en/ljspeech/tacotron2-DDC`
- For voice variety: `tts_models/en/vctk/vits` (multi-speaker)
- For best overall results: `tts_models/multilingual/multi-dataset/xtts_v2` (requires more VRAM)

## How It Works

1. **Audio Extraction**: The script extracts the audio track from the input video
2. **Transcription/Translation**: Whisper processes the audio to generate timestamped text segments
3. **Voice Synthesis**: Each text segment is synthesized using Coqui TTS
4. **Audio Alignment**: Synthesized speech is aligned with the original timing
5. **Video Merging**: The new audio track is merged with the original video

## Tips for Best Results

- Use a higher-quality Whisper model (`medium` or `large`) for better transcription accuracy
- For multi-speaker TTS models, try different speakers to find the most suitable voice
- The XTTS v2 model provides the most natural-sounding results but requires more GPU memory
- For videos with background music, the original music will be lost in the dubbed version
- If you encounter CUDA out-of-memory errors, try using a smaller model or the `--cpu` option

## Troubleshooting

- **CUDA out of memory**: Try a smaller Whisper model, or use `--cpu` for TTS
- **Missing speakers/languages**: Use `--list_speakers` or `--list_languages` to see available options
- **FFmpeg errors**: Ensure FFmpeg is properly installed and in your PATH
- **Slow processing**: Ensure your GPU drivers are up to date for optimal performance

## License

This project uses:
- OpenAI Whisper under the MIT License
- Coqui TTS under the Mozilla Public License 2.0

## Acknowledgements

This tool combines OpenAI's Whisper and Coqui.ai's TTS to create a seamless video dubbing solution.
