# Coqui Video Dubber

A powerful tool for dubbing videos into English using Whisper for transcription/translation, Demucs for music extraction, and Coqui.ai TTS for high-quality speech synthesis.

## Overview

Coqui Video Dubber automates the process of dubbing videos into English by combining three state-of-the-art AI technologies:

- **OpenAI Whisper**: For accurate transcription and translation of the original audio
- **Facebook Demucs**: For high-quality music extraction and vocal removal
- **Coqui.ai TTS**: For high-quality, natural-sounding speech synthesis

This tool is optimized for NVIDIA GPUs, including the RTX 2060, providing efficient processing for transcription, music separation, and speech synthesis tasks.

## Features

- Extract audio from video files
- Preserve original background music while replacing speech
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
- `--whisper_model`: Whisper model size (choices: tiny, base, small, medium, large; default: medium)

### TTS Options
- `--tts_model`: Coqui TTS model to use (default: tts_models/en/ljspeech/tacotron2-DDC)
- `--speaker`: Speaker ID for multi-speaker TTS models
- `--language`: Language ID for multi-language TTS models
- `--cpu`: Force CPU usage for TTS (default: use GPU if available)

### Audio Mixing Options
- `--speech_volume`: Volume level for speech in the final mix (0.0-1.5, default: 1.0)
- `--music_volume`: Volume level for music in the final mix (0.0-1.5, default: 0.5)
- `--no_noise_reduction`: Skip noise reduction preprocessing step
- `--no_music_extraction`: Skip music extraction and preservation

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
2. **Music Extraction**: Demucs separates the background music from vocals
3. **Audio Preprocessing**: Noise reduction is applied to improve transcription accuracy
4. **Transcription/Translation**: Whisper processes the cleaned audio to generate timestamped text segments
5. **Voice Synthesis**: Each text segment is synthesized using Coqui TTS
6. **Audio Mixing**: Synthesized speech is mixed with the extracted background music
7. **Audio Alignment**: The mixed audio is aligned with the original timing
8. **Video Merging**: The new audio track is merged with the original video

## Tips for Best Results

- Use a higher-quality Whisper model (`medium` or `large`) for better transcription accuracy
- For multi-speaker TTS models, try different speakers to find the most suitable voice
- The XTTS v2 model provides the most natural-sounding results but requires more GPU memory
- Adjust `--speech_volume` and `--music_volume` to find the perfect balance for your video
- For videos with complex audio, try different Demucs models by editing the script
- If you encounter CUDA out-of-memory errors, try using a smaller model or the `--cpu` option
- For videos without background music, use `--no_music_extraction` to speed up processing

## Music Extraction Feature

One of the key features of this tool is the ability to preserve the original background music while replacing the speech. This is accomplished using:

- **Facebook's Demucs**: A state-of-the-art music source separation model that can separate audio into drums, bass, other instruments, and vocals
- **Advanced Vocal Removal**: Multiple techniques to ensure no speech from the original video bleeds into the background music
- **Dynamic Audio Mixing**: Intelligent mixing that adjusts music volume during speech pauses

You can control this feature with:
- `--music_volume`: Adjust how prominent the background music is (0.5 by default)
- `--no_music_extraction`: Skip music extraction if the video doesn't have background music

## Troubleshooting

- **CUDA out of memory**: Try a smaller Whisper model, or use `--cpu` for TTS
- **Missing speakers/languages**: Use `--list_speakers` or `--list_languages` to see available options
- **FFmpeg errors**: Ensure FFmpeg is properly installed and in your PATH
- **Slow processing**: Ensure your GPU drivers are up to date for optimal performance
- **Demucs errors**: If you encounter issues with Demucs, the script will automatically fall back to a simpler method
- **Music too loud/quiet**: Use `--music_volume` to adjust the background music level
- **Speech unclear**: Try increasing `--speech_volume` or decreasing `--music_volume`

## License

This project uses:
- OpenAI Whisper under the MIT License
- Coqui TTS under the Mozilla Public License 2.0

## Acknowledgements

This tool combines OpenAI's Whisper and Coqui.ai's TTS to create a seamless video dubbing solution.
