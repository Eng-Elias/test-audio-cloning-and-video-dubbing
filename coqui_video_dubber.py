#!/usr/bin/env python3
"""
Video dubbing script using Whisper for transcription and Coqui.ai TTS for speech synthesis.

This script takes a video file, extracts the audio, transcribes it using Whisper,
translates to English if needed, and then synthesizes new speech using Coqui.ai TTS.
Finally, it merges the new audio with the original video.

Prerequisites:
    pip install -r requirements_coqui.txt
"""
import argparse
import os
import tempfile
import torch
import numpy as np
import whisper
from TTS.api import TTS
from moviepy.editor import VideoFileClip, AudioFileClip
import soundfile as sf


def extract_audio(video_path, output_path=None):
    """Extract audio from video file."""
    if output_path is None:
        output_path = os.path.splitext(video_path)[0] + "_extracted.wav"
    
    print(f"Extracting audio from {video_path}...")
    video = VideoFileClip(video_path)
    video.audio.write_audiofile(output_path, codec='pcm_s16le', ffmpeg_params=["-ac", "1"])
    video.close()
    
    return output_path


def transcribe_audio(audio_path, model_size="base", task="transcribe", language=None):
    """Transcribe audio using Whisper and return segments with timestamps."""
    print(f"Transcribing audio using Whisper {model_size} model...")
    
    # Set device based on availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load Whisper model
    model = whisper.load_model(model_size, device=device)
    
    # Transcribe or translate
    result = model.transcribe(
        audio_path, 
        task=task,
        language=language,
        verbose=True
    )
    
    return result["segments"]


def initialize_tts(model_name, gpu=True):
    """Initialize Coqui TTS model."""
    print(f"Initializing Coqui TTS model: {model_name}")
    
    # Set device based on availability and user preference
    device = "cuda" if torch.cuda.is_available() and gpu else "cpu"
    print(f"Using device for TTS: {device}")
    
    # Initialize TTS
    tts = TTS(model_name=model_name, progress_bar=True, gpu=gpu)
    
    return tts


def synthesize_speech(tts, text, output_path, speaker=None, language=None):
    """Synthesize speech using Coqui TTS."""
    print(f"Synthesizing: {text[:50]}..." if len(text) > 50 else f"Synthesizing: {text}")
    
    # Handle different model types (some require speaker or language)
    kwargs = {}
    if speaker is not None:
        kwargs["speaker"] = speaker
    if language is not None:
        kwargs["language"] = language
    
    # Synthesize speech
    tts.tts_to_file(text=text, file_path=output_path, **kwargs)
    
    # Get audio data and sample rate
    audio, sample_rate = sf.read(output_path)
    
    return audio, sample_rate


def create_dubbed_audio(segments, tts, total_duration, sample_rate, temp_dir, speaker=None, language=None):
    """Create dubbed audio from segments using Coqui TTS."""
    print("Creating dubbed audio segments...")
    
    # Initialize a silent audio array for the full duration
    full_audio = np.zeros(int(total_duration * sample_rate))
    
    for i, segment in enumerate(segments):
        start_time = segment["start"]
        end_time = segment["end"]
        text = segment["text"]
        
        print(f"Processing segment {i+1}/{len(segments)}: {text}")
        
        # Create temporary file for this segment
        segment_path = os.path.join(temp_dir, f"segment_{i}.wav")
        
        # Synthesize speech for this segment
        audio_segment, sr = synthesize_speech(
            tts,
            text,
            segment_path,
            speaker=speaker,
            language=language
        )
        
        # Calculate start and end samples
        start_sample = int(start_time * sr)
        end_sample = min(int(end_time * sr), len(full_audio))
        segment_duration = end_sample - start_sample
        
        # Adjust audio segment length to match original timing if needed
        if len(audio_segment) > segment_duration:
            # Simple time compression (just truncate for now)
            audio_segment = audio_segment[:segment_duration]
        elif len(audio_segment) < segment_duration:
            # Pad with silence
            padding = np.zeros(segment_duration - len(audio_segment))
            audio_segment = np.concatenate([audio_segment, padding])
        
        # Place the segment at the correct position in the full audio
        full_audio[start_sample:start_sample + len(audio_segment)] = audio_segment
    
    # Save the full dubbed audio
    output_path = os.path.join(temp_dir, "dubbed_audio.wav")
    sf.write(output_path, full_audio, samplerate=sample_rate)
    
    return output_path


def merge_audio_with_video(video_path, audio_path, output_path):
    """Merge the dubbed audio with the original video."""
    print(f"Merging dubbed audio with original video...")
    
    video = VideoFileClip(video_path)
    audio = AudioFileClip(audio_path)
    
    # Set the audio of the video clip
    video = video.set_audio(audio)
    
    # Write the result to a file
    video.write_videofile(
        output_path,
        codec="libx264",
        audio_codec="aac",
        temp_audiofile="temp-audio.m4a",
        remove_temp=True
    )
    
    video.close()
    audio.close()
    
    return output_path


def list_available_tts_models():
    """List available Coqui TTS models."""
    print("Available Coqui TTS models:")
    tts = TTS()
    models = tts.list_models()
    
    # Group models by type
    model_types = {}
    for model in models:
        model_type = model.split("/")[0]
        if model_type not in model_types:
            model_types[model_type] = []
        model_types[model_type].append(model)
    
    # Print models by type
    for model_type, models_list in model_types.items():
        print(f"\n{model_type.upper()} Models:")
        for model in models_list:
            print(f"  - {model}")
    
    return models


def list_speakers_for_model(model_name):
    """List available speakers for a specific model."""
    tts = TTS(model_name=model_name)
    if hasattr(tts, "speakers") and tts.speakers:
        print(f"\nAvailable speakers for {model_name}:")
        for speaker in tts.speakers:
            print(f"  - {speaker}")
        return tts.speakers
    else:
        print(f"\nNo specific speakers available for {model_name}")
        return None


def list_languages_for_model(model_name):
    """List available languages for a specific model."""
    tts = TTS(model_name=model_name)
    if hasattr(tts, "languages") and tts.languages:
        print(f"\nAvailable languages for {model_name}:")
        for language in tts.languages:
            print(f"  - {language}")
        return tts.languages
    else:
        print(f"\nNo specific languages available for {model_name}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Dub a video to English using Whisper for transcription and Coqui.ai for speech synthesis."
    )
    parser.add_argument(
        '--video', type=str, required=False,
        help='Path to input video file'
    )
    parser.add_argument(
        '--output', type=str, default=None,
        help='Output video filename. If not specified, will use input_dubbed.mp4'
    )
    parser.add_argument(
        '--whisper_model', type=str, choices=['tiny', 'base', 'small', 'medium', 'large'], default='base',
        help='Whisper model size to use for transcription.'
    )
    parser.add_argument(
        '--translate', action='store_true',
        help='Translate source audio to English (if not already in English).'
    )
    parser.add_argument(
        '--source_language', type=str, default=None,
        help='Source language code (e.g., "fr" for French). If not specified, Whisper will auto-detect.'
    )
    parser.add_argument(
        '--tts_model', type=str, default="tts_models/en/ljspeech/tacotron2-DDC",
        help='Coqui TTS model to use for speech synthesis.'
    )
    parser.add_argument(
        '--speaker', type=str, default=None,
        help='Speaker ID for multi-speaker TTS models.'
    )
    parser.add_argument(
        '--language', type=str, default=None,
        help='Language ID for multi-language TTS models.'
    )
    parser.add_argument(
        '--cpu', action='store_true',
        help='Force CPU usage for TTS (default: use GPU if available).'
    )
    parser.add_argument(
        '--keep_temp_files', action='store_true',
        help='Keep temporary files (extracted audio, etc.)'
    )
    parser.add_argument(
        '--list_models', action='store_true',
        help='List available Coqui TTS models and exit.'
    )
    parser.add_argument(
        '--list_speakers', type=str, default=None,
        help='List available speakers for the specified model and exit.'
    )
    parser.add_argument(
        '--list_languages', type=str, default=None,
        help='List available languages for the specified model and exit.'
    )
    
    args = parser.parse_args()
    
    # Handle informational commands
    if args.list_models:
        list_available_tts_models()
        return
    
    if args.list_speakers:
        list_speakers_for_model(args.list_speakers)
        return
    
    if args.list_languages:
        list_languages_for_model(args.list_languages)
        return
    
    # Validate required arguments for dubbing
    if not args.video:
        parser.error("--video is required for dubbing")
    
    # Set default output path if not specified
    if args.output is None:
        base_name = os.path.splitext(args.video)[0]
        args.output = f"{base_name}_dubbed.mp4"
    
    # Create temporary directory for intermediate files
    with tempfile.TemporaryDirectory() as temp_dir:
        # Extract audio from video
        extracted_audio = extract_audio(args.video, os.path.join(temp_dir, "extracted_audio.wav"))
        
        # Transcribe and translate audio
        task = "translate" if args.translate else "transcribe"
        segments = transcribe_audio(extracted_audio, args.whisper_model, task, args.source_language)
        
        # Get video duration for full audio synthesis
        video = VideoFileClip(args.video)
        duration = video.duration
        video.close()
        
        # Initialize TTS
        tts = initialize_tts(args.tts_model, not args.cpu)
        
        # Get sample rate from a test synthesis
        test_path = os.path.join(temp_dir, "test.wav")
        _, sample_rate = synthesize_speech(
            tts, 
            "Test", 
            test_path, 
            speaker=args.speaker, 
            language=args.language
        )
        
        # Create dubbed audio
        dubbed_audio = create_dubbed_audio(
            segments, 
            tts, 
            duration, 
            sample_rate, 
            temp_dir,
            speaker=args.speaker,
            language=args.language
        )
        
        # Merge dubbed audio with original video
        output_video = merge_audio_with_video(args.video, dubbed_audio, args.output)
        
        print(f"Dubbed video saved to {output_video}")
        
        # Clean up if needed
        if not args.keep_temp_files and os.path.exists(extracted_audio):
            os.remove(extracted_audio)


if __name__ == '__main__':
    main()
