#!/usr/bin/env python3
"""
Video dubbing script using Whisper for transcription and MARS5-TTS for speech synthesis.

This script takes a video file, extracts the audio, transcribes it using Whisper,
translates to English if needed, and then synthesizes new speech using MARS5-TTS
with a reference voice. Finally, it merges the new audio with the original video.

Prerequisites:
    pip install --upgrade torch torchaudio librosa vocos encodec safetensors regex soundfile whisper moviepy ffmpeg-python
"""
import argparse
import os
import tempfile
import torch
import librosa
import soundfile as sf
import whisper
import numpy as np
from moviepy.editor import VideoFileClip, AudioFileClip


def extract_audio(video_path, output_path=None):
    """Extract audio from video file."""
    if output_path is None:
        output_path = os.path.splitext(video_path)[0] + "_extracted.wav"
    
    print(f"Extracting audio from {video_path}...")
    video = VideoFileClip(video_path)
    video.audio.write_audiofile(output_path, codec='pcm_s16le', ffmpeg_params=["-ac", "1"])
    video.close()
    
    return output_path


def transcribe_audio(audio_path, model_size="base"):
    """Transcribe audio using Whisper and return segments with timestamps."""
    print(f"Transcribing audio using Whisper {model_size} model...")
    model = whisper.load_model(model_size)
    result = model.transcribe(
        audio_path, 
        task="translate" if args.translate else "transcribe",
        language=args.source_language if args.source_language else None,
        verbose=True
    )
    
    return result["segments"]


def synthesize_speech(text, ref_audio_path, ref_transcript="", deep_clone=False, ckpt_format="safetensors"):
    """Synthesize speech using MARS5-TTS."""
    print(f"Synthesizing speech with MARS5-TTS...")
    
    # Load MARS5 model and config
    mars5, config_class = torch.hub.load(
        'Camb-ai/mars5-tts', 'mars5_english', trust_repo=True,
        ckpt_format=ckpt_format
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mars5.to(device)

    # Load reference audio
    wav_np, sr = librosa.load(
        ref_audio_path,
        sr=mars5.sr,
        mono=True
    )
    wav = torch.from_numpy(wav_np)

    # Prepare inference config
    cfg = config_class(
        deep_clone=deep_clone,
        rep_penalty_window=100,
        top_k=100,
        temperature=0.7,
        freq_penalty=3
    )

    # Run TTS
    ar_codes, output_audio = mars5.tts(
        text,
        wav,
        ref_transcript if deep_clone else "",
        cfg=cfg
    )
    
    return output_audio, mars5.sr


def create_dubbed_audio(segments, ref_audio_path, ref_transcript, deep_clone, total_duration, sample_rate, temp_dir):
    """Create dubbed audio from segments using MARS5-TTS."""
    print("Creating dubbed audio segments...")
    
    # Initialize a silent audio array for the full duration
    full_audio = np.zeros(int(total_duration * sample_rate))
    
    for i, segment in enumerate(segments):
        start_time = segment["start"]
        end_time = segment["end"]
        text = segment["text"]
        
        print(f"Processing segment {i+1}/{len(segments)}: {text}")
        
        # Synthesize speech for this segment
        audio_segment, sr = synthesize_speech(
            text, 
            ref_audio_path, 
            ref_transcript, 
            deep_clone
        )
        
        # Convert to numpy array
        audio_np = audio_segment.cpu().numpy()
        
        # Calculate start and end samples
        start_sample = int(start_time * sr)
        end_sample = min(int(end_time * sr), len(full_audio))
        
        # Adjust length if needed
        segment_length = min(len(audio_np), end_sample - start_sample)
        
        # Place the segment at the correct position in the full audio
        full_audio[start_sample:start_sample + segment_length] = audio_np[:segment_length]
    
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


def main():
    parser = argparse.ArgumentParser(
        description="Dub a video to English using Whisper for transcription and MARS5-TTS for speech synthesis."
    )
    parser.add_argument(
        '--video', type=str, required=True,
        help='Path to input video file'
    )
    parser.add_argument(
        '--ref_audio', type=str, required=True,
        help='Path to reference WAV file for voice cloning (1-12s, 24kHz)'
    )
    parser.add_argument(
        '--ref_transcript', type=str, default="",
        help='Transcript of reference audio (for deep clone).'
    )
    parser.add_argument(
        '--deep_clone', action='store_true',
        help='Use deep clone (requires --ref_transcript).'
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
        '--ckpt_format', type=str, choices=['safetensors', 'pt'], default='safetensors',
        help='Checkpoint format to load for MARS5: safetensors or pt.'
    )
    parser.add_argument(
        '--keep_temp_files', action='store_true',
        help='Keep temporary files (extracted audio, etc.)'
    )
    
    global args
    args = parser.parse_args()
    
    # Set default output path if not specified
    if args.output is None:
        base_name = os.path.splitext(args.video)[0]
        args.output = f"{base_name}_dubbed.mp4"
    
    # Create temporary directory for intermediate files
    with tempfile.TemporaryDirectory() as temp_dir:
        # Extract audio from video
        extracted_audio = extract_audio(args.video, os.path.join(temp_dir, "extracted_audio.wav"))
        
        # Transcribe and translate audio
        segments = transcribe_audio(extracted_audio, args.whisper_model)
        
        # Get video duration for full audio synthesis
        video = VideoFileClip(args.video)
        duration = video.duration
        video.close()
        
        # Load MARS5 model to get sample rate
        mars5, _ = torch.hub.load(
            'Camb-ai/mars5-tts', 'mars5_english', trust_repo=True,
            ckpt_format=args.ckpt_format
        )
        sample_rate = mars5.sr
        
        # Create dubbed audio
        dubbed_audio = create_dubbed_audio(
            segments, 
            args.ref_audio, 
            args.ref_transcript, 
            args.deep_clone, 
            duration, 
            sample_rate, 
            temp_dir
        )
        
        # Merge dubbed audio with original video
        output_video = merge_audio_with_video(args.video, dubbed_audio, args.output)
        
        print(f"Dubbed video saved to {output_video}")
        
        # Clean up if needed
        if not args.keep_temp_files and os.path.exists(extracted_audio):
            os.remove(extracted_audio)


if __name__ == '__main__':
    main()
