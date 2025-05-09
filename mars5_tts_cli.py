#!/usr/bin/env python3
"""
Test script for MARS5-TTS inference.

Prerequisites:
    pip install --upgrade torch torchaudio librosa vocos encodec safetensors regex soundfile
"""
import argparse
import torch
import librosa
import soundfile as sf


def main():
    parser = argparse.ArgumentParser(
        description="Run MARS5-TTS inference with a reference audio and text prompt."
    )
    parser.add_argument(
        '--ref_audio', type=str, required=True,
        help='Path to reference WAV file (1-12s, 24kHz)'
    )
    parser.add_argument(
        '--ref_transcript', type=str, default="",
        help='Transcript of reference audio (for deep clone).'
    )
    parser.add_argument(
        '--text', type=str, required=True,
        help='Text to synthesize.'
    )
    parser.add_argument(
        '--deep_clone', action='store_true',
        help='Use deep clone (requires --ref_transcript).'
    )
    parser.add_argument(
        '--output', type=str, default='output.wav',
        help='Output WAV filename.'
    )
    parser.add_argument(
        '--ckpt_format', type=str, choices=['safetensors', 'pt'], default='safetensors',
        help='Checkpoint format to load: safetensors or pt.'
    )
    args = parser.parse_args()

    # Load MARS5 model and config
    mars5, config_class = torch.hub.load(
        'Camb-ai/mars5-tts', 'mars5_english', trust_repo=True,
        ckpt_format=args.ckpt_format
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mars5.to(device)

    # Load reference audio
    wav_np, sr = librosa.load(
        args.ref_audio,
        sr=mars5.sr,
        mono=True
    )
    wav = torch.from_numpy(wav_np)

    # Prepare inference config
    cfg = config_class(
        deep_clone=args.deep_clone,
        rep_penalty_window=100,
        top_k=100,
        temperature=0.7,
        freq_penalty=3
    )

    # Run TTS
    ar_codes, output_audio = mars5.tts(
        args.text,
        wav,
        args.ref_transcript if args.deep_clone else "",
        cfg=cfg
    )

    # Save output audio
    sf.write(
        args.output,
        output_audio.cpu().numpy(),
        samplerate=mars5.sr
    )
    print(f"Synthesized audio saved to {args.output}")


if __name__ == '__main__':
    main()
