#!/usr/bin/env python3
"""
Video dubbing script using Whisper for transcription and Coqui.ai TTS for speech synthesis.

This script takes a video file, extracts the audio, applies preprocessing for noise reduction,
transcribes it using Whisper, translates to English if needed, and then synthesizes new speech
using Coqui.ai TTS. It also extracts and preserves the original music track when possible.
Finally, it merges the new speech and preserved music with the original video.

Prerequisites:
    pip install -r requirements_coqui.txt
"""
import argparse
import os
import tempfile
import subprocess
import torch
import numpy as np
import whisper
import librosa
import noisereduce as nr
from TTS.api import TTS
from moviepy.editor import VideoFileClip, AudioFileClip, CompositeAudioClip
import soundfile as sf
from scipy.signal import butter, filtfilt


def extract_audio(video_path: str, output_path: str = None) -> str:
    """
    Extract audio from a video file and save it as a WAV file.
    
    This function uses MoviePy to extract the audio track from a video file and save it
    as a high-quality WAV file. It preserves stereo audio if available, which is important
    for better music separation in later processing steps.
    
    Algorithm:
    1. If no output path is provided, generate one based on the input video filename
    2. Load the video file using MoviePy's VideoFileClip
    3. Extract the audio track and save it as a PCM WAV file (uncompressed for best quality)
    4. Close the video file to free resources
    5. Return the path to the extracted audio file
    
    Args:
        video_path (str): Path to the input video file
        output_path (str, optional): Path where the extracted audio will be saved.
            If None, a default path will be generated based on the input filename.
    
    Returns:
        str: Path to the extracted audio file
    """
    if output_path is None:
        output_path = os.path.splitext(video_path)[0] + "_extracted.wav"
    
    logger.info(f"Extracting audio from {video_path}...")
    video = VideoFileClip(video_path)
    # Extract stereo audio if available (for better music separation)
    video.audio.write_audiofile(output_path, codec='pcm_s16le')
    video.close()
    
    return output_path


def extract_voice_sample(audio_path: str, output_path: str = None, duration: float = 10.0) -> str:
    """
    Extract a clean voice sample from the input audio for voice cloning purposes.
    
    This function analyzes an audio file to find segments containing clean speech,
    which are optimal for voice cloning. It applies noise reduction to improve quality
    and uses energy-based voice activity detection to identify speech segments.
    The function selects the best segments based on duration and energy level.
    
    Algorithm:
    1. If no output path is provided, generate one based on the input audio filename
    2. Load the audio file using librosa
    3. Apply noise reduction to get cleaner voice
    4. Detect speech segments using energy-based voice activity detection (VAD)
    5. Sort segments by duration (longest first)
    6. Collect multiple speech segments until reaching the target duration
    7. Concatenate the selected segments to create a voice sample
    8. Apply additional noise reduction to the final sample
    9. Save the voice sample to the output path
    
    Args:
        audio_path (str): Path to the input audio file
        output_path (str, optional): Path where the voice sample will be saved.
            If None, a default path will be generated based on the input filename.
        duration (float, optional): Target duration in seconds for the voice sample.
            Defaults to 10.0 seconds.
    
    Returns:
        str: Path to the extracted voice sample file
    """
    if output_path is None:
        base_dir = os.path.dirname(audio_path)
        base_name = os.path.splitext(os.path.basename(audio_path))[0]
        output_path = os.path.join(base_dir, f"{base_name}_voice_sample.wav")
    
    logger.info(f"Extracting voice sample for cloning from {audio_path}...")
    
    # Load the audio file
    y, sr = librosa.load(audio_path, sr=None)
    
    # Apply noise reduction to get cleaner voice
    y_reduced = nr.reduce_noise(
        y=y,
        sr=sr,
        stationary=True,
        prop_decrease=0.75
    )
    
    # Detect speech segments using energy-based VAD
    def detect_speech_segments(audio: np.ndarray, sr: int, frame_length: int = 1024, hop_length: int = 512, threshold: float = 0.05) -> list:
        """
        Detect speech segments in an audio signal based on energy thresholding.
        
        This function identifies segments of audio that contain speech by analyzing
        the energy (RMS amplitude) of the signal and finding regions where the energy
        exceeds a specified threshold. It then groups consecutive frames to form
        continuous speech segments.
        
        Algorithm:
        1. Compute the RMS energy of the audio signal using the specified frame and hop lengths
        2. Find all frames where the energy exceeds the threshold
        3. Group consecutive frames into continuous segments
        4. Convert frame indices to time values (in seconds)
        5. Return a list of speech segments as (start_time, end_time) tuples
        
        Args:
            audio (np.ndarray): Audio signal as a numpy array
            sr (int): Sample rate of the audio signal
            frame_length (int, optional): Length of each frame for RMS calculation. Defaults to 1024.
            hop_length (int, optional): Number of samples between frames. Defaults to 512.
            threshold (float, optional): Energy threshold for speech detection. Defaults to 0.05.
        
        Returns:
            list: List of tuples containing (start_time, end_time) for each detected speech segment
        """
        # Compute the RMS energy
        energy = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
        
        # Find segments with energy above threshold
        speech_frames = np.where(energy > threshold)[0]
        
        if len(speech_frames) == 0:
            return []
        
        # Group consecutive frames
        speech_segments = []
        start_frame = speech_frames[0]
        prev_frame = speech_frames[0]
        
        for frame in speech_frames[1:]:
            if frame > prev_frame + 1:  # Non-consecutive frame
                # Convert frames to time
                start_time = start_frame * hop_length / sr
                end_time = prev_frame * hop_length / sr
                speech_segments.append((start_time, end_time))
                start_frame = frame
            prev_frame = frame
        
        # Add the last segment
        start_time = start_frame * hop_length / sr
        end_time = prev_frame * hop_length / sr
        speech_segments.append((start_time, end_time))
        
        return speech_segments
    
    # Find speech segments
    speech_segments = detect_speech_segments(y_reduced, sr)
    
    # Sort segments by duration (descending)
    speech_segments.sort(key=lambda x: x[1] - x[0], reverse=True)
    
    # Collect multiple speech segments to get a better voice sample
    voice_samples = []
    total_duration = 0
    target_duration = min(duration, 20.0)  # Aim for up to 20 seconds of good speech
    
    for start_time, end_time in speech_segments:
        segment_duration = end_time - start_time
        if segment_duration < 0.5:  # Skip very short segments
            continue
            
        # Convert time to samples
        start_sample = int(start_time * sr)
        end_sample = int(end_time * sr)
        
        # Extract the segment
        segment = y_reduced[start_sample:end_sample]
        voice_samples.append(segment)
        
        total_duration += segment_duration
        if total_duration >= target_duration:
            break
    
    if voice_samples:
        # Concatenate all collected segments
        voice_sample = np.concatenate(voice_samples)
        
        # Normalize the audio
        max_val = np.max(np.abs(voice_sample))
        if max_val > 0:
            voice_sample = voice_sample / max_val * 0.9
        
        # Save the voice sample
        sf.write(output_path, voice_sample, sr)
        print(f"Enhanced voice sample extracted to {output_path} (duration: {total_duration:.2f}s)")
        return output_path
    else:
        print("No clear speech segments found. Using the first 10 seconds of audio.")
        # Use the first portion of audio as fallback
        max_samples = min(int(duration * sr), len(y_reduced))
        sf.write(output_path, y_reduced[:max_samples], sr)
        return output_path


def apply_noise_reduction(audio_path: str, output_path: str = None) -> str:
    """
    Apply noise reduction to audio to improve transcription accuracy.
    
    This function processes an audio file to reduce background noise and remove
    low-frequency rumble, which significantly improves the quality of speech
    transcription. It uses a combination of spectral noise reduction and
    high-pass filtering techniques.
    
    Algorithm:
    1. If no output path is provided, generate one based on the input audio filename
    2. Load the audio file using librosa
    3. Estimate a noise profile from the first 2 seconds of audio (assuming this
       portion contains representative background noise)
    4. Apply spectral noise reduction using the noisereduce library
    5. Apply a high-pass filter (80Hz cutoff) to remove low-frequency rumble
       using a Butterworth filter
    6. Save the processed audio to the output path
    
    Args:
        audio_path (str): Path to the input audio file
        output_path (str, optional): Path where the noise-reduced audio will be saved.
            If None, a default path will be generated based on the input filename.
    
    Returns:
        str: Path to the noise-reduced audio file
    """
    if output_path is None:
        base, ext = os.path.splitext(audio_path)
        output_path = f"{base}_cleaned{ext}"
    
    logger.info("Applying noise reduction...")
    # Load audio
    y, sr = librosa.load(audio_path, sr=None)
    
    # Estimate noise profile from a presumably non-speech portion (first 2 seconds)
    noise_sample = y[:min(len(y), int(2 * sr))]
    
    # Apply noise reduction
    reduced_noise = nr.reduce_noise(
        y=y,
        sr=sr,
        stationary=True,
        prop_decrease=0.75,
        n_std_thresh_stationary=1.5
    )
    
    # Apply a high-pass filter to remove low-frequency rumble
    def butter_highpass(cutoff: float, fs: int, order: int = 5) -> tuple:
        """
        Design a Butterworth high-pass filter for audio processing.
        
        This function creates the coefficients for a digital Butterworth high-pass filter,
        which can be used to attenuate low-frequency components in an audio signal.
        The filter is designed using the scipy.signal.butter function.
        
        Algorithm:
        1. Calculate the Nyquist frequency (half the sampling rate)
        2. Normalize the cutoff frequency by dividing by the Nyquist frequency
        3. Generate filter coefficients using the scipy.signal.butter function
        4. Return the filter coefficients as a tuple of numerator and denominator polynomials
        
        Args:
            cutoff (float): Cutoff frequency in Hz (frequencies below this will be attenuated)
            fs (int): Sampling rate of the audio signal in Hz
            order (int, optional): Order of the filter. Higher orders give sharper cutoffs
                but may introduce more ringing artifacts. Defaults to 5.
        
        Returns:
            tuple: A tuple (b, a) containing the filter coefficients, where b is the numerator
                  polynomial and a is the denominator polynomial
        """
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='high', analog=False)
        return b, a
    
    b, a = butter_highpass(cutoff=80, fs=sr, order=4)
    filtered_audio = filtfilt(b, a, reduced_noise)
    
    # Save the processed audio
    sf.write(output_path, filtered_audio, sr)
    logger.info(f"Noise-reduced audio saved to {output_path}")
    
    return output_path


def extract_music(audio_path: str, output_path: str = None) -> str:
    """
    Extract music from the audio while removing vocal content.
    
    This function attempts to separate the music track from the input audio file,
    removing vocals to create a clean instrumental track. It uses a two-tiered approach:
    1. Primary method: Facebook's Demucs model for high-quality source separation
    2. Fallback method: Librosa's HPSS (Harmonic-Percussive Source Separation) with
       additional spectral masking for vocal reduction
    
    The function applies several audio processing techniques to enhance the extracted music:
    - Spectral vocal reduction to further minimize vocal content
    - Dynamic range compression for consistent volume
    - Frequency filtering to focus on musical elements
    - Normalization to prevent clipping
    
    Algorithm (Primary Method - Demucs):
    1. Load the Demucs model (tries htdemucs_ft, htdemucs, or mdx_extra in that order)
    2. Load and preprocess the audio file (resampling to 44.1kHz if needed)
    3. Apply the Demucs model to separate audio into drums, bass, other, and vocals
    4. Combine all non-vocal sources with appropriate gain levels
    5. Apply additional spectral masking to further reduce any remaining vocal content
    6. Apply dynamic range compression and normalization
    7. Save the processed music track
    
    Algorithm (Fallback Method - HPSS):
    1. Use Librosa's HPSS to separate harmonic and percussive components
    2. Apply spectral masking to the harmonic component to reduce vocal frequencies
    3. Mix the processed harmonic component with some percussive content
    4. Apply bandpass filtering to focus on typical music frequencies
    5. Apply dynamic range compression and normalization
    6. Save the processed music track
    
    Args:
        audio_path (str): Path to the input audio file
        output_path (str, optional): Path where the extracted music will be saved.
            If None, a default path will be generated based on the input filename.
    
    Returns:
        str or None: Path to the extracted music file if successful, None if both
                    extraction methods fail
    """
    if output_path is None:
        base_dir = os.path.dirname(audio_path)
        base_name = os.path.splitext(os.path.basename(audio_path))[0]
        output_path = os.path.join(base_dir, f"{base_name}_music.wav")
    
    # Create a directory for Demucs output
    output_dir = os.path.join(os.path.dirname(output_path), "demucs_output")
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info("Extracting music from audio using Demucs with enhanced vocal removal...")
    try:
        # First try to use Demucs (Facebook's state-of-the-art source separation)
        import torch
        from demucs.pretrained import get_model
        from demucs.apply import apply_model
        import torchaudio
        
        logger.info("Loading Demucs model...")
        # Try to load the best model (htdemucs_ft)
        try:
            model = get_model("htdemucs_ft")
            logger.info("Using htdemucs_ft model (highest quality)")
        except:
            try:
                model = get_model("htdemucs")
                logger.info("Using htdemucs model")
            except:
                model = get_model("mdx_extra")
                logger.info("Using mdx_extra model")
        
        # Move model to GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        
        # Load audio
        logger.info("Loading audio file...")
        waveform, sample_rate = torchaudio.load(audio_path)
        
        # Resample if needed (Demucs expects 44.1kHz)
        if sample_rate != 44100:
            logger.info(f"Resampling from {sample_rate}Hz to 44100Hz")
            resampler = torchaudio.transforms.Resample(sample_rate, 44100)
            waveform = resampler(waveform)
            sample_rate = 44100
        
        # Convert to expected format
        waveform = waveform.to(device)
        
        # Apply the separation model
        logger.info("Applying source separation (this may take a while)...")
        with torch.no_grad():
            sources = apply_model(model, waveform.unsqueeze(0), device=device)[0]
        
        # Sources will be in order: drums, bass, other, vocals
        # We want everything except vocals for the music track
        drums = sources[0].cpu().numpy()
        bass = sources[1].cpu().numpy()
        other = sources[2].cpu().numpy()
        
        # Combine all non-vocal sources with appropriate levels
        logger.info("Combining instrumental tracks...")
        drums_gain = 1.0
        bass_gain = 1.2
        other_gain = 1.0
        
        # Create a balanced mix
        music = drums * drums_gain + bass * bass_gain + other * other_gain
        
        # Apply additional vocal removal using spectral masking
        def apply_spectral_vocal_reduction(audio, sr=44100):
            """Apply spectral masking to further reduce any vocal content"""
            logger.info("Applying spectral vocal reduction...")
            # Convert to mono if stereo
            if len(audio.shape) > 1:
                audio_mono = np.mean(audio, axis=0)
            else:
                audio_mono = audio
                
            # Convert to frequency domain
            D = librosa.stft(audio_mono)
            
            # Get magnitude and phase
            magnitude, phase = librosa.magphase(D)
            
            # Create a spectral mask that reduces frequencies in the speech range
            freq_bins = librosa.fft_frequencies(sr=sr, n_fft=2048)
            mask = np.ones_like(magnitude)
            
            # Identify bins in the vocal frequency range (200-3500 Hz)
            vocal_bins = np.where((freq_bins >= 200) & (freq_bins <= 3500))[0]
            
            # Apply a gentler reduction to these frequencies
            for i in vocal_bins:
                if i < mask.shape[0]:
                    mask[i, :] = 0.5  # Reduce by 50%
            
            # Apply the mask
            magnitude_masked = magnitude * mask
            
            # Reconstruct the signal
            D_masked = magnitude_masked * phase
            processed_audio = librosa.istft(D_masked)
            
            # If original was stereo, convert back to stereo
            if len(audio.shape) > 1:
                processed_audio = np.stack([processed_audio, processed_audio])
            
            return processed_audio
        
        # Apply additional vocal reduction
        music = apply_spectral_vocal_reduction(music, sr=sample_rate)
        
        # Apply dynamic range compression to make the music more consistent
        def apply_dynamic_compression(audio, threshold=0.3, ratio=2.0):
            """Apply simple dynamic range compression"""
            logger.info("Applying dynamic range compression...")
            # Handle stereo or mono
            if len(audio.shape) > 1:
                # Process each channel
                compressed = np.zeros_like(audio)
                for i in range(audio.shape[0]):
                    # Compute the amplitude envelope
                    envelope = np.abs(audio[i])
                    
                    # Apply compression when envelope exceeds threshold
                    mask = envelope > threshold
                    compressed[i] = audio[i].copy()
                    if np.any(mask):
                        compressed[i, mask] = threshold + (compressed[i, mask] - threshold) / ratio
            else:
                # Mono processing
                envelope = np.abs(audio)
                mask = envelope > threshold
                compressed = audio.copy()
                if np.any(mask):
                    compressed[mask] = threshold + (compressed[mask] - threshold) / ratio
            
            return compressed
        
        # Apply compression
        music = apply_dynamic_compression(music, threshold=0.5, ratio=1.5)
        
        # Normalize the music to prevent clipping
        logger.info("Normalizing audio levels...")
        if len(music.shape) > 1:
            # Stereo normalization
            max_val = np.max(np.abs(music))
            if max_val > 0:
                music = music / max_val * 0.9
        else:
            # Mono normalization
            max_val = np.max(np.abs(music))
            if max_val > 0:
                music = music / max_val * 0.9
        
        # Apply a slight boost to make music more prominent
        music = music * 1.5
        
        # Save the processed music
        logger.info(f"Saving music track to {output_path}")
        if len(music.shape) > 1:
            # Transpose if needed for soundfile
            sf.write(output_path, music.T, sample_rate)
        else:
            sf.write(output_path, music, sample_rate)
            
        logger.info(f"Enhanced music track extracted to {output_path}")
        return output_path
    except Exception as e:
        logger.error(f"Error extracting music with Demucs: {e}")
        # Fallback method: try to extract music using harmonic-percussive source separation
        try:
            logger.info("Falling back to librosa's HPSS for music extraction...")
            y, sr = librosa.load(audio_path, sr=None)
            
            logger.info("Performing harmonic-percussive source separation...")
            # Compute the harmonic and percussive components
            # Harmonic content often contains the musical elements we want to preserve
            y_harmonic, y_percussive = librosa.effects.hpss(y)
            
            # Apply vocal reduction on the harmonic component
            # Convert to frequency domain
            logger.info("Applying vocal reduction filter...")
            D = librosa.stft(y_harmonic)
            
            # Get magnitude and phase
            magnitude, phase = librosa.magphase(D)
            
            # Create a spectral mask that reduces frequencies in the speech range
            freq_bins = librosa.fft_frequencies(sr=sr, n_fft=2048)
            mask = np.ones_like(magnitude)
            
            # Identify bins in the vocal frequency range (200-3500 Hz)
            vocal_bins = np.where((freq_bins >= 200) & (freq_bins <= 3500))[0]
            
            # Reduce these frequencies (but don't eliminate completely)
            for i in vocal_bins:
                if i < mask.shape[0]:
                    mask[i, :] = 0.4  # Reduce by 60%
            
            # Apply the mask
            magnitude_masked = magnitude * mask
            
            # Reconstruct the signal
            D_masked = magnitude_masked * phase
            y_harmonic_filtered = librosa.istft(D_masked)
            
            # Mix with some percussive content for a more balanced sound
            logger.info("Mixing harmonic and percussive components...")
            music = y_harmonic_filtered * 0.8 + y_percussive * 0.5
            
            # Apply a band-pass filter to focus on typical music frequencies
            logger.info("Applying frequency filtering...")
            def butter_bandpass(lowcut, highcut, fs, order=5):
                nyq = 0.5 * fs
                low = lowcut / nyq
                high = highcut / nyq
                b, a = butter(order, [low, high], btype='band')
                return b, a
            
            # Music often has more energy in 60-10000 Hz range
            b, a = butter_bandpass(lowcut=60, highcut=10000, fs=sr, order=4)
            filtered_music = filtfilt(b, a, music)
            
            # Apply dynamic range compression
            logger.info("Applying dynamic range compression...")
            def apply_simple_compression(audio, threshold=0.3, ratio=2.0):
                envelope = np.abs(audio)
                mask = envelope > threshold
                compressed = audio.copy()
                if np.any(mask):
                    compressed[mask] = threshold + (compressed[mask] - threshold) / ratio
                return compressed
            
            compressed_music = apply_simple_compression(filtered_music, threshold=0.5, ratio=1.5)
            
            # Boost the volume for better audibility
            compressed_music = compressed_music * 2.0
            
            # Normalize to prevent clipping
            max_val = np.max(np.abs(compressed_music))
            if max_val > 0:
                compressed_music = compressed_music / max_val * 0.9
            
            sf.write(output_path, compressed_music, sr)
            logger.info(f"Enhanced fallback music extraction saved to {output_path}")
            return output_path
        except Exception as e2:
            logger.error(f"Fallback music extraction also failed: {e2}")
            return None


def transcribe_audio(audio_path: str, model_size: str = "base", task: str = "transcribe", language: str = None) -> list:
    """
    Transcribe audio using OpenAI's Whisper model and return segments with timestamps.
    
    This function loads the specified Whisper model and processes the audio file to generate
    a transcription or translation with timestamps. The function automatically selects
    the appropriate device (GPU or CPU) based on availability.
    
    Algorithm:
    1. Determine whether to use GPU or CPU based on CUDA availability
    2. Load the specified Whisper model size (tiny, base, small, medium, large)
    3. Process the audio file using Whisper's transcribe method
    4. Extract and return the segments containing text and timestamp information
    
    Args:
        audio_path (str): Path to the audio file to transcribe
        model_size (str, optional): Size of the Whisper model to use. Defaults to "base".
            Options: tiny, base, small, medium, large
        task (str, optional): Task to perform - either "transcribe" or "translate". 
            Defaults to "transcribe".
        language (str, optional): Language code for transcription. If None, Whisper will
            auto-detect the language. Defaults to None.
    
    Returns:
        list: List of segment dictionaries containing text, start and end timestamps, etc.
    """
    logger.info(f"Transcribing audio using Whisper {model_size} model...")
    
    # Set device based on availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
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


def initialize_tts(model_name: str, gpu: bool = True, reference_wav: str = None, reference_speaker_lang: str = None) -> TTS:
    """
    Initialize a Coqui TTS model with optional voice cloning support.
    
    This function initializes a Text-to-Speech model from the Coqui TTS library,
    configuring it based on the specified parameters. It handles device selection
    (GPU/CPU), loads the requested model, and configures voice cloning if reference
    audio is provided. It also logs information about available speakers and languages
    for the selected model.
    
    Algorithm:
    1. Determine the appropriate device (CUDA GPU or CPU) based on availability and user preference
    2. Initialize the TTS model with the specified model name and device
    3. Store reference audio information for voice cloning if provided
    4. Log available speakers and languages for the model if applicable
    5. Return the initialized TTS model ready for speech synthesis
    
    Args:
        model_name (str): Name of the Coqui TTS model to initialize
        gpu (bool, optional): Whether to use GPU acceleration if available. Defaults to True.
        reference_wav (str, optional): Path to a reference audio file for voice cloning.
            Only used with models that support voice cloning like YourTTS. Defaults to None.
        reference_speaker_lang (str, optional): Language code of the speaker in the reference
            audio. Used for voice cloning with multilingual models. Defaults to None.
    
    Returns:
        TTS: Initialized Coqui TTS model ready for speech synthesis
    """
    logger.info(f"Initializing Coqui TTS model: {model_name}")
    
    # Set device based on availability and user preference
    device = "cuda" if torch.cuda.is_available() and gpu else "cpu"
    logger.info(f"Using device for TTS: {device}")
    
    # Initialize TTS
    tts = TTS(model_name=model_name, progress_bar=True, gpu=gpu)
    
    # Store reference audio info for YourTTS voice cloning if provided
    tts.reference_wav = reference_wav
    tts.reference_speaker_lang = reference_speaker_lang
    
    # Log available speakers and languages for the model
    if hasattr(tts, 'speakers') and tts.speakers:
        logger.info(f"Available speakers: {', '.join(tts.speakers[:5])}{'...' if len(tts.speakers) > 5 else ''}")
    
    if hasattr(tts, 'languages') and tts.languages:
        logger.info(f"Available languages: {', '.join(tts.languages[:5])}{'...' if len(tts.languages) > 5 else ''}")
    
    return tts


def synthesize_speech(tts: TTS, text: str, output_path: str, speaker: str = None, language: str = None) -> tuple:
    """
    Synthesize speech using Coqui TTS with optional voice cloning support.
    
    This function generates speech from text using a Coqui TTS model, handling various
    model-specific requirements such as speaker selection, language settings, and voice
    cloning parameters. It includes automatic speaker selection for multi-speaker models
    and applies post-processing to improve speech quality and timing.
    
    Algorithm:
    1. Prepare synthesis parameters based on the model type and available options
    2. For multi-speaker models (like YourTTS):
       a. If no speaker is specified, automatically select an appropriate default speaker
       b. If the model supports multiple languages, ensure a valid language is selected
    3. For voice cloning with YourTTS:
       a. Configure voice cloning parameters using the reference audio
       b. Set speech rate and voice similarity parameters to improve quality
    4. Synthesize speech using the TTS model and save to the specified output path
    5. Apply post-processing if needed:
       a. For voice cloning, check if speech is too fast and apply time stretching
    6. Return the processed audio data and sample rate
    
    Args:
        tts (TTS): Initialized Coqui TTS model
        text (str): Text to synthesize into speech
        output_path (str): Path where the synthesized audio will be saved
        speaker (str, optional): Speaker ID for multi-speaker models. If None, a default
            speaker will be automatically selected for models that require it. Defaults to None.
        language (str, optional): Language code for multilingual models. If None, a default
            language will be selected for models that require it. Defaults to None.
    
    Returns:
        tuple: A tuple containing (audio_data, sample_rate) where audio_data is a numpy array
               of the synthesized audio and sample_rate is the sample rate in Hz
    """
    logger.info(f"Synthesizing: {text[:50]}..." if len(text) > 50 else f"Synthesizing: {text}")
    
    # Handle different model types (some require speaker or language)
    kwargs = {}
    
    # YourTTS requires a speaker even when using voice cloning
    if 'your_tts' in tts.model_name or 'xtts_v2' in tts.model_name:
        # If no speaker is provided, use a default speaker
        if speaker is None:
            # Get the first available speaker for YourTTS
            if hasattr(tts, 'speakers') and tts.speakers:
                # Use the first speaker in the list
                speaker = tts.speakers[0]
                logger.info(f"Using default speaker: {speaker}")
            else:
                # Fallback to a known speaker in YourTTS (female-en-5 is usually available)
                speaker = "female-en-5"
                logger.info(f"Using fallback speaker: {speaker}")
    
    if speaker is not None:
        kwargs["speaker"] = speaker
    
    # Handle language parameter
    if 'your_tts' in tts.model_name or 'xtts_v2' in tts.model_name:
        # For YourTTS, make sure we have a valid language
        if language is None and hasattr(tts, 'languages') and tts.languages:
            # Default to English if available, otherwise use the first language
            if 'en' in tts.languages:
                language = 'en'
            else:
                language = tts.languages[0]
            logger.info(f"Using default language: {language}")
    
    if language is not None:
        kwargs["language"] = language
    
    # Add voice cloning parameters if using YourTTS and reference audio is available
    if hasattr(tts, 'reference_wav') and tts.reference_wav and ('your_tts' in tts.model_name or 'xtts_v2' in tts.model_name):
        logger.info(f"Using voice cloning with reference audio: {tts.reference_wav}")
        kwargs["speaker_wav"] = tts.reference_wav  # Use speaker_wav instead of reference_wav
        
        # If reference speaker language is provided, use it for the input language
        if hasattr(tts, 'reference_speaker_lang') and tts.reference_speaker_lang:
            logger.info(f"Using reference speaker language: {tts.reference_speaker_lang}")
            # Note: We don't override the output language here anymore
        
        # Add parameters to improve voice cloning quality and speech rate
        if hasattr(tts, 'model_name') and ('your_tts' in tts.model_name or 'xtts_v2' in tts.model_name):
            # Adjust speech rate (0.8 = 20% slower than normal)
            kwargs["speed"] = 0.8
            
            # Increase speaker similarity weight (if supported)
            # This makes the output voice more similar to the reference
            if hasattr(tts, 'synthesizer') and hasattr(tts.synthesizer, 'voice_similarity'):
                tts.synthesizer.voice_similarity = 1.0
    
    # Log the final parameters being used
    logger.info(f"TTS parameters: {kwargs}")
    
    # Synthesize speech
    tts.tts_to_file(text=text, file_path=output_path, **kwargs)
    
    # Get audio data and sample rate
    audio, sample_rate = sf.read(output_path)
    
    # Apply additional post-processing to adjust speed if needed
    if hasattr(tts, 'reference_wav') and tts.reference_wav and ('your_tts' in tts.model_name or 'xtts_v2' in tts.model_name):
        # Check if the speech is still too fast
        if len(audio) / sample_rate < len(text) * 0.05:  # Rough estimate of expected duration
            logger.info("Speech seems too fast, applying additional time stretching...")
            # Use librosa to stretch the audio without changing pitch
            import librosa
            audio_stretched = librosa.effects.time_stretch(audio, rate=0.85)  # Stretch by 15%
            sf.write(output_path, audio_stretched, sample_rate)
            audio = audio_stretched
    
    return audio, sample_rate


def create_dubbed_audio(segments: list, tts: TTS, total_duration: float, sample_rate: int, temp_dir: str, speaker: str = None, language: str = None) -> str:
    """
    Create a complete dubbed audio track from transcribed segments using Coqui TTS.
    
    This function takes the transcribed segments (with text and timestamps) and synthesizes
    speech for each segment, placing them at the correct timestamps in a full audio track.
    It handles timing adjustments by either truncating or padding the synthesized audio
    to match the original segment duration.
    
    Algorithm:
    1. Create an empty (silent) audio array for the full video duration
    2. For each transcribed segment:
       a. Extract the text and timestamp information
       b. Synthesize speech for the segment text using the TTS model
       c. Calculate the start and end sample positions based on timestamps
       d. Adjust the synthesized audio length to match the original segment duration
          - If too long: truncate the audio (simple time compression)
          - If too short: pad with silence at the end
       e. Place the adjusted audio segment at the correct position in the full audio
    3. Save the complete dubbed audio to a file
    
    Args:
        segments (list): List of segment dictionaries from Whisper transcription
        tts (TTS): Initialized Coqui TTS model
        total_duration (float): Total duration of the video in seconds
        sample_rate (int): Sample rate for the audio in Hz
        temp_dir (str): Directory to store temporary files
        speaker (str, optional): Speaker ID for multi-speaker models. Defaults to None.
        language (str, optional): Language code for multilingual models. Defaults to None.
    
    Returns:
        str: Path to the generated dubbed audio file
    """
    logger.info("Creating dubbed audio segments...")
    
    # Initialize a silent audio array for the full duration
    full_audio = np.zeros(int(total_duration * sample_rate))
    
    for i, segment in enumerate(segments):
        start_time = segment["start"]
        end_time = segment["end"]
        text = segment["text"]
        
        logger.info(f"Processing segment {i+1}/{len(segments)}: {text}")
        
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
            logger.debug(f"Segment {i+1} is too long, truncating from {len(audio_segment)} to {segment_duration} samples")
            audio_segment = audio_segment[:segment_duration]
        elif len(audio_segment) < segment_duration:
            # Pad with silence
            logger.debug(f"Segment {i+1} is too short, padding from {len(audio_segment)} to {segment_duration} samples")
            padding = np.zeros(segment_duration - len(audio_segment))
            audio_segment = np.concatenate([audio_segment, padding])
        
        # Place the segment at the correct position in the full audio
        full_audio[start_sample:start_sample + len(audio_segment)] = audio_segment
    
    # Save the full dubbed audio
    output_path = os.path.join(temp_dir, "dubbed_audio.wav")
    sf.write(output_path, full_audio, samplerate=sample_rate)
    logger.info(f"Complete dubbed audio saved to {output_path}")
    
    return output_path


def mix_audio_tracks(speech_path: str, music_path: str, output_path: str, speech_volume: float = 1.0, music_volume: float = 0.5) -> str:
    """
    Mix speech and music audio tracks with intelligent volume control and dynamic processing.
    
    This function combines the synthesized speech track with the extracted music track,
    applying various audio processing techniques to create a professional-sounding mix.
    It includes dynamic range compression for consistent speech volume, adaptive music
    volume that increases during silent speech segments, equalization to enhance speech
    clarity, and soft clipping to prevent digital distortion.
    
    Algorithm:
    1. Load both speech and music audio files
    2. Resample music to match speech sample rate if necessary
    3. Adjust lengths to ensure both tracks are the same duration
    4. Apply dynamic range compression to speech for consistent volume
    5. Detect silent segments in the speech track
    6. Create a dynamic volume envelope for music that increases during speech silences
    7. Apply volume adjustments to both tracks
    8. Mix the tracks together
    9. Apply equalization to enhance speech clarity
    10. Apply soft clipping to prevent harsh digital distortion
    11. Save the final mixed audio
    
    Args:
        speech_path (str): Path to the synthesized speech audio file
        music_path (str): Path to the extracted music track
        output_path (str): Path where the mixed audio will be saved
        speech_volume (float, optional): Volume multiplier for speech. Defaults to 1.0.
        music_volume (float, optional): Base volume multiplier for music. Defaults to 0.5.
    
    Returns:
        str: Path to the mixed audio file
    """
    logger.info(f"Mixing speech and music tracks with enhanced process...")
    
    # Load audio files
    speech, speech_sr = librosa.load(speech_path, sr=None)
    music, music_sr = librosa.load(music_path, sr=None)
    
    # Resample music to match speech sample rate if needed
    if music_sr != speech_sr:
        logger.info(f"Resampling music from {music_sr}Hz to {speech_sr}Hz")
        music = librosa.resample(music, orig_sr=music_sr, target_sr=speech_sr)
    
    # Adjust lengths to match
    if len(music) > len(speech):
        music = music[:len(speech)]
    elif len(music) < len(speech):
        # Pad music with zeros to match speech length
        padding = np.zeros(len(speech) - len(music))
        music = np.concatenate([music, padding])
    
    # Apply dynamic range compression to speech to make it more consistent
    def compress_dynamic_range(audio: np.ndarray, threshold: float = 0.3, ratio: float = 2.0) -> np.ndarray:
        """
        Apply simple dynamic range compression to audio.
        
        This function implements a basic dynamic range compressor that reduces the
        volume of loud parts of the audio while leaving quieter parts unchanged.
        It works by attenuating signal amplitudes that exceed a specified threshold
        according to a compression ratio.
        
        Algorithm:
        1. Compute the absolute amplitude envelope of the audio signal
        2. Create a mask identifying samples where the amplitude exceeds the threshold
        3. For samples exceeding the threshold, apply compression using the formula:
           output = threshold + (input - threshold) / ratio
        4. Return the compressed audio signal
        
        Args:
            audio (np.ndarray): Input audio signal as a numpy array
            threshold (float, optional): Amplitude threshold above which compression
                is applied. Defaults to 0.3.
            ratio (float, optional): Compression ratio. Higher values result in more
                compression. Defaults to 2.0.
        
        Returns:
            np.ndarray: Compressed audio signal
        """
        # Compute the amplitude envelope
        envelope = np.abs(audio)
        
        # Apply compression when envelope exceeds threshold
        mask = envelope > threshold
        compressed = np.copy(audio)
        if np.any(mask):
            compressed[mask] = threshold + (compressed[mask] - threshold) / ratio
        
        return compressed
    
    # Compress speech for more consistent volume
    speech = compress_dynamic_range(speech)
    
    # Find silent segments in speech (for adaptive music volume)
    def find_silent_segments(audio: np.ndarray, threshold: float = 0.02, min_length: float = 0.5, sr: int = 22050) -> list:
        """
        Find segments in audio where speech is silent or very quiet.
        
        This function analyzes an audio signal to identify periods of silence or low energy,
        which can be used for adaptive music volume control. It works by computing the RMS
        energy of the signal and finding regions where the energy falls below a specified
        threshold for a minimum duration.
        
        Algorithm:
        1. Compute the RMS energy of the audio signal using fixed frame and hop lengths
        2. Find all frames where the RMS energy is below the threshold
        3. Group consecutive silent frames into continuous segments
        4. Filter segments to keep only those exceeding the minimum length
        5. Convert frame indices to sample positions
        6. Return a list of silent segments as (start_sample, end_sample) tuples
        
        Args:
            audio (np.ndarray): Input audio signal as a numpy array
            threshold (float, optional): Energy threshold below which frames are considered
                silent. Defaults to 0.02.
            min_length (float, optional): Minimum duration in seconds for a segment to be
                considered a valid silent segment. Defaults to 0.5.
            sr (int, optional): Sample rate of the audio signal in Hz. Defaults to 22050.
        
        Returns:
            list: List of tuples containing (start_sample, end_sample) for each silent segment
        """
        # Compute RMS energy
        frame_length = 2048
        hop_length = 512
        rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
        
        # Find frames below threshold
        silent_frames = rms < threshold
        
        # Convert to time segments
        silent_segments = []
        in_silence = False
        start_frame = 0
        
        for i, is_silent in enumerate(silent_frames):
            if is_silent and not in_silence:
                # Start of silence
                in_silence = True
                start_frame = i
            elif not is_silent and in_silence:
                # End of silence
                in_silence = False
                duration = (i - start_frame) * hop_length / sr
                if duration >= min_length:
                    silent_segments.append((start_frame * hop_length, i * hop_length))
        
        # Handle if we end in silence
        if in_silence:
            duration = (len(silent_frames) - start_frame) * hop_length / sr
            if duration >= min_length:
                silent_segments.append((start_frame * hop_length, len(audio)))
        
        return silent_segments
    
    # Find silent segments
    silent_segments = find_silent_segments(speech, sr=speech_sr)
    
    # Create a volume envelope for music (higher during silent speech segments)
    music_envelope = np.ones_like(music) * music_volume
    
    # Increase music volume during silent segments
    for start, end in silent_segments:
        if start < len(music_envelope) and end <= len(music_envelope):
            # Fade in and out for smooth transitions
            fade_samples = min(int(0.1 * speech_sr), (end - start) // 4)  # 100ms fade or 1/4 of segment
            
            # Only apply if segment is long enough for fades
            if end - start > 2 * fade_samples:
                # Set higher volume in silent segment
                music_envelope[start+fade_samples:end-fade_samples] = music_volume * 1.5
                
                # Create fade in
                fade_in = np.linspace(music_volume, music_volume * 1.5, fade_samples)
                music_envelope[start:start+fade_samples] = fade_in
                
                # Create fade out
                fade_out = np.linspace(music_volume * 1.5, music_volume, fade_samples)
                music_envelope[end-fade_samples:end] = fade_out
    
    # Apply volume adjustments with envelope for music
    speech = speech * speech_volume
    music = music * music_envelope
    
    # Mix the tracks
    mixed = speech + music
    
    # Apply a subtle EQ to make speech more clear
    def apply_speech_clarity_eq(audio: np.ndarray, sr: int) -> np.ndarray:
        """
        Apply a subtle equalization to enhance speech clarity in audio.
        
        This function implements a simple but effective equalization technique to improve
        speech intelligibility by boosting mid-high frequencies where speech consonants
        and articulation details reside. It uses a high-pass filter followed by a mixing
        technique to create an effect similar to a high-shelf EQ.
        
        Algorithm:
        1. Apply a Butterworth high-pass filter with a cutoff at 1000 Hz to isolate
           mid-high frequencies where speech articulation details are most present
        2. Boost the filtered signal by a factor of 1.3 to enhance these frequencies
        3. Mix the boosted high-frequency content with the original signal at a ratio
           of 30% boosted to 70% original
        4. This creates a subtle clarity enhancement without making the audio sound thin
        5. Include error handling to return the original audio if processing fails
        
        Args:
            audio (np.ndarray): Input audio signal as a numpy array
            sr (int): Sample rate of the audio signal in Hz
        
        Returns:
            np.ndarray: Equalized audio with enhanced speech clarity
        """
        try:
            # Use a standard high-pass filter instead of highshelf (which is not supported)
            nyquist = sr / 2
            # Focus on speech frequencies (above 1000 Hz)
            cutoff = 1000 / nyquist
            b, a = butter(2, cutoff, btype='high', analog=False)
            filtered = filtfilt(b, a, audio)
            
            # Boost the filtered signal slightly and mix back with original
            # This creates a similar effect to a high shelf filter
            boosted = filtered * 1.3
            return audio * 0.7 + boosted * 0.3
        except Exception as e:
            logger.warning(f"Speech clarity EQ skipped: {e}")
            return audio  # Return unmodified audio if there's an error
    
    # Apply EQ for speech clarity
    try:
        mixed = apply_speech_clarity_eq(mixed, speech_sr)
    except Exception as e:
        logger.error(f"Error applying speech clarity EQ: {e}")
        # Continue without EQ if there's an error
    
    # Normalize to prevent clipping but preserve dynamics
    max_val = np.max(np.abs(mixed))
    if max_val > 1.0:
        # Soft clipping instead of hard normalization to preserve dynamics better
        def soft_clip(x: np.ndarray, threshold: float = 0.9) -> np.ndarray:
            """
            Apply soft clipping to audio to prevent harsh digital distortion.
            
            This function implements a soft clipping algorithm that gradually compresses
            signal amplitudes as they approach the maximum value, rather than hard clipping
            which creates harsh distortion and audible artifacts. It uses a hyperbolic tangent
            (tanh) function to smoothly transition from linear to compressed as signals
            exceed the threshold.
            
            Algorithm:
            1. Compute the absolute value of the input signal
            2. Create a mask identifying samples where the amplitude exceeds the threshold
            3. For samples exceeding the threshold, apply a smooth compression formula:
               output = sign(input) * (threshold + (1-threshold) * tanh((|input|-threshold)/(1-threshold)))
            4. This creates a smooth transition that asymptotically approaches ±1.0
            5. Return the soft-clipped audio signal
            
            Args:
                x (np.ndarray): Input audio signal as a numpy array
                threshold (float, optional): Amplitude threshold above which soft clipping
                    begins to be applied. Defaults to 0.9.
            
            Returns:
                np.ndarray: Soft-clipped audio signal
            """
            x_abs = np.abs(x)
            mask = x_abs > threshold
            x_clip = np.copy(x)
            if np.any(mask):
                x_clip[mask] = np.sign(x[mask]) * (threshold + (1 - threshold) * 
                                                  np.tanh((x_abs[mask] - threshold) / (1 - threshold)))
            return x_clip
        
        # Apply soft clipping
        mixed = soft_clip(mixed / max_val) * 0.95  # Leave some headroom
    
    # Save the mixed audio
    sf.write(output_path, mixed, speech_sr)
    logger.info(f"Enhanced mixed audio saved to {output_path}")
    
    return output_path


def merge_audio_with_video(video_path: str, speech_path: str, output_path: str, music_path: str = None, speech_volume: float = 1.0, music_volume: float = 0.3) -> str:
    """
    Merge the dubbed speech audio and optional music track with the original video.
    
    This function takes the synthesized speech audio and optionally a separated music track,
    mixes them together with the specified volume levels, and then merges the resulting
    audio with the original video to create the final dubbed video output.
    
    Algorithm:
    1. Load the original video using MoviePy
    2. Handle audio mixing based on available tracks:
       a. If a music track is available:
          - Mix the speech and music tracks with specified volume levels
          - Load the mixed audio as the new soundtrack
       b. If no music track is available:
          - Use only the speech track as the new soundtrack
    3. Set the mixed audio as the video's audio track
    4. Export the final video with the new audio track
    5. Clean up temporary resources
    
    Args:
        video_path (str): Path to the original video file
        speech_path (str): Path to the synthesized speech audio file
        output_path (str): Path where the final dubbed video will be saved
        music_path (str, optional): Path to the extracted music track. Defaults to None.
        speech_volume (float, optional): Volume level for the speech track. Defaults to 1.0.
        music_volume (float, optional): Volume level for the music track. Defaults to 0.3.
    
    Returns:
        str: Path to the final dubbed video file
    """
    logger.info(f"Merging audio with original video...")
    
    video = VideoFileClip(video_path)
    
    if music_path and os.path.exists(music_path):
        # If we have a music track, mix it with the speech
        logger.info("Using extracted music track")
        mixed_path = os.path.join(os.path.dirname(speech_path), "mixed_audio.wav")
        mix_audio_tracks(speech_path, music_path, mixed_path, speech_volume, music_volume)
        audio = AudioFileClip(mixed_path)
    else:
        # Otherwise just use the speech track
        logger.info("No music track available, using speech only")
        audio = AudioFileClip(speech_path)
    
    # Set the audio of the video clip
    video = video.set_audio(audio)
    
    # Write the result to a file
    logger.info(f"Exporting final video to {output_path}")
    video.write_videofile(
        output_path,
        codec="libx264",
        audio_codec="aac",
        temp_audiofile="temp-audio.m4a",
        remove_temp=True
    )
    
    video.close()
    audio.close()
    logger.info(f"Video export complete")
    
    return output_path


def list_available_tts_models() -> list:
    """
    List all available Coqui TTS models organized by model type.
    
    This function initializes the TTS API and retrieves a list of all available models,
    then organizes them by model type (e.g., tts_models, vocoder_models) for better readability.
    The models are both logged to the console and returned as a list.
    
    Algorithm:
    1. Initialize the TTS API without loading any specific model
    2. Retrieve the complete list of available models
    3. Group the models by their type based on the first part of their path
    4. Log the models organized by type
    5. Return the complete list of models
    
    Returns:
        list: A list of all available Coqui TTS model names as strings
    """
    logger.info("Available Coqui TTS models:")
    tts = TTS()
    models = tts.list_models()
    
    # Group models by type
    model_types = {}
    for model in models:
        model_type = model.split("/")[0]
        if model_type not in model_types:
            model_types[model_type] = []
        model_types[model_type].append(model)
    
    # Log models by type
    for model_type, models_list in model_types.items():
        logger.info(f"\n{model_type.upper()} Models:")
        for model in models_list:
            logger.info(f"  - {model}")
    
    return models


def list_speakers_for_model(model_name: str) -> list:
    """
    List all available speakers for a specific Coqui TTS model.
    
    This function loads the specified TTS model and retrieves its list of available speakers,
    if the model supports multiple speakers. Some models are single-speaker models and will
    not have a list of speakers available.
    
    Algorithm:
    1. Initialize the TTS API with the specified model
    2. Check if the model has a 'speakers' attribute and if it contains any speakers
    3. If speakers are available, log and return the list of speakers
    4. If no speakers are available, log that information and return None
    
    Args:
        model_name (str): The name of the Coqui TTS model to check for speakers
    
    Returns:
        list or None: A list of available speaker names if the model supports multiple speakers,
                     or None if the model doesn't support multiple speakers
    """
    tts = TTS(model_name=model_name)
    if hasattr(tts, "speakers") and tts.speakers:
        logger.info(f"Available speakers for {model_name}:")
        for speaker in tts.speakers:
            logger.info(f"  - {speaker}")
        return tts.speakers
    else:
        logger.info(f"No specific speakers available for {model_name}")
        return None


def list_languages_for_model(model_name: str) -> list:
    """
    List all available languages for a specific Coqui TTS model.
    
    This function loads the specified TTS model and retrieves its list of available languages,
    if the model supports multiple languages. Some models are single-language models and will
    not have a list of languages available.
    
    Algorithm:
    1. Initialize the TTS API with the specified model
    2. Check if the model has a 'languages' attribute and if it contains any languages
    3. If languages are available, log and return the list of languages
    4. If no languages are available, log that information and return None
    
    Args:
        model_name (str): The name of the Coqui TTS model to check for languages
    
    Returns:
        list or None: A list of available language codes if the model supports multiple languages,
                     or None if the model doesn't support multiple languages
    """
    tts = TTS(model_name=model_name)
    if hasattr(tts, "languages") and tts.languages:
        logger.info(f"Available languages for {model_name}:")
        for language in tts.languages:
            logger.info(f"  - {language}")
        return tts.languages
    else:
        logger.info(f"No specific languages available for {model_name}")
        return None


def main():
    """
    Main entry point for the video dubbing application.
    
    This function parses command-line arguments, orchestrates the video dubbing process,
    and handles the complete pipeline from audio extraction to final video generation.
    It supports various options including voice cloning, language translation, and
    music preservation.
    
    Algorithm:
    1. Parse command-line arguments
    2. Handle informational commands (listing models, speakers, languages)
    3. Validate required arguments
    4. Create a temporary directory for intermediate files
    5. Extract audio from the input video
    6. Extract music track if requested
    7. Apply noise reduction for better transcription if requested
    8. Transcribe and optionally translate the audio using Whisper
    9. Handle voice cloning if requested
    10. Initialize the TTS model
    11. Create dubbed audio by synthesizing speech for each segment
    12. Merge the dubbed speech with the original video and music
    13. Clean up temporary files if requested
    
    Returns:
        None
    """
    # Configure logging
    import logging
    global logger
    logger = logging.getLogger("coqui_video_dubber")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
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
        '--whisper_model', type=str, choices=['tiny', 'base', 'small', 'medium', 'large'], default='medium',
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
        '--tts_model', type=str,
        # default="tts_models/en/ljspeech/tacotron2-DDC",
        # default="tts_models/multilingual/multi-dataset/your_tts",
        default="tts_models/multilingual/multi-dataset/xtts_v2",
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
        '--target_language', type=str, default='en',
        help='Target language code for translation (e.g., "en" for English).'
    )
    parser.add_argument(
        '--clone_voice', action='store_true',
        help='Clone the voice from the input video to the output language using YourTTS.'
    )
    parser.add_argument(
        '--voice_sample_duration', type=float, default=20.0,
        help='Duration in seconds of the voice sample to extract for cloning (default: 20.0).'
    )
    parser.add_argument(
        '--speech_rate', type=float, default=0.8,
        help='Speech rate for voice cloning (0.8 = 20% slower, 1.0 = normal, 1.2 = 20% faster).'
    )
    parser.add_argument(
        '--cpu', action='store_true',
        help='Force CPU usage for TTS (default: use GPU if available).'
    )
    parser.add_argument(
        '--no_noise_reduction', action='store_true',
        help='Skip noise reduction preprocessing step.'
    )
    parser.add_argument(
        '--no_music_extraction', action='store_true',
        help='Skip music extraction and preservation.'
    )
    parser.add_argument(
        '--speech_volume', type=float, default=1.0,
        help='Volume level for speech in the final mix (0.0-1.5).'
    )
    parser.add_argument(
        '--music_volume', type=float, default=0.5,
        help='Volume level for music in the final mix (0.0-1.5).'
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
        
        # Extract music track if requested
        music_path = None
        if not args.no_music_extraction:
            try:
                music_path = extract_music(extracted_audio, os.path.join(temp_dir, "music.wav"))
                logger.info(f"Music extraction {'succeeded' if music_path else 'failed'}")
            except Exception as e:
                logger.error(f"Music extraction error: {e}")
                music_path = None
        
        # Apply noise reduction for better transcription if requested
        transcription_audio = extracted_audio
        if not args.no_noise_reduction:
            try:
                cleaned_audio = apply_noise_reduction(extracted_audio, os.path.join(temp_dir, "cleaned_audio.wav"))
                transcription_audio = cleaned_audio
            except Exception as e:
                logger.error(f"Noise reduction error: {e}")
        
        # Transcribe and translate audio
        task = "translate" if args.translate else "transcribe"
        target_language = args.target_language if args.translate else None
        segments = transcribe_audio(transcription_audio, args.whisper_model, task, args.source_language)
        
        # If voice cloning is enabled, set the language for synthesis to the target language
        if args.clone_voice and args.translate:
            args.language = args.target_language
            
            # If no speaker is specified and we're using voice cloning, set a default speaker
            # We'll let the synthesize_speech function handle selecting an appropriate speaker
        
        # Get video duration for full audio synthesis
        video = VideoFileClip(args.video)
        duration = video.duration
        video.close()
        
        # Handle voice cloning if requested
        reference_wav = None
        reference_speaker_lang = None
        
        if args.clone_voice:
            # Override TTS model to use YourTTS for voice cloning
            if 'your_tts' not in args.tts_model or 'xtts_v2' not in args.tts_model:
                logger.info("Voice cloning requested, switching to YourTTS model")
                args.tts_model = "tts_models/multilingual/multi-dataset/your_tts"
            
            # Extract a voice sample from the input video
            try:
                logger.info("Extracting voice sample for cloning...")
                reference_wav = extract_voice_sample(
                    extracted_audio, 
                    os.path.join(temp_dir, "voice_sample.wav"),
                    duration=args.voice_sample_duration
                )
                
                # Set the reference speaker language based on source language or auto-detected language
                if args.source_language:
                    reference_speaker_lang = args.source_language
                else:
                    # Auto-detect the language using Whisper
                    logger.info("Auto-detecting source language for voice cloning...")
                    audio = whisper.load_audio(reference_wav)
                    audio = whisper.pad_or_trim(audio)

                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    mel = whisper.log_mel_spectrogram(audio).to(device)
                    _, probs = whisper.detect_language(whisper.model, mel)
                    detected_lang = max(probs, key=probs.get)
                    logger.info(f"Detected language: {detected_lang} (confidence: {probs[detected_lang]:.2f})")
                    reference_speaker_lang = detected_lang
            except Exception as e:
                logger.error(f"Error extracting voice sample: {e}")
                logger.warning("Continuing without voice cloning")
        
        # Initialize TTS
        tts = initialize_tts(
            args.tts_model, 
            not args.cpu,
            reference_wav=reference_wav,
            reference_speaker_lang=reference_speaker_lang
        )
        
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
        dubbed_speech = create_dubbed_audio(
            segments, 
            tts, 
            duration, 
            sample_rate, 
            temp_dir,
            speaker=args.speaker,
            language=args.language
        )
        
        # Merge dubbed speech with original video and music if available
        output_video = merge_audio_with_video(
            args.video, 
            dubbed_speech, 
            args.output, 
            music_path=music_path,
            speech_volume=args.speech_volume,
            music_volume=args.music_volume
        )
        
        logger.info(f"Dubbed video saved to {output_video}")
        
        # Clean up if needed
        if not args.keep_temp_files:
            temp_files = [f for f in [extracted_audio, transcription_audio, music_path, dubbed_speech] if f and os.path.exists(f)]
            for file in temp_files:
                try:
                    os.remove(file)
                except:
                    pass


if __name__ == '__main__':
    main()
