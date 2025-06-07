"""TranscribeAI - Universal Media Transcription and Summarization Library

A professional-grade library for transcribing audio/video files using multiple backends
(OpenAI Whisper and HuggingFace Transformers) with optional AI-powered summarization.

Basic Usage:
    from TranscribeAI import transcribe_media
    
    # Simple transcription
    result = transcribe_media('video.mp4')
    
    # Advanced usage with HuggingFace backend
    result = transcribe_media(
        'video.mp4',
        backend='huggingface',
        model_size='large-v3',
        return_timestamps=True,
        language='english'
    )

Features:
- Multiple transcription backends (Whisper, HuggingFace Transformers)
- GPU acceleration support with CPU fallback
- Multiple audio/video format support
- AI-powered summarization with Ollama
- Professional logging and error handling
- Model caching for better performance
"""

from .transcribe_factory import transcribe_media, TranscriptionFactory, TranscriptionBackend
from .transcribe_base import AUDIO_EXTENSIONS, VIDEO_EXTENSIONS, SUPPORTED_FORMATS
from .transcribe_whisper import WhisperTranscriber
from .transcribe_hf import HuggingFaceTranscriber

__version__ = "2.0.0"
__author__ = "TranscribeAI Team"

# Main public API
__all__ = [
    'transcribe_media',
    'TranscriptionFactory', 
    'TranscriptionBackend',
    'WhisperTranscriber',
    'HuggingFaceTranscriber',
    'AUDIO_EXTENSIONS',
    'VIDEO_EXTENSIONS', 
    'SUPPORTED_FORMATS'
] 