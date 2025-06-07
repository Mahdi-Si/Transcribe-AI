"""Legacy compatibility module for TranscribeAI.

This module provides backward compatibility for the old transcribe_media.py interface
while using the new modular architecture under the hood.

For new projects, use the new interface:
    from TranscribeAI import transcribe_media
"""

import warnings
from .transcribe_factory import transcribe_media as new_transcribe_media
from .transcribe_whisper import WhisperTranscriber
from .transcribe_base import AUDIO_EXTENSIONS, VIDEO_EXTENSIONS, logger

# Maintain backward compatibility
DEVICE = "cuda" if WhisperTranscriber().device_manager.cuda_available else "cpu"
CUDA_AVAILABLE = WhisperTranscriber().device_manager.cuda_available

# Issue deprecation warning
warnings.warn(
    "Importing from transcribe_media.py is deprecated. Use 'from TranscribeAI import transcribe_media' instead.",
    DeprecationWarning,
    stacklevel=2
)

# Legacy function wrappers - delegate to new architecture
def media_to_audio(media_file, audio_file='audio.wav'):
    """Convert any media file (video/audio) to audio format.
    
    DEPRECATED: Use MediaProcessor.media_to_audio() instead.
    """
    from .transcribe_base import MediaProcessor
    return MediaProcessor.media_to_audio(media_file, audio_file)

def transcribe_audio(audio_file, output_txt='transcript.txt', model_size='base'):
    """Transcribe audio file using Whisper with GPU acceleration if available.
    
    DEPRECATED: Use WhisperTranscriber.transcribe_audio() instead.
    """
    transcriber = WhisperTranscriber()
    return transcriber.transcribe_audio(audio_file, output_txt, model_size)

def summarize_text_with_library(text, output_file, model='llama3.2'):
    """Summarize text using Ollama Python library and save to file.
    
    DEPRECATED: Use SummarizationEngine.summarize_with_library() instead.
    """
    from .transcribe_base import SummarizationEngine
    summarizer = SummarizationEngine()
    return summarizer.summarize_with_library(text, output_file, model)

def summarize_text(text, output_file, model='llama3.2', ollama_url='http://localhost:11434'):
    """Summarize text using Ollama REST API and save to file.
    
    DEPRECATED: Use SummarizationEngine.summarize_with_api() instead.
    """
    from .transcribe_base import SummarizationEngine
    summarizer = SummarizationEngine(ollama_url)
    return summarizer.summarize_with_api(text, output_file, model)

def process_media_file(media_file, output_txt=None, model_size='base', 
                      summarize=True, ollama_model='llama3.2', ollama_method='library'):
    """Complete pipeline: convert media to audio, transcribe, and optionally summarize.
    
    DEPRECATED: Use transcribe_media() function or WhisperTranscriber.process_media_file() instead.
    """
    return new_transcribe_media(
        media_file=media_file,
        backend='whisper',
        output_txt=output_txt,
        summarize=summarize,
        ollama_model=ollama_model,
        ollama_method=ollama_method,
        model_size=model_size
    )

if __name__ == "__main__":
    # Legacy main execution - redirects to new architecture
    logger.warning("DEPRECATED: Running legacy transcribe_media.py")
    logger.warning("For new projects, use: python -m TranscribeAI.main")
    
    media_file = r'C:\Users\mahdi\Desktop\Mahdi-Si-Projects\Transcribe-AI\2025-06-05 17-30-36.mp4'
    
    try:
        result = new_transcribe_media(
            media_file,
            backend='whisper',
            summarize=True,
            ollama_model='phi4-reasoning:latest',
            ollama_method='library'
        )
        
        logger.info(f"\nProcess completed successfully!")
        logger.info(f"📄 Transcript: {result['transcript']}")
        if 'summary' in result:
            logger.info(f"📋 Summary: {result['summary']}")
        
    except Exception as e:
        logger.error(f"\nError: {e}")
