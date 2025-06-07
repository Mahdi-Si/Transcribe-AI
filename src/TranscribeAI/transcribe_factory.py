from enum import Enum
from typing import Optional, Dict, Any
from .transcribe_whisper import WhisperTranscriber
from .transcribe_hf import HuggingFaceTranscriber
from .transcribe_base import logger

class TranscriptionBackend(Enum):
    """Available transcription backends."""
    WHISPER = "whisper"
    HUGGINGFACE = "huggingface"

class TranscriptionFactory:
    """Factory for creating transcription instances based on backend type."""
    
    _instances: Dict[TranscriptionBackend, Any] = {}
    
    @classmethod
    def create_transcriber(cls, backend: TranscriptionBackend = TranscriptionBackend.WHISPER):
        """Create or retrieve a transcriber instance.
        
        Args:
            backend (TranscriptionBackend): The transcription backend to use.
            
        Returns:
            BaseTranscriber: Configured transcriber instance.
        """
        if backend not in cls._instances:
            logger.info(f"Initializing {backend.value} transcription backend...")
            
            if backend == TranscriptionBackend.WHISPER:
                cls._instances[backend] = WhisperTranscriber()
            elif backend == TranscriptionBackend.HUGGINGFACE:
                cls._instances[backend] = HuggingFaceTranscriber()
            else:
                raise ValueError(f"Unsupported backend: {backend}")
        
        return cls._instances[backend]
    
    @classmethod
    def get_available_backends(cls):
        """Get list of available transcription backends."""
        return [backend.value for backend in TranscriptionBackend]

def transcribe_media(media_file: str, 
                    backend: str = "whisper",
                    output_txt: Optional[str] = None,
                    summarize: bool = True,
                    summary_provider: str = 'ollama',
                    summary_method: str = 'library', 
                    summary_model: Optional[str] = None,
                    # Backward compatibility parameters
                    ollama_model: Optional[str] = None,
                    ollama_method: Optional[str] = None,
                    **kwargs) -> Dict[str, str]:
    """Main entry point for media transcription with automatic backend selection.
    
    Args:
        media_file (str): Path to the media file to transcribe.
        backend (str): Transcription backend to use ('whisper' or 'huggingface').
        output_txt (str): Optional output file path for transcript.
        summarize (bool): Whether to generate a summary.
        summary_provider (str): AI provider for summarization ('ollama', 'openai', 'anthropic', 'gemini').
        summary_method (str): Method for Ollama ('library' or 'api'), ignored for others.
        summary_model (str): Model to use for summarization (provider-specific defaults if None).
        ollama_model (str): DEPRECATED - use summary_model with summary_provider='ollama'.
        ollama_method (str): DEPRECATED - use summary_method.
        **kwargs: Additional parameters specific to the chosen backend.
        
    Returns:
        dict: Dictionary containing paths to generated files.
        
    Examples:
        # Basic usage with Whisper
        result = transcribe_media('video.mp4')
        
        # Advanced usage with OpenAI summarization
        result = transcribe_media(
            'video.mp4',
            backend='huggingface',
            model_size='large-v3',
            summary_provider='openai',
            summary_model='gpt-4o-mini'
        )
        
        # Using Anthropic Claude
        result = transcribe_media(
            'video.mp4',
            summary_provider='anthropic',
            summary_model='claude-3-5-sonnet-20241022'
        )
    """
    # Handle backward compatibility
    if ollama_model is not None:
        import warnings
        warnings.warn("ollama_model parameter is deprecated. Use summary_model with summary_provider='ollama'", 
                     DeprecationWarning, stacklevel=2)
        if summary_provider == 'ollama' and summary_model is None:
            summary_model = ollama_model
    
    if ollama_method is not None:
        import warnings
        warnings.warn("ollama_method parameter is deprecated. Use summary_method", 
                     DeprecationWarning, stacklevel=2)
        summary_method = ollama_method
    
    try:
        backend_enum = TranscriptionBackend(backend.lower())
    except ValueError:
        available = TranscriptionFactory.get_available_backends()
        raise ValueError(f"Invalid backend '{backend}'. Available options: {available}")
    
    transcriber = TranscriptionFactory.create_transcriber(backend_enum)
    
    return transcriber.process_media_file(
        media_file=media_file,
        output_txt=output_txt,
        summarize=summarize,
        summary_provider=summary_provider,
        summary_method=summary_method,
        summary_model=summary_model,
        **kwargs
    ) 