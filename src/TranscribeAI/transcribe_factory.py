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
                    ollama_model: str = 'llama3.2',
                    ollama_method: str = 'library',
                    **kwargs) -> Dict[str, str]:
    """Main entry point for media transcription with automatic backend selection.
    
    Args:
        media_file (str): Path to the media file to transcribe.
        backend (str): Transcription backend to use ('whisper' or 'huggingface').
        output_txt (str): Optional output file path for transcript.
        summarize (bool): Whether to generate a summary.
        ollama_model (str): Ollama model for summarization.
        ollama_method (str): Summarization method ('library' or 'api').
        **kwargs: Additional parameters specific to the chosen backend.
        
    Returns:
        dict: Dictionary containing paths to generated files.
        
    Example:
        # Basic usage with Whisper
        result = transcribe_media('video.mp4')
        
        # Advanced usage with HuggingFace backend
        result = transcribe_media(
            'video.mp4',
            backend='huggingface',
            model_size='large-v3',
            return_timestamps=True,
            language='english'
        )
    """
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
        ollama_model=ollama_model,
        ollama_method=ollama_method,
        **kwargs
    ) 