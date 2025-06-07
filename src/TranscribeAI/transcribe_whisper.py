import whisper
import logging
from .transcribe_base import BaseTranscriber, logger

class WhisperTranscriber(BaseTranscriber):
    """Whisper-based transcription implementation using OpenAI Whisper."""
    
    def __init__(self):
        super().__init__()
        self.model_cache = {}  # Cache loaded models for reuse
    
    def transcribe_audio(self, audio_file, output_txt='transcript.txt', model_size='base'):
        """Transcribe audio file using Whisper with GPU acceleration if available.
        
        Args:
            audio_file (str): Path to audio file to transcribe.
            output_txt (str): Output file path for transcript.
            model_size (str): Whisper model size to use ('tiny', 'base', 'small', 'medium', 'large').
            
        Returns:
            str: Path to the transcript file.
        """
        logger.info(f'Transcribing {audio_file} using Whisper on {self.device_manager.device.upper()}...')
        
        try:
            model = self._load_model(model_size)
            result = model.transcribe(audio_file)
            transcript = result['text']
            
            self._save_transcript(transcript, output_txt, model_size)
            logger.info(f'Transcription complete, saved as {output_txt}')
            return output_txt
            
        except Exception as e:
            return self._handle_transcription_error(e, audio_file, output_txt, model_size)
    
    def _load_model(self, model_size):
        """Load Whisper model with caching and device optimization."""
        cache_key = f"{model_size}_{self.device_manager.device}"
        
        if cache_key not in self.model_cache:
            logger.info(f"Loading Whisper model: {model_size}")
            self.model_cache[cache_key] = whisper.load_model(
                model_size, 
                device=self.device_manager.device
            )
        
        return self.model_cache[cache_key]
    
    def _save_transcript(self, transcript, output_txt, model_size):
        """Save transcript with metadata."""
        with open(output_txt, 'w', encoding='utf-8') as f:
            f.write(f"=== TRANSCRIPT (OpenAI Whisper) ===\n")
            f.write(f"Model: {model_size}\n")
            f.write(f"Device: {self.device_manager.device}\n")
            f.write("=" * 50 + "\n\n")
            f.write(transcript)
    
    def _handle_transcription_error(self, error, audio_file, output_txt, model_size):
        """Handle transcription errors with fallback to CPU."""
        if self.device_manager.device == "cuda":
            logger.warning(f"GPU transcription failed: {str(error)}")
            logger.warning("Falling back to CPU transcription...")
            
            # Fallback to CPU
            model = whisper.load_model(model_size, device="cpu")
            result = model.transcribe(audio_file)
            transcript = result['text']
            
            self._save_transcript(transcript, output_txt, model_size)
            logger.info(f'Transcription complete (CPU fallback), saved as {output_txt}')
            return output_txt
        else:
            raise error 