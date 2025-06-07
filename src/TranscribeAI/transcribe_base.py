import os
import logging
import requests
from abc import ABC, abstractmethod
from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.audio.io.AudioFileClip import AudioFileClip
import torch

# Configure logging
def setup_logging():
    """Setup consistent logging configuration across all modules."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    return logging.getLogger(__name__)

logger = setup_logging()

# Try to import ollama library
try:
    import ollama
    OLLAMA_LIBRARY_AVAILABLE = True
    logger.info("Ollama library detected - both API and library methods available")
except ImportError:
    OLLAMA_LIBRARY_AVAILABLE = False
    logger.info("Ollama library not found - only API method available")

# Constants
AUDIO_EXTENSIONS = {'.wav', '.mp3', '.aac', '.flac', '.ogg', '.m4a', '.wma', '.aiff'}
VIDEO_EXTENSIONS = {'.mp4', '.mov', '.avi', '.mkv', '.webm', '.wmv', '.flv', '.m4v', '.3gp', '.mpg', '.mpeg', '.ts', '.mts'}
SUPPORTED_FORMATS = AUDIO_EXTENSIONS | VIDEO_EXTENSIONS

class DeviceManager:
    """Manages CUDA/CPU device selection and provides device information."""
    
    def __init__(self):
        self.cuda_available = torch.cuda.is_available()
        self.device = "cuda" if self.cuda_available else "cpu"
        self.torch_dtype = torch.float16 if self.cuda_available else torch.float32
        self._log_device_info()
    
    def _log_device_info(self):
        """Log device information for debugging."""
        if self.cuda_available:
            logger.info(f"CUDA detected - GPU acceleration enabled (Device: {torch.cuda.get_device_name()})")
            logger.info(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        else:
            logger.info("CUDA not available - using CPU for transcription")
        logger.info(f"PyTorch version: {torch.__version__}")

class MediaProcessor:
    """Handles media file processing and audio extraction."""
    
    @staticmethod
    def media_to_audio(media_file, audio_file='audio.wav'):
        """Convert any media file (video/audio) to audio format.
        
        Args:
            media_file (str): Path to the media file to convert.
            audio_file (str): Output audio file path.
            
        Returns:
            str: Path to the converted audio file.
        """
        logger.info(f'Processing {media_file}...')
        
        file_ext = os.path.splitext(media_file)[1].lower()
        
        if file_ext not in SUPPORTED_FORMATS:
            supported_formats = sorted(SUPPORTED_FORMATS)
            logger.warning(f"Warning: '{file_ext}' may not be supported.")
            logger.warning(f"Supported formats: {', '.join(supported_formats)}")
            logger.warning("Attempting conversion anyway...")
        
        if file_ext in AUDIO_EXTENSIONS:
            logger.info(f'{media_file} is already an audio file.')
            if file_ext == '.wav' and audio_file.endswith('.wav'):
                logger.info('No conversion needed.')
                return media_file
            else:
                logger.info(f'Converting {file_ext} to .wav format...')
                audio_clip = AudioFileClip(media_file)
                audio_clip.write_audiofile(audio_file)
                audio_clip.close()
        else:
            logger.info(f'Extracting audio from {file_ext} video file...')
            video_clip = VideoFileClip(media_file)
            video_clip.audio.write_audiofile(audio_file)
            video_clip.close()
        
        logger.info('Audio processing complete.')
        return audio_file

class SummarizationEngine:
    """Handles text summarization using Ollama."""
    
    def __init__(self, ollama_url='http://localhost:11434'):
        self.ollama_url = ollama_url
    
    def summarize_with_library(self, text, output_file, model='llama3.2'):
        """Summarize text using Ollama Python library.
        
        Args:
            text (str): Text to summarize.
            output_file (str): Path to save the summary.
            model (str): Ollama model name to use.
            
        Returns:
            str: Path to the summary file.
        """
        if not OLLAMA_LIBRARY_AVAILABLE:
            raise ImportError("Ollama library not installed. Install with: pip install ollama")
        
        logger.info(f'Summarizing text using Ollama library with model: {model}...')
        
        prompt = self._create_summarization_prompt(text)
        
        try:
            response = ollama.generate(model=model, prompt=prompt)
            summary = response.get('response', '').strip()
            
            if not summary:
                raise ValueError("Empty response from Ollama model")
            
            self._save_summary(summary, output_file, model, "Library Method")
            logger.info(f'Summary complete, saved as {output_file}')
            return output_file
            
        except ollama.ResponseError as e:
            logger.error(f"Ollama model error: {str(e)}")
            logger.error("Make sure the model is installed (ollama pull llama3.2)")
            raise
        except Exception as e:
            logger.error(f"Error during summarization with library: {str(e)}")
            raise
    
    def summarize_with_api(self, text, output_file, model='llama3.2'):
        """Summarize text using Ollama REST API.
        
        Args:
            text (str): Text to summarize.
            output_file (str): Path to save the summary.
            model (str): Ollama model name to use.
            
        Returns:
            str: Path to the summary file.
        """
        logger.info(f'Summarizing text using Ollama API with model: {model}...')
        
        prompt = self._create_summarization_prompt(text)
        
        try:
            payload = {"model": model, "prompt": prompt, "stream": False}
            response = requests.post(f"{self.ollama_url}/api/generate", json=payload, timeout=300)
            response.raise_for_status()
            
            result = response.json()
            summary = result.get('response', '').strip()
            
            if not summary:
                raise ValueError("Empty response from Ollama model")
            
            self._save_summary(summary, output_file, model, "API Method")
            logger.info(f'Summary complete, saved as {output_file}')
            return output_file
            
        except requests.exceptions.ConnectionError:
            logger.error("Cannot connect to Ollama. Make sure Ollama is running (ollama serve)")
            logger.error("Install Ollama from: https://ollama.ai")
            raise
        except requests.exceptions.Timeout:
            logger.error("Ollama request timed out. The text might be too long or model is slow")
            raise
        except Exception as e:
            logger.error(f"Error during summarization: {str(e)}")
            raise
    
    def _create_summarization_prompt(self, text):
        """Create a comprehensive summarization prompt."""
        return f"""Please provide a comprehensive summary of the following transcript. Include:
1. Main topics discussed
2. Key points and important information
3. Any conclusions or decisions mentioned
4. Overall context and purpose

Transcript:
{text}

Summary:"""
    
    def _save_summary(self, summary, output_file, model, method):
        """Save summary to file with metadata."""
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"=== TRANSCRIPT SUMMARY ({method}) ===\n")
            f.write(f"Model: {model}\n")
            f.write(f"Generated: {os.path.basename(output_file)}\n")
            f.write("=" * 50 + "\n\n")
            f.write(summary)

class BaseTranscriber(ABC):
    """Abstract base class for transcription implementations."""
    
    def __init__(self):
        self.device_manager = DeviceManager()
        self.media_processor = MediaProcessor()
        self.summarizer = SummarizationEngine()
    
    @abstractmethod
    def transcribe_audio(self, audio_file, output_txt, **kwargs):
        """Transcribe audio file to text.
        
        Args:
            audio_file (str): Path to audio file.
            output_txt (str): Output transcript file path.
            **kwargs: Additional transcription parameters.
            
        Returns:
            str: Path to the transcript file.
        """
        pass
    
    def process_media_file(self, media_file, output_txt=None, summarize=True, 
                          ollama_model='llama3.2', ollama_method='library', **kwargs):
        """Complete pipeline: convert media to audio, transcribe, and optionally summarize.
        
        Args:
            media_file (str): Path to the media file.
            output_txt (str): Path for transcript output.
            summarize (bool): Whether to generate summary.
            ollama_model (str): Ollama model for summarization.
            ollama_method (str): Method to use - 'library' or 'api'.
            **kwargs: Additional parameters for transcription.
            
        Returns:
            dict: Paths to generated files.
        """
        if not os.path.exists(media_file):
            raise FileNotFoundError(f"Media file not found: {media_file}")
        
        self._log_file_info(media_file)
        base_name = os.path.splitext(os.path.basename(media_file))[0]
        audio_file = f"{base_name}_audio.wav"
        
        if output_txt is None:
            output_txt = f"{base_name}_transcript.txt"
        
        try:
            # Convert to audio
            audio = self.media_processor.media_to_audio(media_file, audio_file)
            
            # Transcribe
            transcript_file = self.transcribe_audio(audio, output_txt, **kwargs)
            
            result = {'transcript': transcript_file, 'audio': audio_file}
            
            # Summarize if requested
            if summarize:
                summary_file = f"summary_{base_name}.txt"
                result['summary'] = self._generate_summary(
                    transcript_file, summary_file, ollama_model, ollama_method
                )
            
            return result
        
        except Exception as e:
            self._handle_processing_error(e, media_file)
            raise
    
    def _log_file_info(self, media_file):
        """Log media file information."""
        file_ext = os.path.splitext(media_file)[1].lower()
        file_size_mb = os.path.getsize(media_file) / (1024 * 1024)
        logger.info(f"\nFile: {os.path.basename(media_file)}")
        logger.info(f"Format: {file_ext.upper()[1:]} ({file_size_mb:.1f} MB)")
    
    def _generate_summary(self, transcript_file, summary_file, ollama_model, ollama_method):
        """Generate summary from transcript file."""
        try:
            with open(transcript_file, 'r', encoding='utf-8') as f:
                transcript_text = f.read()
            
            if len(transcript_text.strip()) > 50:
                if ollama_method.lower() == 'library':
                    if OLLAMA_LIBRARY_AVAILABLE:
                        return self.summarizer.summarize_with_library(transcript_text, summary_file, ollama_model)
                    else:
                        logger.warning("Ollama library not available, falling back to API method")
                        return self.summarizer.summarize_with_api(transcript_text, summary_file, ollama_model)
                else:
                    return self.summarizer.summarize_with_api(transcript_text, summary_file, ollama_model)
            else:
                logger.warning("Transcript too short for summarization")
                return None
                
        except Exception as e:
            logger.warning(f"Summarization failed: {str(e)}")
            logger.warning("Continuing without summary...")
            return None
    
    def _handle_processing_error(self, error, media_file):
        """Handle and log processing errors with helpful tips."""
        file_ext = os.path.splitext(media_file)[1].lower()
        logger.error(f"\n❌ Error processing {file_ext.upper()} file:")
        logger.error(f"   {str(error)}")
        
        if file_ext == '.wmv':
            logger.warning("   💡 Tip: WMV files may require additional codecs. Try converting to MP4 first.")
        elif file_ext not in SUPPORTED_FORMATS:
            logger.warning("   💡 Tip: This format may not be supported. Try a common format like MP4, MOV, or MP3.") 