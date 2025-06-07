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

# Try to import optional libraries with graceful fallback
try:
    import ollama
    OLLAMA_LIBRARY_AVAILABLE = True
    logger.info("Ollama library detected - both API and library methods available")
except ImportError:
    OLLAMA_LIBRARY_AVAILABLE = False
    logger.info("Ollama library not found - only API method available")

try:
    import openai
    OPENAI_AVAILABLE = True
    logger.info("OpenAI library detected")
except ImportError:
    OPENAI_AVAILABLE = False
    logger.info("OpenAI library not found - install with: pip install openai")

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
    logger.info("Anthropic library detected")
except ImportError:
    ANTHROPIC_AVAILABLE = False
    logger.info("Anthropic library not found - install with: pip install anthropic")

try:
    from google import genai
    GEMINI_AVAILABLE = True
    logger.info("Google Gemini library detected")
except ImportError:
    GEMINI_AVAILABLE = False
    logger.info("Google Gemini library not found - install with: pip install google-genai")

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
    """Handles text summarization using multiple AI providers."""
    
    def __init__(self, ollama_url='http://localhost:11434', 
                 openai_api_key=None, anthropic_api_key=None, gemini_api_key=None):
        self.ollama_url = ollama_url
        self.openai_client = None
        self.anthropic_client = None
        self.gemini_client = None
        
        # Initialize OpenAI client
        if openai_api_key or OPENAI_AVAILABLE:
            try:
                self.openai_client = openai.OpenAI(api_key=openai_api_key or os.getenv('OPENAI_API_KEY'))
            except Exception as e:
                logger.warning(f"OpenAI client initialization failed: {e}")
        
        # Initialize Anthropic client  
        if anthropic_api_key or ANTHROPIC_AVAILABLE:
            try:
                self.anthropic_client = anthropic.Anthropic(api_key=anthropic_api_key or os.getenv('ANTHROPIC_API_KEY'))
            except Exception as e:
                logger.warning(f"Anthropic client initialization failed: {e}")
        
        # Initialize Gemini client
        if gemini_api_key or GEMINI_AVAILABLE:
            try:
                self.gemini_client = genai.Client(api_key=gemini_api_key or os.getenv('GEMINI_API_KEY'))
            except Exception as e:
                logger.warning(f"Gemini client initialization failed: {e}")
    
    def summarize_with_openai(self, text, output_file, model='gpt-4o-mini', max_tokens=1000):
        """Summarize text using OpenAI API.
        
        Args:
            text (str): Text to summarize.
            output_file (str): Path to save the summary.
            model (str): OpenAI model to use (e.g., 'gpt-4o-mini', 'gpt-4').
            max_tokens (int): Maximum tokens for response.
            
        Returns:
            str: Path to the summary file.
        """
        if not self.openai_client:
            raise ImportError("OpenAI client not available. Install with: pip install openai")
        
        logger.info(f'Summarizing text using OpenAI {model}...')
        prompt = self._create_summarization_prompt(text)
        
        try:
            response = self.openai_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0.3
            )
            
            summary = response.choices[0].message.content.strip()
            if not summary:
                raise ValueError("Empty response from OpenAI model")
            
            self._save_summary(summary, output_file, model, "OpenAI API")
            logger.info(f'Summary complete, saved as {output_file}')
            return output_file
            
        except Exception as e:
            logger.error(f"Error during OpenAI summarization: {str(e)}")
            raise
    
    def summarize_with_anthropic(self, text, output_file, model='claude-3-5-haiku-20241022', max_tokens=1000):
        """Summarize text using Anthropic Claude API.
        
        Args:
            text (str): Text to summarize.
            output_file (str): Path to save the summary.
            model (str): Claude model to use (e.g., 'claude-3-5-haiku-20241022', 'claude-3-5-sonnet-20241022').
            max_tokens (int): Maximum tokens for response.
            
        Returns:
            str: Path to the summary file.
        """
        if not self.anthropic_client:
            raise ImportError("Anthropic client not available. Install with: pip install anthropic")
        
        logger.info(f'Summarizing text using Anthropic {model}...')
        prompt = self._create_summarization_prompt(text)
        
        try:
            response = self.anthropic_client.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=0.3,
                messages=[{"role": "user", "content": prompt}]
            )
            
            summary = response.content[0].text.strip()
            if not summary:
                raise ValueError("Empty response from Anthropic model")
            
            self._save_summary(summary, output_file, model, "Anthropic API")
            logger.info(f'Summary complete, saved as {output_file}')
            return output_file
            
        except Exception as e:
            logger.error(f"Error during Anthropic summarization: {str(e)}")
            raise
    
    def summarize_with_gemini(self, text, output_file, model='gemini-2.0-flash', **kwargs):
        """Summarize text using Google Gemini API.
        
        Args:
            text (str): Text to summarize.
            output_file (str): Path to save the summary.  
            model (str): Gemini model to use (e.g., 'gemini-2.0-flash', 'gemini-1.5-flash', 'gemini-1.5-pro').
            **kwargs: Additional generation parameters.
            
        Returns:
            str: Path to the summary file.
        """
        if not self.gemini_client:
            raise ImportError("Google Gemini client not available. Install with: pip install google-genai")
        
        logger.info(f'Summarizing text using Google {model}...')
        prompt = self._create_summarization_prompt(text)
        
        try:
            response = self.gemini_client.models.generate_content(
                model=model,
                contents=prompt
            )
            
            summary = response.text.strip()
            if not summary:
                raise ValueError("Empty response from Gemini model")
            
            self._save_summary(summary, output_file, model, "Google Gemini API")
            logger.info(f'Summary complete, saved as {output_file}')
            return output_file
            
        except Exception as e:
            logger.error(f"Error during Gemini summarization: {str(e)}")
            raise
    
    def summarize_text(self, text, output_file, provider='ollama', method='library', **kwargs):
        """Universal summarization method that routes to appropriate provider.
        
        Args:
            text (str): Text to summarize.
            output_file (str): Path to save the summary.
            provider (str): Provider to use ('ollama', 'openai', 'anthropic', 'gemini').
            method (str): Method for Ollama ('library' or 'api'), ignored for others.
            **kwargs: Provider-specific parameters.
            
        Returns:
            str: Path to the summary file.
        """
        provider = provider.lower()
        
        if provider == 'ollama':
            model = kwargs.get('model', 'llama3.2')
            if method.lower() == 'library':
                return self.summarize_with_library(text, output_file, model)
            else:
                return self.summarize_with_api(text, output_file, model)
        elif provider == 'openai':
            model = kwargs.get('model', 'gpt-4o-mini')
            max_tokens = kwargs.get('max_tokens', 1000)
            return self.summarize_with_openai(text, output_file, model, max_tokens)
        elif provider == 'anthropic':
            model = kwargs.get('model', 'claude-3-5-haiku-20241022')
            max_tokens = kwargs.get('max_tokens', 1000)
            return self.summarize_with_anthropic(text, output_file, model, max_tokens)
        elif provider == 'gemini':
            model = kwargs.get('model', 'gemini-2.0-flash')
            return self.summarize_with_gemini(text, output_file, model, **kwargs)
        else:
            raise ValueError(f"Unsupported provider: {provider}. Choose from: ollama, openai, anthropic, gemini")
    
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
                          summary_provider='ollama', summary_method='library', summary_model=None, **kwargs):
        """Complete pipeline: convert media to audio, transcribe, and optionally summarize.
        
        Args:
            media_file (str): Path to the media file.
            output_txt (str): Path for transcript output.
            summarize (bool): Whether to generate summary.
            summary_provider (str): AI provider for summarization ('ollama', 'openai', 'anthropic', 'gemini').
            summary_method (str): Method for Ollama ('library' or 'api'), ignored for others.
            summary_model (str): Model to use for summarization (provider-specific defaults if None).
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
                    transcript_file, summary_file, summary_provider, summary_method, summary_model
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
    
    def _generate_summary(self, transcript_file, summary_file, provider, method, model):
        """Generate summary from transcript file using specified provider."""
        try:
            with open(transcript_file, 'r', encoding='utf-8') as f:
                transcript_text = f.read()
            
            if len(transcript_text.strip()) < 50:
                logger.warning("Transcript too short for summarization")
                return None
            
            # Set default models if not specified
            provider_defaults = {
                'ollama': 'llama3.2',
                'openai': 'gpt-4o-mini', 
                'anthropic': 'claude-3-5-haiku-20241022',
                'gemini': 'gemini-2.0-flash'
            }
            
            if model is None:
                model = provider_defaults.get(provider.lower(), 'llama3.2')
            
            return self.summarizer.summarize_text(
                transcript_text, 
                summary_file, 
                provider=provider, 
                method=method, 
                model=model
            )
                
        except Exception as e:
            logger.warning(f"Summarization with {provider} failed: {str(e)}")
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