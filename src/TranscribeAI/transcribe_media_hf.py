import os
from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.audio.io.AudioFileClip import AudioFileClip
import torch
import logging
import requests
import json
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import ollama library, but don't fail if not available
try:
    import ollama
    OLLAMA_LIBRARY_AVAILABLE = True
    logger.info("Ollama library detected - both API and library methods available")
except ImportError:
    OLLAMA_LIBRARY_AVAILABLE = False
    logger.info("Ollama library not found - only API method available")

# Check CUDA availability
CUDA_AVAILABLE = torch.cuda.is_available()
if CUDA_AVAILABLE:
    DEVICE = "cuda"
    TORCH_DTYPE = torch.float16
    logger.info(f"CUDA detected - GPU acceleration enabled (Device: {torch.cuda.get_device_name()})")
    logger.info(f"CUDA device count: {torch.cuda.device_count()}")
    logger.info(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA version: {torch.version.cuda}")
else:
    DEVICE = "cpu"
    TORCH_DTYPE = torch.float32
    logger.info("CUDA not available - using CPU for transcription")
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info("CUDA diagnostics:")
    logger.info(f"  torch.cuda.is_available(): {torch.cuda.is_available()}")
    logger.info(f"  torch.version.cuda: {torch.version.cuda}")
    logger.info(f"  torch.backends.cudnn.enabled: {torch.backends.cudnn.enabled}")

AUDIO_EXTENSIONS = {'.wav', '.mp3', '.aac', '.flac', '.ogg', '.m4a', '.wma', '.aiff'}

VIDEO_EXTENSIONS = {'.mp4', '.mov', '.avi', '.mkv', '.webm', '.wmv', '.flv', '.m4v', '.3gp', '.mpg', '.mpeg', '.ts', '.mts'}

# Available Whisper models on Hugging Face
WHISPER_MODELS = {
    'tiny': 'openai/whisper-tiny',
    'base': 'openai/whisper-base', 
    'small': 'openai/whisper-small',
    'medium': 'openai/whisper-medium',
    'large': 'openai/whisper-large-v2',
    'large-v3': 'openai/whisper-large-v3'
}

def media_to_audio(media_file, audio_file='audio.wav'):
    """Convert any media file (video/audio) to audio format.
    
    Args:
        media_file (str): Path to the media file to convert.
        audio_file (str): Output audio file path.
        
    Returns:
        str: Path to the converted audio file.
    """
    logger.info(f'Processing {media_file}...')
    
    # Get file extension
    file_ext = os.path.splitext(media_file)[1].lower()
    
    # Check if file format is supported
    if file_ext not in AUDIO_EXTENSIONS and file_ext not in VIDEO_EXTENSIONS:
        supported_formats = sorted(AUDIO_EXTENSIONS | VIDEO_EXTENSIONS)
        logger.warning(f"Warning: '{file_ext}' may not be supported.")
        logger.warning(f"Supported formats: {', '.join(supported_formats)}")
        logger.warning("Attempting conversion anyway...")
    
    # If it's already an audio file, we can either copy it or load it directly
    if file_ext in AUDIO_EXTENSIONS:
        logger.info(f'{media_file} is already an audio file.')
        # If it's already wav, we can skip conversion
        if file_ext == '.wav' and audio_file.endswith('.wav'):
            logger.info('No conversion needed.')
            return media_file
        else:
            # Convert audio format if needed
            logger.info(f'Converting {file_ext} to .wav format...')
            audio_clip = AudioFileClip(media_file)
            audio_clip.write_audiofile(audio_file)
            audio_clip.close()
    else:
        # It's a video file, extract audio
        logger.info(f'Extracting audio from {file_ext} video file...')
        video_clip = VideoFileClip(media_file)
        video_clip.audio.write_audiofile(audio_file)
        video_clip.close()
    
    logger.info('Audio processing complete.')
    return audio_file

def load_whisper_pipeline(model_size='base', use_chunked=True, chunk_length_s=30):
    """Load Whisper model and create transcription pipeline.
    
    Args:
        model_size (str): Whisper model size to use.
        use_chunked (bool): Whether to use chunked processing for long audio.
        chunk_length_s (int): Chunk length in seconds for long-form audio.
        
    Returns:
        pipeline: Configured Whisper transcription pipeline.
    """
    global DEVICE, TORCH_DTYPE
    
    model_id = WHISPER_MODELS.get(model_size, WHISPER_MODELS['base'])
    logger.info(f'Loading Whisper model: {model_id} on {DEVICE}...')
    
    # Add device verification
    logger.info(f'Current device setting: {DEVICE}')
    if DEVICE == "cuda":
        logger.info(f'Available CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
        if torch.cuda.is_available():
            logger.info('CUDA is available for model loading')
        else:
            logger.warning('CUDA was detected at startup but is no longer available!')
    
    try:
        # Load model and processor
        logger.info('Loading model from Hugging Face...')
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id,
            torch_dtype=TORCH_DTYPE,
            low_cpu_mem_usage=True,
            use_safetensors=True
        )
        logger.info(f'Model loaded successfully, moving to device: {DEVICE}')
        model.to(DEVICE)
        
        # Verify model is on correct device
        if hasattr(model, 'device'):
            logger.info(f'Model device after .to(): {model.device}')
        
        logger.info('Loading processor...')
        processor = AutoProcessor.from_pretrained(model_id)
        
        # Create pipeline
        logger.info('Creating pipeline...')
        pipe_kwargs = {
            "model": model,
            "tokenizer": processor.tokenizer,
            "feature_extractor": processor.feature_extractor,
            "torch_dtype": TORCH_DTYPE,
            "device": DEVICE,
        }
        
        # Add chunking for long-form audio
        if use_chunked:
            pipe_kwargs["chunk_length_s"] = chunk_length_s
        
        pipe = pipeline("automatic-speech-recognition", **pipe_kwargs)
        
        # Verify pipeline device
        if hasattr(pipe.model, 'device'):
            logger.info(f'Pipeline model device: {pipe.model.device}')
        
        logger.info(f'Whisper pipeline ready with {"chunked" if use_chunked else "sequential"} processing')
        return pipe
        
    except Exception as e:
        logger.error(f"Failed to load Whisper model: {str(e)}")
        logger.error(f"Error type: {type(e).__name__}")
        if DEVICE == "cuda":
            logger.warning("Falling back to CPU...")
            logger.info("CUDA failure reason analysis:")
            if "CUDA out of memory" in str(e):
                logger.warning("  → CUDA memory insufficient for this model size")
                logger.warning(f"  → Try a smaller model (current: {model_size})")
            elif "CUDA" in str(e):
                logger.warning("  → CUDA-related error during model loading")
            else:
                logger.warning(f"  → Non-CUDA error: {str(e)}")
            DEVICE = "cpu"
            TORCH_DTYPE = torch.float32
            return load_whisper_pipeline(model_size, use_chunked, chunk_length_s)
        else:
            raise

def transcribe_audio_hf(audio_file, output_txt='transcript.txt', model_size='base', 
                       return_timestamps=False, language=None, task='transcribe',
                       use_chunked=True, batch_size=None):
    """Transcribe audio file using Hugging Face Transformers Whisper implementation.
    
    Args:
        audio_file (str): Path to audio file to transcribe.
        output_txt (str): Output file path for transcript.
        model_size (str): Whisper model size to use.
        return_timestamps (bool or str): Whether to return timestamps ('word' for word-level).
        language (str): Source audio language (e.g., 'english', 'french').
        task (str): Task type - 'transcribe' or 'translate'.
        use_chunked (bool): Use chunked processing for long audio.
        batch_size (int): Batch size for processing multiple chunks.
        
    Returns:
        str: Path to the transcript file.
    """
    global DEVICE, TORCH_DTYPE
    
    logger.info(f'Transcribing {audio_file} using HF Transformers Whisper on {DEVICE}...')
    
    try:
        # Load pipeline
        pipe = load_whisper_pipeline(model_size, use_chunked)
        
        # Prepare generation kwargs
        generate_kwargs = {
            "max_new_tokens": 448,
            "num_beams": 1,
            "condition_on_prev_tokens": False,
            "compression_ratio_threshold": 1.35,
            "temperature": (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
            "logprob_threshold": -1.0,
            "no_speech_threshold": 0.6,
        }
        
        # Add language and task if specified
        if language:
            generate_kwargs["language"] = language
        if task:
            generate_kwargs["task"] = task
            
        # Configure pipeline kwargs
        pipe_kwargs = {"generate_kwargs": generate_kwargs}
        if return_timestamps:
            pipe_kwargs["return_timestamps"] = return_timestamps
        if batch_size:
            pipe_kwargs["batch_size"] = batch_size
        
        # Transcribe
        result = pipe(audio_file, **pipe_kwargs)
        
        # Extract text and format output
        if isinstance(result, dict):
            transcript = result.get('text', '')
            chunks = result.get('chunks', [])
        else:
            transcript = str(result)
            chunks = []
        
        # Save transcript
        with open(output_txt, 'w', encoding='utf-8') as f:
            f.write(f"=== TRANSCRIPT (HF Transformers Whisper) ===\n")
            f.write(f"Model: {model_size} ({WHISPER_MODELS.get(model_size, 'unknown')})\n")
            f.write(f"Device: {DEVICE}\n")
            f.write(f"Language: {language or 'auto-detected'}\n")
            f.write(f"Task: {task}\n")
            f.write(f"Timestamps: {return_timestamps}\n")
            f.write("=" * 50 + "\n\n")
            f.write(transcript)
            
            # Add timestamp information if available
            if chunks and return_timestamps:
                f.write("\n\n=== TIMESTAMPS ===\n")
                for chunk in chunks:
                    timestamp = chunk.get('timestamp', '')
                    text = chunk.get('text', '')
                    if timestamp:
                        f.write(f"[{timestamp[0]:.2f} - {timestamp[1]:.2f}] {text}\n")

        logger.info(f'Transcription complete, saved as {output_txt}')
        return output_txt
        
    except Exception as e:
        logger.error(f"Transcription failed: {str(e)}")
        if DEVICE == "cuda":
            logger.warning("GPU transcription failed, falling back to CPU...")
            DEVICE = "cpu"
            TORCH_DTYPE = torch.float32
            return transcribe_audio_hf(audio_file, output_txt, model_size, 
                                     return_timestamps, language, task, use_chunked, batch_size)
        else:
            raise

def summarize_text_with_library(text, output_file, model='llama3.2'):
    """Summarize text using Ollama Python library and save to file.
    
    Args:
        text (str): The text to summarize.
        output_file (str): Path to save the summary.
        model (str): Ollama model name to use.
        
    Returns:
        str: Path to the summary file.
    """
    if not OLLAMA_LIBRARY_AVAILABLE:
        raise ImportError("Ollama library not installed. Install with: pip install ollama")
    
    logger.info(f'Summarizing text using Ollama library with model: {model}...')
    
    # Create summarization prompt
    prompt = f"""Please provide a comprehensive summary of the following transcript. Include:
                1. Main topics discussed
                2. Key points and important information
                3. Any conclusions or decisions mentioned
                4. Overall context and purpose

                Transcript:
                {text}

                Summary:"""

    try:
        # Use Ollama library to generate summary
        response = ollama.generate(
            model=model,
            prompt=prompt
        )
        
        summary = response.get('response', '').strip()
        
        if not summary:
            raise ValueError("Empty response from Ollama model")
        
        # Save summary to file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"=== TRANSCRIPT SUMMARY (HF + Library Method) ===\n")
            f.write(f"Model: {model}\n")
            f.write(f"Generated: {os.path.basename(output_file)}\n")
            f.write("=" * 50 + "\n\n")
            f.write(summary)
        
        logger.info(f'Summary complete, saved as {output_file}')
        return output_file
        
    except ollama.ResponseError as e:
        logger.error(f"Ollama model error: {str(e)}")
        logger.error("Make sure the model is installed (ollama pull llama3.2)")
        raise
    except Exception as e:
        logger.error(f"Error during summarization with library: {str(e)}")
        raise

def summarize_text(text, output_file, model='llama3.2', ollama_url='http://localhost:11434'):
    """Summarize text using Ollama REST API and save to file.
    
    Args:
        text (str): The text to summarize.
        output_file (str): Path to save the summary.
        model (str): Ollama model name to use.
        ollama_url (str): Ollama server URL.
        
    Returns:
        str: Path to the summary file.
    """
    logger.info(f'Summarizing text using Ollama API with model: {model}...')
    
    # Create summarization prompt
    prompt = f"""
    Please provide a comprehensive summary of the following transcript. Include:
    1. Main topics discussed
    2. Key points and important information
    3. Any conclusions or decisions mentioned
    4. Overall context and purpose

    Transcript:
    {text}

    Summary:
    """

    try:
        # Prepare request for Ollama API
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False
        }
        
        # Make request to Ollama
        response = requests.post(f"{ollama_url}/api/generate", json=payload, timeout=300)
        response.raise_for_status()
        
        # Extract summary from response
        result = response.json()
        summary = result.get('response', '').strip()
        
        if not summary:
            raise ValueError("Empty response from Ollama model")
        
        # Save summary to file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"Model: {model}\n")
            f.write(f"Generated: {os.path.basename(output_file)}\n")
            f.write(summary)
        
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

def process_media_file_hf(media_file, output_txt=None, model_size='base', 
                         summarize=True, ollama_model='llama3.2', ollama_method='library',
                         return_timestamps=False, language=None, task='transcribe',
                         use_chunked=True, batch_size=None):
    """Complete pipeline: convert media to audio, transcribe with HF Transformers, and optionally summarize.
    
    Args:
        media_file (str): Path to the media file.
        output_txt (str): Path for transcript output.
        model_size (str): Whisper model size to use.
        summarize (bool): Whether to generate summary.
        ollama_model (str): Ollama model for summarization.
        ollama_method (str): Method to use - 'library' or 'api'.
        return_timestamps (bool or str): Whether to return timestamps.
        language (str): Source audio language.
        task (str): Task type - 'transcribe' or 'translate'.
        use_chunked (bool): Use chunked processing for long audio.
        batch_size (int): Batch size for processing.
        
    Returns:
        dict: Paths to generated files.
    """
    if not os.path.exists(media_file):
        raise FileNotFoundError(f"Media file not found: {media_file}")
    
    # Display file info
    file_ext = os.path.splitext(media_file)[1].lower()
    file_size_mb = os.path.getsize(media_file) / (1024 * 1024)
    logger.info(f"\nFile: {os.path.basename(media_file)}")
    logger.info(f"Format: {file_ext.upper()[1:]} ({file_size_mb:.1f} MB)")
    logger.info(f"Processing method: HF Transformers ({'chunked' if use_chunked else 'sequential'})")
    
    # Generate output filenames based on input filename
    base_name = os.path.splitext(os.path.basename(media_file))[0]
    audio_file = f"{base_name}_audio.wav"
    if output_txt is None:
        output_txt = f"{base_name}_transcript_hf.txt"
    summary_file = f"summary_hf_{base_name}.txt"
    
    try:
        # Convert to audio
        audio = media_to_audio(media_file, audio_file)
        
        # Transcribe with HF Transformers
        transcript_file = transcribe_audio_hf(
            audio, output_txt, model_size, return_timestamps, 
            language, task, use_chunked, batch_size
        )
        
        result = {
            'transcript': transcript_file,
            'audio': audio_file
        }
        
        # Summarize if requested
        if summarize:
            try:
                # Read transcript for summarization
                with open(transcript_file, 'r', encoding='utf-8') as f:
                    transcript_text = f.read()
                
                if len(transcript_text.strip()) > 50:  # Only summarize if there's substantial content
                    # Choose summarization method
                    if ollama_method.lower() == 'library':
                        if OLLAMA_LIBRARY_AVAILABLE:
                            summary_path = summarize_text_with_library(transcript_text, summary_file, ollama_model)
                        else:
                            logger.warning("Ollama library not available, falling back to API method")
                            summary_path = summarize_text(transcript_text, summary_file, ollama_model)
                    else:  # API method
                        summary_path = summarize_text(transcript_text, summary_file, ollama_model)
                    
                    result['summary'] = summary_path
                    logger.info(f"📋 Summary saved as: {summary_path}")
                else:
                    logger.warning("Transcript too short for summarization")
                    
            except Exception as e:
                logger.warning(f"Summarization failed: {str(e)}")
                logger.warning("Continuing without summary...")
        
        return result
    
    except Exception as e:
        logger.error(f"\n❌ Error processing {file_ext.upper()} file:")
        logger.error(f"   {str(e)}")
        if file_ext == '.wmv':
            logger.warning("   💡 Tip: WMV files may require additional codecs. Try converting to MP4 first.")
        elif file_ext not in AUDIO_EXTENSIONS and file_ext not in VIDEO_EXTENSIONS:
            logger.warning("   💡 Tip: This format may not be supported. Try a common format like MP4, MOV, or MP3.")
        raise

if __name__ == "__main__":
    media_file = r'examples\files\2025-03-19 12-16-11.mkv'  # Could be .mp4, .avi, .mp3, etc.
    try:
        result = process_media_file_hf(
            media_file,
            model_size='small',
            summarize=True,
            ollama_model='phi4-reasoning:latest',
            ollama_method='library',
            use_chunked=True,
            return_timestamps=True
        )
        
        logger.info(f"\nProcess completed successfully!")
        logger.info(f"Transcript: {result['transcript']}")
        if 'summary' in result:
            logger.info(f"Summary: {result['summary']}")
        
        # Example 2: French to English translation with word-level timestamps
        # result = process_media_file_hf(
        #     media_file,
        #     model_size='large-v3',
        #     language='french',
        #     task='translate',
        #     return_timestamps='word',
        #     summarize=False
        # )
        
        # Example 3: Batch processing optimization
        # result = process_media_file_hf(
        #     media_file,
        #     model_size='medium',
        #     use_chunked=True,
        #     batch_size=2,
        #     summarize=True
        # )
        
    except Exception as e:
        logger.error(f"\nError: {e}") 