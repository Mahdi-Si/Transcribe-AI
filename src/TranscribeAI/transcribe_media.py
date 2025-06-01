import os
from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.audio.io.AudioFileClip import AudioFileClip
import whisper
import torch
import logging
import requests
import json

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
    logger.info(f"CUDA detected - GPU acceleration enabled (Device: {torch.cuda.get_device_name()})")
else:
    DEVICE = "cpu"
    logger.info("CUDA not available - using CPU for transcription")

AUDIO_EXTENSIONS = {'.wav', '.mp3', '.aac', '.flac', '.ogg', '.m4a', '.wma', '.aiff'}

VIDEO_EXTENSIONS = {'.mp4', '.mov', '.avi', '.mkv', '.webm', '.wmv', '.flv', '.m4v', '.3gp', '.mpg', '.mpeg', '.ts', '.mts'}

def media_to_audio(media_file, audio_file='audio.wav'):
    """Convert any media file (video/audio) to audio format."""
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

def transcribe_audio(audio_file, output_txt='transcript.txt', model_size='base'):
    """Transcribe audio file using Whisper with GPU acceleration if available.
    
    Args:
        audio_file (str): Path to audio file to transcribe.
        output_txt (str): Output file path for transcript.
        model_size (str): Whisper model size to use.
        
    Returns:
        str: Path to the transcript file.
    """
    logger.info(f'Transcribing {audio_file} using Whisper on {DEVICE.upper()}...')
    
    try:
        # Load model with device specification
        model = whisper.load_model(model_size, device=DEVICE)
        result = model.transcribe(audio_file)
        transcript = result['text']

        with open(output_txt, 'w', encoding='utf-8') as f:
            f.write(transcript)

        logger.info(f'Transcription complete, saved as {output_txt}')
        return output_txt
        
    except Exception as e:
        if DEVICE == "cuda":
            logger.warning(f"GPU transcription failed: {str(e)}")
            logger.warning("Falling back to CPU transcription...")
            # Fallback to CPU
            model = whisper.load_model(model_size, device="cpu")
            result = model.transcribe(audio_file)
            transcript = result['text']

            with open(output_txt, 'w', encoding='utf-8') as f:
                f.write(transcript)

            logger.info(f'Transcription complete (CPU fallback), saved as {output_txt}')
            return output_txt
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
            f.write(f"=== TRANSCRIPT SUMMARY (Library Method) ===\n")
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
    prompt = f"""Please provide a comprehensive summary of the following transcript. Include:
1. Main topics discussed
2. Key points and important information
3. Any conclusions or decisions mentioned
4. Overall context and purpose

Transcript:
{text}

Summary:"""

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
            f.write(f"=== TRANSCRIPT SUMMARY (API Method) ===\n")
            f.write(f"Model: {model}\n")
            f.write(f"Generated: {os.path.basename(output_file)}\n")
            f.write("=" * 50 + "\n\n")
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

def process_media_file(media_file, output_txt=None, model_size='base', 
                      summarize=True, ollama_model='llama3.2', ollama_method='library'):
    """Complete pipeline: convert media to audio, transcribe, and optionally summarize.
    
    Args:
        media_file (str): Path to the media file.
        output_txt (str): Path for transcript output.
        model_size (str): Whisper model size.
        summarize (bool): Whether to generate summary.
        ollama_model (str): Ollama model for summarization.
        ollama_method (str): Method to use - 'library' or 'api'.
        
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
    
    # Generate output filenames based on input filename
    base_name = os.path.splitext(os.path.basename(media_file))[0]
    audio_file = f"{base_name}_audio.wav"
    if output_txt is None:
        output_txt = f"{base_name}_transcript.txt"
    summary_file = f"summary_{base_name}.txt"
    
    try:
        # Convert to audio
        audio = media_to_audio(media_file, audio_file)
        
        # Transcribe
        transcript_file = transcribe_audio(audio, output_txt, model_size)
        
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
    # Example usage - you can change this to any media file
    media_file = r'examples\files\2025-03-19 12-16-11.mkv'  # Could be .mp4, .avi, .mp3, etc.
    
    logger.info("TranscribeAI - Universal Media Transcription")
    logger.info("=" * 50)
    logger.info("Supported formats:")
    logger.info(f"Video: MP4, MOV, AVI, MKV, WebM, WMV, FLV, M4V, 3GP, MPG, MPEG, TS, MTS")
    logger.info(f" Audio: WAV, MP3, AAC, FLAC, OGG, M4A, WMA, AIFF")
    logger.info("=" * 50)
    
    try:
        # Process with summarization using library method (default)
        result = process_media_file(media_file, summarize=True, 
                                  ollama_model='phi4-reasoning:latest', ollama_method='library')
        
        logger.info(f"\nProcess completed successfully!")
        logger.info(f"📄 Transcript: {result['transcript']}")
        if 'summary' in result:
            logger.info(f"Summary: {result['summary']}")
        
        # Alternative: Use API method
        # result = process_media_file(media_file, summarize=True, ollama_method='api')
        
        # Alternative: Process without summarization
        # result = process_media_file(media_file, summarize=False)
        
    except Exception as e:
        logger.error(f"\nError: {e}")
