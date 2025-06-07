import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from .transcribe_base import BaseTranscriber, logger

# Available Whisper models on Hugging Face
WHISPER_MODELS = {
    'tiny': 'openai/whisper-tiny',
    'base': 'openai/whisper-base', 
    'small': 'openai/whisper-small',
    'medium': 'openai/whisper-medium',
    'large': 'openai/whisper-large-v2',
    'large-v3': 'openai/whisper-large-v3'
}

class HuggingFaceTranscriber(BaseTranscriber):
    """HuggingFace Transformers-based transcription implementation."""
    
    def __init__(self):
        super().__init__()
        self.pipeline_cache = {}  # Cache loaded pipelines for reuse
    
    def transcribe_audio(self, audio_file, output_txt='transcript.txt', model_size='base', 
                        return_timestamps=False, language=None, task='transcribe',
                        use_chunked=True, batch_size=None, chunk_length_s=30):
        """Transcribe audio file using HuggingFace Transformers Whisper implementation.
        
        Args:
            audio_file (str): Path to audio file to transcribe.
            output_txt (str): Output file path for transcript.
            model_size (str): Whisper model size to use.
            return_timestamps (bool or str): Whether to return timestamps ('word' for word-level).
            language (str): Source audio language (e.g., 'english', 'french').
            task (str): Task type - 'transcribe' or 'translate'.
            use_chunked (bool): Use chunked processing for long audio.
            batch_size (int): Batch size for processing multiple chunks.
            chunk_length_s (int): Chunk length in seconds for long-form audio.
            
        Returns:
            str: Path to the transcript file.
        """
        logger.info(f'Transcribing {audio_file} using HF Transformers Whisper on {self.device_manager.device.upper()}...')
        
        try:
            pipe = self._load_pipeline(model_size, use_chunked, chunk_length_s)
            result = self._perform_transcription(pipe, audio_file, return_timestamps, language, task, batch_size)
            self._save_transcript(result, output_txt, model_size, return_timestamps, language, task)
            
            logger.info(f'Transcription complete, saved as {output_txt}')
            return output_txt
            
        except Exception as e:
            return self._handle_transcription_error(e, audio_file, output_txt, model_size, 
                                                  return_timestamps, language, task, use_chunked, batch_size, chunk_length_s)
    
    def _load_pipeline(self, model_size, use_chunked, chunk_length_s):
        """Load HuggingFace pipeline with caching."""
        cache_key = f"{model_size}_{self.device_manager.device}_{use_chunked}_{chunk_length_s}"
        
        if cache_key not in self.pipeline_cache:
            model_id = WHISPER_MODELS.get(model_size, WHISPER_MODELS['base'])
            logger.info(f'Loading Whisper model: {model_id} on {self.device_manager.device}...')
            
            # Load model and processor
            model = AutoModelForSpeechSeq2Seq.from_pretrained(
                model_id,
                torch_dtype=self.device_manager.torch_dtype,
                low_cpu_mem_usage=True,
                use_safetensors=True
            )
            model.to(self.device_manager.device)
            
            processor = AutoProcessor.from_pretrained(model_id)
            
            # Create pipeline
            pipe_kwargs = {
                "model": model,
                "tokenizer": processor.tokenizer,
                "feature_extractor": processor.feature_extractor,
                "torch_dtype": self.device_manager.torch_dtype,
                "device": self.device_manager.device,
            }
            
            if use_chunked:
                pipe_kwargs["chunk_length_s"] = chunk_length_s
            
            self.pipeline_cache[cache_key] = pipeline("automatic-speech-recognition", **pipe_kwargs)
            logger.info(f'HuggingFace pipeline ready with {"chunked" if use_chunked else "sequential"} processing')
        
        return self.pipeline_cache[cache_key]
    
    def _perform_transcription(self, pipe, audio_file, return_timestamps, language, task, batch_size):
        """Perform transcription with specified parameters."""
        generate_kwargs = {
            "max_new_tokens": 448,
            "num_beams": 1,
            "condition_on_prev_tokens": False,
            "compression_ratio_threshold": 1.35,
            "temperature": (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
            "logprob_threshold": -1.0,
            "no_speech_threshold": 0.6,
        }
        
        if language:
            generate_kwargs["language"] = language
        if task:
            generate_kwargs["task"] = task
        
        pipe_kwargs = {"generate_kwargs": generate_kwargs}
        if return_timestamps:
            pipe_kwargs["return_timestamps"] = return_timestamps
        if batch_size:
            pipe_kwargs["batch_size"] = batch_size
        
        return pipe(audio_file, **pipe_kwargs)
    
    def _save_transcript(self, result, output_txt, model_size, return_timestamps, language, task):
        """Save transcript with metadata and timestamps if available."""
        if isinstance(result, dict):
            transcript = result.get('text', '')
            chunks = result.get('chunks', [])
        else:
            transcript = str(result)
            chunks = []
        
        with open(output_txt, 'w', encoding='utf-8') as f:
            f.write(f"=== TRANSCRIPT (HF Transformers Whisper) ===\n")
            f.write(f"Model: {model_size} ({WHISPER_MODELS.get(model_size, 'unknown')})\n")
            f.write(f"Device: {self.device_manager.device}\n")
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
    
    def _handle_transcription_error(self, error, audio_file, output_txt, model_size, 
                                   return_timestamps, language, task, use_chunked, batch_size, chunk_length_s):
        """Handle transcription errors with fallback to CPU."""
        logger.error(f"Transcription failed: {str(error)}")
        
        if self.device_manager.device == "cuda":
            logger.warning("GPU transcription failed, falling back to CPU...")
            
            # Update device manager for CPU fallback
            self.device_manager.device = "cpu"
            self.device_manager.torch_dtype = torch.float32
            
            return self.transcribe_audio(audio_file, output_txt, model_size, 
                                       return_timestamps, language, task, use_chunked, batch_size, chunk_length_s)
        else:
            raise error 