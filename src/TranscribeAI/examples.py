#!/usr/bin/env python3
"""Example usage of TranscribeAI with different backends and configurations.

This file demonstrates various use cases and professional patterns for using
the TranscribeAI library.
"""

from TranscribeAI import transcribe_media, TranscriptionFactory, TranscriptionBackend, logger

def example_basic_usage():
    """Basic usage example with default settings."""
    logger.info("=== Basic Usage Example ===")
    
    media_file = r'C:\Users\mahdi\Desktop\Mahdi-Si-Projects\Transcribe-AI\2025-06-05 17-30-36.mp4'
    
    # Simple transcription with default Whisper backend
    result = transcribe_media(media_file)
    
    logger.info(f"Transcript saved to: {result['transcript']}")
    if 'summary' in result:
        logger.info(f"Summary saved to: {result['summary']}")

def example_whisper_backend():
    """Whisper backend with different model sizes."""
    logger.info("=== Whisper Backend Examples ===")
    
    media_file = r'C:\Users\mahdi\Desktop\Mahdi-Si-Projects\Transcribe-AI\2025-06-05 17-30-36.mp4'
    
    # Different model sizes
    for model_size in ['tiny', 'base', 'small']:
        logger.info(f"Testing Whisper with {model_size} model...")
        result = transcribe_media(
            media_file,
            backend='whisper',
            model_size=model_size,
            summarize=False  # Skip summarization for speed
        )
        logger.info(f"  Result: {result['transcript']}")

def example_huggingface_backend():
    """HuggingFace backend with advanced features."""
    logger.info("=== HuggingFace Backend Examples ===")
    
    media_file = r'C:\Users\mahdi\Desktop\Mahdi-Si-Projects\Transcribe-AI\2025-06-05 17-30-36.mp4'
    
    # Basic HuggingFace transcription
    result = transcribe_media(
        media_file,
        backend='huggingface',
        model_size='small',
        summarize=True,
        olmake_model='phi4-reasoning:latest'
    )
    logger.info(f"Basic HF result: {result['transcript']}")
    
    # Advanced HuggingFace with timestamps
    result = transcribe_media(
        media_file,
        backend='huggingface',
        model_size='base',
        return_timestamps=True,
        use_chunked=True,
        chunk_length_s=30,
        summarize=False
    )
    logger.info(f"Timestamped result: {result['transcript']}")

def example_multilingual_support():
    """Multilingual transcription and translation examples."""
    logger.info("=== Multilingual Support Examples ===")
    
    # This would work with a French audio file
    french_file = "path/to/french_audio.wav"  # Replace with actual file
    
    try:
        # Transcribe French audio
        result = transcribe_media(
            french_file,
            backend='huggingface',
            model_size='medium',
            language='french',
            task='transcribe',
            summarize=False
        )
        logger.info(f"French transcription: {result['transcript']}")
        
        # Translate French to English
        result = transcribe_media(
            french_file,
            backend='huggingface',
            model_size='medium',
            language='french',
            task='translate',
            summarize=False
        )
        logger.info(f"English translation: {result['transcript']}")
        
    except FileNotFoundError:
        logger.info("French audio file not found - skipping multilingual example")

def example_factory_pattern():
    """Using the factory pattern directly for advanced control."""
    logger.info("=== Factory Pattern Example ===")
    
    # Create transcriber instances directly
    whisper_transcriber = TranscriptionFactory.create_transcriber(TranscriptionBackend.WHISPER)
    hf_transcriber = TranscriptionFactory.create_transcriber(TranscriptionBackend.HUGGINGFACE)
    
    media_file = r'C:\Users\mahdi\Desktop\Mahdi-Si-Projects\Transcribe-AI\2025-06-05 17-30-36.mp4'
    
    # Use Whisper transcriber
    logger.info("Using Whisper transcriber directly...")
    result1 = whisper_transcriber.process_media_file(
        media_file,
        model_size='base',
        summarize=False
    )
    logger.info(f"Whisper result: {result1['transcript']}")
    
    # Use HuggingFace transcriber
    logger.info("Using HuggingFace transcriber directly...")
    result2 = hf_transcriber.process_media_file(
        media_file,
        model_size='base',
        return_timestamps=True,
        summarize=False
    )
    logger.info(f"HuggingFace result: {result2['transcript']}")

def example_error_handling():
    """Demonstrate proper error handling patterns."""
    logger.info("=== Error Handling Examples ===")
    
    # Invalid file
    try:
        result = transcribe_media("nonexistent_file.mp4")
    except FileNotFoundError as e:
        logger.error(f"Expected error for missing file: {e}")
    
    # Invalid backend
    try:
        result = transcribe_media(
            r'C:\Users\mahdi\Desktop\Mahdi-Si-Projects\Transcribe-AI\2025-06-05 17-30-36.mp4',
            backend='invalid_backend'
        )
    except ValueError as e:
        logger.error(f"Expected error for invalid backend: {e}")

def example_batch_processing():
    """Process multiple files efficiently."""
    logger.info("=== Batch Processing Example ===")
    
    # List of media files to process
    media_files = [
        r'C:\Users\mahdi\Desktop\Mahdi-Si-Projects\Transcribe-AI\2025-06-05 17-30-36.mp4',
        # Add more files here
    ]
    
    # Create transcriber once and reuse it
    transcriber = TranscriptionFactory.create_transcriber(TranscriptionBackend.WHISPER)
    
    results = []
    for i, media_file in enumerate(media_files, 1):
        try:
            logger.info(f"Processing file {i}/{len(media_files)}: {media_file}")
            result = transcriber.process_media_file(
                media_file,
                model_size='base',
                summarize=True,
                ollama_model='llama3.2'
            )
            results.append(result)
            logger.info(f"✅ Completed: {result['transcript']}")
            
        except Exception as e:
            logger.error(f"❌ Failed to process {media_file}: {e}")
            continue
    
    logger.info(f"Batch processing completed. {len(results)} files processed successfully.")

def main():
    """Run all examples."""
    logger.info("TranscribeAI Examples - Professional Architecture Demo")
    logger.info("=" * 80)
    
    try:
        example_basic_usage()
        example_whisper_backend()
        example_huggingface_backend()
        example_multilingual_support()
        example_factory_pattern()
        example_error_handling()
        example_batch_processing()
        
    except Exception as e:
        logger.error(f"Example execution failed: {e}")
    
    logger.info("=" * 80)
    logger.info("All examples completed!")

if __name__ == "__main__":
    main() 