#!/usr/bin/env python3
"""Main entry point for TranscribeAI application.

This demonstrates the clean, professional architecture with multiple backend support.
"""

import sys
import argparse
from pathlib import Path
from TranscribeAI import transcribe_media, TranscriptionFactory, logger

def create_parser():
    """Create command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="TranscribeAI - Universal Media Transcription and Summarization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py video.mp4
  python main.py video.mp4 --backend huggingface --model-size large-v3
  python main.py audio.mp3 --no-summary --backend whisper
  python main.py video.mov --timestamps --language english --task translate
        """
    )
    
    parser.add_argument('media_file', help='Path to media file to transcribe')
    parser.add_argument('--backend', choices=['whisper', 'huggingface'], default='whisper',
                       help='Transcription backend to use (default: whisper)')
    parser.add_argument('--model-size', default='base',
                       help='Model size to use (default: base)')
    parser.add_argument('--output', help='Output transcript file path')
    parser.add_argument('--no-summary', action='store_true',
                       help='Skip summarization step')
    
    # Summarization options
    summary_group = parser.add_argument_group('Summarization Options')
    summary_group.add_argument('--summary-provider', choices=['ollama', 'openai', 'anthropic', 'gemini'], 
                              default='ollama', help='AI provider for summarization (default: ollama)')
    summary_group.add_argument('--summary-model', 
                              help='Model to use for summarization (provider-specific defaults if not specified)')
    summary_group.add_argument('--summary-method', choices=['library', 'api'], default='library',
                              help='Method for Ollama (ignored for other providers, default: library)')
    
    # Backward compatibility (deprecated)
    summary_group.add_argument('--ollama-model', help='DEPRECATED: Use --summary-model with --summary-provider ollama')
    summary_group.add_argument('--ollama-method', choices=['library', 'api'], 
                              help='DEPRECATED: Use --summary-method')
    
    # HuggingFace specific options
    hf_group = parser.add_argument_group('HuggingFace Options')
    hf_group.add_argument('--timestamps', action='store_true',
                         help='Include timestamps in output')
    hf_group.add_argument('--word-timestamps', action='store_true',
                         help='Include word-level timestamps')
    hf_group.add_argument('--language', help='Source audio language')
    hf_group.add_argument('--task', choices=['transcribe', 'translate'], default='transcribe',
                         help='Task type (default: transcribe)')
    hf_group.add_argument('--no-chunked', action='store_true',
                         help='Disable chunked processing')
    hf_group.add_argument('--batch-size', type=int, help='Batch size for processing')
    hf_group.add_argument('--chunk-length', type=int, default=30,
                         help='Chunk length in seconds (default: 30)')
    
    return parser

def main():
    """Main application entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    # Validate input file
    media_path = Path(args.media_file)
    if not media_path.exists():
        logger.error(f"File not found: {args.media_file}")
        sys.exit(1)
    
    # Display startup information
    logger.info("TranscribeAI v2.0 - Universal Media Transcription")
    logger.info("=" * 60)
    logger.info(f"Available backends: {TranscriptionFactory.get_available_backends()}")
    logger.info(f"Selected backend: {args.backend}")
    logger.info("=" * 60)
    
    try:
        # Prepare transcription parameters
        kwargs = {
            'model_size': args.model_size
        }
        
        # Add HuggingFace specific parameters
        if args.backend == 'huggingface':
            if args.timestamps:
                kwargs['return_timestamps'] = True
            elif args.word_timestamps:
                kwargs['return_timestamps'] = 'word'
            
            if args.language:
                kwargs['language'] = args.language
            if args.task:
                kwargs['task'] = args.task
            if not args.no_chunked:
                kwargs['use_chunked'] = True
                kwargs['chunk_length_s'] = args.chunk_length
            if args.batch_size:
                kwargs['batch_size'] = args.batch_size
        
        # Handle backward compatibility and prepare summarization args
        summary_kwargs = {}
        if args.ollama_model:
            import warnings
            warnings.warn("--ollama-model is deprecated. Use --summary-model with --summary-provider ollama", 
                         DeprecationWarning)
            if args.summary_provider == 'ollama':
                summary_kwargs['summary_model'] = args.ollama_model
        if args.ollama_method:
            import warnings
            warnings.warn("--ollama-method is deprecated. Use --summary-method", DeprecationWarning)
            summary_kwargs['summary_method'] = args.ollama_method
        
        # Process the media file
        result = transcribe_media(
            media_file=str(media_path),
            backend=args.backend,
            output_txt=args.output,
            summarize=not args.no_summary,
            summary_provider=args.summary_provider,
            summary_method=args.summary_method,
            summary_model=args.summary_model,
            **summary_kwargs,
            **kwargs
        )
        
        # Display results
        logger.info("\n🎉 Processing completed successfully!")
        logger.info("=" * 60)
        logger.info(f"📄 Transcript: {result['transcript']}")
        if 'summary' in result and result['summary']:
            logger.info(f"📋 Summary: {result['summary']}")
        logger.info(f"🔊 Audio: {result['audio']}")
        logger.info("=" * 60)
        
    except KeyboardInterrupt:
        logger.info("\nProcess interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
