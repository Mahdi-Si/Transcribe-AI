#!/usr/bin/env python3
"""Examples demonstrating multi-provider summarization capabilities in TranscribeAI."""

import os
from TranscribeAI import transcribe_media, logger

def example_ollama_summarization():
    """Example using Ollama for summarization (default behavior)."""
    logger.info("=== Ollama Summarization Example ===")
    
    media_file = r'C:\Users\mahdi\Desktop\Mahdi-Si-Projects\Transcribe-AI\2025-06-05 17-30-36.mp4'
    
    try:
        result = transcribe_media(
            media_file,
            backend='whisper',
            model_size='base',
            summary_provider='ollama',
            summary_method='library',  # or 'api'
            summary_model='llama3.2'
        )
        logger.info(f"✅ Ollama summary saved: {result.get('summary')}")
    except Exception as e:
        logger.error(f"❌ Ollama summarization failed: {e}")

def example_openai_summarization():
    """Example using OpenAI GPT models for summarization."""
    logger.info("=== OpenAI Summarization Example ===")
    
    # Ensure you have OPENAI_API_KEY in environment or pass it during initialization
    if not os.getenv('OPENAI_API_KEY'):
        logger.warning("OPENAI_API_KEY not found in environment variables")
        return
    
    media_file = r'C:\Users\mahdi\Desktop\Mahdi-Si-Projects\Transcribe-AI\2025-06-05 17-30-36.mp4'
    
    try:
        result = transcribe_media(
            media_file,
            backend='whisper',
            model_size='base',
            summary_provider='openai',
            summary_model='gpt-4o-mini'  # Cost-effective option
        )
        logger.info(f"✅ OpenAI summary saved: {result.get('summary')}")
    except Exception as e:
        logger.error(f"❌ OpenAI summarization failed: {e}")

def example_anthropic_summarization():
    """Example using Anthropic Claude for summarization."""
    logger.info("=== Anthropic Claude Summarization Example ===")
    
    # Ensure you have ANTHROPIC_API_KEY in environment
    if not os.getenv('ANTHROPIC_API_KEY'):
        logger.warning("ANTHROPIC_API_KEY not found in environment variables")
        return
    
    media_file = r'C:\Users\mahdi\Desktop\Mahdi-Si-Projects\Transcribe-AI\2025-06-05 17-30-36.mp4'
    
    try:
        result = transcribe_media(
            media_file,
            backend='whisper',
            model_size='base',
            summary_provider='anthropic',
            summary_model='claude-3-5-haiku-20241022'  # Fast and cost-effective
        )
        logger.info(f"✅ Claude summary saved: {result.get('summary')}")
    except Exception as e:
        logger.error(f"❌ Anthropic summarization failed: {e}")

def example_gemini_summarization():
    """Example using Google Gemini for summarization."""
    logger.info("=== Google Gemini Summarization Example ===")
    
    # Ensure you have GEMINI_API_KEY in environment
    if not os.getenv('GEMINI_API_KEY'):
        logger.warning("GEMINI_API_KEY not found in environment variables")
        return
    
    media_file = r'C:\Users\mahdi\Desktop\Mahdi-Si-Projects\Transcribe-AI\2025-06-05 17-30-36.mp4'
    
    try:
        result = transcribe_media(
            media_file,
            backend='whisper',
            model_size='base',
            summary_provider='gemini',
            summary_model='gemini-2.0-flash'  # Latest and most capable
        )
        logger.info(f"✅ Gemini summary saved: {result.get('summary')}")
    except Exception as e:
        logger.error(f"❌ Gemini summarization failed: {e}")

def example_provider_comparison():
    """Compare different providers on the same transcript."""
    logger.info("=== Provider Comparison Example ===")
    
    media_file = r'C:\Users\mahdi\Desktop\Mahdi-Si-Projects\Transcribe-AI\2025-06-05 17-30-36.mp4'
    
    providers = [
        ('ollama', 'llama3.2'),
        ('openai', 'gpt-4o-mini'),
        ('anthropic', 'claude-3-5-haiku-20241022'),
        ('gemini', 'gemini-2.0-flash')
    ]
    
    for provider, model in providers:
        try:
            # Check if API key is available for cloud providers
            if provider == 'openai' and not os.getenv('OPENAI_API_KEY'):
                logger.warning(f"Skipping {provider}: API key not found")
                continue
            elif provider == 'anthropic' and not os.getenv('ANTHROPIC_API_KEY'):
                logger.warning(f"Skipping {provider}: API key not found")
                continue
            elif provider == 'gemini' and not os.getenv('GEMINI_API_KEY'):
                logger.warning(f"Skipping {provider}: API key not found")
                continue
            
            logger.info(f"Testing {provider} with {model}...")
            result = transcribe_media(
                media_file,
                backend='whisper',
                model_size='base',
                summary_provider=provider,
                summary_model=model
            )
            logger.info(f"✅ {provider.capitalize()} summary: {result.get('summary')}")
            
        except Exception as e:
            logger.error(f"❌ {provider.capitalize()} failed: {e}")

def main():
    """Run all summarization examples."""
    logger.info("TranscribeAI Multi-Provider Summarization Examples")
    logger.info("=" * 60)
    
    # Run individual examples
    example_ollama_summarization()
    example_openai_summarization()
    example_anthropic_summarization()
    example_gemini_summarization()
    
    # Run comparison
    example_provider_comparison()
    
    logger.info("\n💡 Tips for API Usage:")
    logger.info("   • Set environment variables: OPENAI_API_KEY, ANTHROPIC_API_KEY, GEMINI_API_KEY")
    logger.info("   • Install optional dependencies: pip install openai anthropic google-genai")
    logger.info("   • Ollama runs locally and is free (requires 'ollama serve' to be running)")

if __name__ == "__main__":
    main() 