# TranscribeAI v2.0 - Universal Media Transcription & Summarization

A professional-grade Python library for transcribing audio/video files using multiple AI backends with optional AI-powered summarization.

## ✨ Features

- **Multiple Transcription Backends**: OpenAI Whisper and HuggingFace Transformers
- **GPU Acceleration**: Automatic CUDA detection with CPU fallback
- **Universal Format Support**: 20+ audio/video formats (MP4, MOV, AVI, MP3, WAV, etc.)
- **AI-Powered Summarization**: Integration with Ollama models
- **Professional Architecture**: Clean, modular, and extensible design
- **Model Caching**: Efficient memory usage with intelligent model caching
- **Comprehensive Error Handling**: Robust error handling with helpful diagnostics

## 🚀 Quick Start

### Basic Usage

```python
from TranscribeAI import transcribe_media

# Simple transcription with default Whisper backend
result = transcribe_media('video.mp4')
print(f"Transcript: {result['transcript']}")
print(f"Summary: {result['summary']}")
```

### Advanced Usage

```python
# HuggingFace backend with timestamps and language detection
result = transcribe_media(
    'video.mp4',
    backend='huggingface',
    model_size='large-v3',
    return_timestamps=True,
    language='english',
    task='translate',
    summarize=True,
    ollama_model='llama3.2'
)
```

### Command Line Interface

```bash
# Basic transcription
python -m TranscribeAI.main video.mp4

# Advanced options
python -m TranscribeAI.main video.mp4 \
  --backend huggingface \
  --model-size large-v3 \
  --timestamps \
  --language english \
  --ollama-model phi4-reasoning:latest
```

## 🏗️ Architecture

The new v2.0 architecture follows professional software engineering principles:

### Core Components

- **`BaseTranscriber`**: Abstract base class for all transcription implementations
- **`WhisperTranscriber`**: OpenAI Whisper implementation
- **`HuggingFaceTranscriber`**: HuggingFace Transformers implementation  
- **`TranscriptionFactory`**: Factory pattern for backend selection
- **`MediaProcessor`**: Handles audio/video conversion
- **`SummarizationEngine`**: AI-powered text summarization
- **`DeviceManager`**: CUDA/CPU device management

### Design Patterns Used

- **Factory Pattern**: For backend selection and instantiation
- **Strategy Pattern**: For different transcription algorithms
- **Singleton Pattern**: For model caching and resource management
- **Template Method**: For common transcription pipeline
- **Dependency Injection**: For flexible component composition

## 📚 API Reference

### Main Function

```python
def transcribe_media(
    media_file: str,
    backend: str = "whisper",
    output_txt: Optional[str] = None,
    summarize: bool = True,
    ollama_model: str = 'llama3.2',
    ollama_method: str = 'library',
    **kwargs
) -> Dict[str, str]:
```

### Backend-Specific Parameters

#### Whisper Backend
- `model_size`: 'tiny', 'base', 'small', 'medium', 'large'

#### HuggingFace Backend
- `model_size`: 'tiny', 'base', 'small', 'medium', 'large', 'large-v3'
- `return_timestamps`: True, False, or 'word'
- `language`: Source language code (e.g., 'english', 'french')
- `task`: 'transcribe' or 'translate'
- `use_chunked`: Enable chunked processing for long audio
- `batch_size`: Batch size for processing
- `chunk_length_s`: Chunk length in seconds

## 🔧 Installation

```bash
# Core dependencies
pip install torch whisper transformers moviepy requests

# Optional: Ollama Python library for summarization
pip install ollama

# Optional: CUDA support (if you have NVIDIA GPU)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## 💡 Examples

See `src/TranscribeAI/examples.py` for comprehensive usage examples including:

- Basic transcription workflows
- Advanced HuggingFace features
- Multilingual support
- Batch processing
- Error handling patterns
- Factory pattern usage

## 🔄 Migration from v1.0

The new architecture maintains backward compatibility while providing deprecation warnings:

```python
# Old way (still works, but deprecated)
from TranscribeAI.transcribe_media import process_media_file
result = process_media_file('video.mp4')

# New way (recommended)
from TranscribeAI import transcribe_media
result = transcribe_media('video.mp4')
```

## 🎯 Performance Optimizations

- **Model Caching**: Models are cached after first load
- **GPU Memory Management**: Automatic memory cleanup and fallback
- **Chunked Processing**: Efficient handling of long audio files
- **Batch Processing**: Support for processing multiple files
- **Resource Pooling**: Reuse transcriber instances for better performance

## 🛠️ Contributing

This codebase follows professional software engineering practices:

- Clean, modular architecture with clear separation of concerns
- Comprehensive docstrings and type hints
- Professional error handling and logging
- Extensible design for adding new backends
- Test-driven development patterns

## 📄 License

MIT License - see LICENSE file for details.

---

**TranscribeAI v2.0** - Professional-grade transcription with the simplicity you need and the power you want.
