# TranscribeAI

🎵 **AI-Powered Media Transcription and Summarization Tool**

TranscribeAI is a Python application that converts audio and video files into text transcripts using OpenAI's Whisper models, with optional AI-powered summarization through Ollama integration.

## ✨ Features

### 🎯 Core Functionality
- **Media-to-Audio Conversion**: Extract audio from video files or process audio files directly
- **AI Transcription**: High-accuracy speech-to-text using Whisper models
- **Intelligent Summarization**: Generate comprehensive summaries using local Ollama models
- **GPU Acceleration**: Automatic CUDA detection with CPU fallback
- **Multiple Formats**: Support for common audio/video formats

### 🔧 Transcription Methods
- **Standard Whisper**: Direct OpenAI Whisper integration (`transcribe_media.py`)
- **Hugging Face Transformers**: Enhanced Whisper with advanced features (`transcribe_media_hf.py`)
  - Chunked processing for long-form audio
  - Word-level timestamps
  - Language specification
  - Translation capabilities
  - Batch processing

### 📊 Summarization Options
- **Ollama Library**: Python library integration for local AI models
- **Ollama API**: REST API integration with configurable endpoints
- **Comprehensive Summaries**: Main topics, key points, conclusions, and context

## 🚀 Quick Start

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd TranscribeAI
```

2. **Set up Python environment**
```bash
# Install uv package manager (recommended)
pip install uv

# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -r requirements.txt
```

3. **Install Ollama (optional, for summarization)**
```bash
# Install Ollama from https://ollama.ai
# Pull a model for summarization
ollama pull llama3.2
```

### Basic Usage

```python
from src.TranscribeAI.transcribe_media import process_media_file

# Transcribe and summarize a media file
result = process_media_file(
    media_file="path/to/your/video.mp4",
    output_txt="transcript.txt",
    model_size="base",
    summarize=True,
    ollama_model="llama3.2"
)
```

## 📚 Detailed Usage

### Standard Whisper Method

```python
from src.TranscribeAI.transcribe_media import process_media_file

# Basic transcription only
process_media_file(
    media_file="audio.wav",
    model_size="small",  # tiny, base, small, medium, large
    summarize=False
)

# With summarization using Ollama library
process_media_file(
    media_file="video.mp4",
    model_size="base",
    summarize=True,
    ollama_model="llama3.2",
    ollama_method="library"  # or "api"
)
```

### Hugging Face Enhanced Method

```python
from src.TranscribeAI.transcribe_media_hf import process_media_file_hf

# Advanced transcription with timestamps
process_media_file_hf(
    media_file="long_video.mp4",
    model_size="large-v3",
    return_timestamps="word",  # "word" or True for segment timestamps
    language="english",
    task="transcribe",  # or "translate"
    use_chunked=True,
    batch_size=8
)
```

## 🎛️ Configuration

### Supported Media Formats

**Audio**: `.wav`, `.mp3`, `.aac`, `.flac`, `.ogg`, `.m4a`, `.wma`, `.aiff`

**Video**: `.mp4`, `.mov`, `.avi`, `.mkv`, `.webm`, `.wmv`, `.flv`, `.m4v`, `.3gp`, `.mpg`, `.mpeg`, `.ts`, `.mts`

### Whisper Model Sizes

| Model | Parameters | VRAM | Speed | Accuracy |
|-------|------------|------|-------|----------|
| `tiny` | 39M | ~1GB | Fastest | Basic |
| `base` | 74M | ~1GB | Fast | Good |
| `small` | 244M | ~2GB | Moderate | Better |
| `medium` | 769M | ~5GB | Slower | High |
| `large` | 1550M | ~10GB | Slowest | Highest |
| `large-v3` | 1550M | ~10GB | Slowest | Highest (latest) |

### GPU Acceleration

TranscribeAI automatically detects CUDA availability:
- **GPU Available**: Uses CUDA acceleration for faster processing
- **GPU Unavailable**: Falls back to CPU processing
- **GPU Error**: Automatic fallback to CPU with warning

## 📁 Project Structure

```
TranscribeAI/
├── src/
│   └── TranscribeAI/
│       ├── main.py                 # Entry point
│       ├── transcribe_media.py     # Standard Whisper implementation
│       └── transcribe_media_hf.py  # Hugging Face enhanced implementation
├── examples/                       # Example files and usage
├── tests/                         # Unit tests
├── pyproject.toml                 # Project configuration
└── README.md                      # This file
```

## 🛠️ Development

### Running Tests
```bash
python -m pytest tests/
```

### Code Style
```bash
# Format code
black src/
# Lint code
flake8 src/
```

## 🗺️ Roadmap

### Planned Features

#### 🔮 Future API Integrations
- **OpenAI API**: Integration with GPT models for enhanced summarization
  - Support for GPT-4 and GPT-3.5-turbo
  - Configurable prompt templates
  - Cost-aware usage tracking

- **Anthropic API**: Claude integration for alternative summarization
  - Claude-3 Sonnet and Haiku models
  - Context-aware summarization
  - Safety-focused content analysis

#### 🚀 Enhanced Features
- **Web Interface**: Browser-based UI for easy file upload and processing
- **Batch Processing**: Process multiple files simultaneously
- **Custom Prompts**: User-defined summarization templates
- **Export Formats**: JSON, SRT, VTT subtitle formats
- **Real-time Processing**: Live audio transcription
- **Speaker Diarization**: Identify and separate multiple speakers

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [OpenAI Whisper](https://github.com/openai/whisper) for speech recognition
- [Hugging Face Transformers](https://huggingface.co/transformers/) for enhanced Whisper implementation
- [Ollama](https://ollama.ai) for local AI model integration
- [MoviePy](https://zulko.github.io/moviepy/) for media processing

## 📞 Support

For questions, issues, or contributions, please open an issue on the GitHub repository.

---

*Built with ❤️ for accessible AI-powered transcription*
