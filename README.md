# YouTube Playlist Processing Pipeline

A modular, automated pipeline for downloading, processing, and transcribing YouTube playlists. Perfect for creating clean audio libraries, generating transcripts, and building study materials from educational content.

## Features

- **Playlist Download**: Download entire YouTube playlists or individual videos
- **Audio Conversion**: Convert to MP3 with customizable quality
- **Noise Removal**: Remove background noise using DeepFilterNet
- **Loudness Normalization**: Consistent volume across all files
- **Speech-to-Text**: High-quality transcription using Faster-Whisper
- **Summarization**: Generate summaries and key points using LLM
- **Modular Design**: Run individual steps or the full pipeline

## Architecture

```
YouTube Playlist
        │
        ▼
Download (yt-dlp)
        │
        ▼
Convert to MP3 (FFmpeg)
        │
        ▼
Noise Removal (DeepFilterNet)
        │
        ▼
Normalize Volume (FFmpeg)
        │
        ▼
Speech-to-Text (Faster-Whisper)
        │
        ▼
Summarization (LLM)
```

## Prerequisites

### FFmpeg

FFmpeg must be installed and added to your system PATH.

**Windows:**
1. Download FFmpeg from https://ffmpeg.org/download.html
2. Extract to a folder (e.g., `C:\ffmpeg`)
3. Add `C:\ffmpeg\bin` to your system PATH
4. Verify: `ffmpeg -version`

## Installation

### Option 1: Local Installation

1. Clone this repository:
```bash
git clone https://github.com/creative-h/noiseFilterYT.git
cd noiseFilterYT
```

2. Install Python dependencies:
```bash
pip install -r requirements.txt
```

3. Install FFmpeg and add to PATH (see Prerequisites below)

4. (Optional) Set up OpenAI API key for summarization:
```bash
# Windows
setx OPENAI_API_KEY "your-api-key-here"

# Or set it temporarily in your terminal
set OPENAI_API_KEY=your-api-key-here
```

### Option 2: Google Colab (Recommended for Large Playlists)

For large playlists or if you don't have a powerful local machine, use Google Colab with GPU acceleration:

1. Open the notebook: [colab_notebook.ipynb](colab_notebook.ipynb)
2. Click "Open in Colab" button
3. Run cells sequentially to set up and process your playlist

**Colab Benefits:**
- Free GPU acceleration for faster transcription
- No local installation required
- Handles large downloads without storage issues
- Easy to download results as ZIP

## Project Structure

```
youtube_pipeline/
│
├── config.py           # Configuration settings
├── downloader.py       # Playlist/video downloader
├── convert.py          # Audio conversion
├── denoise.py          # Noise removal
├── normalize.py        # Loudness normalization
├── transcribe.py       # Speech-to-text
├── summarize.py        # LLM summarization
├── main.py             # Pipeline orchestration
├── colab_notebook.ipynb # Google Colab notebook
├── requirements.txt    # Python dependencies
├── README.md           # This file
│
├── raw/                # Downloaded audio files
├── mp3/                # Converted MP3 files
├── clean/              # Denoised and normalized audio
├── transcript/         # Text transcripts
├── summary/            # Summaries and key points
└── logs/               # Pipeline logs
```

## Usage

### Process Entire Playlist

```bash
python main.py --playlist https://youtube.com/playlist?list=YOUR_PLAYLIST_ID
```

### Process Specific Range

```bash
python main.py --playlist https://youtube.com/playlist?list=YOUR_PLAYLIST_ID --start 1 --end 5
```

### Process Single Video

```bash
python main.py --video https://youtube.com/watch?v=YOUR_VIDEO_ID
```

### Skip Specific Steps

```bash
# Skip denoising and summarization
python main.py --playlist https://youtube.com/playlist?list=YOUR_PLAYLIST_ID --skip-denoise --skip-summarize

# Skip transcription and summarization (audio processing only)
python main.py --playlist https://youtube.com/playlist?list=YOUR_PLAYLIST_ID --skip-transcribe --skip-summarize
```

### Command-Line Options

```
--playlist URL       YouTube playlist URL
--video URL          Single YouTube video URL
--start N            Start index (1-based)
--end N              End index (1-based)
--skip-download      Skip download step
--skip-conversion    Skip conversion step
--skip-denoise       Skip denoising step
--skip-normalize     Skip normalization step
--skip-transcribe    Skip transcription step
--skip-summarize     Skip summarization step
```

## Configuration

Edit `config.py` to customize:

- **Audio quality**: Bitrate, sample rate
- **Whisper model**: Model size (tiny, base, small, medium, large-v3)
- **LLM settings**: Model, temperature, max tokens
- **Processing options**: Skip existing files, remove silence
- **Directory paths**: Output directories

## Individual Module Usage

You can also use individual modules in your own scripts:

```python
from downloader import PlaylistDownloader
from convert import AudioConverter
from denoise import AudioDenoiser
from normalize import AudioNormalizer
from transcribe import AudioTranscriber
from summarize import TranscriptSummarizer

# Download playlist
downloader = PlaylistDownloader()
files = downloader.download_playlist("https://youtube.com/playlist?list=YOUR_PLAYLIST")

# Convert to MP3
converter = AudioConverter()
mp3_files = converter.batch_convert(Path("raw"))

# Denoise
denoiser = AudioDenoiser()
clean_files = denoiser.batch_denoise(Path("mp3"))

# Normalize
normalizer = AudioNormalizer()
normalized_files = normalizer.batch_normalize(Path("clean"))

# Transcribe
transcriber = AudioTranscriber()
transcripts = transcriber.batch_transcribe(Path("clean"))

# Summarize
summarizer = TranscriptSummarizer()
summaries = summarizer.batch_summarize(Path("transcript"))
```

## Technology Stack

| Step                | Tool                    | Purpose                          |
| ------------------- | ----------------------- | -------------------------------- |
| Download            | yt-dlp                  | Reliable playlist download       |
| Audio Extraction    | FFmpeg                  | Fast, lossless processing        |
| Noise Removal       | DeepFilterNet           | Speech enhancement               |
| Normalization       | FFmpeg (loudnorm)       | Consistent volume                |
| Speech-to-Text      | Faster-Whisper          | Fast, accurate transcription     |
| Summarization       | OpenAI GPT              | Generate notes and summaries     |

## Troubleshooting

### FFmpeg not found
- Ensure FFmpeg is installed and in your PATH
- Run `ffmpeg -version` to verify

### DeepFilterNet errors
- Ensure the package is installed: `pip install deepfilternet`
- Check logs in `logs/pipeline.log`

### Whisper CUDA errors
- Install CUDA-compatible PyTorch
- Or set `WHISPER_DEVICE = "cpu"` in config.py

### Summarization not working
- Set `OPENAI_API_KEY` environment variable
- Or configure in `config.py`

## Performance Tips

- **Faster transcription**: Use `distil-large-v3` model in config.py
- **CPU-only**: Set `WHISPER_DEVICE = "cpu"` and `WHISPER_COMPUTE_TYPE = "int8"`
- **Batch processing**: Process playlists in chunks using `--start` and `--end`
- **Skip existing**: Files are skipped by default if they already exist

## Future Enhancements

Potential additions to the pipeline:

- **RAG Integration**: Build a vector database for Q&A
- **Chapter Detection**: Automatically segment lectures
- **Multi-language Support**: Translate transcripts
- **Voice Cloning**: Generate synthetic audio from transcripts
- **Video Processing**: Extract slides, generate thumbnails

## License

MIT License

## Contributing

Contributions are welcome! Feel free to submit issues or pull requests.
