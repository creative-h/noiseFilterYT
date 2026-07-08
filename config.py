"""
Configuration settings for the YouTube playlist processing pipeline.
"""

import os
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).parent

# Directory structure
RAW_DIR = BASE_DIR / "raw"
MP3_DIR = BASE_DIR / "mp3"
CLEAN_DIR = BASE_DIR / "clean"
TRANSCRIPT_DIR = BASE_DIR / "transcript"
SUMMARY_DIR = BASE_DIR / "summary"
LOGS_DIR = BASE_DIR / "logs"

# Create directories if they don't exist
for dir_path in [RAW_DIR, MP3_DIR, CLEAN_DIR, TRANSCRIPT_DIR, SUMMARY_DIR, LOGS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# yt-dlp settings
YDLP_FORMAT = "bestaudio/best"
YDLP_POSTPROCESSORS = [
    {
        "key": "FFmpegExtractAudio",
        "preferredcodec": "mp3",
        "preferredquality": "192",
    }
]

# Audio processing settings
AUDIO_BITRATE = "192k"
AUDIO_SAMPLE_RATE = 44100

# FFmpeg loudnorm settings
LOUDNORM_SETTINGS = {
    "i": "-16",  # Integrated loudness target
    "tp": "-1.5",  # True peak
    "lra": "11",  # Loudness range
}

# DeepFilterNet settings
DEEPFILTER_MODEL = "deepfilter_net"  # Default model

# Whisper settings
WHISPER_MODEL = "large-v3"  # Options: tiny, base, small, medium, large-v1, large-v2, large-v3
WHISPER_DEVICE = "auto"  # auto, cpu, cuda
WHISPER_COMPUTE_TYPE = "float16"  # float16, int8, float32

# LLM settings for summarization
LLM_API_KEY = os.getenv("OPENAI_API_KEY", "")
LLM_MODEL = "gpt-4o"  # or use a local model
LLM_TEMPERATURE = 0.7
LLM_MAX_TOKENS = 2000

# Logging settings
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_FILE = LOGS_DIR / "pipeline.log"

# Processing options
SKIP_EXISTING = True  # Skip files that already exist in output directories
REMOVE_SILENCE = False  # Optional: remove long pauses
SILENCE_THRESHOLD = "-50dB"  # Silence threshold for removal
MIN_SILENCE_DURATION = "1"  # Minimum silence duration to remove
