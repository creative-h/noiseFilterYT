"""
Configuration settings for the YouTube playlist processing pipeline.
"""

import os
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).parent

# Directory structure (refactored for proper workflow)
DOWNLOADS_DIR = BASE_DIR / "downloads"
WORKING_DIR = BASE_DIR / "working"
WAV_DIR = WORKING_DIR / "wav"
DENOISED_DIR = WORKING_DIR / "denoised"
ENHANCED_DIR = WORKING_DIR / "enhanced"
OUTPUT_DIR = BASE_DIR / "output"
FINAL_DIR = OUTPUT_DIR / "final"
TRANSCRIPT_DIR = OUTPUT_DIR / "transcripts"
LOGS_DIR = BASE_DIR / "logs"

# Legacy directories (for backward compatibility)
RAW_DIR = DOWNLOADS_DIR
MP3_DIR = DOWNLOADS_DIR
CLEAN_DIR = FINAL_DIR
SUMMARY_DIR = OUTPUT_DIR / "summaries"

# Create directories if they don't exist
for dir_path in [
    DOWNLOADS_DIR, WORKING_DIR, WAV_DIR, DENOISED_DIR, ENHANCED_DIR,
    OUTPUT_DIR, FINAL_DIR, TRANSCRIPT_DIR, SUMMARY_DIR, LOGS_DIR
]:
    dir_path.mkdir(parents=True, exist_ok=True)

# yt-dlp settings
YDLP_FORMAT = "bestaudio/best"
YDLP_COOKIES_FILE = os.getenv("YDLP_COOKIES_FILE", "cookies.txt")
YDLP_POSTPROCESSORS = []  # Disabled - we'll convert to WAV manually

# Audio processing settings
AUDIO_BITRATE = "256k"  # High quality MP3
AUDIO_SAMPLE_RATE = 48000  # Higher sample rate for better quality
WAV_BIT_DEPTH = 16  # 16-bit or 32-bit float

# FFmpeg loudnorm settings (EBU R128)
LOUDNORM_SETTINGS = {
    "i": "-16",  # Integrated loudness target (LUFS)
    "tp": "-1.5",  # True peak (dBTP)
    "lra": "11",  # Loudness range (LU)
}

# Noise suppression settings
NOISE_SUPPRESSION_ENABLED = True
NOISE_SUPPRESSION_STRENGTH = 0.5  # 0.0 to 1.0
NOISE_SUPPRESSION_METHOD = "ffmpeg"  # "ffmpeg" or "deepfilter" or "openvino"

# DeepFilterNet settings
DEEPFILTER_MODEL = "deepfilter_net"

# Audio super resolution settings
SUPER_RESOLUTION_ENABLED = False
SUPER_RESOLUTION_MODEL = "basic"  # Model to use for enhancement

# Whisper settings
TRANSCRIPTION_ENABLED = False  # CRITICAL: Disabled by default to avoid RecursionError
WHISPER_MODEL = "distil-large-v3"  # Options: tiny, base, small, medium, large-v3, distil-large-v3
WHISPER_DEVICE = "auto"  # auto, cpu, cuda
WHISPER_COMPUTE_TYPE = "int8"  # int8 for CPU compatibility, float16 for GPU

# LLM settings for summarization
SUMMARIZATION_ENABLED = False
LLM_API_KEY = os.getenv("OPENAI_API_KEY", "")
LLM_MODEL = "gpt-4o"
LLM_TEMPERATURE = 0.7
LLM_MAX_TOKENS = 2000

# Processing options
SKIP_EXISTING = True  # Skip files that already exist
DELETE_INTERMEDIATE = False  # Delete WAV files after processing
REMOVE_SILENCE = False
SILENCE_THRESHOLD = "-50dB"
MIN_SILENCE_DURATION = "1"

# Logging settings
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_FILE = LOGS_DIR / "pipeline.log"
CONSOLE_LOG_LEVEL = "INFO"  # Separate console log level
