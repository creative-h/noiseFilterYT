"""
Convert audio files to WAV and MP3 formats using FFmpeg.
"""

import subprocess
import logging
from pathlib import Path
from typing import Optional
from config import (
    WAV_DIR,
    MP3_DIR,
    FINAL_DIR,
    AUDIO_BITRATE,
    AUDIO_SAMPLE_RATE,
    WAV_BIT_DEPTH,
    LOG_LEVEL,
    LOG_FORMAT,
    LOG_FILE,
)

logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT, filename=LOG_FILE)
logger = logging.getLogger(__name__)


class AudioConverter:
    """Convert audio files to WAV and MP3 formats."""

    def __init__(self, wav_dir: Optional[Path] = None, mp3_dir: Optional[Path] = None):
        """
        Initialize the converter.

        Args:
            wav_dir: Directory to save WAV files
            mp3_dir: Directory to save MP3 files
        """
        self.wav_dir = wav_dir or WAV_DIR
        self.mp3_dir = mp3_dir or MP3_DIR
        self.wav_dir.mkdir(parents=True, exist_ok=True)
        self.mp3_dir.mkdir(parents=True, exist_ok=True)

    def convert_to_wav(
        self,
        input_file: Path,
        output_file: Optional[Path] = None,
        sample_rate: int = AUDIO_SAMPLE_RATE,
        bit_depth: int = WAV_BIT_DEPTH,
    ) -> Path:
        """
        Convert an audio file to WAV format for AI processing.

        Args:
            input_file: Path to input audio file
            output_file: Path to output WAV file (optional)
            sample_rate: Sample rate in Hz
            bit_depth: Bit depth (16 or 32)

        Returns:
            Path to the converted WAV file
        """
        if not input_file.exists():
            raise FileNotFoundError(f"Input file not found: {input_file}")

        if output_file is None:
            output_file = self.wav_dir / f"{input_file.stem}.wav"

        # Skip if output already exists
        if output_file.exists():
            logger.info(f"Output file already exists, skipping: {output_file}")
            return output_file

        logger.info(f"Converting {input_file} to WAV")

        # Use PCM codec for lossless intermediate format
        codec = "pcm_s16le" if bit_depth == 16 else "pcm_f32le"
        
        cmd = [
            "ffmpeg",
            "-i",
            str(input_file),
            "-codec:a",
            codec,
            "-ar",
            str(sample_rate),
            "-ac",
            "1",  # Mono for speech processing
            "-y",
            str(output_file),
        ]

        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            logger.info(f"Successfully converted to WAV: {output_file}")
            return output_file

        except subprocess.CalledProcessError as e:
            logger.error(f"Error converting to WAV: {e.stderr}")
            raise

    def convert_to_mp3(
        self,
        input_file: Path,
        output_file: Optional[Path] = None,
        bitrate: str = AUDIO_BITRATE,
        sample_rate: int = AUDIO_SAMPLE_RATE,
    ) -> Path:
        """
        Convert an audio file to MP3.

        Args:
            input_file: Path to input audio file
            output_file: Path to output MP3 file (optional)
            bitrate: Audio bitrate (e.g., "192k")
            sample_rate: Sample rate in Hz

        Returns:
            Path to the converted MP3 file
        """
        if not input_file.exists():
            raise FileNotFoundError(f"Input file not found: {input_file}")

        if output_file is None:
            output_file = self.mp3_dir / f"{input_file.stem}.mp3"

        # Skip if output already exists
        if output_file.exists():
            logger.info(f"Output file already exists, skipping: {output_file}")
            return output_file

        logger.info(f"Converting {input_file} to MP3")

        cmd = [
            "ffmpeg",
            "-i",
            str(input_file),
            "-codec:a",
            "libmp3lame",
            "-b:a",
            bitrate,
            "-ar",
            str(sample_rate),
            "-y",  # Overwrite output file
            str(output_file),
        ]

        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            logger.info(f"Successfully converted to: {output_file}")
            return output_file

        except subprocess.CalledProcessError as e:
            logger.error(f"Error converting file: {e.stderr}")
            raise

    def batch_convert(
        self,
        input_dir: Path,
        pattern: str = "*",
        to_wav: bool = True,
        bitrate: str = AUDIO_BITRATE,
    ) -> list[Path]:
        """
        Convert all audio files in a directory.

        Args:
            input_dir: Directory containing audio files
            pattern: Glob pattern to match files
            to_wav: If True, convert to WAV; if False, convert to MP3
            bitrate: Audio bitrate (for MP3 conversion)

        Returns:
            List of converted file paths
        """
        logger.info(f"Batch converting files in {input_dir} to {'WAV' if to_wav else 'MP3'}")

        converted_files = []
        for input_file in input_dir.glob(pattern):
            if input_file.is_file():
                try:
                    if to_wav:
                        output_file = self.convert_to_wav(input_file)
                    else:
                        output_file = self.convert_to_mp3(input_file, bitrate=bitrate)
                    converted_files.append(output_file)
                except Exception as e:
                    logger.error(f"Error converting {input_file}: {e}")

        logger.info(f"Converted {len(converted_files)} files")
        return converted_files
