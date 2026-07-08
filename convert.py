"""
Convert audio files to MP3 format using FFmpeg.
"""

import subprocess
import logging
from pathlib import Path
from typing import Optional
from config import (
    MP3_DIR,
    AUDIO_BITRATE,
    AUDIO_SAMPLE_RATE,
    LOG_LEVEL,
    LOG_FORMAT,
    LOG_FILE,
)

logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT, filename=LOG_FILE)
logger = logging.getLogger(__name__)


class AudioConverter:
    """Convert audio files to MP3 format."""

    def __init__(self, output_dir: Optional[Path] = None):
        """
        Initialize the converter.

        Args:
            output_dir: Directory to save converted files
        """
        self.output_dir = output_dir or MP3_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)

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
            output_file = self.output_dir / f"{input_file.stem}.mp3"

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
        bitrate: str = AUDIO_BITRATE,
    ) -> list[Path]:
        """
        Convert all audio files in a directory to MP3.

        Args:
            input_dir: Directory containing audio files
            pattern: Glob pattern to match files
            bitrate: Audio bitrate

        Returns:
            List of converted file paths
        """
        logger.info(f"Batch converting files in {input_dir}")

        converted_files = []
        for input_file in input_dir.glob(pattern):
            if input_file.is_file():
                try:
                    output_file = self.convert_to_mp3(input_file, bitrate=bitrate)
                    converted_files.append(output_file)
                except Exception as e:
                    logger.error(f"Error converting {input_file}: {e}")

        logger.info(f"Converted {len(converted_files)} files")
        return converted_files
