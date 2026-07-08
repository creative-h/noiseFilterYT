"""
Remove noise from audio files using DeepFilterNet.
"""

import subprocess
import logging
from pathlib import Path
from typing import Optional
from config import (
    CLEAN_DIR,
    DEEPFILTER_MODEL,
    LOG_LEVEL,
    LOG_FORMAT,
    LOG_FILE,
)

logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT, filename=LOG_FILE)
logger = logging.getLogger(__name__)


class AudioDenoiser:
    """Remove noise from audio files using DeepFilterNet."""

    def __init__(self, output_dir: Optional[Path] = None):
        """
        Initialize the denoiser.

        Args:
            output_dir: Directory to save denoised files
        """
        self.output_dir = output_dir or CLEAN_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def denoise(
        self,
        input_file: Path,
        output_file: Optional[Path] = None,
        model: str = DEEPFILTER_MODEL,
    ) -> Path:
        """
        Remove noise from an audio file using DeepFilterNet.

        Args:
            input_file: Path to input audio file
            output_file: Path to output denoised file (optional)
            model: DeepFilterNet model to use

        Returns:
            Path to the denoised audio file
        """
        if not input_file.exists():
            raise FileNotFoundError(f"Input file not found: {input_file}")

        if output_file is None:
            output_file = self.output_dir / f"{input_file.stem}_clean.mp3"

        # Skip if output already exists
        if output_file.exists():
            logger.info(f"Output file already exists, skipping: {output_file}")
            return output_file

        logger.info(f"Denoising {input_file} using DeepFilterNet")

        cmd = [
            "deepFilter",
            str(input_file),
            "-o",
            str(output_file),
        ]

        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            logger.info(f"Successfully denoised to: {output_file}")
            return output_file

        except subprocess.CalledProcessError as e:
            logger.error(f"Error denoising file: {e.stderr}")
            raise

    def batch_denoise(
        self,
        input_dir: Path,
        pattern: str = "*.mp3",
        model: str = DEEPFILTER_MODEL,
    ) -> list[Path]:
        """
        Denoise all audio files in a directory.

        Args:
            input_dir: Directory containing audio files
            pattern: Glob pattern to match files
            model: DeepFilterNet model to use

        Returns:
            List of denoised file paths
        """
        logger.info(f"Batch denoising files in {input_dir}")

        denoised_files = []
        for input_file in input_dir.glob(pattern):
            if input_file.is_file():
                try:
                    output_file = self.denoise(input_file, model=model)
                    denoised_files.append(output_file)
                except Exception as e:
                    logger.error(f"Error denoising {input_file}: {e}")

        logger.info(f"Denoised {len(denoised_files)} files")
        return denoised_files
