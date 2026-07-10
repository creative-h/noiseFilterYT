"""
Remove noise from audio files using DeepFilterNet or FFmpeg.
"""

import subprocess
import shutil
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
    """Remove noise from audio files using DeepFilterNet or FFmpeg."""

    def __init__(self, output_dir: Optional[Path] = None):
        """
        Initialize the denoiser.

        Args:
            output_dir: Directory to save denoised files
        """
        self.output_dir = output_dir or CLEAN_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Check if deepFilter is available
        self.deepfilter_available = shutil.which("deepFilter") is not None
        if not self.deepfilter_available:
            logger.warning("deepFilter command not found. Will use FFmpeg for basic noise reduction instead.")

    def denoise(
        self,
        input_file: Path,
        output_file: Optional[Path] = None,
        model: str = DEEPFILTER_MODEL,
    ) -> Path:
        """
        Remove noise from an audio file using DeepFilterNet or FFmpeg.

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

        # Try DeepFilterNet first if available
        if self.deepfilter_available:
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
                logger.error(f"Error denoising with DeepFilterNet: {e.stderr}")
                # Fall through to FFmpeg method
        else:
            logger.info(f"DeepFilterNet not available, using FFmpeg for basic noise reduction")

        # Fallback to FFmpeg for basic noise reduction
        logger.info(f"Applying FFmpeg noise reduction to {input_file}")
        return self._denoise_ffmpeg(input_file, output_file)

    def _denoise_ffmpeg(self, input_file: Path, output_file: Path) -> Path:
        """
        Apply speech enhancement using FFmpeg with multiple filters.

        For speech/podcast audio, this applies:
        1. Highpass filter (80Hz) - removes rumble and low-frequency noise
        2. Lowpass filter (8000Hz) - removes high-frequency hiss
        3. afftdn (FFT denoise) - removes background noise
        4. compand (compressor) - evens out volume levels
        5. equalizer - boosts speech frequencies (2-4kHz)

        Args:
            input_file: Path to input audio file
            output_file: Path to output denoised file

        Returns:
            Path to the denoised audio file
        """
        # Apply speech enhancement filters
        # afftdn: FFT-based noise reduction with adaptive noise floor
        # highpass/lowpass: Remove frequencies outside speech range (80Hz-8000Hz)
        # compand: Dynamic range compression for clearer speech
        # equalizer: Boost speech-critical frequencies (2.5kHz)
        audio_filter = (
            "highpass=f=80,lowpass=f=8000,"
            "afftdn=nf=-25:tn=1,"
            "compand=.3|.3:1| -60/-60|0/-10:6: -90/-60/-60/-60/-60:0.001:0.01:0.01,"
            "equalizer=f=2500:width_type=h:width=500:g=3"
        )
        
        cmd = [
            "ffmpeg",
            "-i",
            str(input_file),
            "-af",
            audio_filter,
            "-y",
            str(output_file),
        ]

        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            logger.info(f"Successfully applied speech enhancement to: {output_file}")
            return output_file
        except subprocess.CalledProcessError as e:
            logger.error(f"Error applying speech enhancement: {e.stderr}")
            # Try simpler filter chain if complex one fails
            logger.info("Trying simpler filter chain...")
            return self._denoise_ffmpeg_simple(input_file, output_file)

    def _denoise_ffmpeg_simple(self, input_file: Path, output_file: Path) -> Path:
        """
        Apply simpler noise reduction using FFmpeg (fallback).

        Args:
            input_file: Path to input audio file
            output_file: Path to output denoised file

        Returns:
            Path to the denoised audio file
        """
        # Simpler filter: just frequency filtering and basic noise reduction
        audio_filter = (
            "highpass=f=100,lowpass=f=7000,"
            "afftdn=nf=-20:tn=1"
        )
        
        cmd = [
            "ffmpeg",
            "-i",
            str(input_file),
            "-af",
            audio_filter,
            "-y",
            str(output_file),
        ]

        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            logger.info(f"Successfully applied simple noise reduction to: {output_file}")
            return output_file
        except subprocess.CalledProcessError as e:
            logger.error(f"Error applying simple noise reduction: {e.stderr}")
            # If FFmpeg also fails, just copy the file
            logger.warning(f"Copying original file to output as fallback")
            shutil.copy2(input_file, output_file)
            logger.info(f"Copied original file to: {output_file}")
            return output_file

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
