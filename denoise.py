"""
Remove noise from audio files using AI-based or FFmpeg methods.
"""

import subprocess
import shutil
import logging
from pathlib import Path
from typing import Optional
from config import (
    DENOISED_DIR,
    DEEPFILTER_MODEL,
    NOISE_SUPPRESSION_ENABLED,
    NOISE_SUPPRESSION_STRENGTH,
    NOISE_SUPPRESSION_METHOD,
    LOG_LEVEL,
    LOG_FORMAT,
    LOG_FILE,
)

logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT, filename=LOG_FILE)
logger = logging.getLogger(__name__)


class AudioDenoiser:
    """Remove noise from audio files using AI-based or FFmpeg methods."""

    def __init__(self, output_dir: Optional[Path] = None, enabled: bool = NOISE_SUPPRESSION_ENABLED):
        """
        Initialize the denoiser.

        Args:
            output_dir: Directory to save denoised files
            enabled: Whether noise suppression is enabled
        """
        self.output_dir = output_dir or DENOISED_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.enabled = enabled
        self.strength = NOISE_SUPPRESSION_STRENGTH
        self.method = NOISE_SUPPRESSION_METHOD
        
        # Check if deepFilter is available
        self.deepfilter_available = shutil.which("deepFilter") is not None
        if not self.deepfilter_available and self.method == "deepfilter":
            logger.warning("deepFilter command not found. Falling back to FFmpeg.")
            self.method = "ffmpeg"
        
        # Note: OpenVINO integration research
        # Audacity's OpenVINO AI Effects plugin uses OpenVINO for noise suppression
        # However, it's designed as an Audacity effect and doesn't provide a direct Python API
        # For automated batch processing, we use FFmpeg with advanced filters as the primary method
        # If OpenVINO direct integration becomes available, it can be added here

    def denoise(
        self,
        input_file: Path,
        output_file: Optional[Path] = None,
        model: str = DEEPFILTER_MODEL,
    ) -> Path:
        """
        Remove noise from an audio file using AI-based or FFmpeg methods.

        Args:
            input_file: Path to input audio file
            output_file: Path to output denoised file (optional)
            model: DeepFilterNet model to use

        Returns:
            Path to the denoised audio file
        """
        if not input_file.exists():
            raise FileNotFoundError(f"Input file not found: {input_file}")

        if not self.enabled:
            logger.info("Noise suppression disabled, copying original file")
            if output_file is None:
                output_file = self.output_dir / f"{input_file.stem}.wav"
            shutil.copy2(input_file, output_file)
            return output_file

        if output_file is None:
            output_file = self.output_dir / f"{input_file.stem}.wav"

        # Skip if output already exists
        if output_file.exists():
            logger.info(f"Output file already exists, skipping: {output_file}")
            return output_file

        # Try DeepFilterNet first if available and method is deepfilter
        if self.method == "deepfilter" and self.deepfilter_available:
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
            logger.info(f"Using FFmpeg for noise reduction (method: {self.method})")

        # Fallback to FFmpeg for noise reduction
        logger.info(f"Applying FFmpeg noise reduction to {input_file}")
        return self._denoise_ffmpeg(input_file, output_file)

    def _denoise_ffmpeg(self, input_file: Path, output_file: Path) -> Path:
        """
        Apply speech enhancement using FFmpeg with advanced filters.
        
        This filter chain is designed to approximate Audacity's OpenVINO noise suppression
        for speech/podcast audio by applying:
        1. Highpass filter (80Hz) - removes rumble and low-frequency noise
        2. Lowpass filter (8000Hz) - removes high-frequency hiss
        3. afftdn (FFT denoise) - removes background noise with adaptive noise floor
        4. deesser - reduces sibilance (harsh 's' sounds)
        5. compand (compressor) - evens out volume levels
        6. equalizer - boosts speech frequencies (2-4kHz)

        Args:
            input_file: Path to input audio file
            output_file: Path to output denoised file

        Returns:
            Path to the denoised audio file
        """
        # Adjust noise reduction based on strength setting
        noise_floor = int(-20 - (self.strength * 15))  # Range: -20 to -35
        
        # Apply speech enhancement filters
        audio_filter = (
            "highpass=f=80,lowpass=f=8000,"
            f"afftdn=nf={noise_floor}:tn=1,"
            "adeclip,"
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
        pattern: str = "*.wav",
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
