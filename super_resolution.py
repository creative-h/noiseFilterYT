"""
Audio super-resolution module for enhancing low-quality audio.

This module provides optional audio enhancement capabilities to improve
the quality of low-bitrate or low-sample-rate audio sources.

Note: This is an optional feature and should only be used when the source
audio quality is poor and would benefit from enhancement.
"""

import subprocess
import logging
from pathlib import Path
from typing import Optional
from config import (
    ENHANCED_DIR,
    SUPER_RESOLUTION_ENABLED,
    SUPER_RESOLUTION_MODEL,
    AUDIO_SAMPLE_RATE,
    LOG_LEVEL,
    LOG_FORMAT,
    LOG_FILE,
)

logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT, filename=LOG_FILE)
logger = logging.getLogger(__name__)


class AudioEnhancer:
    """Enhance audio quality using super-resolution techniques."""

    def __init__(
        self,
        output_dir: Optional[Path] = None,
        enabled: bool = SUPER_RESOLUTION_ENABLED,
        model: str = SUPER_RESOLUTION_MODEL,
    ):
        """
        Initialize the audio enhancer.

        Args:
            output_dir: Directory to save enhanced files
            enabled: Whether super-resolution is enabled
            model: Model to use for enhancement
        """
        self.output_dir = output_dir or ENHANCED_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.enabled = enabled
        self.model = model
        self.target_sample_rate = AUDIO_SAMPLE_RATE

    def enhance(
        self,
        input_file: Path,
        output_file: Optional[Path] = None,
        target_sr: int = None,
    ) -> Path:
        """
        Enhance audio quality using FFmpeg-based upsampling.

        This method uses FFmpeg's resampling filters to improve audio quality
        by upsampling to a higher sample rate and applying gentle enhancement.

        Note: True AI-based super-resolution would require models like
        AudioSR or similar, which are not included here due to complexity
        and resource requirements. This implementation provides a practical
        alternative using FFmpeg's high-quality resampling.

        Args:
            input_file: Path to input audio file
            output_file: Path to output enhanced file (optional)
            target_sr: Target sample rate (uses config default if None)

        Returns:
            Path to the enhanced audio file
        """
        if not input_file.exists():
            raise FileNotFoundError(f"Input file not found: {input_file}")

        if not self.enabled:
            logger.info("Super-resolution disabled, copying original file")
            if output_file is None:
                output_file = self.output_dir / f"{input_file.stem}.wav"
            import shutil
            shutil.copy2(input_file, output_file)
            return output_file

        if output_file is None:
            output_file = self.output_dir / f"{input_file.stem}.wav"

        # Skip if output already exists
        if output_file.exists():
            logger.info(f"Output file already exists, skipping: {output_file}")
            return output_file

        target_sr = target_sr or self.target_sample_rate
        logger.info(f"Enhancing {input_file} to {target_sr}Hz")

        # Use FFmpeg's high-quality resampling with gentle enhancement
        # The 'aresample' filter with high-quality settings
        # Optional: Add subtle EQ enhancement
        audio_filter = f"aresample=resampler=soxr:osf=32:tsf=32:precision=28:cheby=1"

        cmd = [
            "ffmpeg",
            "-i",
            str(input_file),
            "-af",
            audio_filter,
            "-ar",
            str(target_sr),
            "-y",
            str(output_file),
        ]

        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            logger.info(f"Successfully enhanced audio: {output_file}")
            return output_file

        except subprocess.CalledProcessError as e:
            logger.error(f"Error enhancing audio: {e.stderr}")
            # Fallback: just resample without enhancement
            logger.info("Trying basic resampling...")
            return self._basic_resample(input_file, output_file, target_sr)

    def _basic_resample(
        self,
        input_file: Path,
        output_file: Path,
        target_sr: int,
    ) -> Path:
        """
        Basic resampling fallback.

        Args:
            input_file: Path to input audio file
            output_file: Path to output file
            target_sr: Target sample rate

        Returns:
            Path to the resampled audio file
        """
        cmd = [
            "ffmpeg",
            "-i",
            str(input_file),
            "-ar",
            str(target_sr),
            "-y",
            str(output_file),
        ]

        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            logger.info(f"Successfully resampled to: {output_file}")
            return output_file
        except subprocess.CalledProcessError as e:
            logger.error(f"Error resampling: {e.stderr}")
            # If even basic resampling fails, copy the file
            import shutil
            logger.warning(f"Copying original file as fallback")
            shutil.copy2(input_file, output_file)
            return output_file

    def batch_enhance(
        self,
        input_dir: Path,
        pattern: str = "*.wav",
        target_sr: int = None,
    ) -> list[Path]:
        """
        Enhance all audio files in a directory.

        Args:
            input_dir: Directory containing audio files
            pattern: Glob pattern to match files
            target_sr: Target sample rate

        Returns:
            List of enhanced file paths
        """
        logger.info(f"Batch enhancing files in {input_dir}")

        enhanced_files = []
        for input_file in input_dir.glob(pattern):
            if input_file.is_file():
                try:
                    output_file = self.enhance(input_file, target_sr=target_sr)
                    enhanced_files.append(output_file)
                except Exception as e:
                    logger.error(f"Error enhancing {input_file}: {e}")

        logger.info(f"Enhanced {len(enhanced_files)} files")
        return enhanced_files
