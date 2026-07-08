"""
Normalize audio loudness using FFmpeg loudnorm filter.
"""

import subprocess
import logging
from pathlib import Path
from typing import Optional
from config import (
    CLEAN_DIR,
    LOUDNORM_SETTINGS,
    LOG_LEVEL,
    LOG_FORMAT,
    LOG_FILE,
)

logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT, filename=LOG_FILE)
logger = logging.getLogger(__name__)


class AudioNormalizer:
    """Normalize audio loudness using FFmpeg."""

    def __init__(self, output_dir: Optional[Path] = None):
        """
        Initialize the normalizer.

        Args:
            output_dir: Directory to save normalized files
        """
        self.output_dir = output_dir or CLEAN_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def normalize(
        self,
        input_file: Path,
        output_file: Optional[Path] = None,
        loudnorm_settings: Optional[dict] = None,
    ) -> Path:
        """
        Normalize audio loudness using FFmpeg loudnorm.

        Args:
            input_file: Path to input audio file
            output_file: Path to output normalized file (optional)
            loudnorm_settings: Dictionary with loudnorm settings

        Returns:
            Path to the normalized audio file
        """
        if not input_file.exists():
            raise FileNotFoundError(f"Input file not found: {input_file}")

        if output_file is None:
            output_file = self.output_dir / f"{input_file.stem}_normalized.mp3"

        # Skip if output already exists
        if output_file.exists():
            logger.info(f"Output file already exists, skipping: {output_file}")
            return output_file

        settings = loudnorm_settings or LOUDNORM_SETTINGS

        logger.info(f"Normalizing {input_file}")

        # Two-pass loudnorm for accurate normalization
        # First pass: measure loudness
        cmd_pass1 = [
            "ffmpeg",
            "-i",
            str(input_file),
            "-af",
            f"loudnorm=I={settings['i']}:TP={settings['tp']}:LRA={settings['lra']}:print_format=json",
            "-f",
            "null",
            "-",
        ]

        try:
            result = subprocess.run(
                cmd_pass1, check=True, capture_output=True, text=True
            )

            # Parse the loudnorm stats from stderr
            import json
            import re

            match = re.search(r"\{.*\}", result.stderr)
            if match:
                stats = json.loads(match.group())
                measured_i = stats.get("input_i", settings["i"])
                measured_tp = stats.get("input_tp", settings["tp"])
                measured_lra = stats.get("input_lra", settings["lra"])
                measured_thresh = stats.get("input_thresh", settings["i"])

                # Second pass: apply normalization
                cmd_pass2 = [
                    "ffmpeg",
                    "-i",
                    str(input_file),
                    "-af",
                    f"loudnorm=I={settings['i']}:TP={settings['tp']}:LRA={settings['lra']}:measured_I={measured_i}:measured_TP={measured_tp}:measured_LRA={measured_lra}:measured_thresh={measured_thresh}",
                    "-y",
                    str(output_file),
                ]

                subprocess.run(cmd_pass2, check=True, capture_output=True, text=True)
                logger.info(f"Successfully normalized to: {output_file}")
                return output_file
            else:
                # Fallback to single-pass if stats parsing fails
                logger.warning("Could not parse loudnorm stats, using single-pass")
                cmd_fallback = [
                    "ffmpeg",
                    "-i",
                    str(input_file),
                    "-af",
                    f"loudnorm=I={settings['i']}:TP={settings['tp']}:LRA={settings['lra']}",
                    "-y",
                    str(output_file),
                ]
                subprocess.run(cmd_fallback, check=True, capture_output=True, text=True)
                logger.info(f"Successfully normalized (single-pass) to: {output_file}")
                return output_file

        except subprocess.CalledProcessError as e:
            logger.error(f"Error normalizing file: {e.stderr}")
            raise

    def batch_normalize(
        self,
        input_dir: Path,
        pattern: str = "*.mp3",
        loudnorm_settings: Optional[dict] = None,
    ) -> list[Path]:
        """
        Normalize all audio files in a directory.

        Args:
            input_dir: Directory containing audio files
            pattern: Glob pattern to match files
            loudnorm_settings: Dictionary with loudnorm settings

        Returns:
            List of normalized file paths
        """
        logger.info(f"Batch normalizing files in {input_dir}")

        normalized_files = []
        for input_file in input_dir.glob(pattern):
            if input_file.is_file():
                try:
                    output_file = self.normalize(input_file, loudnorm_settings=loudnorm_settings)
                    normalized_files.append(output_file)
                except Exception as e:
                    logger.error(f"Error normalizing {input_file}: {e}")

        logger.info(f"Normalized {len(normalized_files)} files")
        return normalized_files

    def remove_silence(
        self,
        input_file: Path,
        output_file: Optional[Path] = None,
        threshold: str = "-50dB",
        min_duration: str = "1",
    ) -> Path:
        """
        Remove silence from audio file.

        Args:
            input_file: Path to input audio file
            output_file: Path to output file (optional)
            threshold: Silence threshold
            min_duration: Minimum silence duration to remove

        Returns:
            Path to the processed audio file
        """
        if not input_file.exists():
            raise FileNotFoundError(f"Input file not found: {input_file}")

        if output_file is None:
            output_file = self.output_dir / f"{input_file.stem}_nosilence.mp3"

        logger.info(f"Removing silence from {input_file}")

        cmd = [
            "ffmpeg",
            "-i",
            str(input_file),
            "-af",
            f"silenceremove=start_periods=1:start_silence=0.1:start_threshold={threshold}:detection=peak,aformat=dblp,areverse,silenceremove=start_periods=1:start_silence=0.1:start_threshold={threshold}:detection=peak,aformat=dblp,areverse",
            "-y",
            str(output_file),
        ]

        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            logger.info(f"Successfully removed silence: {output_file}")
            return output_file

        except subprocess.CalledProcessError as e:
            logger.error(f"Error removing silence: {e.stderr}")
            raise
