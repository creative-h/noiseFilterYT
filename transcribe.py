"""
Transcribe audio files using Faster-Whisper.

This module implements lazy initialization to avoid loading the Whisper model
unless transcription is actually enabled and needed. This prevents the
RecursionError that occurs when the model is loaded unconditionally.
"""

import logging
from pathlib import Path
from typing import Optional, Generator
from config import (
    TRANSCRIPT_DIR,
    WHISPER_MODEL,
    WHISPER_DEVICE,
    WHISPER_COMPUTE_TYPE,
    TRANSCRIPTION_ENABLED,
    LOG_LEVEL,
    LOG_FORMAT,
    LOG_FILE,
)

logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT, filename=LOG_FILE)
logger = logging.getLogger(__name__)


class AudioTranscriber:
    """Transcribe audio files using Faster-Whisper with lazy initialization."""

    def __init__(
        self,
        output_dir: Optional[Path] = None,
        model: str = WHISPER_MODEL,
        device: str = WHISPER_DEVICE,
        compute_type: str = WHISPER_COMPUTE_TYPE,
        enabled: bool = TRANSCRIPTION_ENABLED,
    ):
        """
        Initialize the transcriber.

        IMPORTANT: The Whisper model is NOT loaded here. It will be loaded
        lazily on first use to avoid RecursionError and unnecessary resource usage.

        Args:
            output_dir: Directory to save transcripts
            model: Whisper model size
            device: Device to run on (auto, cpu, cuda)
            compute_type: Compute type (float16, int8, float32)
            enabled: Whether transcription is enabled
        """
        self.output_dir = output_dir or TRANSCRIPT_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = model
        self.device = device
        self.compute_type = compute_type
        self.enabled = enabled
        self._model = None  # Lazy-loaded model

    def _load_model(self):
        """Load the Whisper model lazily."""
        if self._model is None:
            try:
                logger.info(f"Loading Whisper model: {self.model_name}")
                from faster_whisper import WhisperModel
                self._model = WhisperModel(
                    self.model_name,
                    device=self.device,
                    compute_type=self.compute_type
                )
                logger.info("Whisper model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load Whisper model: {e}")
                self._model = None
                raise

    @property
    def model(self):
        """Get the Whisper model, loading it lazily if needed."""
        if self._model is None:
            self._load_model()
        return self._model

    def transcribe(
        self,
        input_file: Path,
        output_file: Optional[Path] = None,
        language: Optional[str] = None,
        task: str = "transcribe",
        word_timestamps: bool = False,
    ) -> Optional[Path]:
        """
        Transcribe an audio file to text.

        Args:
            input_file: Path to input audio file
            output_file: Path to output transcript file (optional)
            language: Language code (e.g., 'en', 'es'). Auto-detect if None
            task: Task type (transcribe or translate)
            word_timestamps: Include word-level timestamps

        Returns:
            Path to the transcript file, or None if transcription is disabled
        """
        if not self.enabled:
            logger.info("Transcription disabled, skipping")
            return None

        if not input_file.exists():
            raise FileNotFoundError(f"Input file not found: {input_file}")

        if output_file is None:
            output_file = self.output_dir / f"{input_file.stem}.txt"

        # Skip if output already exists
        if output_file.exists():
            logger.info(f"Output file already exists, skipping: {output_file}")
            return output_file

        logger.info(f"Transcribing {input_file}")

        try:
            segments, info = self.model.transcribe(
                str(input_file),
                language=language,
                task=task,
                word_timestamps=word_timestamps,
            )

            logger.info(f"Detected language: {info.language} (probability: {info.language_probability:.2f})")

            # Write transcript to file
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(f"Language: {info.language}\n")
                f.write(f"Duration: {info.duration:.2f} seconds\n")
                f.write("=" * 80 + "\n\n")

                for segment in segments:
                    if word_timestamps:
                        f.write(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}\n")
                    else:
                        f.write(f"{segment.text}\n")

            logger.info(f"Successfully transcribed to: {output_file}")
            return output_file

        except Exception as e:
            logger.error(f"Error transcribing file: {e}")
            # Return None instead of raising to allow pipeline to continue
            return None

    def transcribe_with_timestamps(
        self,
        input_file: Path,
        output_file: Optional[Path] = None,
        language: Optional[str] = None,
    ) -> Path:
        """
        Transcribe with segment-level timestamps.

        Args:
            input_file: Path to input audio file
            output_file: Path to output transcript file (optional)
            language: Language code

        Returns:
            Path to the transcript file
        """
        return self.transcribe(
            input_file,
            output_file=output_file,
            language=language,
            word_timestamps=True,
        )

    def batch_transcribe(
        self,
        input_dir: Path,
        pattern: str = "*.mp3",
        language: Optional[str] = None,
    ) -> list[Path]:
        """
        Transcribe all audio files in a directory.

        Args:
            input_dir: Directory containing audio files
            pattern: Glob pattern to match files
            language: Language code

        Returns:
            List of transcript file paths
        """
        logger.info(f"Batch transcribing files in {input_dir}")

        transcript_files = []
        for input_file in input_dir.glob(pattern):
            if input_file.is_file():
                try:
                    output_file = self.transcribe(input_file, language=language)
                    transcript_files.append(output_file)
                except Exception as e:
                    logger.error(f"Error transcribing {input_file}: {e}")

        logger.info(f"Transcribed {len(transcript_files)} files")
        return transcript_files

    def get_segments(
        self,
        input_file: Path,
        language: Optional[str] = None,
    ) -> Generator:
        """
        Get transcription segments as a generator.

        Args:
            input_file: Path to input audio file
            language: Language code

        Yields:
            Transcription segments
        """
        if not input_file.exists():
            raise FileNotFoundError(f"Input file not found: {input_file}")

        logger.info(f"Getting segments for {input_file}")

        segments, info = self.model.transcribe(
            str(input_file),
            language=language,
        )

        logger.info(f"Detected language: {info.language}")

        for segment in segments:
            yield segment
