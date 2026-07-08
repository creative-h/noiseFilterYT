"""
Transcribe audio files using Faster-Whisper.
"""

import logging
from pathlib import Path
from typing import Optional, Generator
from faster_whisper import WhisperModel
from config import (
    TRANSCRIPT_DIR,
    WHISPER_MODEL,
    WHISPER_DEVICE,
    WHISPER_COMPUTE_TYPE,
    LOG_LEVEL,
    LOG_FORMAT,
    LOG_FILE,
)

logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT, filename=LOG_FILE)
logger = logging.getLogger(__name__)


class AudioTranscriber:
    """Transcribe audio files using Faster-Whisper."""

    def __init__(
        self,
        output_dir: Optional[Path] = None,
        model: str = WHISPER_MODEL,
        device: str = WHISPER_DEVICE,
        compute_type: str = WHISPER_COMPUTE_TYPE,
    ):
        """
        Initialize the transcriber.

        Args:
            output_dir: Directory to save transcripts
            model: Whisper model size
            device: Device to run on (auto, cpu, cuda)
            compute_type: Compute type (float16, int8, float32)
        """
        self.output_dir = output_dir or TRANSCRIPT_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = model
        self.device = device
        self.compute_type = compute_type

        logger.info(f"Loading Whisper model: {model}")
        self.model = WhisperModel(model, device=device, compute_type=compute_type)

    def transcribe(
        self,
        input_file: Path,
        output_file: Optional[Path] = None,
        language: Optional[str] = None,
        task: str = "transcribe",
        word_timestamps: bool = False,
    ) -> Path:
        """
        Transcribe an audio file to text.

        Args:
            input_file: Path to input audio file
            output_file: Path to output transcript file (optional)
            language: Language code (e.g., 'en', 'es'). Auto-detect if None
            task: Task type (transcribe or translate)
            word_timestamps: Include word-level timestamps

        Returns:
            Path to the transcript file
        """
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
            raise

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
