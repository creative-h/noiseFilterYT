"""
Summarize transcripts using LLM.
"""

import logging
from pathlib import Path
from typing import Optional
from openai import OpenAI
from config import (
    SUMMARY_DIR,
    LLM_API_KEY,
    LLM_MODEL,
    LLM_TEMPERATURE,
    LLM_MAX_TOKENS,
    LOG_LEVEL,
    LOG_FORMAT,
    LOG_FILE,
)

logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT, filename=LOG_FILE)
logger = logging.getLogger(__name__)


class TranscriptSummarizer:
    """Summarize transcripts using LLM."""

    def __init__(
        self,
        output_dir: Optional[Path] = None,
        api_key: Optional[str] = None,
        model: str = LLM_MODEL,
        temperature: float = LLM_TEMPERATURE,
        max_tokens: int = LLM_MAX_TOKENS,
    ):
        """
        Initialize the summarizer.

        Args:
            output_dir: Directory to save summaries
            api_key: OpenAI API key (or set OPENAI_API_KEY env var)
            model: LLM model to use
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
        """
        self.output_dir = output_dir or SUMMARY_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

        api_key = api_key or LLM_API_KEY
        if not api_key:
            logger.warning("No API key provided. Summarization will not work.")
            self.client = None
        else:
            self.client = OpenAI(api_key=api_key)

    def summarize(
        self,
        transcript_file: Path,
        output_file: Optional[Path] = None,
        summary_type: str = "detailed",
    ) -> Path:
        """
        Summarize a transcript file.

        Args:
            transcript_file: Path to transcript file
            output_file: Path to output summary file (optional)
            summary_type: Type of summary (brief, detailed, bullet_points)

        Returns:
            Path to the summary file
        """
        if not transcript_file.exists():
            raise FileNotFoundError(f"Transcript file not found: {transcript_file}")

        if output_file is None:
            output_file = self.output_dir / f"{transcript_file.stem}_summary.md"

        # Skip if output already exists
        if output_file.exists():
            logger.info(f"Output file already exists, skipping: {output_file}")
            return output_file

        if not self.client:
            raise ValueError("No API key configured. Cannot summarize.")

        logger.info(f"Summarizing {transcript_file}")

        # Read transcript
        with open(transcript_file, "r", encoding="utf-8") as f:
            transcript_text = f.read()

        # Define prompts based on summary type
        prompts = {
            "brief": "Provide a concise 2-3 sentence summary of the following transcript:",
            "detailed": "Provide a detailed summary of the following transcript, including key points, main topics, and important takeaways:",
            "bullet_points": "Summarize the following transcript using bullet points, highlighting the main ideas and key information:",
        }

        prompt = prompts.get(summary_type, prompts["detailed"])

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that summarizes educational content accurately and concisely.",
                    },
                    {"role": "user", "content": f"{prompt}\n\n{transcript_text}"},
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

            summary = response.choices[0].message.content

            # Write summary to file
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(f"# Summary\n\n")
                f.write(f"**Source:** {transcript_file.name}\n\n")
                f.write(f"**Type:** {summary_type}\n\n")
                f.write("---\n\n")
                f.write(summary)

            logger.info(f"Successfully summarized to: {output_file}")
            return output_file

        except Exception as e:
            logger.error(f"Error summarizing transcript: {e}")
            raise

    def extract_key_points(
        self,
        transcript_file: Path,
        output_file: Optional[Path] = None,
    ) -> Path:
        """
        Extract key points from a transcript.

        Args:
            transcript_file: Path to transcript file
            output_file: Path to output file (optional)

        Returns:
            Path to the key points file
        """
        if not transcript_file.exists():
            raise FileNotFoundError(f"Transcript file not found: {transcript_file}")

        if output_file is None:
            output_file = self.output_dir / f"{transcript_file.stem}_keypoints.md"

        # Skip if output already exists
        if output_file.exists():
            logger.info(f"Output file already exists, skipping: {output_file}")
            return output_file

        if not self.client:
            raise ValueError("No API key configured. Cannot extract key points.")

        logger.info(f"Extracting key points from {transcript_file}")

        with open(transcript_file, "r", encoding="utf-8") as f:
            transcript_text = f.read()

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that extracts key learning points from educational content.",
                    },
                    {
                        "role": "user",
                        "content": f"Extract the key learning points from the following transcript. Format as a numbered list:\n\n{transcript_text}",
                    },
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

            key_points = response.choices[0].message.content

            with open(output_file, "w", encoding="utf-8") as f:
                f.write(f"# Key Learning Points\n\n")
                f.write(f"**Source:** {transcript_file.name}\n\n")
                f.write("---\n\n")
                f.write(key_points)

            logger.info(f"Successfully extracted key points to: {output_file}")
            return output_file

        except Exception as e:
            logger.error(f"Error extracting key points: {e}")
            raise

    def batch_summarize(
        self,
        input_dir: Path,
        pattern: str = "*.txt",
        summary_type: str = "detailed",
    ) -> list[Path]:
        """
        Summarize all transcript files in a directory.

        Args:
            input_dir: Directory containing transcript files
            pattern: Glob pattern to match files
            summary_type: Type of summary

        Returns:
            List of summary file paths
        """
        logger.info(f"Batch summarizing files in {input_dir}")

        summary_files = []
        for transcript_file in input_dir.glob(pattern):
            if transcript_file.is_file():
                try:
                    output_file = self.summarize(transcript_file, summary_type=summary_type)
                    summary_files.append(output_file)
                except Exception as e:
                    logger.error(f"Error summarizing {transcript_file}: {e}")

        logger.info(f"Summarized {len(summary_files)} files")
        return summary_files
