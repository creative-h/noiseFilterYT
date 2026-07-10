"""
Main orchestration script for the YouTube playlist processing pipeline.
"""

import argparse
import logging
from pathlib import Path
from config import (
    SKIP_EXISTING,
    YDLP_COOKIES_FILE,
    LOG_LEVEL,
    LOG_FORMAT,
    LOG_FILE,
)
from downloader import PlaylistDownloader
from convert import AudioConverter
from denoise import AudioDenoiser
from normalize import AudioNormalizer
from transcribe import AudioTranscriber
from summarize import TranscriptSummarizer

logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT, filename=LOG_FILE)
logger = logging.getLogger(__name__)


class Pipeline:
    """Orchestrate the entire YouTube playlist processing pipeline."""

    def __init__(
        self,
        skip_existing: bool = SKIP_EXISTING,
    ):
        """
        Initialize the pipeline.

        Args:
            skip_existing: Skip files that already exist in output directories
        """
        self.skip_existing = skip_existing

        # Initialize components
        self.downloader = PlaylistDownloader(cookies_file=YDLP_COOKIES_FILE)
        self.converter = AudioConverter()
        self.denoiser = AudioDenoiser()
        self.normalizer = AudioNormalizer()
        self.transcriber = AudioTranscriber()
        self.summarizer = TranscriptSummarizer()

    def run_full_pipeline(
        self,
        playlist_url: str,
        start: int = None,
        end: int = None,
        skip_download: bool = False,
        skip_conversion: bool = False,
        skip_denoise: bool = False,
        skip_normalize: bool = False,
        skip_transcribe: bool = False,
        skip_summarize: bool = False,
    ):
        """
        Run the complete pipeline on a YouTube playlist.

        Args:
            playlist_url: URL of the YouTube playlist
            start: Start index (1-based)
            end: End index (1-based)
            skip_download: Skip download step
            skip_conversion: Skip conversion step
            skip_denoise: Skip denoising step
            skip_normalize: Skip normalization step
            skip_transcribe: Skip transcription step
            skip_summarize: Skip summarization step
        """
        logger.info("=" * 80)
        logger.info("Starting full pipeline")
        logger.info("=" * 80)

        # Step 1: Download playlist
        if not skip_download:
            logger.info("Step 1: Downloading playlist")
            downloaded_files = self.downloader.download_playlist(
                playlist_url,
                start=start,
                end=end,
            )
            logger.info(f"Downloaded {len(downloaded_files)} files")
        else:
            logger.info("Skipping download step")
            downloaded_files = list(Path("raw").glob("*.mp3"))

        # Step 2: Convert to MP3 (if not already MP3)
        if not skip_conversion:
            logger.info("Step 2: Converting to MP3")
            mp3_files = self.converter.batch_convert(Path("raw"))
            logger.info(f"Converted {len(mp3_files)} files")
        else:
            logger.info("Skipping conversion step")
            mp3_files = list(Path("mp3").glob("*.mp3"))

        # Step 3: Denoise audio
        if not skip_denoise:
            logger.info("Step 3: Denoising audio")
            clean_files = self.denoiser.batch_denoise(Path("mp3"))
            logger.info(f"Denoised {len(clean_files)} files")
        else:
            logger.info("Skipping denoising step")
            clean_files = list(Path("clean").glob("*.mp3"))

        # Step 4: Normalize audio
        if not skip_normalize:
            logger.info("Step 4: Normalizing audio")
            normalized_files = self.normalizer.batch_normalize(Path("clean"))
            logger.info(f"Normalized {len(normalized_files)} files")
        else:
            logger.info("Skipping normalization step")

        # Step 5: Transcribe audio
        if not skip_transcribe:
            logger.info("Step 5: Transcribing audio")
            transcript_files = self.transcriber.batch_transcribe(Path("clean"))
            logger.info(f"Transcribed {len(transcript_files)} files")
        else:
            logger.info("Skipping transcription step")

        # Step 6: Summarize transcripts
        if not skip_summarize:
            logger.info("Step 6: Summarizing transcripts")
            summary_files = self.summarizer.batch_summarize(Path("transcript"))
            logger.info(f"Summarized {len(summary_files)} files")
        else:
            logger.info("Skipping summarization step")

        logger.info("=" * 80)
        logger.info("Pipeline completed successfully!")
        logger.info("=" * 80)

    def process_single_video(
        self,
        video_url: str,
        skip_denoise: bool = False,
        skip_normalize: bool = False,
        skip_transcribe: bool = False,
        skip_summarize: bool = False,
    ):
        """
        Process a single video through the pipeline.

        Args:
            video_url: URL of the YouTube video
            skip_denoise: Skip denoising step
            skip_normalize: Skip normalization step
            skip_transcribe: Skip transcription step
            skip_summarize: Skip summarization step
        """
        logger.info("=" * 80)
        logger.info("Processing single video")
        logger.info("=" * 80)

        # Download
        logger.info("Downloading video")
        downloaded_file = self.downloader.download_video(video_url)

        # Convert
        logger.info("Converting to MP3")
        mp3_file = self.converter.convert_to_mp3(downloaded_file)

        # Denoise
        if not skip_denoise:
            logger.info("Denoising audio")
            clean_file = self.denoiser.denoise(mp3_file)
        else:
            clean_file = mp3_file

        # Normalize
        if not skip_normalize:
            logger.info("Normalizing audio")
            normalized_file = self.normalizer.normalize(clean_file)
        else:
            normalized_file = clean_file

        # Transcribe
        if not skip_transcribe:
            logger.info("Transcribing audio")
            transcript_file = self.transcriber.transcribe(normalized_file)
        else:
            transcript_file = None

        # Summarize
        if not skip_summarize and transcript_file:
            logger.info("Summarizing transcript")
            summary_file = self.summarizer.summarize(transcript_file)
        else:
            summary_file = None

        logger.info("=" * 80)
        logger.info("Single video processing completed!")
        logger.info("=" * 80)


def main():
    """Main entry point for the pipeline."""
    parser = argparse.ArgumentParser(
        description="YouTube Playlist Processing Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process entire playlist
  python main.py --playlist https://youtube.com/playlist?list=YOUR_PLAYLIST

  # Process specific range
  python main.py --playlist https://youtube.com/playlist?list=YOUR_PLAYLIST --start 1 --end 5

  # Process single video
  python main.py --video https://youtube.com/watch?v=YOUR_VIDEO

  # Skip certain steps
  python main.py --playlist https://youtube.com/playlist?list=YOUR_PLAYLIST --skip-denoise --skip-summarize
        """,
    )

    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--playlist", help="YouTube playlist URL")
    input_group.add_argument("--video", help="Single YouTube video URL")

    # Range options
    parser.add_argument("--start", type=int, help="Start index (1-based)")
    parser.add_argument("--end", type=int, help="End index (1-based)")

    # Skip options
    parser.add_argument("--skip-download", action="store_true", help="Skip download step")
    parser.add_argument("--skip-conversion", action="store_true", help="Skip conversion step")
    parser.add_argument("--skip-denoise", action="store_true", help="Skip denoising step")
    parser.add_argument("--skip-normalize", action="store_true", help="Skip normalization step")
    parser.add_argument("--skip-transcribe", action="store_true", help="Skip transcription step")
    parser.add_argument("--skip-summarize", action="store_true", help="Skip summarization step")

    args = parser.parse_args()

    # Initialize pipeline
    pipeline = Pipeline()

    # Run pipeline
    if args.playlist:
        pipeline.run_full_pipeline(
            playlist_url=args.playlist,
            start=args.start,
            end=args.end,
            skip_download=args.skip_download,
            skip_conversion=args.skip_conversion,
            skip_denoise=args.skip_denoise,
            skip_normalize=args.skip_normalize,
            skip_transcribe=args.skip_transcribe,
            skip_summarize=args.skip_summarize,
        )
    elif args.video:
        pipeline.process_single_video(
            video_url=args.video,
            skip_denoise=args.skip_denoise,
            skip_normalize=args.skip_normalize,
            skip_transcribe=args.skip_transcribe,
            skip_summarize=args.skip_summarize,
        )


if __name__ == "__main__":
    main()
