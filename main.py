"""
Main orchestration script for the YouTube playlist processing pipeline.

This refactored version implements:
- Lazy initialization of AI models (only load when needed)
- Per-video error isolation (one failure doesn't stop the playlist)
- WAV intermediate format for quality preservation
- Progress tracking with console output
- Stage skipping based on existing files
- Optional transcription (disabled by default)
"""

import argparse
import logging
import shutil
from pathlib import Path
from typing import Optional
from config import (
    SKIP_EXISTING,
    YDLP_COOKIES_FILE,
    DOWNLOADS_DIR,
    WAV_DIR,
    DENOISED_DIR,
    ENHANCED_DIR,
    FINAL_DIR,
    TRANSCRIPT_DIR,
    NOISE_SUPPRESSION_ENABLED,
    SUPER_RESOLUTION_ENABLED,
    TRANSCRIPTION_ENABLED,
    SUMMARIZATION_ENABLED,
    DELETE_INTERMEDIATE,
    AUDIO_BITRATE,
)
from downloader import PlaylistDownloader
from convert import AudioConverter
from denoise import AudioDenoiser
from super_resolution import AudioEnhancer
from normalize import AudioNormalizer
from transcribe import AudioTranscriber
from summarize import TranscriptSummarizer
from utils import setup_logging, ProgressTracker, check_file_exists

# Setup logging
logger = setup_logging()


class Pipeline:
    """Orchestrate the entire YouTube playlist processing pipeline."""

    def __init__(
        self,
        skip_existing: bool = SKIP_EXISTING,
        noise_suppression_enabled: bool = NOISE_SUPPRESSION_ENABLED,
        super_resolution_enabled: bool = SUPER_RESOLUTION_ENABLED,
        transcription_enabled: bool = TRANSCRIPTION_ENABLED,
        summarization_enabled: bool = SUMMARIZATION_ENABLED,
    ):
        """
        Initialize the pipeline with lazy component initialization.

        IMPORTANT: AI models are NOT loaded here. They will be loaded lazily
        when their features are actually used. This prevents RecursionError
        and unnecessary resource usage.

        Args:
            skip_existing: Skip files that already exist
            noise_suppression_enabled: Enable noise suppression
            super_resolution_enabled: Enable audio enhancement
            transcription_enabled: Enable transcription
            summarization_enabled: Enable summarization
        """
        self.skip_existing = skip_existing
        self.noise_suppression_enabled = noise_suppression_enabled
        self.super_resolution_enabled = super_resolution_enabled
        self.transcription_enabled = transcription_enabled
        self.summarization_enabled = summarization_enabled

        # Initialize non-AI components immediately
        self.downloader = PlaylistDownloader(cookies_file=YDLP_COOKIES_FILE)
        self.converter = AudioConverter()

        # AI components - initialized lazily or conditionally
        self.denoiser = None
        self.enhancer = None
        self.normalizer = None
        self.transcriber = None
        self.summarizer = None

    def _get_denoiser(self):
        """Lazy initialize denoiser."""
        if self.denoiser is None:
            self.denoiser = AudioDenoiser(enabled=self.noise_suppression_enabled)
        return self.denoiser

    def _get_enhancer(self):
        """Lazy initialize enhancer."""
        if self.enhancer is None:
            self.enhancer = AudioEnhancer(enabled=self.super_resolution_enabled)
        return self.enhancer

    def _get_normalizer(self):
        """Lazy initialize normalizer."""
        if self.normalizer is None:
            self.normalizer = AudioNormalizer()
        return self.normalizer

    def _get_transcriber(self):
        """Lazy initialize transcriber."""
        if self.transcriber is None and self.transcription_enabled:
            self.transcriber = AudioTranscriber(enabled=self.transcription_enabled)
        return self.transcriber

    def _get_summarizer(self):
        """Lazy initialize summarizer."""
        if self.summarizer is None and self.summarization_enabled:
            self.summarizer = TranscriptSummarizer()
        return self.summarizer

    def run_full_pipeline(
        self,
        playlist_url: str,
        start: int = None,
        end: int = None,
        skip_download: bool = False,
        skip_denoise: bool = False,
        skip_enhance: bool = False,
        skip_normalize: bool = False,
        skip_transcribe: bool = False,
        skip_summarize: bool = False,
    ):
        """
        Run the complete pipeline on a YouTube playlist with per-video error isolation.

        Args:
            playlist_url: URL of the YouTube playlist
            start: Start index (1-based)
            end: End index (1-based)
            skip_download: Skip download step
            skip_denoise: Skip denoising step
            skip_enhance: Skip enhancement step
            skip_normalize: Skip normalization step
            skip_transcribe: Skip transcription step
            skip_summarize: Skip summarization step
        """
        logger.info("=" * 80)
        logger.info("Starting full pipeline")
        logger.info("=" * 80)

        # Get playlist info
        playlist_info = self.downloader.get_playlist_info(playlist_url)
        total_videos = playlist_info["video_count"]
        logger.info(f"Playlist: {playlist_info['title']}")
        logger.info(f"Total videos: {total_videos}")

        # Apply range if specified
        if start is not None:
            start_idx = max(0, start - 1)
        else:
            start_idx = 0
        if end is not None:
            end_idx = min(end, total_videos)
        else:
            end_idx = total_videos

        videos_to_process = playlist_info["entries"][start_idx:end_idx]
        logger.info(f"Processing videos {start_idx + 1} to {end_idx} ({len(videos_to_process)} videos)")

        # Initialize progress tracker
        progress = ProgressTracker(len(videos_to_process), "Processing Playlist")

        # Process each video independently
        for idx, video_info in enumerate(videos_to_process):
            video_url = video_info["url"]
            video_title = video_info["title"]
            video_index = start_idx + idx + 1

            try:
                logger.info(f"\n[{video_index}/{end_idx}] Processing: {video_title}")
                
                # Download
                if not skip_download:
                    downloaded_file = self.downloader.download_video(video_url)
                    logger.info(f"  Downloaded: {downloaded_file.name}")
                else:
                    # Use existing file
                    downloaded_files = list(DOWNLOADS_DIR.glob(f"{video_index:03d}-*"))
                    if downloaded_files:
                        downloaded_file = downloaded_files[0]
                        logger.info(f"  Using existing: {downloaded_file.name}")
                    else:
                        logger.warning(f"  No existing file found for video {video_index}")
                        progress.update("fail", f"Video {video_index}: No file found")
                        continue

                # Convert to WAV
                wav_file = WAV_DIR / f"{downloaded_file.stem}.wav"
                if self.skip_existing and check_file_exists(wav_file):
                    logger.info(f"  WAV already exists, skipping conversion")
                else:
                    wav_file = self.converter.convert_to_wav(downloaded_file)
                    logger.info(f"  Converted to WAV")

                # Denoise
                if not skip_denoise:
                    denoised_file = DENOISED_DIR / f"{wav_file.stem}.wav"
                    if self.skip_existing and check_file_exists(denoised_file):
                        logger.info(f"  Denoised file already exists, skipping")
                    else:
                        denoised_file = self._get_denoiser().denoise(wav_file)
                        logger.info(f"  Denoised audio")
                    current_audio = denoised_file
                else:
                    current_audio = wav_file

                # Enhance (super resolution)
                if not skip_enhance:
                    enhanced_file = ENHANCED_DIR / f"{current_audio.stem}.wav"
                    if self.skip_existing and check_file_exists(enhanced_file):
                        logger.info(f"  Enhanced file already exists, skipping")
                    else:
                        enhanced_file = self._get_enhancer().enhance(current_audio)
                        logger.info(f"  Enhanced audio")
                    current_audio = enhanced_file

                # Normalize
                if not skip_normalize:
                    normalized_file = FINAL_DIR / f"{current_audio.stem}.mp3"
                    if self.skip_existing and check_file_exists(normalized_file):
                        logger.info(f"  Normalized file already exists, skipping")
                    else:
                        normalized_file = self._get_normalizer().normalize(current_audio)
                        logger.info(f"  Normalized audio")
                    current_audio = normalized_file
                else:
                    # Convert to MP3 if not already
                    if current_audio.suffix != ".mp3":
                        current_audio = self.converter.convert_to_mp3(current_audio)

                # Transcribe
                transcript_file = None
                if not skip_transcribe and self.transcription_enabled:
                    transcript_file = TRANSCRIPT_DIR / f"{current_audio.stem}.txt"
                    if self.skip_existing and check_file_exists(transcript_file):
                        logger.info(f"  Transcript already exists, skipping")
                    else:
                        transcript_file = self._get_transcriber().transcribe(current_audio)
                        if transcript_file:
                            logger.info(f"  Transcribed audio")
                        else:
                            logger.warning(f"  Transcription failed")

                # Summarize
                if not skip_summarize and self.summarization_enabled and transcript_file:
                    summary_file = self._get_summarizer().summarize(transcript_file)
                    if summary_file:
                        logger.info(f"  Summarized transcript")

                # Delete intermediate files if configured
                if DELETE_INTERMEDIATE:
                    if wav_file.exists():
                        wav_file.unlink()
                    if current_audio.suffix == ".wav" and current_audio != wav_file:
                        current_audio.unlink()

                progress.update("success", f"Video {video_index}: {video_title}")

            except Exception as e:
                logger.error(f"Error processing video {video_index}: {e}")
                progress.update("fail", f"Video {video_index}: {str(e)}")
                continue

        # Print summary
        logger.info(progress.summary())
        logger.info("=" * 80)
        logger.info("Pipeline completed!")
        logger.info("=" * 80)

    def process_single_video(
        self,
        video_url: str,
        skip_denoise: bool = False,
        skip_enhance: bool = False,
        skip_normalize: bool = False,
        skip_transcribe: bool = False,
        skip_summarize: bool = False,
    ):
        """
        Process a single video through the pipeline.

        Args:
            video_url: URL of the YouTube video
            skip_denoise: Skip denoising step
            skip_enhance: Skip enhancement step
            skip_normalize: Skip normalization step
            skip_transcribe: Skip transcription step
            skip_summarize: Skip summarization step
        """
        logger.info("=" * 80)
        logger.info("Processing single video")
        logger.info("=" * 80)

        try:
            # Download
            logger.info("Downloading video")
            downloaded_file = self.downloader.download_video(video_url)
            logger.info(f"Downloaded: {downloaded_file.name}")

            # Convert to WAV
            wav_file = self.converter.convert_to_wav(downloaded_file)
            logger.info("Converted to WAV")
            current_audio = wav_file

            # Denoise
            if not skip_denoise:
                denoised_file = self._get_denoiser().denoise(wav_file)
                logger.info("Denoised audio")
                current_audio = denoised_file

            # Enhance
            if not skip_enhance:
                enhanced_file = self._get_enhancer().enhance(current_audio)
                logger.info("Enhanced audio")
                current_audio = enhanced_file

            # Normalize
            if not skip_normalize:
                normalized_file = self._get_normalizer().normalize(current_audio)
                logger.info("Normalized audio")
                current_audio = normalized_file
            else:
                # Convert to MP3 if not already
                if current_audio.suffix != ".mp3":
                    current_audio = self.converter.convert_to_mp3(current_audio)

            # Transcribe
            if not skip_transcribe and self.transcription_enabled:
                transcript_file = self._get_transcriber().transcribe(current_audio)
                if transcript_file:
                    logger.info("Transcribed audio")
                else:
                    logger.warning("Transcription failed")

            # Summarize
            if not skip_summarize and self.summarization_enabled and transcript_file:
                summary_file = self._get_summarizer().summarize(transcript_file)
                if summary_file:
                    logger.info("Summarized transcript")

            logger.info("=" * 80)
            logger.info("Single video processing completed!")
            logger.info("=" * 80)
            logger.info(f"Final output: {current_audio}")

        except Exception as e:
            logger.error(f"Error processing video: {e}")
            raise


def main():
    """Main entry point for the pipeline."""
    parser = argparse.ArgumentParser(
        description="YouTube Playlist Processing Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process entire playlist (audio processing only, no transcription)
  python main.py --playlist https://youtube.com/playlist?list=YOUR_PLAYLIST

  # Process specific range
  python main.py --playlist https://youtube.com/playlist?list=YOUR_PLAYLIST --start 1 --end 5

  # Process single video
  python main.py --video https://youtube.com/watch?v=YOUR_VIDEO

  # Enable transcription
  python main.py --playlist https://youtube.com/playlist?list=YOUR_PLAYLIST --enable-transcribe

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

    # Feature enable/disable options
    parser.add_argument("--enable-transcribe", action="store_true", help="Enable transcription")
    parser.add_argument("--enable-summarize", action="store_true", help="Enable summarization")
    parser.add_argument("--enable-enhance", action="store_true", help="Enable audio enhancement")
    parser.add_argument("--disable-denoise", action="store_true", help="Disable noise suppression")

    # Skip options
    parser.add_argument("--skip-download", action="store_true", help="Skip download step")
    parser.add_argument("--skip-denoise", action="store_true", help="Skip denoising step")
    parser.add_argument("--skip-enhance", action="store_true", help="Skip enhancement step")
    parser.add_argument("--skip-normalize", action="store_true", help="Skip normalization step")
    parser.add_argument("--skip-transcribe", action="store_true", help="Skip transcription step")
    parser.add_argument("--skip-summarize", action="store_true", help="Skip summarization step")

    args = parser.parse_args()

    # Initialize pipeline with feature flags
    pipeline = Pipeline(
        noise_suppression_enabled=not args.disable_denoise,
        super_resolution_enabled=args.enable_enhance,
        transcription_enabled=args.enable_transcribe,
        summarization_enabled=args.enable_summarize,
    )

    # Run pipeline
    if args.playlist:
        pipeline.run_full_pipeline(
            playlist_url=args.playlist,
            start=args.start,
            end=args.end,
            skip_download=args.skip_download,
            skip_denoise=args.skip_denoise,
            skip_enhance=args.skip_enhance,
            skip_normalize=args.skip_normalize,
            skip_transcribe=args.skip_transcribe,
            skip_summarize=args.skip_summarize,
        )
    elif args.video:
        pipeline.process_single_video(
            video_url=args.video,
            skip_denoise=args.skip_denoise,
            skip_enhance=args.skip_enhance,
            skip_normalize=args.skip_normalize,
            skip_transcribe=args.skip_transcribe,
            skip_summarize=args.skip_summarize,
        )


if __name__ == "__main__":
    main()
