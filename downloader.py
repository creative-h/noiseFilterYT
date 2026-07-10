"""
Download YouTube playlists using yt-dlp.
"""

import yt_dlp
import logging
import os
from pathlib import Path
from typing import Optional
from config import (
    RAW_DIR,
    YDLP_FORMAT,
    YDLP_POSTPROCESSORS,
    LOG_LEVEL,
    LOG_FORMAT,
    LOG_FILE,
)

logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT, filename=LOG_FILE)
logger = logging.getLogger(__name__)


class PlaylistDownloader:
    """Download YouTube playlists and individual videos."""

    def __init__(
        self,
        output_dir: Optional[Path] = None,
        format: str = YDLP_FORMAT,
        postprocessors: Optional[list] = None,
        cookies_file: Optional[str] = None,
    ):
        """
        Initialize the downloader.

        Args:
            output_dir: Directory to save downloaded files
            format: yt-dlp format string
            postprocessors: List of post-processing options
            cookies_file: Path to cookies.txt file for authentication
        """
        self.output_dir = output_dir or RAW_DIR
        self.format = format
        self.postprocessors = postprocessors or YDLP_POSTPROCESSORS
        self.cookies_file = cookies_file or os.getenv("YDLP_COOKIES_FILE")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def download_playlist(
        self,
        playlist_url: str,
        start: Optional[int] = None,
        end: Optional[int] = None,
    ) -> list[Path]:
        """
        Download an entire YouTube playlist.

        Args:
            playlist_url: URL of the YouTube playlist
            start: Start index (1-based)
            end: End index (1-based)

        Returns:
            List of downloaded file paths
        """
        logger.info(f"Starting playlist download: {playlist_url}")

        ydl_opts = {
            "format": self.format,
            "outtmpl": str(self.output_dir / "%(playlist_index)03d-%(title)s.%(ext)s"),
            "postprocessors": self.postprocessors,
            "quiet": False,
            "no_warnings": False,
            "progress": True,
        }

        # Add cookies if available
        if self.cookies_file and Path(self.cookies_file).exists():
            ydl_opts["cookiefile"] = self.cookies_file
            logger.info(f"Using cookies from: {self.cookies_file}")

        if start is not None or end is not None:
            ydl_opts["playliststart"] = start or 1
            ydl_opts["playlistend"] = end

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(playlist_url, download=True)
                
                # Get list of downloaded files
                downloaded_files = []
                if "entries" in info:
                    for entry in info["entries"]:
                        if entry:
                            filename = ydl.prepare_filename(entry)
                            # yt-dlp may have changed extension during post-processing
                            downloaded_files.append(Path(filename).with_suffix(".mp3"))
                
                logger.info(f"Downloaded {len(downloaded_files)} files")
                return downloaded_files

        except Exception as e:
            logger.error(f"Error downloading playlist: {e}")
            raise

    def download_video(self, video_url: str) -> Path:
        """
        Download a single video.

        Args:
            video_url: URL of the YouTube video

        Returns:
            Path to the downloaded file
        """
        logger.info(f"Starting video download: {video_url}")

        ydl_opts = {
            "format": self.format,
            "outtmpl": str(self.output_dir / "%(title)s.%(ext)s"),
            "postprocessors": self.postprocessors,
            "quiet": False,
            "no_warnings": False,
        }

        # Add cookies if available
        if self.cookies_file and Path(self.cookies_file).exists():
            ydl_opts["cookiefile"] = self.cookies_file
            logger.info(f"Using cookies from: {self.cookies_file}")

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(video_url, download=True)
                filename = ydl.prepare_filename(info)
                logger.info(f"Downloaded: {filename}")
                return Path(filename).with_suffix(".mp3")

        except Exception as e:
            logger.error(f"Error downloading video: {e}")
            raise

    def get_playlist_info(self, playlist_url: str) -> dict:
        """
        Get information about a playlist without downloading.

        Args:
            playlist_url: URL of the YouTube playlist

        Returns:
            Dictionary with playlist information
        """
        ydl_opts = {
            "quiet": True,
            "extract_flat": True,
        }

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(playlist_url, download=False)
                return {
                    "title": info.get("title"),
                    "id": info.get("id"),
                    "video_count": len(info.get("entries", [])),
                    "entries": [
                        {
                            "title": entry.get("title"),
                            "id": entry.get("id"),
                            "url": entry.get("url"),
                            "duration": entry.get("duration"),
                        }
                        for entry in info.get("entries", [])
                    ],
                }
        except Exception as e:
            logger.error(f"Error getting playlist info: {e}")
            raise
