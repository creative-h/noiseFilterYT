"""
Utility functions for the YouTube playlist processing pipeline.
"""

import logging
import time
from pathlib import Path
from typing import Optional
from config import LOG_LEVEL, LOG_FORMAT, LOG_FILE, CONSOLE_LOG_LEVEL


def setup_logging():
    """Setup logging with both file and console handlers."""
    # Create logs directory if it doesn't exist
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    
    # Configure root logger
    logger = logging.getLogger()
    logger.setLevel(LOG_LEVEL)
    
    # Remove existing handlers
    logger.handlers.clear()
    
    # File handler
    file_handler = logging.FileHandler(LOG_FILE, encoding='utf-8')
    file_handler.setLevel(LOG_LEVEL)
    file_handler.setFormatter(logging.Formatter(LOG_FORMAT))
    logger.addHandler(file_handler)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(CONSOLE_LOG_LEVEL)
    console_handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
    logger.addHandler(console_handler)
    
    return logger


class ProgressTracker:
    """Track progress of playlist processing."""
    
    def __init__(self, total: int, task_name: str = "Processing"):
        """
        Initialize progress tracker.
        
        Args:
            total: Total number of items to process
            task_name: Name of the task being tracked
        """
        self.total = total
        self.current = 0
        self.task_name = task_name
        self.start_time = time.time()
        self.success_count = 0
        self.skip_count = 0
        self.fail_count = 0
        self.logger = logging.getLogger(__name__)
    
    def update(self, status: str = "in_progress", message: str = ""):
        """
        Update progress.
        
        Args:
            status: Status of current item (in_progress, success, skip, fail)
            message: Optional message to display
        """
        if status == "success":
            self.success_count += 1
        elif status == "skip":
            self.skip_count += 1
        elif status == "fail":
            self.fail_count += 1
        
        self.current += 1
        self._print_progress(message)
    
    def _print_progress(self, message: str = ""):
        """Print progress to console."""
        elapsed = time.time() - self.start_time
        if self.current > 0:
            avg_time = elapsed / self.current
            remaining = (self.total - self.current) * avg_time
            remaining_str = f"{remaining:.0f}s remaining"
        else:
            remaining_str = "calculating..."
        
        progress_msg = f"[{self.current}/{self.total}] {self.task_name}"
        if message:
            progress_msg += f" - {message}"
        
        self.logger.info(progress_msg)
        self.logger.info(f"  Success: {self.success_count}, Skipped: {self.skip_count}, Failed: {self.fail_count}, {remaining_str}")
    
    def summary(self) -> str:
        """Generate summary of processing."""
        elapsed = time.time() - self.start_time
        summary = (
            f"\n{'='*80}\n"
            f"{self.task_name} Summary\n"
            f"{'='*80}\n"
            f"Total items: {self.total}\n"
            f"Successfully processed: {self.success_count}\n"
            f"Skipped: {self.skip_count}\n"
            f"Failed: {self.fail_count}\n"
            f"Total time: {elapsed:.2f}s\n"
            f"{'='*80}\n"
        )
        return summary


def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename by removing/replacing problematic characters.
    
    Args:
        filename: Original filename
        
    Returns:
        Sanitized filename
    """
    # Replace problematic characters
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, '_')
    
    # Remove leading/trailing spaces and dots
    filename = filename.strip('. ')
    
    # Limit length
    if len(filename) > 200:
        filename = filename[:200]
    
    return filename


def check_file_exists(filepath: Path) -> bool:
    """
    Check if file exists and is not empty.
    
    Args:
        filepath: Path to check
        
    Returns:
        True if file exists and is not empty
    """
    if not filepath.exists():
        return False
    return filepath.stat().st_size > 0


def format_duration(seconds: float) -> str:
    """
    Format duration in seconds to human-readable string.
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted duration string
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


def get_audio_duration(filepath: Path) -> Optional[float]:
    """
    Get audio file duration using FFmpeg.
    
    Args:
        filepath: Path to audio file
        
    Returns:
        Duration in seconds, or None if failed
    """
    import subprocess
    
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(filepath)
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return float(result.stdout.strip())
    except (subprocess.CalledProcessError, ValueError):
        return None
