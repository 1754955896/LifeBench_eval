"""
Logging utilities.
"""
import logging
from pathlib import Path
from typing import Optional


def setup_logger(log_file: Optional[Path] = None, level: int = logging.INFO) -> logging.Logger:
    """
    Setup logger with file and console handlers.

    Args:
        log_file: Path to log file (optional)
        level: Logging level

    Returns:
        Logger instance
    """
    logger = logging.getLogger("lifebench_eval")
    logger.setLevel(level)
    logger.handlers = []

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    # Fix for Windows ASCII encoding issue with Chinese characters
    try:
        import sys
        if sys.platform == 'win32':
            import io
            console_handler.stream = io.TextIOWrapper(
                console_handler.stream.buffer if hasattr(console_handler.stream, 'buffer') else console_handler.stream,
                encoding='utf-8',
                errors='replace'
            )
    except Exception:
        pass  # Fallback to default encoding if this fails
    logger.addHandler(console_handler)

    return logger


def get_console():
    """Get console for rich output."""
    try:
        from rich.console import Console

        return Console()
    except ImportError:
        return None
