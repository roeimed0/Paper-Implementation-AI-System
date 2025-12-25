"""
Simple Logging System

Structured logging with rich console output.

Key Learning: Professional logging for debugging LLM applications.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from rich.logging import RichHandler
from rich.console import Console


def setup_logging(level: str = "INFO", log_dir: Path = None) -> None:
    """
    Setup logging for the application.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR)
        log_dir: Optional directory for log files
    """
    log_level = getattr(logging, level.upper(), logging.INFO)

    # Rich console handler
    console_handler = RichHandler(
        rich_tracebacks=True,
        console=Console(stderr=True),
        show_time=True,
        show_path=False,
    )
    console_handler.setLevel(log_level)

    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    root_logger.handlers.clear()
    root_logger.addHandler(console_handler)

    # File handler (optional)
    if log_dir:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"app_{datetime.now().strftime('%Y%m%d')}.log"

        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(log_level)
        file_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        )
        root_logger.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    """Get a logger for the specified module."""
    return logging.getLogger(name)


# Initialize on import
setup_logging()
