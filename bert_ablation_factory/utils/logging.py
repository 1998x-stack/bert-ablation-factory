from __future__ import annotations
from loguru import logger
import sys
from pathlib import Path


def setup_logger(log_file: Path | None = None) -> None:
    """配置 Loguru 日志输出。"""
    logger.remove()
    logger.add(sys.stderr, level="INFO",
               format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
                      "<level>{level: <8}</level> | "
                      "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
                      "<level>{message}</level>")
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        logger.add(str(log_file), level="INFO", rotation="10 MB", retention="10 files")
