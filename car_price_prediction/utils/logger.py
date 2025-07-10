"""
Logging utilities for the car price prediction project.
"""

import sys
from pathlib import Path
from typing import Optional

try:
    from loguru import logger
except ImportError:
    # Fallback to standard logging if loguru is not available
    import logging
    
    class LoggerFallback:
        def __init__(self):
            self._logger = logging.getLogger()
            
        def remove(self):
            pass
            
        def add(self, sink, **kwargs):
            if sink == sys.stderr:
                handler = logging.StreamHandler(sys.stderr)
            else:
                handler = logging.FileHandler(sink)
            
            formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(name)s | %(message)s')
            handler.setFormatter(formatter)
            self._logger.addHandler(handler)
            
        def info(self, message):
            self._logger.info(message)
            
        def error(self, message):
            self._logger.error(message)
            
        def warning(self, message):
            self._logger.warning(message)
            
        def debug(self, message):
            self._logger.debug(message)
    
    logger = LoggerFallback()

from ..config import settings


def get_logger(name: str, log_file: Optional[str] = None):
    """
    Get a configured logger instance.
    
    Args:
        name: Logger name
        log_file: Optional log file path
        
    Returns:
        Configured logger instance
    """
    try:
        # Remove default handler
        logger.remove()
        
        # Add console handler
        logger.add(
            sys.stderr,
            format=settings.log_format,
            level=settings.log_level,
            colorize=True
        )
        
        # Add file handler if specified
        if log_file:
            log_path = settings.logs_dir / log_file
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            logger.add(
                log_path,
                format=settings.log_format,
                level=settings.log_level,
                rotation="10 MB",
                retention="1 month",
                compression="zip"
            )
    except:
        # Fallback for standard logging
        pass
    
    return logger


def setup_training_logger(model_type: str, brand: str):
    """
    Setup logger for model training.
    
    Args:
        model_type: Type of model (classification, regression, valuation)
        brand: Car brand
        
    Returns:
        Configured logger
    """
    log_file = f"{model_type}_{brand}_training.log"
    return get_logger(f"{model_type}.{brand}", log_file)
