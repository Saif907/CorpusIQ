import logging
import logging.config
import json
from app import settings

class JSONFormatter(logging.Formatter):
    """
    Formats logs records in JSON strings for machine readability in production.
    """
    def format(self,record: logging.LogRecord)->str:
        log_data = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "func_name": record.funcName,
            "line_number": record.lineno,
        }
        
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
            
        return json.dumps(log_data)

LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "dev": {
            "format": "%(asctime)s [%(levelname)s] %(name)s (%(filename)s:%(lineno)d): %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
        "json": {
            "()": JSONFormatter,
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "dev" if settings.ENV == "development" else "json",
            "level": "DEBUG",
        },
    },
    "loggers": {
        # Root logger definition
        "": {
            "handlers": ["console"],
            "level": "DEBUG" if settings.ENV == "development" else "INFO",
        },
        # Silence verbose third-party logging
        "numba": {
            "level": "WARNING",
        },
        "httpcore": {
            "level": "INFO",
        },
        "httpx": {
            "level": "INFO",
        },
        "openai": {
            "level": "INFO",
        },
        "yfinance": {
            "level": "WARNING",
        },
        # Intercept and unify Uvicorn logs
        "uvicorn": {
            "handlers": ["console"],
            "level": "INFO",
            "propagate": False,
        },
        "uvicorn.access": {
            "handlers": ["console"],
            "level": "INFO",
            "propagate": False,
        },
    },
}

def setup_logging():
    """
    Initializes the centralized logging configuration.
    Call this once in your main.py entrypoint during app startup.
    """
    logging.config.dictConfig(LOGGING_CONFIG)