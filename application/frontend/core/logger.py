import logging
import sys
from loguru import logger
import streamlit

from core.logger_config import (
    LOG_LEVEL, LOG_DIR, LOG_FORMAT,
    LOG_RETENTION_DAYS, LOG_ROTATION_SIZE
)


class InterceptHandler(logging.Handler):
    loglevel_mapping = {
        50: 'CRITICAL',
        40: 'ERROR',
        30: 'WARNING',
        20: 'INFO',
        10: 'DEBUG',
        0: 'NOTSET',
    }

    def emit(self, record):
        try:
            level = logger.level(record.levelname).name
        except AttributeError:
            level = self.loglevel_mapping[record.levelno]

        frame, depth = logging.currentframe(), 2
        while frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        log = logger.bind(request_id='app')
        log.opt(
            depth=depth,
            exception=record.exc_info
        ).log(level, record.getMessage())


class CustomizeLogger:

    @classmethod
    def make_logger(cls, logger_name: str):
        logger.remove()

        logger.add(
            sys.stdout,
            format=LOG_FORMAT,
            level=LOG_LEVEL,
            colorize=True
        )

        logger.add(
            LOG_DIR / logger_name / "app.log",
            format=LOG_FORMAT,
            level=LOG_LEVEL,
            rotation=LOG_ROTATION_SIZE,
            retention=LOG_RETENTION_DAYS,
            compression="zip",
            encoding="utf-8"
        )

        logger.add(
            LOG_DIR / logger_name / "errors.log",
            format=LOG_FORMAT,
            level="ERROR",
            rotation="1 week",
            retention=LOG_RETENTION_DAYS,
            compression="zip",
            encoding="utf-8",
            backtrace=True,
            diagnose=True
        )

        logging.basicConfig(handlers=[InterceptHandler()], level=0)
        for lg in streamlit.logger._loggers:
            logging.getLogger(lg).handlers = [InterceptHandler()]

        return logger.bind(request_id=None, method=None)
