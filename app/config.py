import os
import logging


def get_logging_level():
    return os.getenv("LOGGING_LEVEL", "INFO")


def configure_logging():
    logging_level = get_logging_level().upper()
    numeric_level = getattr(logging, logging_level, None)

    if not isinstance(numeric_level, int):
        raise Exception(f"Invalid log level: {numeric_level}")

    logging.basicConfig(
        level=numeric_level,
        datefmt="%Y-%m-%d %H:%M:%S",
        format="[%(asctime)s.%(msecs)03d] [%(levelname)s] [%(module)s] #%(funcName)s @%(lineno)d: %(message)s",
    )
    # [%(process)s]
    logging.info(f"Logging level: {logging_level}")
