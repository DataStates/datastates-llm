import logging
import sys

def get_logger(logger_name) -> logging.Logger:
    logging_level = logging.INFO
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] [%(filename)s:%(lineno)d:%(funcName)s] %(message)s")
    channel = logging.StreamHandler(stream=sys.stdout)
    channel.setLevel(logging_level)
    channel.setFormatter(formatter)
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging_level)
    logger.addHandler(channel)
    return logger