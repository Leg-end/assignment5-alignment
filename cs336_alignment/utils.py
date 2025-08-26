import logging
import os
from glob import glob


def set_logger(verbose="info", log_path="./stdout.txt"):
    level = getattr(logging, verbose.upper(), None)

    handler1 = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(levelname)s - %(asctime)s - %(message)s",
        datefmt="%H:%M:%S"
    )
    handler1.setFormatter(formatter)
    logger = logging.getLogger()
    logger.addHandler(handler1)
    logger.setLevel(level)
    if log_path is not None:
        if os.path.exists(log_path):
            name, suffix = os.path.splitext(log_path)
            n = len(glob(f"{name}*{suffix}"))
            log_path = f"{name}_{n}{suffix}"
        handler2 = logging.FileHandler(log_path)
        handler2.setFormatter(formatter)
        logger.addHandler(handler2)