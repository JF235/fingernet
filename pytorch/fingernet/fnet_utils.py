from typing import Optional
import sys
import logging
import time
import os
import torch

DEFAULT_WEIGHTS_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../models/released_version/Model.pth")
)
DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Setup a custom logger for the package
def get_fingernet_logger(name: str, level: Optional[int] = None) -> logging.Logger:
    """Return a logger that prefixes messages with '[Fingernet]' and can set level.

    Usage:
      logger = get_fingernet_logger(__name__, level=logging.INFO)
      logger.info('message')
    """
    logger = logging.getLogger(name)
    if level is not None:
        logger.setLevel(level)

    # add one stream handler only if none exists
    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        #fmt = "[Fingernet] %(levelname)s: %(message)s (at %(filename)s)"
        fmt = "[Fingernet] %(levelname)s: %(message)s"
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter(fmt))
        h.setLevel(logging.NOTSET)
        logger.addHandler(h)

    logger.propagate = False
    return logger

class FnetTimer:
    """ Timer to use with util and logger in DEBUG level """
    def __init__(self, name: str, logger: logging.Logger):
        self.name = name
        self.logger = logger

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.end = time.perf_counter()
        self.interval = self.end - self.start
        self.logger.debug(f"{self.name} - {self.interval:.3f} s")