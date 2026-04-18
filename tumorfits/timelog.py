# SPDX-FileCopyrightText: 2025 Abhinav Mishra
# SPDX-License-Identifier: MIT
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime


def get_logger(name: str = "tumorfit", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(level)

    h = logging.StreamHandler()
    fmt = logging.Formatter(
        fmt="%(asctime)s [pid=%(process)d] %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    h.setFormatter(fmt)
    logger.addHandler(h)
    logger.propagate = False
    return logger


@dataclass
class Timer:
    t0: float = None  # type: ignore

    def __post_init__(self) -> None:
        self.t0 = time.time()

    def s(self) -> float:
        return time.time() - self.t0


def stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")
